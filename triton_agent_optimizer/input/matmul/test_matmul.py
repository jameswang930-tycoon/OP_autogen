#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
910B3 真机 MatMul 驱动 + 优化数据采集流程
================================================================
依赖: torch_npu + triton-ascend + CANN, 必须在 910B3 服务器运行
kernel: 同目录 triton_kernel.py 的 matmul_kernel

★ 主流程 (一键, 推荐): bash analyzers/run_server_flow.sh
   = msprof op + 通用 msprof → diagnosis.json (roofline 核心诊断)
   自动: 采集 → 解析 → 整合 → 字段校验
   产物: input/matmul/e2e_run/06_diagnosis/diagnosis.json

─────────────────────────────────────────────
0. 每次运行前清理旧产物 (主流程脚本已自动做):
   rm -rf e2e_run ~/.triton
─────────────────────────────────────────────

【真机命令 1 — msprof op (★主源, 默认全量 8 CSV)】
   msprof op --kernel-name=matmul_kernel --warm-up=10 --output=./board_prof python3 test_matmul.py
   注意: 不要指定 --aic-metrics (默认全量; 指定会限制/报错)
   产物: OpBasicInfo/PipeUtilization/ArithmeticUtilization/Memory/MemoryL0/MemoryUB/L2Cache/ResourceConflictRatio
   → board.json (真实带宽/L2/cube/引擎利用率)

【真机命令 2 — msprof 通用 (任务级, op_summary)】
   msprof --output=./task_prof --application="python3 test_matmul.py" --ai-core=on
   注意: 8.5.1 不认 --aic-metrics (退出255), 用 --ai-core=on 拿基础 op_summary
   产物: op_summary/op_statistic/task_time/api_statistic/l2_cache
   → task.json (每kernel耗时/核数/多kernel/launch/L2)

【仿真命令 3 — msprof op simulator (可选, 复杂场景弃用)】
   export LD_LIBRARY_PATH=.../tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
   msprof op simulator --kernel-name=matmul_kernel --soc-version=Ascend910B3 --output=./sim_prof python3 test_matmul.py
   注意: 大负载会卡死 (指令级仿真慢); 只适合小尺寸看指令结构 (建议 MATMUL_M/N/K=64)

【可选 — HIVM (多算子融合时用)】
   多算子 (op_summary 里多个不同 kernel) → 需看内部结构/依赖做融合:
   1. 编译 + 打印 HIVM:
      rm -rf ~/.triton
      export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
      python3 test_matmul.py
      cp ~/.triton/dump/*/kernel.*.mlir ./
      bishengir-compile --target=Ascend910B3 --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True --enable-hfusion-compile=true --enable-hivm-compile=true --enable-triton-kernel-compile=true --bishengir-print-ir-after=hivm-inject-sync kernel.ttadapter.mlir -o /tmp/k.o 2>&1 | tee hivm_try.txt
      (pass 名 hivm-inject-sync 或 hivm-graph-sync-solver 都试)
   2. 过滤成融合专用视图 (删无关, 保留 op+同步+依赖):
      python3 analyzers/filter_hivm_for_fusion.py hivm_try.txt --out hivm_fusion_view.txt
   3. LLM 读 hivm_fusion_view.txt: 找 RAW 链相邻逐元素 op → 融合候选; WAR → 换 buffer
   注意: npuir.mlir 不生成 (pass 改名跟 CANN bishengir 走) → 手动 D 打印是正路

【诊断输出 diagnosis.json (roofline 核心)】
   kernel_summary  每kernel耗时/核数/多kernel/launch/L2
   roofline        memory/compute/latency/balanced 判类型
   engine_util     cube/vec/mte1/2/3/scalar/fixpipe 占比
   transfer_paths  每通路真实带宽 (GM/L1/L2/UB/L0)
   memory_issues   L2命中 + UB冲突
   compute         cube/vec fops
   bottlenecks     每 Tier 处方化 hint
   注意: 只有 kernel 级, 无 per-op (内部 load/matmul/store 拆不开)
     Tier2 内部融合需临时跑 hivm 看依赖

【尺寸可调】 MATMUL_M/N/K env 覆盖 (默认 512^3; simulator 建议 64^3)

常见提示:
   - "not selected via --kernel-name" 是 reference 的 torch.matmul 被过滤, 正常
   - "std::bad_weak_ptr" 是仿真器退出时 teardown 崩溃, 通常数据已写盘
================================================================
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl

from triton_kernel import matmul_kernel

# ---------- 配置 (与 input/matmul/config.json 对齐) ----------
# 注意: 不传 num_warps/num_stages — triton-ascend 后端自动管理 tiling/流水,
#       传了会报 "please do not tune args ['num_warps','num_stages']"
# 尺寸可用环境变量覆盖 (simulator 指令级仿真, 大尺寸会极慢/卡):
#   MATMUL_M=64 MATMUL_N=64 MATMUL_K=64 python3 test_matmul.py
M  = int(os.environ.get("MATMUL_M", 512))
N  = int(os.environ.get("MATMUL_N", 512))
K  = int(os.environ.get("MATMUL_K", 512))
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
DTYPE = torch.float32

torch.npu.set_device(0)   # simulator 只支持 device 0


def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    # fp32 dot 值域限制 [-5, 5], 用 rand-0.5 控制在 ±0.5
    a = torch.rand(M, K, dtype=DTYPE, device="npu") - 0.5
    b = torch.rand(K, N, dtype=DTYPE, device="npu") - 0.5
    c = torch.empty(M, N, dtype=DTYPE, device="npu")

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m * grid_n,)   # 必须用 tuple: triton-ascend 对 int grid 调 len() 会报 "int has no len()"
    print(f"[info] launch grid={grid}  A({M}x{K}) @ B({K}x{N})  block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        # ⚠️ 不要传 num_warps/num_stages: triton-ascend 禁止 tune 这两个参数
    )
    torch.npu.synchronize()
    print("[info] kernel launched & synced OK")

    # 正确性校验默认关闭(仿真下多跑一个 torch.matmul 会触发退出时 bad_weak_ptr 且拖慢仿真)。
    # 真机上想校验: MATMUL_VERIFY=1 python3 test_matmul.py
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        try:
            ref = torch.matmul(a, b)
            diff = (c - ref).abs().max().item()
            print(f"[info] result check: {'PASS' if diff < 0.05 else 'CHECK'}  max|C-ref| = {diff:.5f}")
        except Exception as e:
            print(f"[warn] result check skipped: {e}")


if __name__ == "__main__":
    main()
