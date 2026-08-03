#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
910B3 真机 MatMul 驱动脚本 (triton-ascend + msprof)
================================================================
依赖: torch_npu + triton-ascend + CANN(Toolkit/Kernels 910b), 必须在 910B3 服务器运行
kernel: 同目录 triton_kernel.py 的 matmul_kernel (C = A @ B, 512x512x512)

每次终端先准备环境:
    conda activate triton-npu
    source /usr/local/Ascend/ascend-toolkit/set_env.sh

【真机命令 1/3 — msprof op 单算子调优】 (真实硬件时序, 单算子粒度):
    全指标版 (产出所有 csv):
    msprof op --kernel-name=matmul_kernel \
        --aic-metrics=PipeUtilization,ResourceConflictRatio,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache \
        --output=./board_prof python3 test_matmul.py
    精简版 (只产 PipeUtilization + ResourceConflictRatio):
    msprof op --kernel-name=matmul_kernel \
        --aic-metrics=PipeUtilization,ResourceConflictRatio \
        --output=./board_prof python3 test_matmul.py
    产物: ./board_prof/OPPROF_xxx/
        OpBasicInfo.csv              算子基础信息 (端到端耗时, 真实延迟)
        PipeUtilization.csv          计算/搬运单元耗时占比
        ArithmeticUtilization.csv    Cube/Vector 指令周期占比
        ResourceConflictRatio.csv    UB bank 冲突率
        Memory.csv / MemoryL0.csv / MemoryUB.csv   各级读写带宽率
        L2Cache.csv                  L2 命中率
        visualize_data.bin           MindStudio Insight 可视化
        dump/                        原始数据 + aicore_binary.o
    注意: 单算子模式不产出 op_summary_*.csv, 那是通用 msprof 的产物

【真机命令 2/3 — msprof 通用任务级】 (真实硬件, 算子汇总, 产出 op_summary):
    msprof --output=./task_prof --application="python3 test_matmul.py"
    产物: ./task_prof/PROF_xxx/
        host/data/                          Host 侧原始数据 (无需关注)
        device_{id}/data/                   Device 侧原始数据 (无需关注)
        mindstudio_profiler_output/         ★ 性能数据分析推荐目录
            op_summary_*.csv     AI Core/AI CPU 算子数据, 关键列:
                                 Task Duration(us) / Task Type(AI_CORE|AI_VECTOR_CORE|AI_CPU)
                                 / aicore_time(us) / aiv_time(us) / total_cycles
                                 / Block Dim / Task Start Time(us) / Stream ID / Device ID
            op_statistic_*.csv   算子调用次数/总耗时统计
            msprof_*.json        timeline 主表 (Chrome tracing 打开)
            task_time_*.csv      Task Scheduler 调度信息
            api_statistic_*.csv  CANN 层 API 耗时
            fusion_op_*.csv      算子融合信息
    注意: 通用 msprof 不采集 Python 调用栈 / PyTorch 框架层数据; 具体列名以
          服务器上 head -1 mindstudio_profiler_output/op_summary_*.csv 为准

【仿真命令 3/3 — msprof op simulator】 (CPU 指令级仿真, 不占 NPU):
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
    msprof op simulator --kernel-name=matmul_kernel --soc-version=Ascend910B3 \
        --output=./sim_prof python3 test_matmul.py
    产物: ./sim_prof/OPPROF_xxx/
        simulator/trace.json              全核汇总指令流水 (Chrome tracing)
        simulator/core*.veccore*/         每核一目录 (含 cubecore)
            *_instr_exe.csv   ★ 指令级时序 (pipe/cycles/running_time)
            *_code_exe.csv    代码行耗时
        dump/aicore_binary.o  算子二进制
    注: 只有本命令产出 instr_exe.csv, 是真机 msprof/msprof op 没有的

【方式 4/4 — HIVM IR dump】 (编译中间产物, 静态结构字段来源, 不占 NPU)
    机制: TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 → ~/.triton/dump/<hash>/
         产 kernel.ttir.mlir + kernel.ttadapter.mlir + host 代码(.h/.cxx)。
    ★ 关键事实 (本机 3.2.0 已确认):
        - kernel.npuir.mlir 不生成: 同步 pass 改名 (hivm-inject-sync →
          hivm-graph-sync-solver) 跟着 CANN 的 bishengir 走, 与 triton-ascend
          版本无关 (3.2.0 也中招)。
        - TRITON_DEBUG 有时命中缓存 → 只产 .h/.cxx (host 启动代码, 分析用不到),
          连 ttir/ttadapter 都不产。
        → 不再依赖 TRITON_DEBUG dump, 直接手动对 ttadapter.mlir 跑 bishengir
          打印 HIVM (下方标准流程)。

    拿真实 HIVM 的标准流程 (按序执行; A/B/C 都是给 D 准备输入):
        # A. 找已有 ttadapter.mlir (之前拷回 ir_dump/ 过可直接复用, 跳过 B)
        find . -name 'kernel.ttadapter.mlir' -o -name 'kernel.ttir.mlir' 2>/dev/null

        # B. 没有就强制重编译 (nuke 整个 ~/.triton 才保险; 单清 dump 会命中缓存只产 .h/.cxx)
        rm -rf ~/.triton
        export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
        python3 test_matmul.py 2>&1 | tee run_debug.txt
        find ~/.triton -type f | head -30     # 预期: kernel.ttir.mlir + kernel.ttadapter.mlir

        # C. 确认 bishengir-compile 在 PATH (不在就 find 定位)
        which bishengir-compile || find /usr/local/Ascend -name 'bishengir-compile' 2>/dev/null

        # D. ★核心: 对 ttadapter.mlir 直接跑官方命令打印 HIVM (不依赖 TRITON_DEBUG)
        cd <A 或 B 里 ttadapter 所在目录>
        bishengir-compile --target=Ascend910B3 --enable-auto-multi-buffer=True \
          --enable-auto-bind-sub-block=True --enable-hfusion-compile=true \
          --enable-hivm-compile=true --enable-triton-kernel-compile=true \
          --bishengir-print-ir-after=hivm-inject-sync kernel.ttadapter.mlir \
          -o /tmp/k.o 2>&1 | tee hivm_try.txt
        grep -c 'hivm.hir' hivm_try.txt      # 预期 >0 即成功; 0 → 换 pass hivm-graph-sync-solver 重跑
        # 仍 0: 加 --enable-hivm-graph-sync-solver=true (新版同步求解器默认 false, 不开不产 HIVM)
        # 打印 flag 必须是 --bishengir-print-ir-after=<pass名> (不是 --mlir-print-ir-after-all)
        # hivm_try.txt 即含 hivm.hir 的完整 IR, 拷回本目录直接喂 hivmir_analyzer

    方案 A (兜底, D 拿不到 HIVM 时): 真机 ttir.mlir 喂现有 ttir_to_hivm() → 直接出结构 JSON
        python3 -c "
        from analyzers.ttir_to_hivm import ttir_to_hivm
        hivm_text, ops = ttir_to_hivm(open('kernel.ttir.mlir').read(), 'matmul_kernel')
        open('kernel_hivm_fallback.mlir','w').write(hivm_text)
        print('ops:', len(ops))
        "
        ⚠️ 局限: matmul 的 cube 通路 (L0A/L0B/L0C) 是近似的; 但依赖边/buffer大小/dtype/op类型
           (融合/分块/访存判定核心) 是对的

    怎么看 D 的产物 hivm_try.txt (纯查看, 不需要编译):
        grep -c 'hivm.hir' hivm_try.txt                      # 指令总数, 应 > 0
        grep -n 'memref.alloc'  hivm_try.txt | head          # buffer: size+region+dtype
        grep -E 'hivm.hir.matmul|hivm.hir.load' hivm_try.txt | head   # 指令样例
    真实格式 = hivmir_analyzer 的 Format A, 样例:
        %alloc = memref.alloc() : memref<256x256xf32, #hivm.address_space<ub>>
        hivm.hir.load ins(%arg0 : memref<...>) outs(%alloc : memref<...>)
        hivm.hir.matmul ins(%A, %B : ...) outs(%C : ...) {a_transpose, block_sizes=[16,16,16]}

    怎么解析并映射到 29 字段 (纯 Python, 可在 WSL2/任意机器跑):
        python3 -c "
        from pathlib import Path
        from analyzers.hivmir_analyzer import HIVMIRAnalyzer
        a = HIVMIRAnalyzer()
        rep = a.analyze_file(Path('hivm_try.txt'))
        import json; print(json.dumps(a.to_dict(rep), ensure_ascii=False, indent=2))
        "
        得到: op_id/op_type/instruction/dst/src/src2/size_kb/memory_region/
              dependencies(RAW/WAR/WAW)/buffers(producers/consumers)/dtype/attrs(block_sizes)
        ⚠️ timing 字段 (duration_ns/start/end/带宽) 仍是 "待补充" — 由 方式1/2/3 的
           msprof 数据填, dsl_merger 按 op_id 对齐。

    ⚠️ hivmir_analyzer 对真实 dump 有 3 个待补点 (见 analyzers/hivmir_analyzer.py):
        1) 2D memref alloc (memref<256x256xf32,...>) 不解析 → size_kb 变 0, 需加 2D 分支
        2) address_space 值 cbuf/ca/cb/cc (=L1/L0A/L0B/L0C) 未映射 → L0 通路 region 判错
        3) sync/barrier op (hivm.barrier, 非 hivm.hir.*) 被跳过 → 丢失依赖/串行证据

常见提示说明:
    - "not selected via --kernel-name" 是 reference 的 torch.matmul 被过滤, 正常;
      若担心 matmul_kernel 没采到, 检查 OPPROF_xxx/simulator/ 是否有 instr_exe.csv
    - "terminate called after throwing 'std::bad_weak_ptr'" 是仿真器退出时 teardown
      崩溃, 通常 OPPROF 数据已写盘; 已默认关闭正确性校验降低触发概率
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
M, N, K = 512, 512, 512
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
