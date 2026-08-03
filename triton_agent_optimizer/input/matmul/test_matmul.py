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
    产物: ./task_prof/PROF_xxx/mindstudio_profiler_output/
        op_summary_*.csv     AI Core/AI CPU 算子数据 (按 Task Duration 排序找热点)
        op_statistic_*.csv   算子调用次数/总耗时统计
        msprof_*.json        timeline 主表
        task_time_*.csv      Task Scheduler 调度信息
        api_statistic_*.csv  CANN 层 API 耗时
        fusion_op_*.csv      算子融合信息
    注意: 通用 msprof 不采集 Python 调用栈 / PyTorch 框架层数据

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

(可选) 让 trace 关联源码行 / 同时 dump HIVM IR:
    export TRITON_DISABLE_LINE_INFO=false          # 消除 "kernel missed debug_line" 警告
    export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1   # -> ~/.triton/dump/<hash>/kernel.npuir.mlir

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
