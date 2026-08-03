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

【命令 1/2 — 真机采集】 (真实硬件时序, 含 CUBE 流水):
    msprof op --kernel-name=matmul_kernel \
        --aic-metrics=PipeUtilization,ResourceConflictRatio \
        --output=./board_prof python3 test_matmul.py

【命令 2/2 — CPU 仿真】 (指令级流水, 无真实硬件, CAModel 建议单核):
    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
    msprof op simulator --kernel-name=matmul_kernel --soc-version=Ascend910B3 \
        --output=./sim_prof python3 test_matmul.py

(可选) 让 trace 关联源码行 / 同时 dump HIVM IR:
    export TRITON_DISABLE_LINE_INFO=false          # 消除 "kernel missed debug_line" 警告
    export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1   # -> ~/.triton/dump/<hash>/kernel.npuir.mlir

产物:
    <output>/OPPROF_xxx/simulator/core*.veccore*/instr_exe.csv + trace.json
    (真机模式下 OPPROF_xxx 顶层还有 op_summary_*.csv / PipeUtilization.csv)

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
