#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — RMSNorm + Residual (行级归一化) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y[row,:] = (X[row,:] + Residual[row,:]) / sqrt(mean((X+Res)^2) + eps) * W[:]
#  (残差相加 → 行归约(mean x²) → rsqrt → 乘 weight; 每行一个 program)
#
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块
#  形状: X[M,N], Residual[M,N], W[N] → Y[M,N]
# ═══════════════════════════════════════════════════════════════════════════════
M = int(os.environ.get("RMSR_M", 2048))       # 行数 (batch×seq)
N = int(os.environ.get("RMSR_N", 4096))       # 每行元素数 (特征维, Llama hidden)
DTYPE = torch.float32
EPS = 1e-6
BLOCK_N = 1 << (N - 1).bit_length()           # ≥N 的 2 幂 (tl.arange 要求 2 幂), 整行一块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# RMSNorm + Residual: 每行一个 program, 残差相加 → 行内归约 x² → rsqrt → 乘 weight
@triton.jit
def add_rms_norm_kernel(x_ptr, residual_ptr, w_ptr, y_ptr,
                        M, N, eps,
                        BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    r = tl.load(residual_ptr + row * N + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    c = x + r                                  # 残差相加
    ms = tl.sum(c * c, axis=0) / N             # mean(c²)
    rstd = tl.math.rsqrt(ms + eps)             # 1/sqrt(mean+eps)
    tl.store(y_ptr + row * N + offs, c * rstd * w, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(M, N, dtype=DTYPE, device=npu)) * 0.1
    residual = (torch.randn(M, N, dtype=DTYPE, device=npu)) * 0.1
    w = (torch.randn(N, dtype=DTYPE, device=npu)) * 0.1
    y = torch.empty(M, N, dtype=DTYPE, device=npu)

    grid = (M,)                                # 每行一个 program
    print(f"[info] rms_norm_residual M={M} N={N} dtype={DTYPE} grid={grid[0]} block_N={BLOCK_N}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        add_rms_norm_kernel[grid](x, residual, w, y, M, N, EPS, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("[info] rms_norm_residual launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        c = x + residual
        ref = c / torch.sqrt((c * c).mean(-1, keepdim=True) + EPS) * w
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-3 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
