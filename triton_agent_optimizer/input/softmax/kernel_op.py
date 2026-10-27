#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Softmax (行级数值稳定) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y[row, :] = softmax(X[row, :])  (每行一个 program, 行内 max→sub→exp→sum→div)
#  ★数值稳定: 先减行 max 再 exp (防溢出), 与 torch.softmax(dim=-1) 一致
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
#  形状: X[M, N] → Y[M, N] (每行 N 个元素 softmax)
# ═══════════════════════════════════════════════════════════════════════════════
M = int(os.environ.get("SOFTMAX_M", 2048))       # 行数
N = int(os.environ.get("SOFTMAX_N", 2048))       # 每行元素数
DTYPE = torch.float32
BLOCK_N = 1 << (N - 1).bit_length()              # ≥N 的 2 幂 (tl.arange 要求 2 幂), 整行一块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# Softmax: 每行一个 program, 行内 max→sub→exp→sum→div (数值稳定)
@triton.jit
def softmax_kernel(x_ptr, y_ptr,
                   M, N,
                   BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)                    # 行 max (数值稳定)
    e = tl.exp(x - x_max)                        # masked lane: exp(-inf)=0
    s = tl.sum(e, axis=0)                        # 行和
    tl.store(y_ptr + row * N + offs, e / s, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(M, N, dtype=DTYPE, device=npu)) * 0.1
    y = torch.empty(M, N, dtype=DTYPE, device=npu)

    grid = (M,)                                  # 每行一个 program
    print(f"[info] softmax M={M} N={N} dtype={DTYPE} grid={grid[0]} block_N={BLOCK_N}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        softmax_kernel[grid](x, y, M, N, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("[info] softmax launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        ref = torch.softmax(x, dim=-1)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
