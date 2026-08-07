#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — LayerNorm (逐行归一化 + 逐元素缩放) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: Y[row, :] = gamma * (X[row,:] - mean) / sqrt(var + eps) + beta
#    mean = mean(X[row, :]);  var = mean((X - mean)^2)
#  (一行一个 program; 两遍扫: ① 算 mean ② 算 var → 归一化缩放)
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
#  形状: X[M, N], gamma[N], beta[N] → Y[M, N]
# ═══════════════════════════════════════════════════════════════════════════════
M = int(os.environ.get("LAYERNORM_M", 2048))   # 行数
N = int(os.environ.get("LAYERNORM_N", 2048))   # 每行元素数 (特征维)
DTYPE = torch.float32
EPS = 1e-5
BLOCK_N = 1 << (N - 1).bit_length()            # ≥N 的 2 幂 (tl.arange 要求 2 幂), 整行一块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

# LayerNorm: 每行一个 program. ★两遍扫: ① mean ② var (留 Tier1/4 单遍合并空间)
@triton.jit
def layernorm_kernel(x_ptr, gamma_ptr, beta_ptr, y_ptr,
                     M, N, eps,
                     BLOCK_N: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # ① 第一遍扫: 算 mean
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / N            # fp32 归约

    # ② 第二遍扫: 算 var (★又 load 一次 X — 冗余, Tier1 可 online 单遍省这次 load)
    x2 = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    diff = x2 - mean
    var = tl.sum(diff * diff, axis=0) / N

    # 归一化 + 缩放 + 偏置
    rstd = tl.math.rsqrt(var + eps)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    y = gamma * (x2 - mean) * rstd + beta   # 复用 x2 (第二遍 load 的)
    tl.store(y_ptr + row * N + offs, y, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(M, N, dtype=DTYPE, device=npu)) * 0.1
    gamma = (torch.randn(N, dtype=DTYPE, device=npu)) * 0.1
    beta = (torch.randn(N, dtype=DTYPE, device=npu)) * 0.1
    y = torch.empty(M, N, dtype=DTYPE, device=npu)

    grid = (M,)                               # 每行一个 program
    print(f"[info] LayerNorm M={M} N={N} dtype={DTYPE} grid={grid[0]} block_N={BLOCK_N}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        layernorm_kernel[grid](x, gamma, beta, y, M, N, EPS, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("[info] LayerNorm launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        ref = torch.nn.functional.layer_norm(x, (N,), gamma, beta, EPS)
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-3 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
