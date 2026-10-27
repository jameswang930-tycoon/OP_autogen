#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Matmul + ReLU (2 个分离 kernel) (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算链:
#    Z = X @ W                (matmul)
#    Y = relu(Z)              (逐元素激活)
#  两个 kernel 分离启动, 中间 Z 落 GM
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
#  形状: X[M,K] @ W[K,N] → Z[M,N] → Y[M,N]
# ═══════════════════════════════════════════════════════════════════════════════
M = int(os.environ.get("MATMUL_M", 2048))
N = int(os.environ.get("MATMUL_N", 2048))
K = int(os.environ.get("MATMUL_K", 2048))
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32        # matmul 分块
BLOCK_SIZE = 1024                             # 逐元素 kernel (relu) 分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel (2 个分离 kernel)
# ═══════════════════════════════════════════════════════════════════════════════

# ① Matmul: Z = X @ W (fp32 累加)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ② ReLU: Y = max(0, Z) (逐元素, ★独立 kernel, Z 落 GM 又读回 — 留 Tier2 融合空间)
@triton.jit
def relu_kernel(x_ptr, y_ptr,
                n_elements,
                BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, tl.maximum(x, 0.0), mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步/正确性校验
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    x = (torch.randn(M, K, dtype=DTYPE, device=npu)) * 0.1
    w = (torch.randn(K, N, dtype=DTYPE, device=npu)) * 0.1
    z = torch.empty(M, N, dtype=DTYPE, device=npu)   # ★中间量 Z 落 GM (Tier2 可省)
    y = torch.empty(M, N, dtype=DTYPE, device=npu)

    grid_mm = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    grid_el = (triton.cdiv(M * N, BLOCK_SIZE),)
    print(f"[info] matmul+relu M={M} N={N} K={K} dtype={DTYPE} kernels=2 "
          f"grid_mm={grid_mm[0]} grid_el={grid_el[0]} block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均; 分配只做一次复用)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        # ① Z = X @ W
        matmul_kernel[grid_mm](x, w, z, M, N, K,
                               x.stride(0), x.stride(1), w.stride(0), w.stride(1),
                               z.stride(0), z.stride(1),
                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        # ② Y = relu(Z)  (★Z 落 GM 又读回, Tier2 可并进 matmul epilogue)
        relu_kernel[grid_el](z, y, M * N, BLOCK=BLOCK_SIZE)
    torch.npu.synchronize()
    print("[info] matmul+relu launched & synced OK")

    # 正确性校验 (默认关, verify 设 MATMUL_VERIFY=1 自动跑; 对 torch 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        import torch.nn.functional as F
        ref = F.relu(torch.matmul(x, w))
        abs_diff = (y - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-3 else 'CHECK'}  "
              f"max|Y-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
