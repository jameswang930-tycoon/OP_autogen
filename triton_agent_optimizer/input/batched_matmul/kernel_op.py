#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — Batched Matmul (BMM) (v4.5)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config  ② 算子 kernel  ③ 测试 main
#
#  运算: C[b] = A[b] @ B[b]   (b = 0..B-1, 每个 batch 一个标准 GEMM)
#  算法: 每 program 处理 (batch, m 块, n 块); grid = B × (M/BM) × (N/BN)
#  (中上水平基线: 标准分块 GEMM + fp32 累加; 优化空间留 tier3/4/5)
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 尺寸/精度/分块 (与 bench_910b3 _shapes 对齐)
# ═══════════════════════════════════════════════════════════════════════════════
B = int(os.environ.get("BMM_B", 16))          # batch
M = int(os.environ.get("BMM_M", 512))         # 行
K = int(os.environ.get("BMM_K", 512))         # 内维
N = int(os.environ.get("BMM_N", 512))         # 列
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64        # 分块 (sweep 可扫)
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel
# ═══════════════════════════════════════════════════════════════════════════════

@triton.jit
def bmm_kernel(a_ptr, b_ptr, c_ptr,
               B, M, K, N,
               stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    b = pid // (grid_m * grid_n)
    tmp = pid % (grid_m * grid_n)
    pid_m = tmp // grid_n
    pid_n = tmp % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + b * M * K
    b_base = b_ptr + b * K * N
    a_ptrs = a_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < (K - k)), other=0.0)
        b_tile = tl.load(b_ptrs, mask=(offs_k[:, None] < (K - k)) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b_tile, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + b * M * N + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    npu = torch.device("npu")
    a = (torch.randn(B, M, K, dtype=DTYPE, device=npu) * 0.1)
    b = (torch.randn(B, K, N, dtype=DTYPE, device=npu) * 0.1)
    c = torch.empty(B, M, N, dtype=DTYPE, device=npu)

    grid = (B * triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    print(f"[info] BMM B={B} M={M} K={K} N={N} grid={grid[0]} block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # ★KERNEL_LOOP: verify/bench 用它 (一次 msprof 内循环 N 次, 求单次平均)
    LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
    for _ in range(LOOP):
        bmm_kernel[grid](a, b, c, B, M, K, N,
                         a.stride(1), a.stride(2),
                         b.stride(1), b.stride(2),
                         c.stride(1), c.stride(2),
                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    torch.npu.synchronize()
    print("[info] BMM launched & synced OK")

    # 正确性校验 (MATMUL_VERIFY=1 自动跑; 对 torch.bmm 参考)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        ref = torch.bmm(a, b)
        abs_diff = (c - ref).abs().max().item()
        rel_diff = abs_diff / (ref.abs().max().item() + 1e-6)
        print(f"[info] result check: {'PASS' if rel_diff < 1e-2 else 'CHECK'}  "
              f"max|C-ref|={abs_diff:.6f} rel={rel_diff:.6f}")


if __name__ == "__main__":
    main()
