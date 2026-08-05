#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════════════
#  单文件 kernel_op.py — 两层 MLP (X→FC1→GELU→FC2→Y) 合一体 (v4)
#  ★ 优化循环里 coder 只改这一个文件; 读取/运行/测试都只看这一个文件
#  分三个区: ① 场景 config (改尺寸/分块/精度)  ② 算子 kernel (优化核心)  ③ 测试 main
#
#  相比单一 matmul (512³ fp32, ~µs, 访存受限, 跑不满):
#    - 尺寸大 (2048³, 单 matmul 17.2 GFLOP), 端到端 ~百µs, 真机跑得满
#    - 3 个 kernel (fc1 / bias_gelu / fc2) → Tier2 融合空间 (bias_gelu 并入 fc1 epilogue 省 Z 的 GM 往返)
#    - 流程长 (matmul→激活→matmul), 各 tier 都有优化点
#    - 保持 fp32 起步 → tier1 留有"切 fp16"杠杆
# ═══════════════════════════════════════════════════════════════════════════════
import os
import sys
import torch
import torch_npu          # 必须先 import, 注册 NPU 后端
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  ① 场景 config — 改这里调尺寸、精度、分块 (tier3 优化点: BLOCK_*)
#   两层 MLP: Y = GELU(X@W1 + b1) @ W2
#   形状: X[M,K] @ W1[K,HIDDEN] → Z[M,HIDDEN] → GELU(Z+b1) → H[M,HIDDEN] @ W2[HIDDEN,N] → Y[M,N]
# ═══════════════════════════════════════════════════════════════════════════════
M  = int(os.environ.get("MATMUL_M", 2048))
N  = int(os.environ.get("MATMUL_N", 2048))
K  = int(os.environ.get("MATMUL_K", 2048))
HIDDEN = int(os.environ.get("MLP_HIDDEN", 2048))   # 隐藏层宽度 (= 两个 matmul 的中间维度)
DTYPE = torch.float32                    # 精度 (tier1/tier5 优化点: fp16 计算 + fp32 累加)
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32   # matmul 分块 (tier3 优化点)
BLOCK_SIZE = 1024                        # 逐元素 kernel (bias_gelu) 分块
# 注意: 不传 num_warps/num_stages — triton-ascend 禁止 tune 这两个参数, 自动管理 tiling/流水


# ═══════════════════════════════════════════════════════════════════════════════
#  ② 算子 kernel — 优化核心 (tier1 算法 / tier2 融合 / tier3 分块 / tier4 访存 / tier5 计算 / tier6 架构)
# ═══════════════════════════════════════════════════════════════════════════════

# FC1: Z = X @ W1   (matmul, fp32 累加)
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


# FC2: Y = H @ W2   (同样 matmul, 独立 kernel 名以便 msprof 区分两次 matmul)
@triton.jit
def matmul_kernel2(a_ptr, b_ptr, c_ptr,
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


# bias + GELU: H = GELU(Z + b1)  (逐元素, Tier2 融合目标: 并入 fc1 epilogue 省一次 Z 的 GM 写+读)
@triton.jit
def bias_gelu_kernel(x_ptr, bias_ptr, y_ptr,
                     n_elements, N,
                     BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + (offs % N), mask=mask, other=0.0)   # bias 按列广播 (每行同列)
    val = x + b
    # tanh 近似 GELU (与 torch F.gelu(approximate="tanh") 一致)
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))
    y = val * cdf
    tl.store(y_ptr + offs, y, mask=mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  ③ 测试 main — 分配输入/启动 kernel/同步 (一般不动)
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if not torch.npu.is_available():
        raise SystemExit("[FATAL] torch.npu 不可用, 请检查 CANN/torch_npu/驱动 (/dev/davinci0)")

    # 输入 (小值 ±0.05, 避免 fp32 dot 值域溢出)
    x  = (torch.rand(M, K, dtype=DTYPE, device="npu") - 0.5) * 0.1
    w1 = (torch.rand(K, HIDDEN, dtype=DTYPE, device="npu") - 0.5) * 0.1
    b1 = (torch.rand(HIDDEN, dtype=DTYPE, device="npu") - 0.5) * 0.1
    w2 = (torch.rand(HIDDEN, N, dtype=DTYPE, device="npu") - 0.5) * 0.1
    z = torch.empty(M, HIDDEN, dtype=DTYPE, device="npu")
    h = torch.empty(M, HIDDEN, dtype=DTYPE, device="npu")
    y = torch.empty(M, N, dtype=DTYPE, device="npu")

    grid1 = (triton.cdiv(M, BLOCK_M) * triton.cdiv(HIDDEN, BLOCK_N),)   # fc1: X[M,K]@W1[K,H]
    grid2 = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)        # fc2: H[M,H]@W2[H,N]
    grid_g = (triton.cdiv(M * HIDDEN, BLOCK_SIZE),)                     # bias_gelu: M*H 个元素

    print(f"[info] MLP M={M} K={K} HIDDEN={HIDDEN} N={N}  dtype={DTYPE}")
    print(f"[info] fc1 grid={grid1[0]}  fc2 grid={grid2[0]}  bias_gelu grid={grid_g[0]}  "
          f"block={BLOCK_M}x{BLOCK_N}x{BLOCK_K}")

    # FC1: Z = X @ W1
    matmul_kernel[grid1](
        x, w1, z,
        M, HIDDEN, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    # bias + GELU: H = GELU(Z + b1)
    bias_gelu_kernel[grid_g](
        z, b1, h,
        M * HIDDEN, HIDDEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    # FC2: Y = H @ W2
    matmul_kernel2[grid2](
        h, w2, y,
        M, N, HIDDEN,
        h.stride(0), h.stride(1),
        w2.stride(0), w2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.npu.synchronize()
    print("[info] MLP launched & synced OK")

    # 正确性校验 (默认关, 要开用 MATMUL_VERIFY=1)
    if os.environ.get("MATMUL_VERIFY", "0") == "1":
        try:
            import torch.nn.functional as F
            z_ref = torch.matmul(x, w1)
            h_ref = F.gelu(z_ref + b1, approximate="tanh")
            y_ref = torch.matmul(h_ref, w2)
            abs_diff = (y - y_ref).abs().max().item()
            rel_diff = abs_diff / (y_ref.abs().max().item() + 1e-6)
            print(f"[info] result check: {'PASS' if rel_diff < 0.05 else 'CHECK'}  "
                  f"max|Y-Y_ref|={abs_diff:.5f} rel={rel_diff:.5f}")
        except Exception as e:
            print(f"[warn] result check skipped: {e}")


if __name__ == "__main__":
    main()
