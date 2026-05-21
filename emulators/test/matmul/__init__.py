"""
Matmul Emulator: 2D tiled matrix multiplication
=================================================
Kernel: C[M, N] = A[M, K] @ B[K, N]
Grid:   2D, (grid_m, grid_n), 每个 program 计算 C 的一个 [BLOCK_M x BLOCK_N] tile
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_2d, verify, EmulatorError


# ---- Triton-style Kernel ----

def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    标准 2D tiled matmul kernel。
    每个 program (pid_m, pid_n) 计算 C 的一个 tile:
      C[pid_m*BM:(pid_m+1)*BM, pid_n*BN:(pid_n+1)*BN]
        = sum_k A[..., k:k+BK] @ B[k:k+BK, ...]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 累加器 (在寄存器中)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_offsets = (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak).ravel()
        a_mask = ((offs_m[:, None] < M) & (offs_k[None, :] < K)).ravel()
        a_tile = tl.load(a_ptr, a_offsets, mask=a_mask).reshape(BLOCK_M, BLOCK_K)

        # Load B tile [BLOCK_K, BLOCK_N]
        b_offsets = (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn).ravel()
        b_mask = ((offs_k[:, None] < K) & (offs_n[None, :] < N)).ravel()
        b_tile = tl.load(b_ptr, b_offsets, mask=b_mask).reshape(BLOCK_K, BLOCK_N)

        # 累加
        acc = acc + tl.dot(a_tile, b_tile)

    # Store C tile
    c_offsets = (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn).ravel()
    c_mask = ((offs_m[:, None] < M) & (offs_n[None, :] < N)).ravel()
    tl.store(c_ptr, c_offsets, acc.ravel(), mask=c_mask)


# ---- Emulator 封装 ----

def emulate_matmul(a: np.ndarray, b: np.ndarray,
                   BLOCK_M=32, BLOCK_N=32, BLOCK_K=32) -> np.ndarray:
    """
    在 CPU 上 emulate 2D tiled matmul。
    
    参数:
      a: [M, K], b: [K, N]
    
    错误检查:
      - 输入不是 2D
      - K 维度不匹配
      - 空矩阵
    """
    if a.ndim != 2:
        raise EmulatorError("matmul_kernel", f"a must be 2D, got shape {a.shape}")
    if b.ndim != 2:
        raise EmulatorError("matmul_kernel", f"b must be 2D, got shape {b.shape}")

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise EmulatorError("matmul_kernel",
            f"Inner dimension mismatch: a.shape={a.shape}, b.shape={b.shape}. "
            f"a.shape[1]={K} != b.shape[0]={K2}")
    if M == 0 or N == 0 or K == 0:
        raise EmulatorError("matmul_kernel", f"Zero-sized dimension: M={M}, N={N}, K={K}")

    a_flat = a.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    c_flat = np.zeros(M * N, dtype=np.float32)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    launch_kernel_2d(
        matmul_kernel,
        a_flat, b_flat, c_flat,
        M, N, K,
        K, 1,      # stride_am, stride_ak  (row-major A)
        N, 1,      # stride_bk, stride_bn  (row-major B)
        N, 1,      # stride_cm, stride_cn  (row-major C)
        BLOCK_M, BLOCK_N, BLOCK_K,
        grid=(grid_m, grid_n),
    )

    return c_flat.reshape(M, N)


# ---- Reference ----

def reference_matmul(a, b):
    return (a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Matmul Emulator Test")
    print("=" * 60)

    # Test 1: 方阵
    M, N, K = 64, 64, 64
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    out = emulate_matmul(A, B)
    ref = reference_matmul(A, B)
    verify(out, ref, "matmul_square", rtol=1e-3, atol=1e-4)

    # Test 2: 非方阵
    M, N, K = 48, 96, 32
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    out = emulate_matmul(A, B)
    ref = reference_matmul(A, B)
    verify(out, ref, "matmul_rect", rtol=1e-3, atol=1e-4)

    # Test 3: 非 BLOCK 对齐
    M, N, K = 37, 53, 19
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    out = emulate_matmul(A, B, BLOCK_M=16, BLOCK_N=16, BLOCK_K=16)
    ref = reference_matmul(A, B)
    verify(out, ref, "matmul_unaligned", rtol=1e-3, atol=1e-4)

    # Test 4: K 不匹配报错
    try:
        emulate_matmul(np.zeros((4, 8)), np.zeros((16, 4)))
        print("  [FAIL] Should have raised EmulatorError for K mismatch")
    except EmulatorError:
        print("  [PASS] Correctly caught K dimension mismatch")

    print()


if __name__ == "__main__":
    test()
