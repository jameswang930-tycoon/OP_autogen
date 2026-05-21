"""
RMSNorm Emulator: Root Mean Square Layer Normalization
=======================================================
Kernel: out[i, j] = x[i, j] / sqrt(mean(x[i, :]^2) + eps) * weight[j]
Grid:   1D, 每个 program 处理一行
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def rmsnorm_kernel(x_ptr, weight_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    """
    RMSNorm kernel: 每个 program 处理一行。
    x_ptr:      [n_rows, n_cols] 展平
    weight_ptr: [n_cols]
    out_ptr:    [n_rows, n_cols] 展平
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    row_start = row_idx * n_cols
    ptrs = row_start + col_offs

    # 加载行和权重
    x = tl.load(x_ptr, ptrs, mask=mask)
    w = tl.load(weight_ptr, col_offs, mask=mask)

    # RMS 计算: sqrt(mean(x^2) + eps)
    x_sq = x * x
    mean_sq = tl.sum(x_sq, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)

    # 归一化并乘以权重
    out = x * rrms * w
    tl.store(out_ptr, ptrs, out, mask=mask)


# ---- Emulator 封装 ----

def emulate_rmsnorm(x: np.ndarray, weight: np.ndarray,
                    eps=1e-6, BLOCK_SIZE=None) -> np.ndarray:
    """
    在 CPU 上 emulate RMSNorm。
    
    参数:
      x:      [n_rows, n_cols]
      weight: [n_cols]  (learnable scale parameter)
      eps:    稳定性常数
    
    错误检查:
      - x 不是 2D
      - weight 长度 != n_cols
      - eps <= 0
      - 包含 NaN/Inf
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise EmulatorError("rmsnorm_kernel",
            f"Input x must be 1D or 2D, got shape {x.shape}")

    n_rows, n_cols = x.shape

    if weight.ndim != 1 or weight.shape[0] != n_cols:
        raise EmulatorError("rmsnorm_kernel",
            f"weight shape {weight.shape} does not match n_cols={n_cols}",
            {"expected_weight_shape": f"({n_cols},)",
             "got_weight_shape": str(weight.shape)})

    if eps <= 0:
        raise EmulatorError("rmsnorm_kernel",
            f"eps must be positive, got {eps}")

    if np.any(np.isnan(x)):
        raise EmulatorError("rmsnorm_kernel",
            f"Input x contains {int(np.sum(np.isnan(x)))} NaN values")
    if np.any(np.isinf(x)):
        raise EmulatorError("rmsnorm_kernel",
            f"Input x contains {int(np.sum(np.isinf(x)))} Inf values")

    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2

    x_flat = x.ravel().astype(np.float32)
    weight_flat = weight.ravel().astype(np.float32)
    out_flat = np.zeros_like(x_flat)

    launch_kernel_1d(rmsnorm_kernel, x_flat, weight_flat, out_flat,
                     n_cols, eps, BLOCK_SIZE, grid_size=n_rows)

    return out_flat.reshape(n_rows, n_cols)


# ---- Reference ----

def reference_rmsnorm(x, weight, eps=1e-6):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    variance = np.mean(x ** 2, axis=-1, keepdims=True)
    x_norm = x / np.sqrt(variance + eps)
    return (x_norm * weight).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" RMSNorm Emulator Test")
    print("=" * 60)

    # Test 1: 基本功能
    n_rows, n_cols = 8, 64
    x = np.random.randn(n_rows, n_cols).astype(np.float32)
    w = np.random.randn(n_cols).astype(np.float32)
    out = emulate_rmsnorm(x, w)
    ref = reference_rmsnorm(x, w)
    verify(out, ref, "rmsnorm_basic")

    # Test 2: weight 全 1 (纯归一化)
    w_ones = np.ones(n_cols, dtype=np.float32)
    out2 = emulate_rmsnorm(x, w_ones)
    ref2 = reference_rmsnorm(x, w_ones)
    verify(out2, ref2, "rmsnorm_unit_weight")

    # Test 3: 大维度
    x_big = np.random.randn(4, 4096).astype(np.float32)
    w_big = np.random.randn(4096).astype(np.float32)
    out_big = emulate_rmsnorm(x_big, w_big)
    ref_big = reference_rmsnorm(x_big, w_big)
    verify(out_big, ref_big, "rmsnorm_large_dim")

    # Test 4: weight 维度不匹配报错
    try:
        emulate_rmsnorm(x, np.ones(32, dtype=np.float32))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught weight dimension mismatch")

    # Test 5: 输出 RMS ≈ 1 (当 weight=1 时)
    rms_out = np.sqrt(np.mean(out2 ** 2, axis=-1))
    rms_close_to_1 = np.allclose(rms_out, 1.0, atol=0.05)
    print(f"  RMS of normalized output ≈ 1.0: {'✓' if rms_close_to_1 else '✗'} "
          f"(sample: {rms_out[:3]})")

    print()


if __name__ == "__main__":
    test()
