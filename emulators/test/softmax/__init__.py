"""
Softmax Emulator: row-wise numerically stable softmax
======================================================
Kernel: out[i, j] = exp(x[i,j] - max(x[i,:])) / sum(exp(x[i,:] - max(x[i,:])))
Grid:   1D, 每个 program 处理一行
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    行级 softmax, 数值稳定版本。
    x_ptr: [n_rows, n_cols] 展平后的 1D 数组
    每个 program_id(0) 处理一行。
    BLOCK_SIZE 必须 >= n_cols (一个 program 加载完整一行)。
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    row_start = row_idx * n_cols
    ptrs = row_start + col_offs

    # 加载一行
    row = tl.load(x_ptr, ptrs, mask=mask, other=float('-inf'))

    # 数值稳定: 减去行最大值
    row_max = tl.max(row, axis=0)
    row_safe = row - row_max

    # exp 和归一化
    numerator = tl.exp(row_safe)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(out_ptr, ptrs, result, mask=mask)


# ---- Emulator 封装 ----

def emulate_softmax(x: np.ndarray, BLOCK_SIZE=None) -> np.ndarray:
    """
    在 CPU 上 emulate row-wise softmax。
    
    参数:
      x: [n_rows, n_cols] 或 [n_cols] (自动升维)
      BLOCK_SIZE: 必须 >= n_cols, 默认自动取最小 2 的幂
    
    错误检查:
      - 维度 > 2
      - BLOCK_SIZE < n_cols
      - 包含 NaN
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.ndim != 2:
        raise EmulatorError("softmax_kernel",
            f"Input must be 1D or 2D, got shape {x.shape}",
            {"hint": "Reshape to [batch, features] before calling softmax."})

    if np.any(np.isnan(x)):
        raise EmulatorError("softmax_kernel",
            "Input contains NaN values",
            {"nan_count": int(np.sum(np.isnan(x))),
             "hint": "Check upstream computation for NaN propagation."})

    n_rows, n_cols = x.shape

    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2

    if BLOCK_SIZE < n_cols:
        raise EmulatorError("softmax_kernel",
            f"BLOCK_SIZE={BLOCK_SIZE} < n_cols={n_cols}. "
            f"Each program must load a complete row.",
            {"required_min_BLOCK_SIZE": n_cols})

    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros_like(x_flat)

    launch_kernel_1d(softmax_kernel, x_flat, out_flat, n_cols, BLOCK_SIZE,
                     grid_size=n_rows)

    return out_flat.reshape(n_rows, n_cols)


# ---- Reference ----

def reference_softmax(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    shifted = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return (e / np.sum(e, axis=-1, keepdims=True)).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Softmax Emulator Test")
    print("=" * 60)

    # Test 1: 基本功能
    x = np.random.randn(8, 64).astype(np.float32)
    out = emulate_softmax(x)
    ref = reference_softmax(x)
    verify(out, ref, "softmax_basic")

    # Test 2: 行和 == 1
    row_sums = np.sum(out, axis=-1)
    all_one = np.allclose(row_sums, 1.0, atol=1e-5)
    print(f"  Row sums ≈ 1.0: {'✓' if all_one else '✗'} (sample: {row_sums[:3]})")

    # Test 3: 大数值稳定性 (不应溢出)
    x_large = np.array([[1000, 1001, 1002], [-1000, -999, -998]], dtype=np.float32)
    out_large = emulate_softmax(x_large)
    ref_large = reference_softmax(x_large)
    verify(out_large, ref_large, "softmax_numerical_stability")
    has_nan = np.any(np.isnan(out_large))
    has_inf = np.any(np.isinf(out_large))
    print(f"  NaN check: {'✓ (no NaN)' if not has_nan else '✗ (has NaN)'}")
    print(f"  Inf check: {'✓ (no Inf)' if not has_inf else '✗ (has Inf)'}")

    # Test 4: 1D 输入自动升维
    x1d = np.random.randn(32).astype(np.float32)
    out1d = emulate_softmax(x1d)
    ref1d = reference_softmax(x1d)
    verify(out1d, ref1d, "softmax_1d_auto")

    # Test 5: 非对齐列数
    x_odd = np.random.randn(4, 37).astype(np.float32)
    out_odd = emulate_softmax(x_odd)
    ref_odd = reference_softmax(x_odd)
    verify(out_odd, ref_odd, "softmax_unaligned_cols")

    print()


if __name__ == "__main__":
    test()
