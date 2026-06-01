"""
Mul Emulator: element-wise multiplication
==========================================
Kernel: out[i] = x[i] * y[i]
Grid:   1D, 每 program 处理 BLOCK_SIZE 个元素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr, offsets, mask=mask)
    y = tl.load(y_ptr, offsets, mask=mask)
    output = x * y
    tl.store(output_ptr, offsets, output, mask=mask)


# ---- Emulator 封装 ----

def emulate_mul(x: np.ndarray, y: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    if x.shape != y.shape:
        raise EmulatorError("mul_kernel",
            f"Input shape mismatch: x.shape={x.shape}, y.shape={y.shape}")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    y_flat = y.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)

    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(mul_kernel, x_flat, y_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)

    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_mul(x, y):
    return (x * y).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Mul Emulator Test")
    print("=" * 60)

    # Test 1: 1D
    x = np.random.randn(1024).astype(np.float32)
    y = np.random.randn(1024).astype(np.float32)
    out = emulate_mul(x, y)
    ref = reference_mul(x, y)
    verify(out, ref, "mul_1d")

    # Test 2: 2D
    x2 = np.random.randn(32, 64).astype(np.float32)
    y2 = np.random.randn(32, 64).astype(np.float32)
    out2 = emulate_mul(x2, y2)
    ref2 = reference_mul(x2, y2)
    verify(out2, ref2, "mul_2d")

    # Test 3: non-aligned
    x3 = np.random.randn(100).astype(np.float32)
    y3 = np.random.randn(100).astype(np.float32)
    out3 = emulate_mul(x3, y3, BLOCK_SIZE=32)
    ref3 = reference_mul(x3, y3)
    verify(out3, ref3, "mul_unaligned")

    # Test 4: shape mismatch
    try:
        emulate_mul(np.zeros(10), np.zeros(20))
        print("  [FAIL] Should have raised EmulatorError for shape mismatch")
    except EmulatorError:
        print("  [PASS] Correctly caught shape mismatch error")

    print()


if __name__ == "__main__":
    test()
