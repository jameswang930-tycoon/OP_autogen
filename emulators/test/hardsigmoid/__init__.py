"""
Hardsigmoid Emulator: element-wise Hardsigmoid activation
=========================================================
Kernel: out[i] = min(max(x[i] + 3, 0), 6) / 6
Grid:   1D, 每 program 处理 BLOCK_SIZE 个元素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def hardsigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr, offs, mask=mask)
    out = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(out_ptr, offs, out, mask=mask)


# ---- Emulator 封装 ----

def emulate_hardsigmoid(x: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    if x.size == 0:
        raise EmulatorError("hardsigmoid_kernel", "Empty input tensor")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(hardsigmoid_kernel, x_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_hardsigmoid(x):
    import torch
    return torch.nn.functional.hardsigmoid(
        torch.tensor(x, dtype=torch.float32)).numpy()


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Hardsigmoid Emulator Test")
    print("=" * 60)

    # Test 1: basic (range [0, 1])
    x = np.random.randn(1024).astype(np.float32)
    out = emulate_hardsigmoid(x)
    ref = reference_hardsigmoid(x)
    verify(out, ref, "hardsigmoid_basic")

    # Test 2: all-negative → 0 (all values < -3)
    x_neg = (np.random.randn(256).astype(np.float32) * 0.5 - 5.0)
    out_neg = emulate_hardsigmoid(x_neg)
    assert np.allclose(out_neg, 0.0, atol=1e-6), "hardsigmoid(<-3) should be 0"
    print("  [PASS] All-negative → 0")

    # Test 3: all-positive → 1 (all values > 3)
    x_pos = (np.random.randn(256).astype(np.float32) * 0.5 + 5.0)
    out_pos = emulate_hardsigmoid(x_pos)
    assert np.allclose(out_pos, 1.0, atol=1e-6), "hardsigmoid(>3) should be 1"
    print("  [PASS] All-positive → 1")

    # Test 4: 2D
    x2 = np.random.randn(16, 64).astype(np.float32)
    out2 = emulate_hardsigmoid(x2)
    ref2 = reference_hardsigmoid(x2)
    verify(out2, ref2, "hardsigmoid_2d")

    # Test 5: non-aligned
    x3 = np.random.randn(100).astype(np.float32)
    out3 = emulate_hardsigmoid(x3, BLOCK_SIZE=32)
    ref3 = reference_hardsigmoid(x3)
    verify(out3, ref3, "hardsigmoid_unaligned")

    print()


if __name__ == "__main__":
    test()
