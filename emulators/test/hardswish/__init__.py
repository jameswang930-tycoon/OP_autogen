"""
Hardswish Emulator: element-wise Hardswish activation
======================================================
Kernel: out[i] = x[i] * min(max(x[i] + 3, 0), 6) / 6
Grid:   1D, 每 program 处理 BLOCK_SIZE 个元素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def hardswish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr, offs, mask=mask)
    out = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(out_ptr, offs, out, mask=mask)


# ---- Emulator 封装 ----

def emulate_hardswish(x: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    if x.size == 0:
        raise EmulatorError("hardswish_kernel", "Empty input tensor")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(hardswish_kernel, x_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_hardswish(x):
    import torch
    return torch.nn.functional.hardswish(
        torch.tensor(x, dtype=torch.float32)).numpy()


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Hardswish Emulator Test")
    print("=" * 60)

    # Test 1: basic
    x = np.random.randn(1024).astype(np.float32)
    out = emulate_hardswish(x)
    ref = reference_hardswish(x)
    verify(out, ref, "hardswish_basic")

    # Test 2: all-negative (< -3) → 0
    x_neg = (np.random.randn(256).astype(np.float32) - 5.0)
    out_neg = emulate_hardswish(x_neg)
    ref_neg = reference_hardswish(x_neg)
    verify(out_neg, ref_neg, "hardswish_neg")

    # Test 3: all-positive (> 3) → identity
    x_pos = (np.random.randn(256).astype(np.float32) + 5.0)
    out_pos = emulate_hardswish(x_pos)
    ref_pos = reference_hardswish(x_pos)
    verify(out_pos, ref_pos, "hardswish_pos")

    # Test 4: 2D
    x2 = np.random.randn(16, 64).astype(np.float32)
    out2 = emulate_hardswish(x2)
    ref2 = reference_hardswish(x2)
    verify(out2, ref2, "hardswish_2d")

    # Test 5: non-aligned
    x3 = np.random.randn(100).astype(np.float32)
    out3 = emulate_hardswish(x3, BLOCK_SIZE=32)
    ref3 = reference_hardswish(x3)
    verify(out3, ref3, "hardswish_unaligned")

    print()


if __name__ == "__main__":
    test()
