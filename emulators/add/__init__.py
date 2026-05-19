"""
Add Emulator: element-wise addition
=====================================
Kernel: out[i] = x[i] + y[i]
Grid:   1D, 每个 program 处理 BLOCK_SIZE 个元素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr, offsets, mask=mask)
    y = tl.load(y_ptr, offsets, mask=mask)
    output = x + y
    tl.store(output_ptr, offsets, output, mask=mask)


# ---- Emulator 封装 ----

def emulate_add(x: np.ndarray, y: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    """
    在 CPU 上 emulate element-wise add kernel。
    
    参数:
      x, y:        输入数组, shape 必须一致
      BLOCK_SIZE:  每个 program 处理的元素数
    
    返回:
      out: x + y
    
    错误检查:
      - x, y shape 不一致
      - 空输入
    """
    if x.shape != y.shape:
        raise EmulatorError("add_kernel",
            f"Input shape mismatch: x.shape={x.shape}, y.shape={y.shape}")
    if x.size == 0:
        raise EmulatorError("add_kernel", "Empty input tensor")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    y_flat = y.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)

    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(add_kernel, x_flat, y_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)

    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_add(x, y):
    return (x + y).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Add Emulator Test")
    print("=" * 60)

    # Test 1: 1D
    x = np.random.randn(1024).astype(np.float32)
    y = np.random.randn(1024).astype(np.float32)
    out = emulate_add(x, y)
    ref = reference_add(x, y)
    verify(out, ref, "add_1d")

    # Test 2: 2D (flattened internally)
    x2 = np.random.randn(32, 64).astype(np.float32)
    y2 = np.random.randn(32, 64).astype(np.float32)
    out2 = emulate_add(x2, y2)
    ref2 = reference_add(x2, y2)
    verify(out2, ref2, "add_2d")

    # Test 3: 非对齐大小 (BLOCK_SIZE 不能整除)
    x3 = np.random.randn(100).astype(np.float32)
    y3 = np.random.randn(100).astype(np.float32)
    out3 = emulate_add(x3, y3, BLOCK_SIZE=32)
    ref3 = reference_add(x3, y3)
    verify(out3, ref3, "add_unaligned")

    # Test 4: shape mismatch 应报错
    try:
        emulate_add(np.zeros(10), np.zeros(20))
        print("  [FAIL] Should have raised EmulatorError for shape mismatch")
    except EmulatorError:
        print("  [PASS] Correctly caught shape mismatch error")

    print()


if __name__ == "__main__":
    test()
