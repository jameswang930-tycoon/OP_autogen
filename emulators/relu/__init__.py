"""
ReLU Emulator: element-wise ReLU activation
=============================================
Kernel: out[i] = max(x[i], 0)
Grid:   1D, 每个 program 处理 BLOCK_SIZE 个元素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr, offs, mask=mask)
    # ReLU: max(x, 0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr, offs, out, mask=mask)


def leaky_relu_kernel(x_ptr, out_ptr, n_elements, alpha, BLOCK_SIZE: tl.constexpr):
    """Leaky ReLU 变体: out = x if x > 0 else alpha * x"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr, offs, mask=mask)
    out = tl.where(x > 0, x, x * alpha)
    tl.store(out_ptr, offs, out, mask=mask)


# ---- Emulator 封装 ----

def emulate_relu(x: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    """
    在 CPU 上 emulate ReLU。
    
    参数:
      x: 任意 shape
    
    错误检查:
      - 空输入
      - NaN 输入
    """
    if x.size == 0:
        raise EmulatorError("relu_kernel", "Empty input tensor")
    if np.any(np.isnan(x)):
        raise EmulatorError("relu_kernel",
            "Input contains NaN values",
            {"nan_count": int(np.sum(np.isnan(x)))})

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(relu_kernel, x_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)


def emulate_leaky_relu(x: np.ndarray, alpha=0.01, BLOCK_SIZE=1024) -> np.ndarray:
    """Leaky ReLU: out = x if x > 0 else alpha * x"""
    if x.size == 0:
        raise EmulatorError("leaky_relu_kernel", "Empty input tensor")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(leaky_relu_kernel, x_flat, out_flat, n, alpha, BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_relu(x):
    return np.maximum(x, 0).astype(np.float32)

def reference_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" ReLU Emulator Test")
    print("=" * 60)

    # Test 1: 基本 ReLU
    x = np.random.randn(1024).astype(np.float32)
    out = emulate_relu(x)
    ref = reference_relu(x)
    verify(out, ref, "relu_basic")

    # Test 2: 全负 → 全零
    x_neg = -np.abs(np.random.randn(256).astype(np.float32))
    out_neg = emulate_relu(x_neg)
    assert np.all(out_neg == 0), "ReLU of all-negative should be all-zero"
    print("  [PASS] All-negative → all-zero ✓")

    # Test 3: 全正 → 不变
    x_pos = np.abs(np.random.randn(256).astype(np.float32))
    out_pos = emulate_relu(x_pos)
    verify(out_pos, x_pos, "relu_all_positive")

    # Test 4: 多维
    x_2d = np.random.randn(16, 64).astype(np.float32)
    out_2d = emulate_relu(x_2d)
    ref_2d = reference_relu(x_2d)
    verify(out_2d, ref_2d, "relu_2d")

    # Test 5: Leaky ReLU
    x_lr = np.random.randn(512).astype(np.float32)
    out_lr = emulate_leaky_relu(x_lr, alpha=0.1)
    ref_lr = reference_leaky_relu(x_lr, alpha=0.1)
    verify(out_lr, ref_lr, "leaky_relu")

    # Test 6: Leaky ReLU 负数部分不为零
    negative_vals = out_lr[x_lr < 0]
    assert np.all(negative_vals != 0), "Leaky ReLU negative region should not be zero"
    print("  [PASS] Leaky ReLU preserves negative region ✓")

    print()


if __name__ == "__main__":
    test()
