"""
Reshape Emulator: tensor reshape (view)
=========================================
Reshape 不涉及数据搬运, 只是改变 stride/shape 元信息。
在 Triton kernel 中, reshape 通常体现为 offset 计算方式的改变,
而不是一个独立的 kernel。

本 emulator 提供:
  1. reshape 的正确性验证 (shape 合法性检查)
  2. 一个 copy kernel (将数据按新 layout 重排, 模拟非 contiguous reshape)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel (copy with reshape) ----

def reshape_copy_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    逐元素 copy kernel。
    reshape 本身是 zero-cost 的 (只改 view), 但如果原 tensor 不 contiguous,
    需要一次 copy 来得到新 layout 下的连续数据。
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr, offs, mask=mask)
    tl.store(out_ptr, offs, x, mask=mask)


# ---- Emulator 封装 ----

def emulate_reshape(x: np.ndarray, new_shape) -> np.ndarray:
    """
    在 CPU 上 emulate reshape 操作。
    
    参数:
      x:         输入数组
      new_shape: 目标 shape, 可包含一个 -1 表示自动推断
    
    错误检查:
      - 元素总数不一致
      - 多个 -1
      - new_shape 包含 0 或负数 (除 -1)
    """
    # 处理 -1 维度
    new_shape = list(new_shape)
    neg_count = new_shape.count(-1)

    if neg_count > 1:
        raise EmulatorError("reshape",
            f"Only one dimension can be -1, got {neg_count} in shape {new_shape}")

    if neg_count == 1:
        known_prod = 1
        neg_idx = -1
        for i, s in enumerate(new_shape):
            if s == -1:
                neg_idx = i
            elif s <= 0:
                raise EmulatorError("reshape",
                    f"Invalid dimension {s} at index {i} in shape {new_shape}",
                    {"hint": "Dimensions must be positive (or exactly -1 for inference)."})
            else:
                known_prod *= s
        if x.size % known_prod != 0:
            raise EmulatorError("reshape",
                f"Cannot reshape {x.shape} (size={x.size}) to {new_shape}: "
                f"{x.size} is not divisible by {known_prod}",
                {"known_dims_product": known_prod})
        new_shape[neg_idx] = x.size // known_prod
    else:
        total = 1
        for i, s in enumerate(new_shape):
            if s <= 0:
                raise EmulatorError("reshape",
                    f"Invalid dimension {s} at index {i} in shape {new_shape}")
            total *= s
        if total != x.size:
            raise EmulatorError("reshape",
                f"Cannot reshape {x.shape} (size={x.size}) to {tuple(new_shape)} (size={total}): "
                f"element count mismatch",
                {"input_size": x.size, "target_size": total})

    new_shape = tuple(new_shape)

    # 执行 copy kernel (确保数据连续)
    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    BLOCK = 1024
    grid = tl.cdiv(n, BLOCK)
    launch_kernel_1d(reshape_copy_kernel, x_flat, out_flat, n, BLOCK, grid_size=grid)

    return out_flat.reshape(new_shape)


# ---- Reference ----

def reference_reshape(x, new_shape):
    return x.reshape(new_shape).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Reshape Emulator Test")
    print("=" * 60)

    # Test 1: 基本 reshape
    x = np.random.randn(2, 3, 4).astype(np.float32)
    out = emulate_reshape(x, (6, 4))
    ref = reference_reshape(x, (6, 4))
    verify(out, ref, "reshape_3d_to_2d")

    # Test 2: 展平
    out2 = emulate_reshape(x, (24,))
    ref2 = reference_reshape(x, (24,))
    verify(out2, ref2, "reshape_flatten")

    # Test 3: -1 推断
    out3 = emulate_reshape(x, (8, -1))
    ref3 = reference_reshape(x, (8, -1))
    verify(out3, ref3, "reshape_infer")
    assert out3.shape == (8, 3), f"Shape wrong: {out3.shape}"
    print(f"  Inferred shape: {out3.shape} ✓")

    # Test 4: 元素数不匹配报错
    try:
        emulate_reshape(x, (5, 5))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught element count mismatch")

    # Test 5: 多个 -1 报错
    try:
        emulate_reshape(x, (-1, -1))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught multiple -1 dims")

    print()


if __name__ == "__main__":
    test()
