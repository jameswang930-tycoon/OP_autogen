"""
Vadd_fp16 Emulator: element-wise vector + scalar (fp16 storage)
===============================================================
Kernel: out[i] = x[i] + scalar
Grid:   1D, BLOCK_SIZE elements per program
Dtype:  fp16 (storage) — dtype flows from /triton-plan (dtype=fp16) -> /triton-gen.

Plan-code guidance (cost model, fp16 口径):
  - vec path GM->UB->Vec->UB->GM; bottleneck = ub_to_gm (store); single tile sequential.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError

DTYPE = np.float16   # from plan dtype=fp16


def vadd_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr, offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr, offsets, out, mask=mask)


def emulate_vadd(x: np.ndarray, scalar: float = 1.0, BLOCK_SIZE: int = 8192) -> np.ndarray:
    if x.size == 0:
        raise EmulatorError("vadd_kernel", "Empty input tensor")
    n = x.size
    x_flat = x.ravel().astype(DTYPE)
    out_flat = np.zeros(n, dtype=DTYPE)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(vadd_kernel, x_flat, out_flat, n, DTYPE(scalar), BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)


def reference_vadd(x, scalar=1.0):
    return (x + scalar).astype(DTYPE)


def test():
    print("=" * 60)
    print(" Vadd_fp16 Emulator Test (out = x + scalar, fp16)")
    print("=" * 60)
    np.random.seed(42)

    x = np.random.randn(4096).astype(DTYPE)
    verify(emulate_vadd(x), reference_vadd(x), "vadd_fp16_1d")

    x2 = np.random.randn(4096).astype(DTYPE)
    verify(emulate_vadd(x2, scalar=3.14), reference_vadd(x2, scalar=3.14), "vadd_fp16_scalar")

    x3 = np.random.randn(32, 128).astype(DTYPE)
    verify(emulate_vadd(x3, scalar=-2.0), reference_vadd(x3, scalar=-2.0), "vadd_fp16_2d")

    try:
        emulate_vadd(np.zeros(0))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught empty input")
    print()


if __name__ == "__main__":
    test()
