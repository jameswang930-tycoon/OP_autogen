"""
Vadd Emulator: element-wise vector + scalar
============================================
Kernel: out[i] = x[i] + scalar
Grid:   1D, each program handles BLOCK_SIZE elements

Generated via the skill pipeline:
  /triton-plan  (real plan from cost_emulator; op_kind=vadd, N=4096)
  -> /triton-gen

Plan-code guidance applied:
  - vec path: GM -> UB (load) -> VecUnit (compute) -> UB -> GM (store)
  - gm_to_ub was on the bandwidth ramp (66% util at 8KB < 12KB saturation point),
    so BLOCK_SIZE is set to 8192 (16KB tile) to push the read toward saturation.
  - bottleneck is ub_to_gm (store, 69%) — unavoidable for an elementwise op;
    a single tile is fully sequential (RAW chain load -> compute -> store).
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def vadd_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr, offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr, offsets, out, mask=mask)


# ---- Emulator wrapper ----

def emulate_vadd(x: np.ndarray, scalar: float = 1.0, BLOCK_SIZE: int = 8192) -> np.ndarray:
    """
    CPU-emulate the element-wise vadd kernel (out = x + scalar).

    Args:
      x:          input array
      scalar:     scalar to add (default 1.0, matching the cost-model vadd DSL)
      BLOCK_SIZE: elements per program (8192 -> 16KB tile, saturates GM->UB read BW)

    Returns:
      out: x + scalar

    Errors:
      - empty input
    """
    if x.size == 0:
        raise EmulatorError("vadd_kernel", "Empty input tensor")

    n = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)

    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(vadd_kernel, x_flat, out_flat, n, np.float32(scalar), BLOCK_SIZE, grid_size=grid)

    return out_flat.reshape(x.shape)


# ---- Reference ----

def reference_vadd(x, scalar=1.0):
    return (x + scalar).astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Vadd Emulator Test (out = x + scalar)")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: 1D, default scalar=1.0
    x = np.random.randn(4096).astype(np.float32)
    out = emulate_vadd(x)
    ref = reference_vadd(x)
    verify(out, ref, "vadd_1d_default")

    # Test 2: custom scalar
    x2 = np.random.randn(4096).astype(np.float32)
    out2 = emulate_vadd(x2, scalar=3.14)
    ref2 = reference_vadd(x2, scalar=3.14)
    verify(out2, ref2, "vadd_1d_scalar")

    # Test 3: 2D (flattened internally)
    x3 = np.random.randn(32, 128).astype(np.float32)
    out3 = emulate_vadd(x3, scalar=-2.0)
    ref3 = reference_vadd(x3, scalar=-2.0)
    verify(out3, ref3, "vadd_2d")

    # Test 4: unaligned size (BLOCK_SIZE does not divide n)
    x4 = np.random.randn(100).astype(np.float32)
    out4 = emulate_vadd(x4, scalar=1.0, BLOCK_SIZE=32)
    ref4 = reference_vadd(x4, scalar=1.0)
    verify(out4, ref4, "vadd_unaligned")

    # Test 5: empty input should raise
    try:
        emulate_vadd(np.zeros(0))
        print("  [FAIL] Should have raised EmulatorError for empty input")
    except EmulatorError:
        print("  [PASS] Correctly caught empty input error")

    print()


if __name__ == "__main__":
    test()
