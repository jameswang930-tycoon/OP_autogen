"""
Vadd_fp16 — grid=1 打满单核的上板用例
=====================================
配置: N=32768 (64KB fp16), BLOCK_SIZE=32768, grid=1 → 单 block 单核.
UB 占用: ub_a+ub_c = 64KB+64KB = 128KB peak < 实际 UB 192KB (不 overflow).
cost model 预测 (tile=N, grid=1 正确口径): total ≈ 135 ns.

注: cost model 的 MEMORY_CAPACITY_KB['UB']=512KB 是 placeholder, 实际 910B3 UB=192KB,
所以 BLOCK 选 32768 (非 65536) 以避免实际 UB overflow.

上板对照: msprof 实测 total vs 预测 ~135 ns.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError

DTYPE = np.float16
BLOCK_DEFAULT = 32768   # grid=1 打满单核: N=BLOCK → peak UB 128KB < 192KB


def vadd_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr, offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr, offsets, out, mask=mask)


def emulate_vadd(x: np.ndarray, scalar: float = 1.0, BLOCK_SIZE: int = BLOCK_DEFAULT) -> np.ndarray:
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
    print(" Vadd_fp16 Test (out = x + scalar)")
    print("=" * 60)
    np.random.seed(42)

    # 正确性小 case
    x = np.random.randn(4096).astype(DTYPE)
    verify(emulate_vadd(x, BLOCK_SIZE=8192), reference_vadd(x), "vadd_fp16_small")

    # 打满单核上板点: N=32768, BLOCK=32768, grid=1 (64KB fp16, UB peak 128KB)
    x_big = np.random.randn(32768).astype(DTYPE)
    verify(emulate_vadd(x_big, BLOCK_SIZE=32768), reference_vadd(x_big), "vadd_fp16_grid1_64KB")
    print("  [info] grid=1 打满单核: N=32768 BLOCK=32768 → 64KB, UB peak 128KB<192KB, 预测 ~135ns")

    try:
        emulate_vadd(np.zeros(0))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught empty input")
    print()


if __name__ == "__main__":
    test()
