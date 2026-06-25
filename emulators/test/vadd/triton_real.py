"""
Vadd — real Triton kernel (converted from the emulator version).
===============================================================
Mechanical conversion from emulators/test/vadd/__init__.py (5 rewrites; the
kernel compute logic is identical — only the calling convention changed):

  1. import      : from common import ...  ->  import triton; import triton.language as tl (+torch)
  2. decorator   : def vadd_kernel         ->  @triton.jit def vadd_kernel
  3. load        : tl.load(x_ptr, offsets) ->  tl.load(x_ptr + offsets)
  4. store       : tl.store(out_ptr, off)  ->  tl.store(out_ptr + offsets)
  5. launch      : launch_kernel_1d(...)   ->  vadd_kernel[grid](...); numpy -> torch

NPU constraint self-check: grid = ceil(n / BLOCK_SIZE). For n=4096, BLOCK=8192
-> grid=1, far under the 65535 coreDim limit. No two-level tiling needed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def vadd_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def vadd(x: torch.Tensor, scalar: float = 1.0, BLOCK_SIZE: int = 8192) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    vadd_kernel[grid](x, out, n, scalar, BLOCK_SIZE=BLOCK_SIZE)
    return out


def reference_vadd(x, scalar=1.0):
    return x + scalar


def test():
    torch.manual_seed(42)
    # 1D, default scalar
    x = torch.randn(4096, dtype=torch.float32, device='cpu')
    out = vadd(x)
    torch.testing.assert_close(out, reference_vadd(x))
    print("[PASS] vadd_real_1d_default", tuple(out.shape))

    # custom scalar
    x2 = torch.randn(4096, dtype=torch.float32, device='cpu')
    torch.testing.assert_close(vadd(x2, scalar=3.14), reference_vadd(x2, scalar=3.14))
    print("[PASS] vadd_real_1d_scalar")

    # 2D
    x3 = torch.randn(32, 128, dtype=torch.float32, device='cpu')
    torch.testing.assert_close(vadd(x3, scalar=-2.0), reference_vadd(x3, scalar=-2.0))
    print("[PASS] vadd_real_2d")


if __name__ == "__main__":
    test()
