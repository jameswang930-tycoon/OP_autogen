"""
Vadd_fp16 — real Triton kernel (converted from the emulator version, fp16 storage).
=====================================================================================
5 mechanical rewrites from emulators/test/vadd_fp16/__init__.py (kernel compute logic
identical — only the calling convention changed). fp16 flows end-to-end:
  /triton-plan (dtype=fp16) -> /triton-gen (fp16 emulator) -> /triton-convert (fp16 real).
"""

import torch
import triton
import triton.language as tl

DTYPE = torch.float16   # matches plan dtype=fp16


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
    x = torch.randn(4096, dtype=DTYPE, device='cpu')
    torch.testing.assert_close(vadd(x), reference_vadd(x))
    print("[PASS] vadd_fp16_real_1d", tuple(x.shape))

    x2 = torch.randn(32, 128, dtype=DTYPE, device='cpu')
    torch.testing.assert_close(vadd(x2, scalar=-2.0), reference_vadd(x2, scalar=-2.0))
    print("[PASS] vadd_fp16_real_2d")


if __name__ == "__main__":
    test()
