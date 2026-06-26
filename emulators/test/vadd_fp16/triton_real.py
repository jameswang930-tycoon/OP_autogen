"""
Vadd_fp16 real Triton — grid=1 打满单核上板用例
================================================
N=32768 (64KB fp16), BLOCK_SIZE=32768, grid=1.
UB peak 128KB < 实际 192KB (不 overflow).
转换自 emulator 版 (5 处机械改写, kernel 逻辑不变, fp16).
"""

import torch
import triton
import triton.language as tl

DTYPE = torch.float16
BLOCK_DEFAULT = 32768


@triton.jit
def vadd_kernel(x_ptr, out_ptr, n_elements, scalar, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x + scalar
    tl.store(out_ptr + offsets, out, mask=mask)


def vadd(x: torch.Tensor, scalar: float = 1.0, BLOCK_SIZE: int = BLOCK_DEFAULT) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    vadd_kernel[grid](x, out, n, scalar, BLOCK_SIZE=BLOCK_SIZE)
    return out


def reference_vadd(x, scalar=1.0):
    return x + scalar


def test():
    torch.manual_seed(42)
    # grid=1 打满单核: N=32768, BLOCK=32768 (64KB fp16)
    x = torch.randn(32768, dtype=DTYPE, device='cpu')
    torch.testing.assert_close(vadd(x), reference_vadd(x))
    print("[PASS] vadd_fp16_real grid=1 (N=32768, 64KB)", tuple(x.shape))


if __name__ == "__main__":
    test()
