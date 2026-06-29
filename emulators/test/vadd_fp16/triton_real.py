"""
Vadd_fp16 real Triton — grid=1 打满单核上板用例
================================================
N=32768 (64KB fp16), BLOCK_SIZE=32768, grid=1.
UB peak 128KB < 实际 192KB (不 overflow).
转换自 emulator 版 (5 处机械改写, kernel 逻辑不变, fp16).

注意: 这是真实 triton kernel (@triton.jit), 必须在 GPU/NPU 上运行.
device 由 pick_device() 自动选 (cuda/npu), 不用 cpu (triton 不是为 cpu 设计的).
本地无 triton/硬件时跑不了 —— 上板用 npu (910B3) 或 cuda.
"""

import torch
import triton
import triton.language as tl

try:                       # noqa: SIM105
    import torch_npu       # noqa: F401
except ImportError:
    pass

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


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu'
    raise SystemExit("No CUDA/NPU device. 真实 triton kernel 需 GPU/NPU, 不支持 cpu.")


def test():
    torch.manual_seed(42)
    device = pick_device()
    # grid=1 打满单核: N=32768, BLOCK=32768 (64KB fp16)
    x = torch.randn(32768, dtype=DTYPE, device=device)
    out = vadd(x)
    ref = reference_vadd(x)
    ok = torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
    print(f"[{'PASS' if ok else 'FAIL'}] vadd_fp16_real grid=1 (N=32768, 64KB) device={device}")


if __name__ == "__main__":
    test()
