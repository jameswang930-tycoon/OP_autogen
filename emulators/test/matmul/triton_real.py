"""
Matmul — 真实 Triton 版本（可上板运行）
========================================

与 emulator 版本 (__init__.py) 的 diff 对照：
  1. import:  from common import tl  →  import triton + import triton.language as tl
  2. 装饰器:  def kernel(...)        →  @triton.jit def kernel(...)
  3. load/store:
     emulator: tl.load(base_ptr, offsets, mask=mask)  + .ravel() / .reshape()
     真实:    tl.load(ptr + offsets, mask=mask)        — 无需 ravel/reshape
  4. launch:
     emulator: launch_kernel_2d(kernel, args..., grid=(x,y))
     真实:    kernel[grid](args...)

kernel 核心计算逻辑（offset 计算、mask、dot、累加）完全不变。
"""

import torch
import triton
import triton.language as tl


# ---- [改动1+2] 真实 Triton kernel ----

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # [改动3] 指针算术前移到 tl.load 参数中，去掉 .ravel()/.reshape()
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[:, None] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile)

    # [改动3] store 同理
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[:, None] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[:, None] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ---- [改动4] Launcher（替代 emulate_matmul） ----

def matmul(a: torch.Tensor, b: torch.Tensor,
           BLOCK_M=64, BLOCK_N=64, BLOCK_K=32) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"
    assert a.shape[1] == b.shape[0], f"K mismatch: {a.shape} vs {b.shape}"

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return c


# ---- 验证 ----

def test():
    print("=" * 60)
    print(" Matmul Real Triton Test")
    print("=" * 60)

    for M, N, K, BM, BN, BK in [
        (128, 128, 128, 64, 64, 32),
        (64, 128, 32, 32, 32, 32),
        (37, 53, 19, 16, 16, 16),  # 非对齐
    ]:
        a = torch.randn(M, K, device='cuda', dtype=torch.float32)
        b = torch.randn(K, N, device='cuda', dtype=torch.float32)
        out = matmul(a, b, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
        ref = torch.mm(a, b)
        diff = (out - ref).abs().max().item()
        status = "PASS" if diff < 1e-2 else f"FAIL (max_diff={diff})"
        print(f"  [{status}] M={M}, N={N}, K={K}")

    print()


if __name__ == "__main__":
    test()
