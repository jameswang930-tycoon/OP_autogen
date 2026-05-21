"""
Transpose Emulator: 2D matrix transpose
=========================================
Kernel: out[j, i] = x[i, j]
Grid:   2D, 每个 program 转置一个 [BLOCK_M x BLOCK_N] tile
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_2d, verify, EmulatorError


# ---- Triton-style Kernel ----

def transpose_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    转置 kernel: 读取 x 的 [BLOCK_M, BLOCK_N] tile, 写入 out 的 [BLOCK_N, BLOCK_M] 位置。
    x shape: [M, N], out shape: [N, M]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # 读: x[offs_m, offs_n]
    x_offsets = (offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn).ravel()
    x_mask = ((offs_m[:, None] < M) & (offs_n[None, :] < N)).ravel()
    tile = tl.load(x_ptr, x_offsets, mask=x_mask).reshape(BLOCK_M, BLOCK_N)

    # 写: out[offs_n, offs_m] (注意维度交换)
    # out shape = [N, M], stride_om = M, stride_on = 1
    out_offsets = (offs_n[:, None] * stride_om + offs_m[None, :] * stride_on).ravel()
    out_mask = ((offs_n[:, None] < N) & (offs_m[None, :] < M)).ravel()

    # tile 需要转置后再 ravel
    tile_T = np.asarray(tile).T
    tl.store(out_ptr, out_offsets, xarray(tile_T.ravel(), in_fast_mem=True), mask=out_mask)


# ---- Emulator 封装 ----

def emulate_transpose(x: np.ndarray, BLOCK_M=32, BLOCK_N=32) -> np.ndarray:
    """
    在 CPU 上 emulate 2D transpose。
    
    参数:
      x: [M, N] → out: [N, M]
    
    错误检查:
      - 输入不是 2D
    """
    if x.ndim != 2:
        raise EmulatorError("transpose_kernel",
            f"Input must be 2D, got shape {x.shape}",
            {"hint": "For higher-dim transpose, reshape to 2D first or specify axes."})

    M, N = x.shape
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(M * N, dtype=np.float32)

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    launch_kernel_2d(
        transpose_kernel,
        x_flat, out_flat,
        M, N,
        N, 1,      # stride_xm, stride_xn (row-major [M,N])
        M, 1,      # stride_om, stride_on (row-major [N,M])
        BLOCK_M, BLOCK_N,
        grid=(grid_m, grid_n),
    )

    return out_flat.reshape(N, M)


# ---- Reference ----

def reference_transpose(x):
    return x.T.astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" Transpose Emulator Test")
    print("=" * 60)

    # Test 1: 方阵
    x = np.random.randn(64, 64).astype(np.float32)
    out = emulate_transpose(x)
    ref = reference_transpose(x)
    verify(out, ref, "transpose_square")

    # Test 2: 非方阵
    x2 = np.random.randn(32, 128).astype(np.float32)
    out2 = emulate_transpose(x2)
    ref2 = reference_transpose(x2)
    verify(out2, ref2, "transpose_rect")
    assert out2.shape == (128, 32), f"Shape wrong: {out2.shape}"
    print(f"  Shape check: input {x2.shape} → output {out2.shape} ✓")

    # Test 3: 非对齐
    x3 = np.random.randn(37, 53).astype(np.float32)
    out3 = emulate_transpose(x3, BLOCK_M=16, BLOCK_N=16)
    ref3 = reference_transpose(x3)
    verify(out3, ref3, "transpose_unaligned")

    # Test 4: 转置的转置 == 原矩阵
    x4 = np.random.randn(20, 50).astype(np.float32)
    out4 = emulate_transpose(emulate_transpose(x4))
    verify(out4, x4, "transpose_double")

    print()


if __name__ == "__main__":
    test()
