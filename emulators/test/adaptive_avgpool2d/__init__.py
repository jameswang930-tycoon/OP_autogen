"""
AdaptiveAvgPool2d Emulator: Global average pooling for ResNet18
================================================================
Kernel: y[n, c] = sum_{h,w}(x[n, c, h, w]) / (H * W)
Grid:   1D, grid_size = N * C, each program handles one (n, c) output
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Kernel
# ============================================================

def adaptive_avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid %  C

    total = H * W
    acc = tl.zeros((1,), dtype=tl.float32)

    for hw_start in range(0, total, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < total

        h_idx = offs_hw // W
        w_idx = offs_hw %  W

        x_offsets = (
            n * stride_xn +
            c * stride_xc +
            h_idx * stride_xh +
            w_idx * stride_xw
        )

        vals = tl.load(x_ptr, x_offsets, mask=mask_hw, other=0.0)
        acc = acc + tl.sum(vals, axis=0)

    avg = acc / total

    # Store to output[n, c] at flat position pid
    tl.store(out_ptr, np.array([pid], dtype=np.int64), avg)


# ============================================================
# Emulator wrapper
# ============================================================

def emulate_adaptive_avgpool2d(x: np.ndarray, BLOCK_HW=256) -> np.ndarray:
    """
    CPU-emulate global adaptive average pooling.

    Args:
      x: [N, C, H, W]

    Returns:
      y: [N, C, 1, 1]
    """
    if x.ndim != 4:
        raise EmulatorError("adaptive_avgpool2d_kernel",
            f"x must be 4D [N,C,H,W], got {x.shape}")

    N, C, H, W = x.shape
    if H == 0 or W == 0:
        raise EmulatorError("adaptive_avgpool2d_kernel",
            f"Spatial dimensions must be > 0, got H={H}, W={W}")

    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(N * C, dtype=np.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = C * H * W, H * W, W, 1

    grid_size = N * C
    launch_kernel_1d(
        adaptive_avgpool2d_kernel,
        x_flat, out_flat,
        N, C, H, W,
        stride_xn, stride_xc, stride_xh, stride_xw,
        BLOCK_HW,
        grid_size=grid_size,
    )
    return out_flat.reshape(N, C, 1, 1)


# ============================================================
# Reference (PyTorch)
# ============================================================

def reference_adaptive_avgpool2d(x):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.nn.functional.adaptive_avg_pool2d(x_t, (1, 1))
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" AdaptiveAvgPool2d Emulator Test")
    print("=" * 70)

    # Test 1: ResNet18 final layer (N=1, C=512, H=7, W=7)
    print("\n--- Test 1: ResNet18 final (1, 512, 7, 7) ---")
    x = np.random.randn(1, 512, 7, 7).astype(np.float32)
    out = emulate_adaptive_avgpool2d(x)
    ref = reference_adaptive_avgpool2d(x)
    verify(out, ref, "adaptpool_resnet", rtol=1e-3, atol=1e-4)

    # Test 2: Batch > 1
    print("\n--- Test 2: Batch=4 ---")
    x2 = np.random.randn(4, 512, 7, 7).astype(np.float32)
    out2 = emulate_adaptive_avgpool2d(x2)
    ref2 = reference_adaptive_avgpool2d(x2)
    verify(out2, ref2, "adaptpool_batch4", rtol=1e-3, atol=1e-4)

    # Test 3: 1x1 spatial (identity)
    print("\n--- Test 3: 1x1 spatial (identity) ---")
    x3 = np.random.randn(2, 64, 1, 1).astype(np.float32)
    out3 = emulate_adaptive_avgpool2d(x3)
    ref3 = reference_adaptive_avgpool2d(x3)
    verify(out3, ref3, "adaptpool_1x1", rtol=1e-3, atol=1e-4)

    # Test 4: Large spatial
    print("\n--- Test 4: Large spatial (1, 64, 56, 56) ---")
    x4 = np.random.randn(1, 64, 56, 56).astype(np.float32)
    out4 = emulate_adaptive_avgpool2d(x4)
    ref4 = reference_adaptive_avgpool2d(x4)
    verify(out4, ref4, "adaptpool_large", rtol=1e-3, atol=1e-4)

    # Test 5: Non-square
    print("\n--- Test 5: Non-square (1, 32, 8, 12) ---")
    x5 = np.random.randn(1, 32, 8, 12).astype(np.float32)
    out5 = emulate_adaptive_avgpool2d(x5)
    ref5 = reference_adaptive_avgpool2d(x5)
    verify(out5, ref5, "adaptpool_nonsquare", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 70)
    print(" AdaptiveAvgPool2d test complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
