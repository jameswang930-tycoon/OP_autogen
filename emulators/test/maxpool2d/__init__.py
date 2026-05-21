"""
MaxPool2d Emulator: 2D Max Pooling with stride and padding
============================================================
Kernel: y[n,c,oh,ow] = max_{kh,kw} x[n, c, oh*stride_h+kh-pad_h, ow*stride_w+kw-pad_w]
        H_out = (H + 2*pad_h - kH) // stride_h + 1
Grid:   1D, grid_size = N * C * H_out * W_out, each program computes one output pixel
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Kernel
# ============================================================

def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W, kH, kW,
    stride_h, stride_w, pad_h, pad_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_KK: tl.constexpr,
):
    pid = tl.program_id(0)

    # pid -> (n, c, oh, ow)
    n  = pid // (C * H_out * W_out)
    rn = pid %  (C * H_out * W_out)
    c  = rn // (H_out * W_out)
    rn = rn %  (H_out * W_out)
    oh = rn // W_out
    ow = rn %  W_out

    # Build all window indices at once
    window = kH * kW
    kk = tl.arange(0, BLOCK_KK)
    mask_kk = kk < window

    kh = kk // kW
    kw = kk % kW

    ih = oh * stride_h + kh - pad_h
    iw = ow * stride_w + kw - pad_w

    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    combined_mask = mask_kk & in_bounds

    x_offsets = (
        n * stride_xn +
        c * stride_xc +
        ih * stride_xh +
        iw * stride_xw
    )

    # Load with -inf for out-of-bounds so they never win max
    x_vals = tl.load(x_ptr, x_offsets, mask=combined_mask, other=float('-inf'))

    # Reduce: max over window
    max_val = tl.max(x_vals, axis=0)

    # Store output — tl.max returns (1,) due to keepdims=True
    out_offset = (
        n * stride_outn +
        c * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    )
    tl.store(out_ptr, np.array([out_offset], dtype=np.int64), max_val)


# ============================================================
# Emulator wrapper
# ============================================================

def emulate_maxpool2d(x: np.ndarray, kH=3, kW=3,
                      stride_h=2, stride_w=2,
                      pad_h=0, pad_w=0,
                      BLOCK_KK=32) -> np.ndarray:
    """
    CPU-emulate 2D max pooling.

    Args:
      x: [N, C, H, W]
      kH, kW: pooling kernel size
      stride_h, stride_w: stride
      pad_h, pad_w: padding
    """
    if x.ndim != 4:
        raise EmulatorError("maxpool2d_kernel",
            f"x must be 4D [N,C,H,W], got {x.shape}")

    N, C, H, W = x.shape
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if H_out <= 0 or W_out <= 0:
        raise EmulatorError("maxpool2d_kernel",
            f"Output size invalid: H_out={H_out}, W_out={W_out}")

    x_flat = x.ravel().astype(np.float32)
    out_flat = np.full(N * C * H_out * W_out, float('-inf'), dtype=np.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = C * H * W, H * W, W, 1
    stride_outn = C * H_out * W_out
    stride_outc = H_out * W_out
    stride_outh = W_out
    stride_outw = 1

    grid_size = N * C * H_out * W_out
    launch_kernel_1d(
        maxpool2d_kernel,
        x_flat, out_flat,
        N, C, H, W, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_outn, stride_outc, stride_outh, stride_outw,
        BLOCK_KK,
        grid_size=grid_size,
    )
    return out_flat.reshape(N, C, H_out, W_out)


# ============================================================
# Reference (PyTorch)
# ============================================================

def reference_maxpool2d(x, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=0, pad_w=0):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.nn.functional.max_pool2d(x_t, kernel_size=(kH, kW),
                                          stride=(stride_h, stride_w),
                                          padding=(pad_h, pad_w))
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" MaxPool2d Emulator Test")
    print("=" * 70)

    # Test 1: ResNet18 config (3x3, stride=2, pad=1)
    print("\n--- Test 1: ResNet18 config (3x3, s=2, p=1) ---")
    x = np.random.randn(1, 64, 8, 8).astype(np.float32)
    out = emulate_maxpool2d(x, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref = reference_maxpool2d(x, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    verify(out, ref, "maxpool_resnet_cfg", rtol=1e-3, atol=1e-4)

    # Test 2: 2x2, stride=2, no padding
    print("\n--- Test 2: 2x2, s=2, p=0 ---")
    x2 = np.random.randn(1, 64, 8, 8).astype(np.float32)
    out2 = emulate_maxpool2d(x2, kH=2, kW=2, stride_h=2, stride_w=2)
    ref2 = reference_maxpool2d(x2, kH=2, kW=2, stride_h=2, stride_w=2)
    verify(out2, ref2, "maxpool_2x2_s2", rtol=1e-3, atol=1e-4)

    # Test 3: All-negative input
    print("\n--- Test 3: All-negative input ---")
    x_neg = -np.abs(np.random.randn(1, 4, 4, 4).astype(np.float32))
    out_neg = emulate_maxpool2d(x_neg, kH=2, kW=2, stride_h=2, stride_w=2)
    ref_neg = reference_maxpool2d(x_neg, kH=2, kW=2, stride_h=2, stride_w=2)
    verify(out_neg, ref_neg, "maxpool_all_neg", rtol=1e-3, atol=1e-4)

    # Test 4: Non-aligned spatial
    print("\n--- Test 4: Non-aligned spatial (13x17) ---")
    x4 = np.random.randn(1, 32, 13, 17).astype(np.float32)
    out4 = emulate_maxpool2d(x4, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref4 = reference_maxpool2d(x4, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    verify(out4, ref4, "maxpool_unaligned", rtol=1e-3, atol=1e-4)

    # Test 5: Batch > 1
    print("\n--- Test 5: Batch=4 ---")
    x5 = np.random.randn(4, 64, 8, 8).astype(np.float32)
    out5 = emulate_maxpool2d(x5, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref5 = reference_maxpool2d(x5, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    verify(out5, ref5, "maxpool_batch4", rtol=1e-3, atol=1e-4)

    # Test 6: Larger spatial (ResNet18-like stem output)
    print("\n--- Test 6: ResNet18 stem output shape (N=1, C=64, H=112, W=112) ---")
    x6 = np.random.randn(1, 64, 112, 112).astype(np.float32)
    out6 = emulate_maxpool2d(x6, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref6 = reference_maxpool2d(x6, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    verify(out6, ref6, "maxpool_large", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 70)
    print(" MaxPool2d test complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
