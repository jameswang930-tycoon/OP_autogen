"""
Conv2d Depthwise Emulator: Depthwise 2D Convolution with stride + padding
=========================================================================
Kernel: each output channel c only depends on input channel c.
        y[n,c,oh,ow] = sum_{kh,kw} x[n,c,ih,iw] * w[c,0,kh,kw] + b[c]
        where ih = oh*stride_h + kh - pad_h, iw = ow*stride_w + kw - pad_w
Grid:   1D, grid_size = N * C * H_out * W_out, each program computes one output pixel
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Kernel
# ============================================================

def conv2d_depthwise_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W, kH, kW, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wh, stride_ww,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_KK: tl.constexpr,
):
    pid = tl.program_id(0)

    n  = pid // (C * H_out * W_out)
    rn = pid %  (C * H_out * W_out)
    c  = rn // (H_out * W_out)
    rn = rn %  (H_out * W_out)
    oh = rn // W_out
    ow = rn %  W_out

    window = kH * kW
    kk = tl.arange(0, BLOCK_KK)
    mask_kk = kk < window

    kh = kk // kW
    kw = kk % kW

    ih = oh * stride_h + kh - pad_h
    iw = ow * stride_w + kw - pad_w

    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    combined_mask = mask_kk & in_bounds

    # x: channel c only
    x_offsets = n * stride_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
    # w: [C, 1, kH, kW], channel c's filter
    w_offsets = c * stride_wc + kh * stride_wh + kw * stride_ww

    x_vals = tl.load(x_ptr, x_offsets, mask=combined_mask, other=0.0)
    w_vals = tl.load(w_ptr, w_offsets, mask=mask_kk, other=0.0)

    acc = 0.0
    acc += tl.sum(x_vals * w_vals)

    b_val = tl.load(b_ptr, c)
    out_val = acc + b_val

    out_offset = n * stride_outn + c * stride_outc + oh * stride_outh + ow * stride_outw
    tl.store(out_ptr, out_offset, out_val)


# ============================================================
# Emulator wrapper
# ============================================================

def emulate_conv2d_depthwise(x: np.ndarray, w: np.ndarray, b: np.ndarray = None,
                              stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                              BLOCK_KK=32) -> np.ndarray:
    """
    CPU-emulate depthwise 2D convolution.

    Args:
      x: [N, C, H, W]
      w: [C, 1, kH, kW]  (depthwise weight, 1 filter per channel)
      b: [C] or None
      stride_h, stride_w: stride
      pad_h, pad_w: padding
    """
    if x.ndim != 4:
        raise EmulatorError("conv2d_depthwise_kernel",
            f"x must be 4D [N,C,H,W], got {x.shape}")
    if w.ndim != 4:
        raise EmulatorError("conv2d_depthwise_kernel",
            f"w must be 4D [C,1,kH,kW], got {w.shape}")

    N, C, H, W = x.shape
    wC, w1, kH, kW = w.shape
    if wC != C or w1 != 1:
        raise EmulatorError("conv2d_depthwise_kernel",
            f"w shape {w.shape} must be [C,1,kH,kW] with C={C}")

    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if H_out <= 0 or W_out <= 0:
        raise EmulatorError("conv2d_depthwise_kernel",
            f"Output size invalid: H_out={H_out}, W_out={W_out}")

    if b is None:
        b = np.zeros(C, dtype=np.float32)
    if b.shape != (C,):
        raise EmulatorError("conv2d_depthwise_kernel",
            f"bias shape {b.shape} != ({C},)")

    x_flat = x.ravel().astype(np.float32)
    w_flat = w.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    out_flat = np.zeros(N * C * H_out * W_out, dtype=np.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = C * H * W, H * W, W, 1
    stride_wc, stride_wh, stride_ww = kH * kW, kW, 1
    stride_outn = C * H_out * W_out
    stride_outc = H_out * W_out
    stride_outh = W_out
    stride_outw = 1

    grid_size = N * C * H_out * W_out
    launch_kernel_1d(
        conv2d_depthwise_kernel,
        x_flat, w_flat, b_flat, out_flat,
        N, C, H, W, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wc, stride_wh, stride_ww,
        stride_outn, stride_outc, stride_outh, stride_outw,
        BLOCK_KK,
        grid_size=grid_size,
    )
    return out_flat.reshape(N, C, H_out, W_out)


# ============================================================
# Reference (PyTorch)
# ============================================================

def reference_conv2d_depthwise(x, w, b=None, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    C = x.shape[1]
    b_t = torch.tensor(b, dtype=torch.float32) if b is not None else None
    y_t = torch.nn.functional.conv2d(x_t, w_t, bias=b_t,
                                      stride=(stride_h, stride_w),
                                      padding=(pad_h, pad_w),
                                      groups=C)
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" Conv2d Depthwise Emulator Test")
    print("=" * 70)

    # Test 1: 3x3, stride=2, pad=1 (MobileNetV3 block 0)
    print("\n--- Test 1: 3x3, s=2, p=1 ---")
    x = np.random.randn(1, 16, 56, 56).astype(np.float32)
    w = np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.01
    b = np.random.randn(16).astype(np.float32)
    out = emulate_conv2d_depthwise(x, w, b, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref = reference_conv2d_depthwise(x, w, b, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    verify(out, ref, "dw_3x3_s2_p1", rtol=1e-3, atol=1e-4)

    # Test 2: 5x5, stride=2, pad=2 (MobileNetV3 block 4)
    print("\n--- Test 2: 5x5, s=2, p=2 ---")
    x2 = np.random.randn(1, 96, 28, 28).astype(np.float32)
    w2 = np.random.randn(96, 1, 5, 5).astype(np.float32) * 0.01
    b2 = np.random.randn(96).astype(np.float32)
    out2 = emulate_conv2d_depthwise(x2, w2, b2, stride_h=2, stride_w=2, pad_h=2, pad_w=2)
    ref2 = reference_conv2d_depthwise(x2, w2, b2, stride_h=2, stride_w=2, pad_h=2, pad_w=2)
    verify(out2, ref2, "dw_5x5_s2_p2", rtol=1e-3, atol=1e-4)

    # Test 3: 5x5, stride=1, pad=2 (MobileNetV3 block 5)
    print("\n--- Test 3: 5x5, s=1, p=2 ---")
    x3 = np.random.randn(1, 240, 14, 14).astype(np.float32)
    w3 = np.random.randn(240, 1, 5, 5).astype(np.float32) * 0.01
    out3 = emulate_conv2d_depthwise(x3, w3, stride_h=1, stride_w=1, pad_h=2, pad_w=2)
    ref3 = reference_conv2d_depthwise(x3, w3, stride_h=1, stride_w=1, pad_h=2, pad_w=2)
    verify(out3, ref3, "dw_5x5_s1_p2", rtol=1e-3, atol=1e-4)

    # Test 4: Batch > 1
    print("\n--- Test 4: Batch=4 ---")
    x4 = np.random.randn(4, 16, 8, 8).astype(np.float32)
    w4 = np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.01
    out4 = emulate_conv2d_depthwise(x4, w4, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    ref4 = reference_conv2d_depthwise(x4, w4, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    verify(out4, ref4, "dw_batch4", rtol=1e-3, atol=1e-4)

    # Test 5: No bias
    print("\n--- Test 5: No bias ---")
    out5 = emulate_conv2d_depthwise(x4, w4)
    ref5 = reference_conv2d_depthwise(x4, w4)
    verify(out5, ref5, "dw_no_bias", rtol=1e-3, atol=1e-4)

    # Test 6: 3x3, stride=1, pad=1 (MobileNetV3 block 2)
    print("\n--- Test 6: 3x3, s=1, p=1 ---")
    x6 = np.random.randn(1, 88, 28, 28).astype(np.float32)
    w6 = np.random.randn(88, 1, 3, 3).astype(np.float32) * 0.01
    out6 = emulate_conv2d_depthwise(x6, w6, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    ref6 = reference_conv2d_depthwise(x6, w6, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    verify(out6, ref6, "dw_3x3_s1_p1", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 70)
    print(" Conv2d Depthwise test complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
