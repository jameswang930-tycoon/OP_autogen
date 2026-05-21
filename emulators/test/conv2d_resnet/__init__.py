"""
Conv2d-ResNet Emulator: Generalized 2D Convolution with stride + padding
========================================================================
Kernel: y[N, C_out, H_out, W_out] = x[N, C_in, H, W] * w[C_out, C_in, kH, kW] + b[C_out]
        H_out = (H + 2*pad_h - kH) // stride_h + 1
        W_out = (W + 2*pad_w - kW) // stride_w + 1
Grid:   1D, grid_size = N * C_out * H_out * W_out, each program computes one output pixel
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Kernel
# ============================================================

def conv2d_resnet_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
    pid = tl.program_id(0)

    # pid -> (n, oc, oh, ow)
    n  = pid // (C_out * H_out * W_out)
    rn = pid %  (C_out * H_out * W_out)
    oc = rn // (H_out * W_out)
    rn = rn %  (H_out * W_out)
    oh = rn // W_out
    ow = rn %  W_out

    window = C_in * kH * kW
    acc = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window

        # flat -> (ic, kh, kw)
        ic     = offs // (kH * kW)
        rem_ck = offs %  (kH * kW)
        kh_idx = rem_ck // kW
        kw_idx = rem_ck %  kW

        # Input coordinates with stride and padding
        ih = oh * stride_h + kh_idx - pad_h
        iw = ow * stride_w + kw_idx - pad_w

        # Bounds check: in_bounds implements zero-padding via mask
        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        combined_mask = mask_ck & in_bounds

        # x offsets — only valid when in_bounds
        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            ih * stride_xh +
            iw * stride_xw
        )
        # w offsets
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kh_idx * stride_wkh +
            kw_idx * stride_wkw
        )

        # Load with combined mask: out-of-bounds positions get 0.0 (zero-padding)
        x_vals = tl.load(x_ptr, x_offsets, mask=combined_mask, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    # Add bias
    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    # Store output
    out_offset = (
        n  * stride_outn +
        oc * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    )
    tl.store(out_ptr, np.array([out_offset], dtype=np.int64), out_val)


# ============================================================
# Emulator wrapper
# ============================================================

def emulate_conv2d_resnet(x: np.ndarray, w: np.ndarray, b: np.ndarray = None,
                          stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                          BLOCK_CK=128) -> np.ndarray:
    """
    CPU-emulate generalized 2D convolution with stride + padding.

    Args:
      x: input  [N, C_in, H, W]
      w: weight [C_out, C_in, kH, kW]
      b: bias   [C_out], optional (default zeros)
      stride_h, stride_w: stride
      pad_h, pad_w: zero-padding
    """
    if x.ndim != 4:
        raise EmulatorError("conv2d_resnet_kernel",
            f"x must be 4D [N,C,H,W], got {x.shape}")
    if w.ndim != 4:
        raise EmulatorError("conv2d_resnet_kernel",
            f"w must be 4D [C_out,C_in,kH,kW], got {w.shape}")

    N, C_in, H, W = x.shape
    C_out, C_in2, kH, kW = w.shape
    if C_in != C_in2:
        raise EmulatorError("conv2d_resnet_kernel",
            f"C_in mismatch: x has {C_in}, w has {C_in2}")
    if stride_h <= 0 or stride_w <= 0:
        raise EmulatorError("conv2d_resnet_kernel",
            f"stride must be > 0, got stride_h={stride_h}, stride_w={stride_w}")

    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if H_out <= 0 or W_out <= 0:
        raise EmulatorError("conv2d_resnet_kernel",
            f"Output size invalid: H_out={H_out}, W_out={W_out} "
            f"(H={H}, W={W}, kH={kH}, kW={kW}, stride_h={stride_h}, pad_h={pad_h})")

    if b is None:
        b = np.zeros(C_out, dtype=np.float32)
    if b.shape != (C_out,):
        raise EmulatorError("conv2d_resnet_kernel",
            f"bias shape {b.shape} != ({C_out},)")

    x_flat = x.ravel().astype(np.float32)
    w_flat = w.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    out_flat = np.zeros(N * C_out * H_out * W_out, dtype=np.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = C_in * H * W, H * W, W, 1
    stride_woc, stride_wic, stride_wkh, stride_wkw = C_in * kH * kW, kH * kW, kW, 1
    stride_outn = C_out * H_out * W_out
    stride_outc = H_out * W_out
    stride_outh = W_out
    stride_outw = 1

    grid_size = N * C_out * H_out * W_out
    launch_kernel_1d(
        conv2d_resnet_kernel,
        x_flat, w_flat, b_flat, out_flat,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wic, stride_wkh, stride_wkw,
        stride_outn, stride_outc, stride_outh, stride_outw,
        BLOCK_CK,
        grid_size=grid_size,
    )
    return out_flat.reshape(N, C_out, H_out, W_out)


# ============================================================
# Reference (PyTorch)
# ============================================================

def reference_conv2d_resnet(x, w, b=None, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32) if b is not None else None
    y_t = torch.nn.functional.conv2d(x_t, w_t, bias=b_t,
                                      stride=(stride_h, stride_w),
                                      padding=(pad_h, pad_w))
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" Conv2d-ResNet Emulator Test — Generalized Conv2d with stride+padding")
    print("=" * 70)

    # ----------------------------------------------------------
    # Config B: 3x3, stride=1, pad=1 (most common in ResNet18)
    # ----------------------------------------------------------
    print("\n--- Config B: 3x3, stride=1, pad=1, 64->64 ---")
    N, C_in, H, W = 1, 64, 8, 8
    C_out, kH, kW = 64, 3, 3
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, b, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    ref = reference_conv2d_resnet(x, w, b, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    result_B = verify(out, ref, "conv2d_configB_3x3_s1_p1", rtol=1e-3, atol=1e-4)

    # ----------------------------------------------------------
    # Config A: 7x7, stride=2, pad=3 (ResNet18 stem conv)
    # ----------------------------------------------------------
    print("\n--- Config A: 7x7, stride=2, pad=3, 3->64 ---")
    N, C_in, H, W = 1, 3, 14, 14
    C_out, kH, kW = 64, 7, 7
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=3, pad_w=3)
    ref = reference_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=3, pad_w=3)
    result_A = verify(out, ref, "conv2d_configA_7x7_s2_p3", rtol=1e-3, atol=1e-4)

    # ----------------------------------------------------------
    # Config C: 3x3, stride=2, pad=1 (downsample conv)
    # ----------------------------------------------------------
    print("\n--- Config C: 3x3, stride=2, pad=1, 64->128 ---")
    N, C_in, H, W = 1, 64, 8, 8
    C_out, kH, kW = 128, 3, 3
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    ref = reference_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    result_C = verify(out, ref, "conv2d_configC_3x3_s2_p1", rtol=1e-3, atol=1e-4)

    # ----------------------------------------------------------
    # Config E: 1x1, stride=2, pad=0 (projection shortcut)
    # ----------------------------------------------------------
    print("\n--- Config E: 1x1, stride=2, pad=0, 64->128 ---")
    N, C_in, H, W = 1, 64, 8, 8
    C_out, kH, kW = 128, 1, 1
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=0, pad_w=0)
    ref = reference_conv2d_resnet(x, w, b, stride_h=2, stride_w=2, pad_h=0, pad_w=0)
    result_E = verify(out, ref, "conv2d_configE_1x1_s2_p0", rtol=1e-3, atol=1e-4)

    # ----------------------------------------------------------
    # Batch > 1
    # ----------------------------------------------------------
    print("\n--- Batch size = 4 ---")
    N, C_in, H, W = 4, 64, 8, 8
    C_out, kH, kW = 64, 3, 3
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, b, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    ref = reference_conv2d_resnet(x, w, b, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    verify(out, ref, "conv2d_batch4", rtol=1e-3, atol=1e-4)

    # ----------------------------------------------------------
    # No bias
    # ----------------------------------------------------------
    print("\n--- No bias ---")
    N, C_in, H, W = 1, 64, 8, 8
    C_out, kH, kW = 64, 3, 3
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)

    out = emulate_conv2d_resnet(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    ref = reference_conv2d_resnet(x, w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    verify(out, ref, "conv2d_no_bias", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 70)
    print(" Conv2d-ResNet test complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
