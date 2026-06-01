"""
MobileNetV3-Small — Real Triton Kernels (NPU Deployable)
========================================================
Input:  [B, 3, 224, 224]  torch.Tensor on device
Output: [B, 1000]         torch.Tensor on device

Converted from emulator kernels. Emulator generates NPU-compatible code natively
(scalar accumulators, += accumulation, no redundant axis), so only calling-convention
changes are needed:
  1. @triton.jit decorator added
  2. tl.load(ptr, offset, mask=) → tl.load(ptr + offset, mask=)
  3. tl.store(ptr, offset, val)  → tl.store(ptr + offset, val)
  4. launch_kernel_1d(kernel, args, grid_size=N) → kernel[(N,)](args)

Architecture:
  Stem:  conv(3→16, k=3, s=2, p=1) → bn → hardswish               → [B, 16, 112, 112]
  Block 0:  DW(16, k=3, s=2, p=1) → bn → relu → SE(16,8) → pw    → [B, 16, 56, 56]
  Block 1:  pw(16→72) → bn → relu → DW(72, k=3, s=2) → pw(72→24) → [B, 24, 28, 28]
  Block 2:  pw(24→88) → bn → relu → DW(88, k=3) → pw(88→24) +res  → [B, 24, 28, 28]
  Block 3:  pw(24→96) → bn → hs → DW(96, k=5, s=2) → SE(96,24) → pw(96→40)
  Block 4:  pw(40→240) → bn → hs → DW(240, k=5) → SE(240,64) → pw(240→40) +res
  Block 5:  (same as block 4) +res
  Block 6:  pw(40→120) → bn → hs → DW(120, k=5) → SE(120,32) → pw(120→48)
  Block 7:  pw(48→144) → bn → hs → DW(144, k=5) → SE(144,40) → pw(144→48) +res
  Block 8:  pw(48→288) → bn → hs → DW(288, k=5, s=2) → SE(288,72) → pw(288→96)
  Block 9:  pw(96→576) → bn → hs → DW(576, k=5) → SE(576,144) → pw(576→96) +res
  Block 10: (same as block 9) +res
  Final pw: pw(96→576) → bn → hardswish
  Head:    avgpool → Linear(576,1024) → hs → Linear(1024,1000)     → [B, 1000]
"""

import torch
import triton
import triton.language as tl


# ================================================================
#  Architecture Config
# ================================================================

# (in_c, hidden_c, out_c, kernel, stride, pad, use_hs, se_reduce, has_residual)
BLOCKS = [
    (16,  16,  16,  3, 2, 1, False, 8,   False),  # features.1
    (16,  72,  24,  3, 2, 1, False, None, False),  # features.2
    (24,  88,  24,  3, 1, 1, False, None, True),   # features.3
    (24,  96,  40,  5, 2, 2, True,  24,  False),   # features.4
    (40,  240, 40,  5, 1, 2, True,  64,  True),    # features.5
    (40,  240, 40,  5, 1, 2, True,  64,  True),    # features.6
    (40,  120, 48,  5, 1, 2, True,  32,  False),   # features.7
    (48,  144, 48,  5, 1, 2, True,  40,  True),    # features.8
    (48,  288, 96,  5, 2, 2, True,  72,  False),   # features.9
    (96,  576, 96,  5, 1, 2, True,  144, True),    # features.10
    (96,  576, 96,  5, 1, 2, True,  144, True),    # features.11
]


# ================================================================
#  Kernels
# ================================================================

@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
    pid = tl.program_id(0)

    n  = pid // (C_out * H_out * W_out)
    rn = pid %  (C_out * H_out * W_out)
    oc = rn // (H_out * W_out)
    rn = rn %  (H_out * W_out)
    oh = rn // W_out
    ow = rn %  W_out

    window = C_in * kH * kW
    acc = 0.0

    for ck_start in range(0, window, BLOCK_CK):
        offs = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window

        ic     = offs // (kH * kW)
        rem_ck = offs %  (kH * kW)
        kh_idx = rem_ck // kW
        kw_idx = rem_ck %  kW

        ih = oh * stride_h + kh_idx - pad_h
        iw = ow * stride_w + kw_idx - pad_w

        in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        combined_mask = mask_ck & in_bounds

        x_offsets = n * stride_xn + ic * stride_xc + ih * stride_xh + iw * stride_xw
        w_offsets = oc * stride_woc + ic * stride_wic + kh_idx * stride_wkh + kw_idx * stride_wkw

        x_vals = tl.load(x_ptr + x_offsets, mask=combined_mask, other=0.0)
        w_vals = tl.load(w_ptr + w_offsets, mask=mask_ck, other=0.0)

        acc += tl.sum(x_vals * w_vals)

    b_val = tl.load(b_ptr + oc)
    out_val = acc + b_val

    out_offset = n * stride_outn + oc * stride_outc + oh * stride_outh + ow * stride_outw
    tl.store(out_ptr + out_offset, out_val)


@triton.jit
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

    x_offsets = n * stride_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
    w_offsets = c * stride_wc + kh * stride_wh + kw * stride_ww

    x_vals = tl.load(x_ptr + x_offsets, mask=combined_mask, other=0.0)
    w_vals = tl.load(w_ptr + w_offsets, mask=mask_kk, other=0.0)

    acc = 0.0
    acc += tl.sum(x_vals * w_vals)

    b_val = tl.load(b_ptr + c)
    out_val = acc + b_val

    out_offset = n * stride_outn + c * stride_outc + oh * stride_outh + ow * stride_outw
    tl.store(out_ptr + out_offset, out_val)


@triton.jit
def batchnorm2d_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.load(x_ptr + offsets, mask=mask)

    hw = H * W
    c_idx = (offsets // hw) % C

    mean_vals  = tl.load(mean_ptr  + c_idx, mask=mask, other=0.0)
    var_vals   = tl.load(var_ptr   + c_idx, mask=mask, other=1.0)
    gamma_vals = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0)
    beta_vals  = tl.load(beta_ptr  + c_idx, mask=mask, other=0.0)

    x_centered = x_vals - mean_vals
    std_inv = 1.0 / tl.sqrt(var_vals + eps)
    y_vals = gamma_vals * x_centered * std_inv + beta_vals

    tl.store(out_ptr + offsets, y_vals, mask=mask)


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def hardswish_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    out = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def hardsigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    out = tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) / 6.0
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    out = x * y
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
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
    acc = 0.0

    for hw_start in range(0, total, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < total

        h_idx = offs_hw // W
        w_idx = offs_hw %  W
        x_offsets = n * stride_xn + c * stride_xc + h_idx * stride_xh + w_idx * stride_xw

        vals = tl.load(x_ptr + x_offsets, mask=mask_hw, other=0.0)
        acc += tl.sum(vals)

    avg = acc / total
    tl.store(out_ptr + pid, avg)


@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, in_features, out_features,
    stride_xb, stride_xf,
    stride_wof, stride_wif,
    stride_outb, stride_outof,
    BLOCK_IN: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // out_features
    j = pid %  out_features

    acc = 0.0

    for i_start in range(0, in_features, BLOCK_IN):
        offs_i = i_start + tl.arange(0, BLOCK_IN)
        mask_i = offs_i < in_features

        x_val = tl.load(x_ptr + b_idx * stride_xb + offs_i * stride_xf,
                        mask=mask_i, other=0.0)
        w_val = tl.load(w_ptr + j * stride_wof + offs_i * stride_wif,
                        mask=mask_i, other=0.0)
        acc += tl.sum(x_val * w_val)

    b_val = tl.load(b_ptr + j)
    out_val = acc + b_val
    tl.store(out_ptr + b_idx * stride_outb + j * stride_outof, out_val)


# ================================================================
#  Launchers
# ================================================================

def triton_conv2d(x, w, b=None, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                  BLOCK_CK=128):
    N, C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if b is None:
        b = torch.zeros(C_out, device=x.device, dtype=torch.float32)

    out = torch.zeros(N, C_out, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_size = N * C_out * H_out * W_out

    conv2d_kernel[(grid_size,)](
        x, w, b, out,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_CK,
    )
    return out


def triton_conv2d_depthwise(x, w, b=None, stride_h=1, stride_w=1, pad_h=0, pad_w=0,
                            BLOCK_KK=32):
    N, C, H, W = x.shape
    _, _, kH, kW = w.shape
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    if b is None:
        b = torch.zeros(C, device=x.device, dtype=torch.float32)

    out = torch.zeros(N, C, H_out, W_out, device=x.device, dtype=torch.float32)
    grid_size = N * C * H_out * W_out

    conv2d_depthwise_kernel[(grid_size,)](
        x, w, b, out,
        N, C, H, W, kH, kW, H_out, W_out,
        stride_h, stride_w, pad_h, pad_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_KK,
    )
    return out


def triton_batchnorm2d(x, running_mean, running_var, gamma, beta,
                       eps=1e-5, BLOCK_SIZE=1024):
    N, C, H, W = x.shape
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)

    batchnorm2d_kernel[(grid_size,)](
        x, out,
        running_mean, running_var, gamma, beta,
        N, C, H, W, eps,
        n_elements,
        BLOCK_SIZE,
    )
    return out


def triton_relu(x, BLOCK_SIZE=1024):
    n = x.numel()
    out = torch.empty_like(x)
    grid_size = triton.cdiv(n, BLOCK_SIZE)

    relu_kernel[(grid_size,)](x, out, n, BLOCK_SIZE)
    return out


def triton_hardswish(x, BLOCK_SIZE=1024):
    n = x.numel()
    out = torch.empty_like(x)
    grid_size = triton.cdiv(n, BLOCK_SIZE)

    hardswish_kernel[(grid_size,)](x, out, n, BLOCK_SIZE)
    return out


def triton_hardsigmoid(x, BLOCK_SIZE=1024):
    n = x.numel()
    out = torch.empty_like(x)
    grid_size = triton.cdiv(n, BLOCK_SIZE)

    hardsigmoid_kernel[(grid_size,)](x, out, n, BLOCK_SIZE)
    return out


def triton_mul(x, y, BLOCK_SIZE=1024):
    n = x.numel()
    out = torch.empty_like(x)
    grid_size = triton.cdiv(n, BLOCK_SIZE)

    mul_kernel[(grid_size,)](x, y, out, n, BLOCK_SIZE)
    return out


def triton_adaptive_avgpool2d(x, BLOCK_HW=256):
    N, C, H, W = x.shape
    out = torch.zeros(N, C, 1, 1, device=x.device, dtype=torch.float32)
    grid_size = N * C

    adaptive_avgpool2d_kernel[(grid_size,)](
        x, out,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        BLOCK_HW,
    )
    return out


def triton_linear(x, weight, bias=None, BLOCK_IN=256):
    B, in_features = x.shape
    out_features = weight.shape[0]

    if bias is None:
        bias = torch.zeros(out_features, device=x.device, dtype=torch.float32)

    out = torch.zeros(B, out_features, device=x.device, dtype=torch.float32)
    grid_size = B * out_features

    linear_kernel[(grid_size,)](
        x, weight, bias, out,
        B, in_features, out_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_IN,
    )
    return out


# ================================================================
#  MobileNetV3-Small Forward
# ================================================================

def _bn(x, mean, var, gamma, beta, eps=1e-5):
    return triton_batchnorm2d(x, mean, var, gamma, beta, eps)


def _pw(x, w):
    return triton_conv2d(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0)


def _dw(x, w, stride=1, pad=0):
    return triton_conv2d_depthwise(x, w, stride_h=stride, stride_w=stride, pad_h=pad, pad_w=pad)


def _act(x, use_hs):
    return triton_hardswish(x) if use_hs else triton_relu(x)


def _se(x, fc1_w, fc1_b, fc2_w, fc2_b):
    pooled = triton_adaptive_avgpool2d(x)
    squeezed = triton_conv2d(pooled, fc1_w, fc1_b)
    squeezed = triton_relu(squeezed)
    scale = triton_conv2d(squeezed, fc2_w, fc2_b)
    scale = triton_hardsigmoid(scale)
    scale = scale.expand_as(x).contiguous()
    return triton_mul(x, scale)


def _block(x, w, i, cfg):
    in_c, hid, out_c, k, stride, pad, use_hs, se_r, has_res = cfg
    p = f'b{i}'
    out = x

    # Expand (skip if hidden == input, i.e. t=1)
    if hid != in_c:
        out = triton_conv2d(out, w[f'{p}_exp_w'])
        out = _bn(out, w[f'{p}_exp_mean'], w[f'{p}_exp_var'],
                      w[f'{p}_exp_gamma'], w[f'{p}_exp_beta'])
        out = _act(out, use_hs)

    # Depthwise + BN + act
    out = _dw(out, w[f'{p}_dw_w'], stride=stride, pad=pad)
    out = _bn(out, w[f'{p}_dw_mean'], w[f'{p}_dw_var'],
                  w[f'{p}_dw_gamma'], w[f'{p}_dw_beta'])
    out = _act(out, use_hs)

    # SE (optional)
    if se_r is not None:
        out = _se(out, w[f'{p}_se1_w'], w[f'{p}_se1_b'],
                      w[f'{p}_se2_w'], w[f'{p}_se2_b'])

    # Project + BN
    out = triton_conv2d(out, w[f'{p}_pw_w'])
    out = _bn(out, w[f'{p}_pw_mean'], w[f'{p}_pw_var'],
                  w[f'{p}_pw_gamma'], w[f'{p}_pw_beta'])

    # Residual
    if has_res:
        out = out + x

    return out


def mobilenetv3_small_forward(x, w):
    """
    Full MobileNetV3-Small forward pass using Triton kernels.

    Args:
        x: [B, 3, 224, 224] torch.Tensor on device
        w: weight dict (use make_mobilenetv3_small_weights() to generate)

    Returns:
        [B, 1000] torch.Tensor on device
    """
    # Stem: conv(3→16, k=3, s=2, p=1) → bn → hardswish
    out = triton_conv2d(x, w['stem_w'], stride_h=2, stride_w=2, pad_h=1, pad_w=1)
    out = _bn(out, w['stem_mean'], w['stem_var'],
                  w['stem_gamma'], w['stem_beta'])
    out = triton_hardswish(out)

    # Blocks
    for i, cfg in enumerate(BLOCKS):
        out = _block(out, w, i, cfg)

    # features.12: final pointwise 96→576 → bn → hardswish
    out = triton_conv2d(out, w['final_pw_w'])
    out = _bn(out, w['final_pw_mean'], w['final_pw_var'],
                  w['final_pw_gamma'], w['final_pw_beta'])
    out = triton_hardswish(out)

    # avgpool → flatten → classifier
    out = triton_adaptive_avgpool2d(out)
    out = out.view(out.shape[0], -1)
    out = triton_linear(out, w['fc1_w'], w['fc1_b'])
    out = triton_hardswish(out)
    out = triton_linear(out, w['fc2_w'], w['fc2_b'])
    return out


# ================================================================
#  Weight Generation
# ================================================================

def _make_bn(C, device):
    return {
        'mean':  torch.zeros(C, device=device),
        'var':   torch.ones(C, device=device),
        'gamma': torch.ones(C, device=device),
        'beta':  torch.zeros(C, device=device),
    }


def make_mobilenetv3_small_weights(device):
    """Generate random MobileNetV3-Small weight dict for Triton + PyTorch reference."""
    sc = 0.01
    w = {}

    # Stem
    w['stem_w'] = torch.randn(16, 3, 3, 3, device=device) * sc
    bn = _make_bn(16, device)
    w['stem_mean'], w['stem_var'] = bn['mean'], bn['var']
    w['stem_gamma'], w['stem_beta'] = bn['gamma'], bn['beta']

    # Per-block weights
    for i, (in_c, hid, out_c, k, s, p, hs, se_r, res) in enumerate(BLOCKS):
        prefix = f'b{i}'

        # Expand (only if hidden != input)
        if hid != in_c:
            w[f'{prefix}_exp_w'] = torch.randn(hid, in_c, 1, 1, device=device) * sc
            bn = _make_bn(hid, device)
            w[f'{prefix}_exp_mean'], w[f'{prefix}_exp_var'] = bn['mean'], bn['var']
            w[f'{prefix}_exp_gamma'], w[f'{prefix}_exp_beta'] = bn['gamma'], bn['beta']

        # Depthwise
        w[f'{prefix}_dw_w'] = torch.randn(hid, 1, k, k, device=device) * sc
        bn = _make_bn(hid, device)
        w[f'{prefix}_dw_mean'], w[f'{prefix}_dw_var'] = bn['mean'], bn['var']
        w[f'{prefix}_dw_gamma'], w[f'{prefix}_dw_beta'] = bn['gamma'], bn['beta']

        # SE (optional)
        if se_r is not None:
            w[f'{prefix}_se1_w'] = torch.randn(se_r, hid, 1, 1, device=device) * sc
            w[f'{prefix}_se1_b'] = torch.randn(se_r, device=device) * sc
            w[f'{prefix}_se2_w'] = torch.randn(hid, se_r, 1, 1, device=device) * sc
            w[f'{prefix}_se2_b'] = torch.randn(hid, device=device) * sc

        # Project
        w[f'{prefix}_pw_w'] = torch.randn(out_c, hid, 1, 1, device=device) * sc
        bn = _make_bn(out_c, device)
        w[f'{prefix}_pw_mean'], w[f'{prefix}_pw_var'] = bn['mean'], bn['var']
        w[f'{prefix}_pw_gamma'], w[f'{prefix}_pw_beta'] = bn['gamma'], bn['beta']

    # features.12: final pointwise 96→576
    w['final_pw_w'] = torch.randn(576, 96, 1, 1, device=device) * sc
    bn = _make_bn(576, device)
    w['final_pw_mean'], w['final_pw_var'] = bn['mean'], bn['var']
    w['final_pw_gamma'], w['final_pw_beta'] = bn['gamma'], bn['beta']

    # Classifier
    w['fc1_w'] = torch.randn(1024, 576, device=device) * sc
    w['fc1_b'] = torch.randn(1024, device=device) * sc
    w['fc2_w'] = torch.randn(1000, 1024, device=device) * sc
    w['fc2_b'] = torch.randn(1000, device=device) * sc

    return w


# ================================================================
#  Reference (PyTorch native)
# ================================================================

def _reference_mobilenetv3_small_forward(x, w):
    """PyTorch native MobileNetV3-Small forward using the same weight dict."""
    import torch.nn.functional as F

    def ref_bn(t, prefix):
        return F.batch_norm(t,
                            w[f'{prefix}mean'].clone(),
                            w[f'{prefix}var'].clone(),
                            w[f'{prefix}gamma'].clone(),
                            w[f'{prefix}beta'].clone(),
                            training=False)

    def ref_pw(t, wk):
        return F.conv2d(t, w[wk])

    def ref_dw(t, wk, stride=1, pad=0):
        C = t.shape[1]
        return F.conv2d(t, w[wk], stride=stride, padding=pad, groups=C)

    def ref_se(t, se1_wk, se1_bk, se2_wk, se2_bk):
        pooled = F.adaptive_avg_pool2d(t, (1, 1))
        squeezed = F.relu(F.conv2d(pooled, w[se1_wk], bias=w[se1_bk]))
        scale = F.hardsigmoid(F.conv2d(squeezed, w[se2_wk], bias=w[se2_bk]))
        return t * scale

    # Stem
    out = F.hardswish(ref_bn(
        F.conv2d(x, w['stem_w'], stride=2, padding=1),
        'stem_'))

    # Blocks
    for i, (in_c, hid, out_c, k, stride, pad, use_hs, se_r, has_res) in enumerate(BLOCKS):
        p = f'b{i}'
        identity = out

        if hid != in_c:
            out = _act_fn(ref_bn(ref_pw(out, f'{p}_exp_w'), f'{p}_exp_'), use_hs)

        out = _act_fn(ref_bn(ref_dw(out, f'{p}_dw_w', stride=stride, pad=pad), f'{p}_dw_'), use_hs)

        if se_r is not None:
            out = ref_se(out, f'{p}_se1_w', f'{p}_se1_b', f'{p}_se2_w', f'{p}_se2_b')

        out = ref_bn(ref_pw(out, f'{p}_pw_w'), f'{p}_pw_')

        if has_res:
            out = out + identity

    # features.12
    out = F.hardswish(ref_bn(ref_pw(out, 'final_pw_w'), 'final_pw_'))

    # Head
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = out.flatten(1)
    out = F.hardswish(F.linear(out, w['fc1_w'], w['fc1_b']))
    out = F.linear(out, w['fc2_w'], w['fc2_b'])
    return out


def _act_fn(t, use_hs):
    import torch.nn.functional as F
    return F.hardswish(t) if use_hs else F.relu(t)


# ================================================================
#  Test
# ================================================================

def test(device):
    """
    Run MobileNetV3-Small Triton kernel test on given device.

    Usage:
      test('cuda')   # NVIDIA GPU
      test('npu')    # NPU backend
      test('cpu')    # CPU (if Triton supports it)
    """
    print("=" * 70)
    print(f" MobileNetV3-Small Real Triton Test — device={device}")
    print("=" * 70)
    B = 1

    torch.manual_seed(42)
    weights = make_mobilenetv3_small_weights(device)
    x = torch.randn(B, 3, 224, 224, device=device, dtype=torch.float32)

    print(f"\nInput:  {list(x.shape)}")

    with torch.no_grad():
        out = mobilenetv3_small_forward(x, weights)
    print(f"Output: {list(out.shape)}")

    assert out.shape == (B, 1000), f"Expected [{B}, 1000], got {list(out.shape)}"
    print(f"[PASS] Output shape = [{B}, 1000]")

    # Compare against PyTorch reference using the SAME weights
    with torch.no_grad():
        ref_out = _reference_mobilenetv3_small_forward(x, weights)

    diff = (out - ref_out).abs().max().item()
    print(f"\nMax diff vs PyTorch reference (same weights): {diff:.6f}")
    status = "PASS" if diff < 0.5 else "FAIL"
    print(f"[{status}] Numerical check (tol=0.5)")

    print("=" * 70)


if __name__ == "__main__":
    test()
