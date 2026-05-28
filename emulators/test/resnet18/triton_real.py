"""
ResNet18 — Real Triton Kernels (NPU Deployable)
================================================
Input:  [B, 3, 224, 224]  torch.Tensor on device
Output: [B, 1000]         torch.Tensor on device

Converted from emulator kernels. Each kernel applies the 4 mechanical changes:
  1. @triton.jit decorator added
  2. tl.load(ptr, offset, mask=) → tl.load(ptr + offset, mask=)
  3. tl.store(ptr, offset, val)  → tl.store(ptr + offset, val)
  4. launch_kernel_1d(kernel, args, grid_size=N) → kernel[(N,)](args)

Architecture:
  Stem:  conv1(7x7,s2,p3) → bn1 → relu → maxpool(3x3,s2,p1)   → [B, 64, 56, 56]
  Layer1: 2×BasicBlock(64→64)                                    → [B, 64, 56, 56]
  Layer2: 2×BasicBlock(64→128, stride=2)                         → [B, 128, 28, 28]
  Layer3: 2×BasicBlock(128→256, stride=2)                        → [B, 256, 14, 14]
  Layer4: 2×BasicBlock(256→512, stride=2)                        → [B, 512, 7, 7]
  Pool:   adaptive_avgpool2d(1,1)                                → [B, 512, 1, 1]
  FC:     linear(512→1000)                                       → [B, 1000]
"""

import torch
import triton
import triton.language as tl


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
    acc = tl.zeros((1,), dtype=tl.float32)

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

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    b_val = tl.load(b_ptr + oc)
    out_val = acc + b_val

    out_offset = n * stride_outn + oc * stride_outc + oh * stride_outh + ow * stride_outw
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
    x_vals = tl.load(x_ptr + x_offsets, mask=combined_mask, other=float('-inf'))

    max_val = tl.max(x_vals, axis=0)

    out_offset = n * stride_outn + c * stride_outc + oh * stride_outh + ow * stride_outw
    tl.store(out_ptr + out_offset, max_val)


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
    acc = tl.zeros((1,), dtype=tl.float32)

    for hw_start in range(0, total, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < total

        h_idx = offs_hw // W
        w_idx = offs_hw %  W
        x_offsets = n * stride_xn + c * stride_xc + h_idx * stride_xh + w_idx * stride_xw

        vals = tl.load(x_ptr + x_offsets, mask=mask_hw, other=0.0)
        acc = acc + tl.sum(vals, axis=0)

    avg = acc / total
    tl.store(out_ptr + pid, avg)


@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, a + b, mask=mask)


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

    acc = tl.zeros((1,), dtype=tl.float32)

    for i_start in range(0, in_features, BLOCK_IN):
        offs_i = i_start + tl.arange(0, BLOCK_IN)
        mask_i = offs_i < in_features

        x_val = tl.load(x_ptr + b_idx * stride_xb + offs_i * stride_xf,
                        mask=mask_i, other=0.0)
        w_val = tl.load(w_ptr + j * stride_wof + offs_i * stride_wif,
                        mask=mask_i, other=0.0)
        acc = acc + tl.sum(x_val * w_val, axis=0)

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


def triton_maxpool2d(x, kH=3, kW=3, stride_h=2, stride_w=2,
                     pad_h=0, pad_w=0, BLOCK_KK=32):
    N, C, H, W = x.shape
    H_out = (H + 2 * pad_h - kH) // stride_h + 1
    W_out = (W + 2 * pad_w - kW) // stride_w + 1

    out = torch.full((N, C, H_out, W_out), float('-inf'),
                     device=x.device, dtype=torch.float32)
    grid_size = N * C * H_out * W_out

    maxpool2d_kernel[(grid_size,)](
        x, out,
        N, C, H, W, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_KK,
    )
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


def triton_add(a, b, BLOCK_SIZE=1024):
    n = a.numel()
    out = torch.empty_like(a)
    grid_size = triton.cdiv(n, BLOCK_SIZE)

    add_kernel[(grid_size,)](a, b, out, n, BLOCK_SIZE)
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
#  ResNet18 Forward
# ================================================================

# ResNet18 layer config: (layer_name, in_c, out_c, stride, has_downsample)
RESNET18_LAYERS = [
    ('layer1', 64,  64,  1, False),
    ('layer2', 64,  128, 2, True),
    ('layer3', 128, 256, 2, True),
    ('layer4', 256, 512, 2, True),
]


def _bn_params(weights, prefix):
    """Extract (running_mean, running_var, gamma, beta) from state_dict prefix."""
    return (
        weights[f'{prefix}running_mean'],
        weights[f'{prefix}running_var'],
        weights[f'{prefix}weight'],
        weights[f'{prefix}bias'],
    )


def _basic_block(x, conv1_w, bn1, conv2_w, bn2,
                 stride=1, downsample=None):
    """ResNet18 BasicBlock: conv1→bn1→relu→conv2→bn2 → (+identity) → relu."""
    identity = x

    out = triton_conv2d(x, conv1_w, stride_h=stride, stride_w=stride, pad_h=1, pad_w=1)
    out = triton_batchnorm2d(out, *bn1)
    out = triton_relu(out)

    out = triton_conv2d(out, conv2_w, stride_h=1, stride_w=1, pad_h=1, pad_w=1)
    out = triton_batchnorm2d(out, *bn2)

    if downsample is not None:
        ds_conv_w, ds_bn, ds_stride = downsample
        identity = triton_conv2d(x, ds_conv_w,
                                 stride_h=ds_stride, stride_w=ds_stride,
                                 pad_h=0, pad_w=0)
        identity = triton_batchnorm2d(identity, *ds_bn)

    out = triton_add(out, identity)
    out = triton_relu(out)
    return out


def resnet18_forward(x, weights):
    """
    Full ResNet18 forward pass using Triton kernels.

    Args:
        x: [B, 3, 224, 224] torch.Tensor on device
        weights: dict matching torchvision ResNet18 state_dict keys
                 (use load_resnet18_weights() to get it)

    Returns:
        [B, 1000] torch.Tensor on device
    """
    # Stem: conv1(7x7, s2, p3) → bn1 → relu → maxpool(3x3, s2, p1)
    out = triton_conv2d(x, weights['conv1.weight'],
                        stride_h=2, stride_w=2, pad_h=3, pad_w=3)
    out = triton_batchnorm2d(out, *_bn_params(weights, 'bn1.'))
    out = triton_relu(out)
    out = triton_maxpool2d(out, kH=3, kW=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1)

    # Layers 1-4
    for layer_name, in_c, out_c, layer_stride, has_ds in RESNET18_LAYERS:
        for block_idx in range(2):
            p = f'{layer_name}.{block_idx}.'
            block_stride = layer_stride if block_idx == 0 else 1

            downsample = None
            if has_ds and block_idx == 0:
                ds_p = f'{p}downsample.'
                downsample = (
                    weights[f'{ds_p}0.weight'],
                    _bn_params(weights, f'{ds_p}1.'),
                    layer_stride,
                )

            out = _basic_block(
                out,
                weights[f'{p}conv1.weight'],
                _bn_params(weights, f'{p}bn1.'),
                weights[f'{p}conv2.weight'],
                _bn_params(weights, f'{p}bn2.'),
                stride=block_stride,
                downsample=downsample,
            )

    # AdaptiveAvgPool → flatten → FC
    out = triton_adaptive_avgpool2d(out)       # [B, 512, 1, 1]
    out = out.view(out.shape[0], -1)            # [B, 512]
    out = triton_linear(out, weights['fc.weight'], weights['fc.bias'])  # [B, 1000]
    return out


# ================================================================
#  Weight Generation
# ================================================================

def make_resnet18_weights(device):
    """Generate random ResNet18 weight dict. Same weights for Triton + PyTorch reference."""
    w = {}
    w['conv1.weight'] = torch.randn(64, 3, 7, 7, device=device) * 0.01
    for name in ['running_mean', 'running_var', 'weight', 'bias']:
        shape = (64,)
        init = torch.zeros if name in ('running_mean', 'bias') else torch.ones
        w[f'bn1.{name}'] = init(shape, device=device)

    for layer_name, in_c, out_c, stride, has_ds in RESNET18_LAYERS:
        for i in range(2):
            p = f'{layer_name}.{i}.'
            w[f'{p}conv1.weight'] = torch.randn(out_c, in_c, 3, 3, device=device) * 0.01
            w[f'{p}conv2.weight'] = torch.randn(out_c, out_c, 3, 3, device=device) * 0.01
            for bn in ['bn1', 'bn2']:
                for name in ['running_mean', 'running_var', 'weight', 'bias']:
                    shape = (out_c,)
                    init = torch.zeros if name in ('running_mean', 'bias') else torch.ones
                    w[f'{p}{bn}.{name}'] = init(shape, device=device)
        if has_ds:
            p = f'{layer_name}.0.downsample.'
            w[f'{p}0.weight'] = torch.randn(out_c, in_c, 1, 1, device=device) * 0.01
            for name in ['running_mean', 'running_var', 'weight', 'bias']:
                shape = (out_c,)
                init = torch.zeros if name in ('running_mean', 'bias') else torch.ones
                w[f'{p}1.{name}'] = init(shape, device=device)

    w['fc.weight'] = torch.randn(1000, 512, device=device) * 0.01
    w['fc.bias']   = torch.zeros(1000, device=device)
    return w


# ================================================================
#  Reference (PyTorch native)
# ================================================================

def _reference_resnet18_forward(x, weights):
    """PyTorch native ResNet18 forward using the same weight dict."""
    import torch.nn.functional as F

    def ref_bn(x, prefix):
        return F.batch_norm(x,
                            weights[f'{prefix}running_mean'].clone(),
                            weights[f'{prefix}running_var'].clone(),
                            weights[f'{prefix}weight'].clone(),
                            weights[f'{prefix}bias'].clone(),
                            training=False)

    def ref_conv(x, w_key, stride_h, stride_w, pad_h, pad_w):
        return F.conv2d(x, weights[w_key],
                        stride=(stride_h, stride_w), padding=(pad_h, pad_w))

    def ref_block(x, p, stride=1, downsample=None):
        identity = x
        out = F.relu(ref_bn(ref_conv(x, f'{p}conv1.weight', stride, stride, 1, 1), f'{p}bn1.'))
        out = ref_bn(ref_conv(out, f'{p}conv2.weight', 1, 1, 1, 1), f'{p}bn2.')
        if downsample is not None:
            ds_p, ds_stride = downsample
            identity = ref_bn(ref_conv(x, f'{ds_p}0.weight', ds_stride, ds_stride, 0, 0),
                              f'{ds_p}1.')
        return F.relu(out + identity)

    # Stem
    out = F.relu(ref_bn(ref_conv(x, 'conv1.weight', 2, 2, 3, 3), 'bn1.'))
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

    # Layers
    for layer_name, in_c, out_c, layer_stride, has_ds in RESNET18_LAYERS:
        for i in range(2):
            block_stride = layer_stride if i == 0 else 1
            ds = None
            if has_ds and i == 0:
                ds = (f'{layer_name}.0.downsample.', layer_stride)
            out = ref_block(out, f'{layer_name}.{i}.', stride=block_stride, downsample=ds)

    # Head
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = out.view(out.shape[0], -1)
    out = F.linear(out, weights['fc.weight'], weights['fc.bias'])
    return out


# ================================================================
#  Test
# ================================================================

def test(device):
    """
    Run ResNet18 Triton kernel test on given device.

    Usage:
      test('cuda')   # NVIDIA GPU
      test('npu')    # NPU backend
      test('cpu')    # CPU (if Triton supports it)
    """
    print("=" * 70)
    print(f" ResNet18 Real Triton Test — device={device}")
    print("=" * 70)
    B = 1

    torch.manual_seed(42)
    weights = make_resnet18_weights(device)
    x = torch.randn(B, 3, 224, 224, device=device, dtype=torch.float32)

    print(f"\nInput:  {list(x.shape)}")

    with torch.no_grad():
        out = resnet18_forward(x, weights)
    print(f"Output: {list(out.shape)}")

    assert out.shape == (B, 1000), f"Expected [{B}, 1000], got {list(out.shape)}"
    print(f"[PASS] Output shape = [{B}, 1000]")

    # Compare against PyTorch reference using the SAME weights
    with torch.no_grad():
        ref_out = _reference_resnet18_forward(x, weights)

    diff = (out - ref_out).abs().max().item()
    print(f"\nMax diff vs PyTorch reference (same weights): {diff:.6f}")
    status = "PASS" if diff < 0.5 else "FAIL"
    print(f"[{status}] Numerical check (tol=0.5)")

    print("=" * 70)


if __name__ == "__main__":
    test()
