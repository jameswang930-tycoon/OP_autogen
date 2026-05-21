"""
BatchNorm2d Emulator: Batch Normalization (eval/inference mode)
================================================================
Kernel: y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
Grid:   1D elementwise, each program handles BLOCK_SIZE elements
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Kernel
# ============================================================

def batchnorm2d_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x_vals = tl.load(x_ptr, offsets, mask=mask)

    # Channel index: [N,C,H,W] flat -> c = (flat_pos // (H*W)) % C
    hw = H * W
    c_idx = (offsets // hw) % C

    mean_vals  = tl.load(mean_ptr, c_idx, mask=mask, other=0.0)
    var_vals   = tl.load(var_ptr, c_idx, mask=mask, other=1.0)
    gamma_vals = tl.load(gamma_ptr, c_idx, mask=mask, other=1.0)
    beta_vals  = tl.load(beta_ptr, c_idx, mask=mask, other=0.0)

    x_centered = x_vals - mean_vals
    std_inv = 1.0 / tl.sqrt(var_vals + eps)
    y_vals = gamma_vals * x_centered * std_inv + beta_vals

    tl.store(out_ptr, offsets, y_vals, mask=mask)


# ============================================================
# Emulator wrapper
# ============================================================

def emulate_batchnorm2d(x: np.ndarray,
                        running_mean: np.ndarray,
                        running_var: np.ndarray,
                        gamma: np.ndarray = None,
                        beta: np.ndarray = None,
                        eps: float = 1e-5,
                        BLOCK_SIZE=1024) -> np.ndarray:
    if x.ndim != 4:
        raise EmulatorError("batchnorm2d_kernel",
            f"x must be 4D [N,C,H,W], got {x.shape}")

    N, C, H, W = x.shape
    if running_mean.shape != (C,):
        raise EmulatorError("batchnorm2d_kernel",
            f"running_mean shape {running_mean.shape} != ({C},)")
    if running_var.shape != (C,):
        raise EmulatorError("batchnorm2d_kernel",
            f"running_var shape {running_var.shape} != ({C},)")

    if gamma is None:
        gamma = np.ones(C, dtype=np.float32)
    if beta is None:
        beta = np.zeros(C, dtype=np.float32)

    n_elements = x.size
    x_flat = x.ravel().astype(np.float32)
    out_flat = np.zeros(n_elements, dtype=np.float32)
    mean_flat = running_mean.astype(np.float32)
    var_flat = running_var.astype(np.float32)
    gamma_flat = gamma.astype(np.float32)
    beta_flat = beta.astype(np.float32)

    grid_size = tl.cdiv(n_elements, BLOCK_SIZE)
    launch_kernel_1d(
        batchnorm2d_kernel,
        x_flat, out_flat,
        mean_flat, var_flat, gamma_flat, beta_flat,
        N, C, H, W, eps,
        n_elements,
        BLOCK_SIZE,
        grid_size=grid_size,
    )
    return out_flat.reshape(N, C, H, W)


# ============================================================
# Reference (PyTorch)
# ============================================================

def reference_batchnorm2d(x, running_mean, running_var, gamma=None, beta=None, eps=1e-5):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    mean_t = torch.tensor(running_mean, dtype=torch.float32)
    var_t = torch.tensor(running_var, dtype=torch.float32)
    gamma_t = torch.tensor(gamma, dtype=torch.float32) if gamma is not None else None
    beta_t = torch.tensor(beta, dtype=torch.float32) if beta is not None else None
    y_t = torch.nn.functional.batch_norm(x_t, mean_t, var_t,
                                          weight=gamma_t, bias=beta_t,
                                          training=False, momentum=0.1, eps=eps)
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" BatchNorm2d Emulator Test — Eval Mode")
    print("=" * 70)

    # Test 1: Basic
    N, C, H, W = 2, 64, 8, 8
    x = np.random.randn(N, C, H, W).astype(np.float32)
    running_mean = np.random.randn(C).astype(np.float32) * 0.5
    running_var = np.abs(np.random.randn(C).astype(np.float32)) + 0.5
    gamma = np.random.randn(C).astype(np.float32)
    beta = np.random.randn(C).astype(np.float32)

    print("\n--- Test 1: Basic BN (N=2, C=64, H=8, W=8) ---")
    out = emulate_batchnorm2d(x, running_mean, running_var, gamma, beta)
    ref = reference_batchnorm2d(x, running_mean, running_var, gamma, beta)
    verify(out, ref, "bn2d_basic", rtol=1e-3, atol=1e-4)

    # Test 2: Identity
    print("\n--- Test 2: Identity BN (gamma=1, beta=0) ---")
    out2 = emulate_batchnorm2d(x, running_mean, running_var)
    ref2 = reference_batchnorm2d(x, running_mean, running_var)
    verify(out2, ref2, "bn2d_identity", rtol=1e-3, atol=1e-4)

    # Test 3: Single channel
    print("\n--- Test 3: Single channel ---")
    x3 = np.random.randn(1, 1, 4, 4).astype(np.float32)
    out3 = emulate_batchnorm2d(x3, np.array([0.5]), np.array([1.0]))
    ref3 = reference_batchnorm2d(x3, np.array([0.5]), np.array([1.0]))
    verify(out3, ref3, "bn2d_single_ch", rtol=1e-3, atol=1e-4)

    # Test 4: ResNet18-like shape
    print("\n--- Test 4: ResNet18-like (N=1, C=64, H=56, W=56) ---")
    x4 = np.random.randn(1, 64, 56, 56).astype(np.float32)
    mean4 = np.random.randn(64).astype(np.float32) * 0.3
    var4 = np.abs(np.random.randn(64).astype(np.float32)) + 1.0
    out4 = emulate_batchnorm2d(x4, mean4, var4)
    ref4 = reference_batchnorm2d(x4, mean4, var4)
    verify(out4, ref4, "bn2d_resnet_shape", rtol=1e-3, atol=1e-4)

    print("\n" + "=" * 70)
    print(" BatchNorm2d test complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
