"""
AddRMSNormGamma Emulator: Fused Add + RMSNorm + Gamma
=======================================================
常见于 Transformer 的残差连接后归一化:
  residual = x + residual
  out = rmsnorm(residual) * gamma

这是一个 fused kernel, 将三步合为一个 kernel:
  1. residual add:   residual[i,j] = x[i,j] + residual_in[i,j]
  2. rms normalize:  norm[i,j] = residual[i,j] / sqrt(mean(residual[i,:]^2) + eps)
  3. gamma scale:    out[i,j] = norm[i,j] * gamma[j]

Grid: 1D, 每个 program 处理一行。
输出两个结果: out (归一化后) 和 residual (add 后, 供下一层残差连接使用)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ---- Triton-style Kernel ----

def add_rmsnorm_gamma_kernel(
    x_ptr, residual_ptr, gamma_ptr, out_ptr, residual_out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused add + rmsnorm + gamma kernel。
    
    输入:
      x_ptr:            [n_rows, n_cols] 当前层输出
      residual_ptr:     [n_rows, n_cols] 前一层残差 (同时作为输入和输出)
      gamma_ptr:        [n_cols]          可学习缩放参数
    
    输出:
      out_ptr:          [n_rows, n_cols] 归一化结果
      residual_out_ptr: [n_rows, n_cols] 更新后的残差 (= x + residual_in)
    """
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    row_start = row_idx * n_cols
    ptrs = row_start + col_offs

    # Step 1: Residual Add
    x = tl.load(x_ptr, ptrs, mask=mask)
    residual = tl.load(residual_ptr, ptrs, mask=mask)
    residual_new = x + residual

    # 写回更新后的 residual (供下一层使用)
    tl.store(residual_out_ptr, ptrs, residual_new, mask=mask)

    # Step 2: RMSNorm
    sq = residual_new * residual_new
    mean_sq = tl.sum(sq, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    normed = residual_new * rrms

    # Step 3: Gamma Scale
    gamma = tl.load(gamma_ptr, col_offs, mask=mask)
    out = normed * gamma

    tl.store(out_ptr, ptrs, out, mask=mask)


# ---- 简化版: 不输出 residual (只有 out) ----

def add_rmsnorm_gamma_kernel_simple(
    x_ptr, residual_ptr, gamma_ptr, out_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """简化版: 只输出归一化结果, 不单独输出 residual"""
    row_idx = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_SIZE)
    mask = col_offs < n_cols

    row_start = row_idx * n_cols
    ptrs = row_start + col_offs

    x = tl.load(x_ptr, ptrs, mask=mask)
    residual = tl.load(residual_ptr, ptrs, mask=mask)
    hidden = x + residual

    sq = hidden * hidden
    mean_sq = tl.sum(sq, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    normed = hidden * rrms

    gamma = tl.load(gamma_ptr, col_offs, mask=mask)
    out = normed * gamma

    tl.store(out_ptr, ptrs, out, mask=mask)


# ---- Emulator 封装 ----

def emulate_add_rmsnorm_gamma(
    x: np.ndarray,
    residual: np.ndarray,
    gamma: np.ndarray,
    eps: float = 1e-6,
    BLOCK_SIZE=None,
    return_residual=True,
) -> tuple:
    """
    在 CPU 上 emulate fused add + rmsnorm + gamma。
    
    参数:
      x:         [n_rows, n_cols]  当前层输出
      residual:  [n_rows, n_cols]  残差输入
      gamma:     [n_cols]          缩放参数
      eps:       稳定性常数
      return_residual: 是否同时返回更新后的 residual
    
    返回:
      (out, residual_new) if return_residual else out
    
    错误检查:
      - x, residual shape 不一致
      - gamma 长度不匹配
      - eps <= 0
      - NaN/Inf
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if residual.ndim == 1:
        residual = residual.reshape(1, -1)

    if x.ndim != 2:
        raise EmulatorError("add_rmsnorm_gamma_kernel",
            f"x must be 1D or 2D, got shape {x.shape}")
    if x.shape != residual.shape:
        raise EmulatorError("add_rmsnorm_gamma_kernel",
            f"x shape {x.shape} != residual shape {residual.shape}",
            {"hint": "x and residual must have identical shape."})

    n_rows, n_cols = x.shape

    if gamma.ndim != 1 or gamma.shape[0] != n_cols:
        raise EmulatorError("add_rmsnorm_gamma_kernel",
            f"gamma shape {gamma.shape} does not match n_cols={n_cols}",
            {"expected": f"({n_cols},)", "got": str(gamma.shape)})

    if eps <= 0:
        raise EmulatorError("add_rmsnorm_gamma_kernel", f"eps must be positive, got {eps}")

    for name, arr in [("x", x), ("residual", residual)]:
        if np.any(np.isnan(arr)):
            raise EmulatorError("add_rmsnorm_gamma_kernel",
                f"{name} contains {int(np.sum(np.isnan(arr)))} NaN values")
        if np.any(np.isinf(arr)):
            raise EmulatorError("add_rmsnorm_gamma_kernel",
                f"{name} contains {int(np.sum(np.isinf(arr)))} Inf values")

    if BLOCK_SIZE is None:
        BLOCK_SIZE = 1
        while BLOCK_SIZE < n_cols:
            BLOCK_SIZE *= 2

    x_flat = x.ravel().astype(np.float32)
    res_flat = residual.ravel().astype(np.float32)
    gamma_flat = gamma.ravel().astype(np.float32)
    out_flat = np.zeros_like(x_flat)

    if return_residual:
        res_out_flat = np.zeros_like(x_flat)
        launch_kernel_1d(
            add_rmsnorm_gamma_kernel,
            x_flat, res_flat, gamma_flat, out_flat, res_out_flat,
            n_cols, eps, BLOCK_SIZE,
            grid_size=n_rows,
        )
        return (out_flat.reshape(n_rows, n_cols),
                res_out_flat.reshape(n_rows, n_cols))
    else:
        launch_kernel_1d(
            add_rmsnorm_gamma_kernel_simple,
            x_flat, res_flat, gamma_flat, out_flat,
            n_cols, eps, BLOCK_SIZE,
            grid_size=n_rows,
        )
        return out_flat.reshape(n_rows, n_cols)


# ---- Reference ----

def reference_add_rmsnorm_gamma(x, residual, gamma, eps=1e-6):
    """NumPy reference: add → rmsnorm → gamma"""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if residual.ndim == 1:
        residual = residual.reshape(1, -1)

    # Step 1: add
    hidden = x + residual

    # Step 2: rmsnorm
    variance = np.mean(hidden ** 2, axis=-1, keepdims=True)
    normed = hidden / np.sqrt(variance + eps)

    # Step 3: gamma
    out = normed * gamma

    return out.astype(np.float32), hidden.astype(np.float32)


# ---- Self-test ----

def test():
    print("=" * 60)
    print(" AddRMSNormGamma Emulator Test")
    print("=" * 60)

    n_rows, n_cols = 8, 128

    # Test 1: 基本功能 (完整版, 同时返回 residual)
    x = np.random.randn(n_rows, n_cols).astype(np.float32)
    res = np.random.randn(n_rows, n_cols).astype(np.float32)
    gamma = np.random.randn(n_cols).astype(np.float32)

    out, res_new = emulate_add_rmsnorm_gamma(x, res, gamma)
    ref_out, ref_res = reference_add_rmsnorm_gamma(x, res, gamma)

    verify(out, ref_out, "add_rmsnorm_gamma_output")
    verify(res_new, ref_res, "add_rmsnorm_gamma_residual")

    # Test 2: 验证残差确实是 x + residual
    verify(res_new, x + res, "residual_is_x_plus_res")

    # Test 3: gamma 全 1 (纯 add + rmsnorm)
    gamma_ones = np.ones(n_cols, dtype=np.float32)
    out3, _ = emulate_add_rmsnorm_gamma(x, res, gamma_ones)
    ref3, _ = reference_add_rmsnorm_gamma(x, res, gamma_ones)
    verify(out3, ref3, "add_rmsnorm_unit_gamma")

    # Test 4: residual 全 0 (纯 rmsnorm(x) * gamma)
    res_zero = np.zeros_like(x)
    out4, _ = emulate_add_rmsnorm_gamma(x, res_zero, gamma)
    ref4, _ = reference_add_rmsnorm_gamma(x, res_zero, gamma)
    verify(out4, ref4, "add_rmsnorm_zero_residual")

    # Test 5: 简化版 (不返回 residual)
    out5 = emulate_add_rmsnorm_gamma(x, res, gamma, return_residual=False)
    verify(out5, ref_out, "add_rmsnorm_simple_mode")

    # Test 6: shape 不匹配报错
    try:
        emulate_add_rmsnorm_gamma(x, np.zeros((4, 64)), gamma)
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught x/residual shape mismatch")

    # Test 7: gamma 维度不匹配报错
    try:
        emulate_add_rmsnorm_gamma(x, res, np.ones(64, dtype=np.float32))
        print("  [FAIL] Should have raised EmulatorError")
    except EmulatorError:
        print("  [PASS] Correctly caught gamma dimension mismatch")

    # Test 8: 大维度
    x_big = np.random.randn(4, 4096).astype(np.float32)
    res_big = np.random.randn(4, 4096).astype(np.float32)
    gamma_big = np.random.randn(4096).astype(np.float32)
    out_big, _ = emulate_add_rmsnorm_gamma(x_big, res_big, gamma_big)
    ref_big, _ = reference_add_rmsnorm_gamma(x_big, res_big, gamma_big)
    verify(out_big, ref_big, "add_rmsnorm_large_4096")

    print()


if __name__ == "__main__":
    test()
