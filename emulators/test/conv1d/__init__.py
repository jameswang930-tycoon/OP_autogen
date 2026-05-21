"""
Conv1d Emulator: 1D Convolution (valid padding, stride=1, dilation=1)
======================================================================
Kernel: y[N, C_out, L_out] = x[N, C_in, L] * w[C_out, C_in, kL] + b[C_out]
Grid:   1D, grid_size = N * C_out * L_out, 每个 program 计算一个输出位置
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError, TraceLogger, AggregatedEmulatorError, run_with_feedback


# ============================================================
# Correct Kernel
# ============================================================

def conv1d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    pid = tl.program_id(0)

    # pid -> (n, oc, ol)
    n  = pid // (C_out * L_out)
    rn = pid %  (C_out * L_out)
    oc = rn // L_out
    ol = rn %  L_out

    window = C_in * kL
    acc = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window

        # flat -> (ic, kl)
        ic     = offs // kL
        kl_idx = offs %  kL

        # x offsets: window around position ol
        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (ol + kl_idx) * stride_xl
        )
        # w offsets
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kl_idx * stride_wkl
        )

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    # Add bias
    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    # Store
    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        ol * stride_outl
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


# ============================================================
# Bug Kernel (模拟 LLM 错误)
# ============================================================

def conv1d_kernel_bug_axis(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    """Bug: tl.sum axis=1 对 1D 向量无效"""
    pid = tl.program_id(0)
    n  = pid // (C_out * L_out)
    rn = pid %  (C_out * L_out)
    oc = rn // L_out
    ol = rn %  L_out

    window = C_in * kL
    acc = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window

        ic     = offs // kL
        kl_idx = offs %  kL

        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (ol + kl_idx) * stride_xl
        )
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kl_idx * stride_wkl
        )

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        # BUG: axis=1 越界
        acc = acc + tl.sum(x_vals * w_vals, axis=1)

    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        ol * stride_outl
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


def conv1d_kernel_bug_kl_range(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    """Bug: L_out 公式写错, 导致 grid 覆盖超范围, OOB"""
    pid = tl.program_id(0)
    n  = pid // (C_out * L_out)
    rn = pid %  (C_out * L_out)
    oc = rn // L_out
    ol = rn %  L_out

    window = C_in * kL
    acc = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window

        ic     = offs // kL
        kl_idx = offs %  kL

        # BUG: ol 范围假设 L_out = L - kL (少 1), 但实际 grid 用到正确 L_out,
        # 导致末尾 ol 过大, x 索引 OOB
        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (ol + kl_idx) * stride_xl
        )
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kl_idx * stride_wkl
        )

        # BUG: 不加 mask — OOB 直接暴露
        x_vals = tl.load(x_ptr, x_offsets, mask=None)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        ol * stride_outl
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


# ============================================================
# Emulator 封装
# ============================================================

def emulate_conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray = None,
                   l_out_formula="correct",
                   kernel_fn=conv1d_kernel,
                   BLOCK_CK=128,
                   collect_errors=False) -> np.ndarray:
    if x.ndim != 3:
        raise EmulatorError("conv1d_kernel", f"x must be 3D [N,C,L], got {x.shape}")
    if w.ndim != 3:
        raise EmulatorError("conv1d_kernel", f"w must be 3D [C_out,C_in,kL], got {w.shape}")
    N, C_in, L = x.shape
    C_out, C_in2, kL = w.shape
    if C_in != C_in2:
        raise EmulatorError("conv1d_kernel", f"C_in mismatch: x has {C_in}, w has {C_in2}")

    if l_out_formula == "off_by_one":
        L_out = L - kL       # Bug: 少 +1
    else:
        L_out = L - kL + 1

    if L_out <= 0:
        raise EmulatorError("conv1d_kernel",
            f"L_out={L_out} <= 0 (L={L}, kL={kL})")

    if b is None:
        b = np.zeros(C_out, dtype=np.float32)
    if b.shape != (C_out,):
        raise EmulatorError("conv1d_kernel", f"bias shape {b.shape} != ({C_out},)")

    x_flat = x.ravel().astype(np.float32)
    w_flat = w.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    out_flat = np.zeros(N * C_out * L_out, dtype=np.float32)

    stride_xn, stride_xc, stride_xl = C_in * L, L, 1
    stride_woc, stride_wic, stride_wkl = C_in * kL, kL, 1
    stride_outn, stride_outc, stride_outl = C_out * L_out, L_out, 1

    grid_size = N * C_out * L_out
    launch_kernel_1d(
        kernel_fn,
        x_flat, w_flat, b_flat, out_flat,
        N, C_in, L, C_out, kL, L_out,
        stride_xn, stride_xc, stride_xl,
        stride_woc, stride_wic, stride_wkl,
        stride_outn, stride_outc, stride_outl,
        BLOCK_CK,
        grid_size=grid_size,
        collect_errors=collect_errors,
    )
    return out_flat.reshape(N, C_out, L_out)


# ============================================================
# Reference (torch)
# ============================================================

def reference_conv1d(x, w, b=None):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32) if b is not None else None
    y_t = torch.nn.functional.conv1d(x_t, w_t, bias=b_t, stride=1, padding=0)
    return y_t.numpy()


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 70)
    print(" Conv1d Emulator Test — 含错误兜底评估")
    print("=" * 70)

    N, C_in, L = 2, 3, 16
    C_out, kL = 4, 5
    x = np.random.randn(N, C_in, L).astype(np.float32)
    w = np.random.randn(C_out, C_in, kL).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    # --- Test 0: Correct ---
    print("\n--- Test 0: Correct kernel ---")
    out = emulate_conv1d(x, w, b)
    ref = reference_conv1d(x, w, b)
    result = verify(out, ref, "conv1d_correct", rtol=1e-3, atol=1e-4)
    if result["passed"]:
        print("  => 正确版本通过验证")

    # --- Test 1: L_out off-by-one ---
    print("\n--- Test 1: L_out off-by-one ---")
    try:
        out1 = emulate_conv1d(x, w, b, l_out_formula="off_by_one")
        ref1 = reference_conv1d(x, w, b)
        verify(out1, ref1, "conv1d_bug_l_out")
    except EmulatorError as e:
        print(f"  [EmulatorError] {str(e)[:300]}")

    # --- Test 2: Wrong reduce axis ---
    print("\n--- Test 2: Wrong reduce axis ---")
    try:
        out2 = emulate_conv1d(x, w, b, kernel_fn=conv1d_kernel_bug_axis)
        ref2 = reference_conv1d(x, w, b)
        verify(out2, ref2, "conv1d_bug_axis")
    except EmulatorError as e:
        print(f"  [EmulatorError] {str(e)[:300]}")

    # --- Test 3: Window OOB (no mask) ---
    print("\n--- Test 3: Window OOB (no mask) ---")
    try:
        out3 = emulate_conv1d(x, w, b, kernel_fn=conv1d_kernel_bug_kl_range)
        ref3 = reference_conv1d(x, w, b)
        verify(out3, ref3, "conv1d_bug_kl_oob")
    except EmulatorError as e:
        print(f"  [EmulatorError] {str(e)[:300]}")

    # --- TraceLogger Demo ---
    print("\n--- TraceLogger Demo (正确 kernel, 小数据) ---")
    x_small = np.random.randn(1, 2, 6).astype(np.float32)
    w_small = np.random.randn(1, 2, 3).astype(np.float32)
    b_small = np.random.randn(1).astype(np.float32)

    TraceLogger.enable()
    out_small = emulate_conv1d(x_small, w_small, b_small, BLOCK_CK=8)
    ref_small = reference_conv1d(x_small, w_small, b_small)
    result_small = verify(out_small, ref_small, "conv1d_trace_demo")
    if result_small.get("trace"):
        print(result_small["trace"][:2000])
    TraceLogger.disable()

    # --- Test 4: run_with_feedback 去重演示 (OOB + collect_errors) ---
    print("\n--- Test 4: run_with_feedback + collect_errors (OOB 去重) ---")
    result4 = run_with_feedback(
        lambda: emulate_conv1d(x, w, b, kernel_fn=conv1d_kernel_bug_kl_range, collect_errors=True),
        lambda: reference_conv1d(x, w, b),
        op_name="conv1d_oob_dedup"
    )
    print(f"  passed: {result4['passed']}")
    if result4["feedback"]:
        print(f"  feedback:\n{result4['feedback']}")

    print("\n" + "=" * 70)
    print(" Conv1d 错误兜底评估完成")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
