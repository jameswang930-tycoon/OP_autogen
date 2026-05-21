"""
Conv2d Emulator: 2D Convolution (valid padding, stride=1, dilation=1)
======================================================================
Kernel: y[N, C_out, H_out, W_out] = x[N, C_in, H, W] * w[C_out, C_in, kH, kW] + b[C_out]
Grid:   1D, grid_size = N * C_out * H_out * W_out, 每个 program 计算一个输出像素
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError, TraceLogger, AggregatedEmulatorError, run_with_feedback


# ============================================================
# Correct Kernel
# ============================================================

def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
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

        # x offsets: pick window around (oh, ow)
        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (oh + kh_idx) * stride_xh +
            (ow + kw_idx) * stride_xw
        )
        # w offsets: output channel oc, input channel ic, kernel pos (kh, kw)
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kh_idx * stride_wkh +
            kw_idx * stride_wkw
        )

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    # Add bias
    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    # Store to output
    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


# ============================================================
# Bug Kernels (模拟 LLM 常见错误, 用于评估反馈质量)
# ============================================================

def conv2d_kernel_bug_axis(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
    """Bug: tl.sum axis=1 写成了 axis=0 -> 但这里写 axis=1 才是 bug"""
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

        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (oh + kh_idx) * stride_xh +
            (ow + kw_idx) * stride_xw
        )
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kh_idx * stride_wkh +
            kw_idx * stride_wkw
        )

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        # BUG: axis=1 instead of axis=0 — 对 1D 向量归约, axis=1 越界
        acc = acc + tl.sum(x_vals * w_vals, axis=1)

    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


def conv2d_kernel_bug_window_oob(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
    """Bug: 窗口索引边界 +3, 导致 OOB 且无 mask 保护"""
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

        # BUG: +3 导致大幅越界, 且用 mask=None (无保护) 暴露问题
        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (oh + kh_idx + 3) * stride_xh +
            (ow + kw_idx) * stride_xw
        )
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kh_idx * stride_wkh +
            kw_idx * stride_wkw
        )

        # BUG: mask=None — 无保护, OOB 直接触发 EmulatorError
        x_vals = tl.load(x_ptr, x_offsets, mask=None)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


def conv2d_kernel_bug_weight_stride(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
    """Bug: weight 的 stride_wkh 和 stride_wkw 互换"""
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

        x_offsets = (
            n  * stride_xn +
            ic * stride_xc +
            (oh + kh_idx) * stride_xh +
            (ow + kw_idx) * stride_xw
        )
        # BUG: stride_wkh <-> stride_wkw 互换
        w_offsets = (
            oc     * stride_woc +
            ic     * stride_wic +
            kh_idx * stride_wkw +   # swapped
            kw_idx * stride_wkh      # swapped
        )

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)

        acc = acc + tl.sum(x_vals * w_vals, axis=0)

    b_val = tl.load(b_ptr, oc)
    out_val = acc + b_val

    out_offs = np.array([
        n  * stride_outn +
        oc * stride_outc +
        oh * stride_outh +
        ow * stride_outw
    ], dtype=np.int64)
    tl.store(out_ptr, out_offs, out_val)


# ============================================================
# Emulator 封装
# ============================================================

def emulate_conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray = None,
                   h_out_formula="correct",  # "correct" | "off_by_one"
                   kernel_fn=conv2d_kernel,
                   BLOCK_CK=128,
                   collect_errors=False) -> np.ndarray:
    """
    在 CPU 上 emulate 2D convolution (valid padding, stride=1, dilation=1).

    参数:
      x: input  [N, C_in, H, W]
      w: weight [C_out, C_in, kH, kW]
      b: bias   [C_out], optional
      h_out_formula: "correct" -> H_out = H - kH + 1
                     "off_by_one" -> H_out = H - kH (Bug 1: 故意少 +1)
      kernel_fn: kernel 变体, 用于测试错误 kernel

    返回:
      y: [N, C_out, H_out, W_out]
    """
    if x.ndim != 4:
        raise EmulatorError("conv2d_kernel", f"x must be 4D [N,C,H,W], got {x.shape}")
    if w.ndim != 4:
        raise EmulatorError("conv2d_kernel", f"w must be 4D [C_out,C_in,kH,kW], got {w.shape}")
    N, C_in, H, W = x.shape
    C_out, C_in2, kH, kW = w.shape
    if C_in != C_in2:
        raise EmulatorError("conv2d_kernel",
            f"C_in mismatch: x has {C_in}, w has {C_in2}")

    if h_out_formula == "off_by_one":
        H_out = H - kH       # Bug 1
        W_out = W - kW
    else:
        H_out = H - kH + 1
        W_out = W - kW + 1

    if H_out <= 0 or W_out <= 0:
        raise EmulatorError("conv2d_kernel",
            f"Output size invalid: H_out={H_out}, W_out={W_out} (H={H}, W={W}, kH={kH}, kW={kW})")

    if b is None:
        b = np.zeros(C_out, dtype=np.float32)
    if b.shape != (C_out,):
        raise EmulatorError("conv2d_kernel", f"bias shape {b.shape} != ({C_out},)")

    x_flat = x.ravel().astype(np.float32)
    w_flat = w.ravel().astype(np.float32)
    b_flat = b.ravel().astype(np.float32)
    out_flat = np.zeros(N * C_out * H_out * W_out, dtype=np.float32)

    stride_xn, stride_xc, stride_xh, stride_xw = C_in * H * W, H * W, W, 1
    stride_woc, stride_wic, stride_wkh, stride_wkw = C_in * kH * kW, kH * kW, kW, 1
    stride_outn, stride_outc, stride_outh, stride_outw = C_out * H_out * W_out, H_out * W_out, W_out, 1

    grid_size = N * C_out * H_out * W_out
    launch_kernel_1d(
        kernel_fn,
        x_flat, w_flat, b_flat, out_flat,
        N, C_in, H, W, C_out, kH, kW, H_out, W_out,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wic, stride_wkh, stride_wkw,
        stride_outn, stride_outc, stride_outh, stride_outw,
        BLOCK_CK,
        grid_size=grid_size,
        collect_errors=collect_errors,
    )
    return out_flat.reshape(N, C_out, H_out, W_out)


# ============================================================
# Reference (torch)
# ============================================================

def reference_conv2d(x, w, b=None):
    import torch
    x_t = torch.tensor(x, dtype=torch.float32)
    w_t = torch.tensor(w, dtype=torch.float32)
    b_t = torch.tensor(b, dtype=torch.float32) if b is not None else None
    y_t = torch.nn.functional.conv2d(x_t, w_t, bias=b_t, stride=1, padding=0)
    return y_t.numpy()


# ============================================================
# Self-Test (包含错误兜底评估)
# ============================================================

def test():
    print("=" * 70)
    print(" Conv2d Emulator Test — 含错误兜底评估")
    print("=" * 70)

    N, C_in, H, W = 2, 3, 8, 8
    C_out, kH, kW = 4, 3, 3
    x = np.random.randn(N, C_in, H, W).astype(np.float32)
    w = np.random.randn(C_out, C_in, kH, kW).astype(np.float32)
    b = np.random.randn(C_out).astype(np.float32)

    # ----------------------------------------------------------
    # Test 0: Correct kernel
    # ----------------------------------------------------------
    print("\n--- Test 0: Correct kernel ---")
    out = emulate_conv2d(x, w, b)
    ref = reference_conv2d(x, w, b)
    result = verify(out, ref, "conv2d_correct", rtol=1e-3, atol=1e-4)
    if result["passed"]:
        print("  => 正确版本通过验证")
    else:
        print("  => 正确版本未通过! 请检查实现")
        print(f"  error: {result['error_msg'][:200]}")

    # ----------------------------------------------------------
    # Test 1: Bug — H_out 公式少 +1 (shape 错误)
    # ----------------------------------------------------------
    print("\n--- Test 1: H_out off-by-one (H_out = H - kH, 少 +1) ---")
    try:
        out1 = emulate_conv2d(x, w, b, h_out_formula="off_by_one")
        ref1 = reference_conv2d(x, w, b)
        # 如果 shape 意外能匹配(不太可能), verify 会报 shape mismatch
        verify(out1, ref1, "conv2d_bug_h_out")
    except EmulatorError as e:
        print(f"  [EmulatorError caught] {str(e)[:300]}")
    except Exception as e:
        print(f"  [Other exception] {type(e).__name__}: {str(e)[:300]}")

    # ----------------------------------------------------------
    # Test 2: Bug — reduce axis 写错 (数值错误, 不崩溃)
    # ----------------------------------------------------------
    print("\n--- Test 2: Wrong reduce axis (axis=1 而非 axis=0) ---")
    try:
        out2 = emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_axis)
        ref2 = reference_conv2d(x, w, b)
        verify(out2, ref2, "conv2d_bug_axis", rtol=1e-3, atol=1e-4)
    except EmulatorError as e:
        print(f"  [EmulatorError caught] {str(e)[:300]}")
    except Exception as e:
        print(f"  [Other exception] {type(e).__name__}: {str(e)[:300]}")

    # ----------------------------------------------------------
    # Test 3: Bug — 窗口索引 OOB (运行时崩溃)
    # ----------------------------------------------------------
    print("\n--- Test 3: Window OOB (h_idx +1 导致越界) ---")
    try:
        out3 = emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_window_oob)
        ref3 = reference_conv2d(x, w, b)
        verify(out3, ref3, "conv2d_bug_window_oob", rtol=1e-3, atol=1e-4)
    except EmulatorError as e:
        print(f"  [EmulatorError caught] {str(e)[:300]}")
    except Exception as e:
        print(f"  [Other exception] {type(e).__name__}: {str(e)[:300]}")

    # ----------------------------------------------------------
    # Test 4: Bug — weight stride 互换 (数值错误, 不崩溃)
    # ----------------------------------------------------------
    print("\n--- Test 4: Weight stride swapped (kh/kw stride 互换) ---")
    try:
        out4 = emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_weight_stride)
        ref4 = reference_conv2d(x, w, b)
        verify(out4, ref4, "conv2d_bug_weight_stride", rtol=1e-3, atol=1e-4)
    except EmulatorError as e:
        print(f"  [EmulatorError caught] {str(e)[:300]}")
    except Exception as e:
        print(f"  [Other exception] {type(e).__name__}: {str(e)[:300]}")

    # ----------------------------------------------------------
    # Test with TraceLogger enabled (演示 trace 反馈)
    # ----------------------------------------------------------
    print("\n--- TraceLogger Demo (对 Bug 4 - weight stride 错误 启用 trace) ---")
    TraceLogger.enable()
    out_trace = emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_weight_stride)
    result_trace = verify(out_trace, ref, "conv2d_bug_weight_stride_trace", rtol=1e-3, atol=1e-4)
    if result_trace.get("trace"):
        print(result_trace["trace"][:2000])
    TraceLogger.disable()

    # ----------------------------------------------------------
    # Test 5: run_with_feedback 去重演示 (OOB + collect_errors)
    # ----------------------------------------------------------
    print("\n--- Test 5: run_with_feedback + collect_errors (OOB 去重) ---")
    result5 = run_with_feedback(
        lambda: emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_window_oob, collect_errors=True),
        lambda: reference_conv2d(x, w, b),
        op_name="conv2d_oob_dedup"
    )
    print(f"  passed: {result5['passed']}")
    if result5["feedback"]:
        print(f"  feedback:\n{result5['feedback']}")

    # ----------------------------------------------------------
    # Test 6: run_with_feedback 去重演示 (weight stride 软错误)
    # ----------------------------------------------------------
    print("\n--- Test 6: run_with_feedback (weight stride 软错误去重) ---")
    result6 = run_with_feedback(
        lambda: emulate_conv2d(x, w, b, kernel_fn=conv2d_kernel_bug_weight_stride),
        lambda: reference_conv2d(x, w, b),
        op_name="conv2d_stride_dedup"
    )
    print(f"  passed: {result6['passed']}")
    if result6["feedback"]:
        print(f"  feedback:\n{result6['feedback']}")

    print("\n" + "=" * 70)
    print(" Conv2d 错误兜底评估完成")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
