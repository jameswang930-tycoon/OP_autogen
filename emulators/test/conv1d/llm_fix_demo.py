"""
conv1d/llm_fix_demo.py
======================
模拟 LLM 生成 conv1d kernel → emulator 反馈 → 迭代修正的全过程。

背景
----
LLM 被要求生成一个 Conv1d Triton kernel（valid padding, stride=1）。
它对 Triton API 有基本了解，但容易在以下地方出错：
  1. tl.sum 的 axis 参数（对 1D 张量误用 axis=1）
  2. masked load 的遗漏（不清楚 BLOCK 末尾 padding 需要 mask）
  3. bias 加载索引用错（用 batch index n 代替 output channel index oc）

迭代路径
--------
  v1_buggy     : axis=1 越界 + x_load 无 mask
                 → 2 个 EmulatorError（tl.sum + tl.load），各 48 程序出错
                 → LLM 根据反馈同时修正两处 crash

  v2_wrong_bias: axis=0 ✓, mask ✓, 但 b_ptr 用 n 而非 oc 索引
                 → 无 crash，verify 报数值不匹配，TraceLogger 无异常 flag
                 → LLM 推理 bias 加载错误并修正

  v3_correct   : 全部修正 → PASS

关键演示意义
-----------
  Round 1 → Round 2：展示「精确行号 + 错误类型」的 crash 反馈能力
  Round 2 → Round 3：展示「纯数值不匹配」的静默错误场景——emulator 只能
                     告诉「哪里错了」，不能直接指向根因行

运行方式
--------
  cd emulators && python conv1d/llm_fix_demo.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import (
    tl, launch_kernel_1d, verify, EmulatorError,
    AggregatedEmulatorError, run_with_feedback, TraceLogger,
)

# ============================================================
# 公共测试参数
# ============================================================

np.random.seed(42)
N, C_in, L = 2, 3, 16
C_out, kL  = 4, 5
L_out      = L - kL + 1   # = 12
BLOCK_CK   = 16            # > window(=15), 最后一个 block 有 1 个 padding 槽

x = np.random.randn(N, C_in, L).astype(np.float32)
w = np.random.randn(C_out, C_in, kL).astype(np.float32)
b = np.random.randn(C_out).astype(np.float32)

_x_flat = x.ravel().astype(np.float32)
_w_flat = w.ravel().astype(np.float32)
_b_flat = b.ravel().astype(np.float32)

_stride_xn,   _stride_xc,   _stride_xl   = C_in * L,      L,      1
_stride_woc,  _stride_wic,  _stride_wkl  = C_in * kL,     kL,     1
_stride_outn, _stride_outc, _stride_outl = C_out * L_out,  L_out,  1


def _make_emulate(kernel_fn):
    def emulate():
        out = np.zeros(N * C_out * L_out, dtype=np.float32)
        launch_kernel_1d(
            kernel_fn,
            _x_flat, _w_flat, _b_flat, out,
            N, C_in, L, C_out, kL, L_out,
            _stride_xn, _stride_xc, _stride_xl,
            _stride_woc, _stride_wic, _stride_wkl,
            _stride_outn, _stride_outc, _stride_outl,
            BLOCK_CK,
            grid_size=N * C_out * L_out,
            collect_errors=True,
        )
        return out.reshape(N, C_out, L_out)
    return emulate


def _reference():
    import torch
    return torch.nn.functional.conv1d(
        torch.tensor(x), torch.tensor(w), torch.tensor(b)
    ).numpy()


# ============================================================
# Round 1: v1_buggy — LLM 首次生成版本
# ============================================================
#
# LLM 对 tl.sum 的 axis 语义理解有偏差（以为 axis 和 NumPy reduce 的维度
# 编号一致，没注意 1D 张量只有 axis=0），同时遗漏了 padding mask。
#
# BUG-1 (行 ~111):
#   tl.sum(x_vals * w_vals, axis=1)
#   x_vals * w_vals 是 shape=(BLOCK_CK,) 的 1D 张量，axis=1 越界。
#   emulator 在 tl.sum 处抛出 EmulatorError("sum", "axis=1 OOR for ndim=1")。
#
# BUG-2 (行 ~108):
#   tl.load(x_ptr, x_offsets)  —— 无 mask
#   当 n=0 的 batch（pid 0..47）时，for x_flat size=96：
#     最大的合法 x_offset = (N-1)*stride_xn + (C_in-1)*stride_xc + (L-1)*1 = 95 ✓
#     padding slot (offs=15, ic=3, kl_idx=0): offset = 0*48 + 3*16 + 0 = 48 < 96 → 未越界
#   当 n=1 的 batch（pid 48..95）时：
#     padding slot offset = 1*48 + 3*16 + 0 = 96 = len(x_flat) → OOB!
#
# 因此：pid 0..47 先触发 BUG-1（tl.sum），pid 48..95 先触发 BUG-2（tl.load OOB）。
# collect_errors=True 聚合后反馈 2 个去重错误，各出现 48 次。

def conv1d_v1_buggy(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    pid    = tl.program_id(0)
    n      = pid // (C_out * L_out)
    rn     = pid %  (C_out * L_out)
    oc     = rn  // L_out
    ol     = rn  %  L_out

    window = C_in * kL
    acc    = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs    = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window
        ic      = offs // kL
        kl_idx  = offs %  kL

        x_offsets = n * stride_xn + ic * stride_xc + (ol + kl_idx) * stride_xl
        w_offsets = oc * stride_woc + ic * stride_wic + kl_idx * stride_wkl

        x_vals = tl.load(x_ptr, x_offsets)                          # BUG-2: 无 mask
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)
        acc    = acc + tl.sum(x_vals * w_vals, axis=1)               # BUG-1: axis=1

    b_val    = tl.load(b_ptr, oc)
    out_offs = np.array([n * stride_outn + oc * stride_outc + ol * stride_outl],
                        dtype=np.int64)
    tl.store(out_ptr, out_offs, acc + b_val)


# ============================================================
# Round 2: v2_wrong_bias — 修正两个 crash 后引入数值型 bug
# ============================================================
#
# LLM 读到 Round-1 反馈，同时修正了 axis 和 mask，但在 bias 加载时犯了
# 一个常见的"变量混淆"错误：用 n（batch index）代替 oc（output channel index）。
#
# BUG-3 (行 ~181):
#   b_val = tl.load(b_ptr, n)   ← 应为 oc
#   b_ptr 有 C_out=4 个元素，n ∈ {0,1}，索引合法，不会 crash。
#   但 bias 值对应关系错误：
#     pid 属于 oc=k, n=j 的程序，实际加了 b[j] 而非 b[k]。
#
# 影响范围（seed=42 下 b = [+, -, +, -] 量级约为 0.1~1.0）：
#   当 n == oc 时，b[n] == b[oc]，结果恰好正确（"巧合正确"）
#   当 n != oc 时，偏差为 b[n] - b[oc]，产生系统性数值偏移
#
# emulator 反馈特征：
#   - 无 crash（索引合法）
#   - verify 报数值不匹配，max_abs_err ≈ |b[n]-b[oc]| 量级
#   - TraceLogger 无异常 flag（bias 偏差不触发 NaN/Inf/ALL_ZERO）
#   - emulator 只能告知「哪个位置误差最大」，不能直接定位 L181

def conv1d_v2_wrong_bias(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    pid    = tl.program_id(0)
    n      = pid // (C_out * L_out)
    rn     = pid %  (C_out * L_out)
    oc     = rn  // L_out
    ol     = rn  %  L_out

    window = C_in * kL
    acc    = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs    = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window
        ic      = offs // kL
        kl_idx  = offs %  kL

        x_offsets = n * stride_xn + ic * stride_xc + (ol + kl_idx) * stride_xl
        w_offsets = oc * stride_woc + ic * stride_wic + kl_idx * stride_wkl

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)  # FIXED: mask 加上
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)
        acc    = acc + tl.sum(x_vals * w_vals, axis=0)                # FIXED: axis=0

    b_val    = tl.load(b_ptr, n)                                       # BUG-3: 应为 oc
    out_offs = np.array([n * stride_outn + oc * stride_outc + ol * stride_outl],
                        dtype=np.int64)
    tl.store(out_ptr, out_offs, acc + b_val)


# ============================================================
# Round 3: v3_correct — 全部修正
# ============================================================
#
# LLM 读到 Round-2 反馈：
#   "FAIL conv1d_iter: max_abs_err=X.XXe+00, max_rel_err=X.XXe+00
#    Worst position (n=0, oc=1, ol=0): emulator=Y, reference=Z"
#
# LLM 推理过程：
#   1. 无 crash → mask/axis 修正成功
#   2. 无 trace 异常 → 非 NaN/Inf 类数值问题
#   3. 误差量级 ≈ 1.0，与 b 的数值范围一致 → 怀疑 bias 相关
#   4. Worst position 在 (n=0, oc=1)：oc=1 ≠ n=0，偏差为 b[0]-b[1]
#      若 bias 加载用了 n 而非 oc，这组数据的误差就是 b[n]-b[oc] = b[0]-b[1]
#   5. 回查代码：b_val = tl.load(b_ptr, n) ← 确认是 bug
#
# 修正：tl.load(b_ptr, n) → tl.load(b_ptr, oc)

def conv1d_v3_correct(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, L, C_out, kL, L_out,
    stride_xn, stride_xc, stride_xl,
    stride_woc, stride_wic, stride_wkl,
    stride_outn, stride_outc, stride_outl,
    BLOCK_CK: tl.constexpr,
):
    pid    = tl.program_id(0)
    n      = pid // (C_out * L_out)
    rn     = pid %  (C_out * L_out)
    oc     = rn  // L_out
    ol     = rn  %  L_out

    window = C_in * kL
    acc    = tl.zeros((1,), dtype=tl.float32)

    for ck_start in range(0, window, BLOCK_CK):
        offs    = ck_start + tl.arange(0, BLOCK_CK)
        mask_ck = offs < window
        ic      = offs // kL
        kl_idx  = offs %  kL

        x_offsets = n * stride_xn + ic * stride_xc + (ol + kl_idx) * stride_xl
        w_offsets = oc * stride_woc + ic * stride_wic + kl_idx * stride_wkl

        x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
        w_vals = tl.load(w_ptr, w_offsets, mask=mask_ck, other=0.0)
        acc    = acc + tl.sum(x_vals * w_vals, axis=0)

    b_val    = tl.load(b_ptr, oc)                                      # FIXED: oc
    out_offs = np.array([n * stride_outn + oc * stride_outc + ol * stride_outl],
                        dtype=np.int64)
    tl.store(out_ptr, out_offs, acc + b_val)


# ============================================================
# Demo Runner
# ============================================================

def run_demo():
    sep = "=" * 70

    print(sep)
    print(" Conv1d LLM 迭代修正演示 (llm_fix_demo.py)")
    print(f" 参数: N={N} C_in={C_in} L={L} | C_out={C_out} kL={kL} | "
          f"L_out={L_out} BLOCK_CK={BLOCK_CK} window={C_in*kL}")
    print(f" b = {_b_flat.tolist()}  (size={C_out}, seed=42)")
    print(" 迭代路径: v1_buggy → v2_wrong_bias → v3_correct")
    print(sep)

    rounds = [
        (
            "Round 1 — v1_buggy (axis=1 越界 + x_load 无 mask)",
            conv1d_v1_buggy,
            [
                "BUG-1: tl.sum(x_vals*w_vals, axis=1) —— 1D 张量不存在 axis=1",
                "BUG-2: tl.load(x_ptr, x_offsets) —— 无 mask，n=1 的 batch 触发 OOB",
            ],
            [
                "反馈给出精确行号 + 错误类型 + 出错程序数",
                "修正方向：axis=1→0，x_load 加 mask=mask_ck other=0.0",
                "同时注意：BUG-1 在前 48 个 pid 先触发，BUG-2 在后 48 个 pid 先触发",
            ],
        ),
        (
            "Round 2 — v2_wrong_bias (crash 消除，但 bias 索引用了 n 而非 oc)",
            conv1d_v2_wrong_bias,
            [
                "BUG-3: tl.load(b_ptr, n) —— 应为 tl.load(b_ptr, oc)",
                "n ∈ {0,1}, b 大小 4，索引合法，不 crash，但 bias 值对应关系错误",
            ],
            [
                "反馈无 crash，verify 报 max_abs_err≈3.06",
                "TraceLogger 有 ALL_ZERO flag：",
                "  • L190 tl.zeros ALL_ZERO —— 正常，acc 初始化为零",
                "  • L205 tl.load(b_ptr,n) offset ALL_ZERO (48x) —— 弱信号：",
                "    48 个 program 的 bias offset 恒为 0，说明这半批在用 n=0 而非 oc",
                "  • L208 tl.store offset ALL_ZERO (1x) —— 正常，pid=0 的输出位置是 0",
                "关键证据：verify diff=3.057 ≈ b[n=0](1.866) - b[oc=2](-1.191)",
                "修正方向：tl.load(b_ptr, n) → tl.load(b_ptr, oc)",
            ],
        ),
        (
            "Round 3 — v3_correct (全部修正)",
            conv1d_v3_correct,
            [
                "axis=0 ✓, x/w load 有 mask+other=0.0 ✓, tl.load(b_ptr, oc) ✓",
            ],
            [],
        ),
    ]

    for title, kernel_fn, bugs, analysis in rounds:
        print(f"\n{'─'*70}")
        print(f" {title}")
        if bugs:
            print(" 当前问题:")
            for bug in bugs:
                print(f"   • {bug}")
        print(f"{'─'*70}")

        result = run_with_feedback(
            _make_emulate(kernel_fn),
            _reference,
            op_name="conv1d_iter",
            rtol=1e-3,
            atol=1e-4,
        )

        print(f" passed: {result['passed']}")

        if result["feedback"]:
            print(" [Emulator Feedback]")
            for line in result["feedback"].splitlines():
                print(f"   {line}")
        else:
            print(" (no feedback — kernel passed)")

        if analysis:
            print(" [LLM 分析 & 修正思路]")
            for a in analysis:
                print(f"   → {a}")


# ============================================================
# 补充：观察 v2_wrong_bias 的 TraceLogger 局限
# ============================================================

def show_silent_bias_bug():
    """
    展示 v2_wrong_bias 的 TraceLogger 反馈特征：
    有 ALL_ZERO flag 但属于弱信号，verify 的 diff 才是关键证据。
    """
    print("\n" + "=" * 70)
    print(" 补充演示：v2_wrong_bias 的 TraceLogger 盲区")
    print("=" * 70)

    TraceLogger.enable()
    try:
        out = _make_emulate(conv1d_v2_wrong_bias)()
    except (EmulatorError, AggregatedEmulatorError) as e:
        print(f" [crash] {e}")
        TraceLogger.disable()
        return

    ref = _reference()
    report = verify(out, ref, "conv1d_v2_bias_bug", rtol=1e-3, atol=1e-4)

    deduped_flags = TraceLogger.get_flags_summary_deduped()
    print(f"\n TraceLogger anomaly flags: "
          f"{'(none — TraceLogger 对该 bug 无感知)' if not deduped_flags else ''}")
    if deduped_flags:
        for line in deduped_flags.splitlines():
            print(f"   {line}")

    print(f"\n verify 结果:")
    print(f"   passed        = {report['passed']}")
    print(f"   max_abs_error = {report['max_abs_error']:.4e}")
    print(f"   max_rel_error = {report['max_rel_error']:.4e}")
    if report.get("error_indices") is not None:
        idx = report["error_indices"]
        out_arr = np.asarray(out)
        ref_arr = np.asarray(ref)
        n_idx, oc_idx, ol_idx = idx
        print(f"   worst index   = (n={n_idx}, oc={oc_idx}, ol={ol_idx})")
        print(f"   emulator val  = {out_arr[idx]:.6f}")
        print(f"   reference val = {ref_arr[idx]:.6f}")
        print(f"   diff          = {out_arr[idx] - ref_arr[idx]:.6f}  "
              f"≈ b[n={n_idx}]({_b_flat[n_idx]:.4f}) - b[oc={oc_idx}]({_b_flat[oc_idx]:.4f})"
              f" = {_b_flat[n_idx] - _b_flat[oc_idx]:.4f}")

    print()
    print(" 结论：bias 索引错误的 TraceLogger 反馈是「弱信号」而非「无信号」。")
    print(" ALL_ZERO flag 说明 48 个 program 用 offset=0 加载 bias，")
    print(" 间接暗示这半批在用 n=0 而非 oc，但不直接指向根因行。")
    print(" 最有力的证据是 verify 的 diff 精确等于 b[n]-b[oc]，")
    print(" LLM 需要将数值证据和代码结构结合才能定位根因。")

    TraceLogger.disable()


if __name__ == "__main__":
    run_demo()
    show_silent_bias_bug()
