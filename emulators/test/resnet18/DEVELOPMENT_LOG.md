# ResNet18 Convolution Layer Development Log

> Developer: Claude (via triton-gen skill)
> Date: 2026-05-21
> Design doc: `docs/resnet18_conv_dev_plan.md`

---

## Goals

1. Validate whether the design doc is sufficient to guide actual kernel development
2. Test whether the emulator's stub API can catch real bugs during development
3. Capture any issues or improvement suggestions for the emulator framework

---

## Phase 1: Operator Development

### 1.1 Conv2d Generalized (stride + padding) — PASS (first try)

**File:** `emulators/test/conv2d_resnet/__init__.py`

All 4 ResNet18 conv configurations passed on first run:

| Config | Parameters | max_abs |
|--------|-----------|---------|
| B (3x3, s=1, p=1) | 64->64ch, 8x8 input | 6.10e-05 |
| A (7x7, s=2, p=3) | 3->64ch, 14x14 input | 1.34e-05 |
| C (3x3, s=2, p=1) | 64->128ch, 8x8 input | 7.63e-05 |
| E (1x1, s=2, p=0) | 64->128ch, 8x8 input | 5.72e-06 |

**Iteration count:** 0. The design doc's padding-via-mask trick (`tl.load(mask=in_bounds, other=0.0)`) worked exactly as described.

---

### 1.2 BatchNorm2d (eval mode) — PASS (1 iteration)

**File:** `emulators/test/batchnorm2d/__init__.py`

| Test | Shape | max_abs |
|------|-------|---------|
| Basic | (2, 64, 8, 8) | 9.54e-07 |
| Identity (gamma=1,beta=0) | (2, 64, 8, 8) | 4.77e-07 |
| Single channel | (1, 1, 4, 4) | 1.19e-07 |
| ResNet18-like | (1, 64, 56, 56) | 4.77e-07 |

**Iteration count:** 1. First version had a reference function bug: `torch.tensor(x, dtype=np.float32)` — used numpy dtype instead of torch dtype. This was a Python-level typo, not a kernel logic error. The kernel itself was correct from the start.

**Key learning:** The channel index derivation `c = (flat_pos // (H*W)) % C` is correct and simple. The emulator's elementwise load pattern handles this naturally.

---

### 1.3 MaxPool2d (stride + padding) — PASS (1 iteration)

**File:** `emulators/test/maxpool2d/__init__.py`

| Test | Shape | max_abs |
|------|-------|---------|
| ResNet18 cfg (3x3, s=2, p=1) | (1, 64, 8, 8) | 0.00e+00 |
| 2x2, s=2, p=0 | (1, 64, 8, 8) | 0.00e+00 |
| All-negative input | (1, 4, 4, 4) | 0.00e+00 |
| Non-aligned (13x17) | (1, 32, 13, 17) | 0.00e+00 |
| Batch=4 | (4, 64, 8, 8) | 0.00e+00 |
| Large (112x112) | (1, 64, 112, 112) | 0.00e+00 |

**Iteration count:** 1. First version used a running `tl.maximum(acc, x_vals)` in a loop, but `acc` shape `(1,)` broadcasting with `x_vals` shape `(BLOCK_KK,)` produced `(BLOCK_KK,)` instead of staying `(1,)`. The emulator caught this precisely:

```
[Triton Emulator Error] in tl.store():
  values shape (32,) != offsets shape (1,)
  at: L78: tl.store(out_ptr, np.array([out_offset], dtype=np.int64), acc)
```

This is a **real developer mistake that the emulator caught exactly at the right line**. Fix: replaced loop with single-load + `tl.max(x_vals, axis=0)` reduction. Max pooling is exact (max comparison, no float accumulation), so max_abs=0.

---

### 1.4 AdaptiveAvgPool2d (global) — PASS (first try)

**File:** `emulators/test/adaptive_avgpool2d/__init__.py`

| Test | Shape | max_abs |
|------|-------|---------|
| ResNet18 final (512, 7, 7) | (1, 512, 7, 7) | 4.47e-08 |
| Batch=4 | (4, 512, 7, 7) | 5.96e-08 |
| 1x1 spatial (identity) | (2, 64, 1, 1) | 0.00e+00 |
| Large spatial | (1, 64, 56, 56) | 1.12e-08 |
| Non-square | (1, 32, 8, 12) | 2.98e-08 |

**Iteration count:** 0. Simplest operator. The blocked reduction pattern (iterate `H*W` in `BLOCK_HW` chunks) worked correctly.

---

## Phase 2: Integration

### 2.1 BasicBlock + Stem — ALL PASSED

**File:** `emulators/test/resnet18/__init__.py`

| Test | Description | max_abs | max_rel |
|------|-------------|---------|---------|
| Stem | conv1(7x7,s2,p3)+bn+relu+maxpool(3x3,s2,p1) | 4.77e-07 | 1.14e-04 |
| Block (no down) | 64->64, 2x conv3x3 + residual | 2.38e-07 | 3.28e-05 |
| Block (downsample) | 64->128, conv3x3 s2 + 1x1 proj shortcut | 8.34e-07 | 1.82e-05 |
| Two blocks | layer1.0 + layer1.1 chained | 4.77e-07 | 1.65e-04 |
| Full chain | stem -> block -> avgpool | 1.19e-07 | 2.51e-06 |

**Iteration count:** 0. Composition of individual operators worked correctly on first try. The chained operator test demonstrates that error does not accumulate significantly even after 5+ operator calls in sequence.

---

## Findings Summary

### Design Doc Effectiveness

**Score: 9/10 — Highly effective.**

- The padding-via-mask trick described in the doc worked exactly as designed. No ambiguity.
- The output size formula `H_out = (H + 2*pad_h - kH) // stride_h + 1` was correct for all configurations.
- The 4 new operators (conv2d_resnet, batchnorm2d, maxpool2d, adaptive_avgpool2d) covered all ResNet18 patterns.
- The BasicBlock integration test matched the doc's description of the operator chain.
- **One gap:** The doc didn't mention the `tl.maximum` broadcasting pitfall that I hit in maxpool2d. This is an important pattern to document — when doing running reductions, `tl.maximum((1,), (N,))` broadcasts to `(N,)` instead of reducing. The fix is to use `tl.max(x, axis=0)` for reduction instead of `tl.maximum` in a loop.

### Emulator API Coverage

**Score: 8/10 — Sufficient for ResNet18.**

- All required APIs (`tl.load`, `tl.store`, `tl.sum`, `tl.max`, `tl.maximum`, `tl.sqrt`, `tl.zeros`, `tl.full`, `tl.arange`, `tl.program_id`, `tl.cdiv`) worked correctly.
- No new API additions were needed. The existing API surface fully covers ResNet18 convolution layers.
- **The emulator caught a real bug** during maxpool2d development: shape mismatch between accumulator and store target. Error message was precise (line number, shapes).

### Issues & Suggestions

#### 1. `tl.sum` / `tl.max` keepdims=True behavior (Low severity, design choice)

The emulator uses `keepdims=True` for all reductions. This is a deliberate design choice to maintain shape consistency: reduction outputs are `(1,)`, which naturally aligns with `tl.store`'s `(1,)` offsets and `tl.zeros((1,))` accumulators. Numerical results are equivalent. Kernels use `np.array([offset])` for single-element stores, which works cleanly with this convention.

#### 2. `tl.maximum` broadcasting surprise (Medium severity)

`tl.maximum(acc_shape(1,), vals_shape(N,))` returns shape `(N,)` instead of reducing. This is correct numpy broadcasting behavior, but it's a common trap when writing reduction kernels. The emulator catches the downstream store shape mismatch, but the error is reported at the `tl.store` line, not at the `tl.maximum` line where the shape diverged.

**Suggestion:** Consider adding a shape-change warning in TraceLogger when an operation changes the accumulator's shape unexpectedly. This would help LLMs diagnose faster.

#### 3. Operator composition pattern (Low severity)

Currently, composing operators requires manually calling `emulate_xxx` sequentially and passing flat arrays + shapes. For longer chains (like a full ResNet18 forward pass), this gets verbose. A lightweight "pipeline" helper that chains emulators would reduce boilerplate.

**This is not blocking** — the current pattern works fine for the scope we tested.

#### 4. Performance for large grids (Observation, not bug)

The 112x112 maxpool test (grid_size = 1 * 64 * 56 * 56 = 200,704 programs) ran in reasonable time. Larger sizes (224x224 -> 802,816 programs) would be noticeably slow due to serial program execution. This is a known design tradeoff of the CPU emulator.

**Not a problem for testing purposes**, but worth noting for users who want to test with full-resolution inputs.

---

## Conclusion

1. **The design doc is sufficient** — all 4 operators and the BasicBlock integration were implemented successfully following the doc, with at most 1 iteration per operator.
2. **The emulator caught real bugs** — specifically the maxpool2d shape mismatch from `tl.maximum` broadcasting. The error was precise and actionable.
3. **No new `tl.*` APIs needed** — ResNet18's entire convolution pipeline fits within the existing API surface.
