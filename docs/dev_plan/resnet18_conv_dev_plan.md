# ResNet18 Convolution Layer Development Plan

> Target: Use ResNet18 convolution layers as complex test cases to validate the OP_autogen emulator architecture.

---

## 1. ResNet18 Convolution Layer Operator Structure

### 1.1 ResNet18 Architecture Overview

```
Input: [N, 3, 224, 224]
  |
  +-- conv1: Conv2d(3, 64, 7x7, stride=2, padding=3)    <- NEW: stride>1, padding
  +-- bn1:   BatchNorm2d(64)                              <- NEW
  +-- relu                                                <- EXISTING
  +-- maxpool: MaxPool2d(3x3, stride=2, padding=1)        <- NEW
  |
  +-- layer1.0: BasicBlock(64, 64)
  |   +-- conv1: Conv2d(64, 64, 3x3, stride=1, padding=1)   <- NEW: padding
  |   +-- bn1 + relu
  |   +-- conv2: Conv2d(64, 64, 3x3, stride=1, padding=1)
  |   +-- bn2
  |   +-- residual add + relu                                <- EXISTING: add + relu
  |
  +-- layer1.1: BasicBlock(64, 64)   [same structure]
  |
  +-- layer2.0: BasicBlock(64, 128, stride=2)
  |   +-- conv1: Conv2d(64, 128, 3x3, stride=2, padding=1)  <- NEW: stride>1
  |   +-- bn1 + relu
  |   +-- conv2: Conv2d(128, 128, 3x3, stride=1, padding=1)
  |   +-- bn2
  |   +-- downsample:                                        <- NEW: 1x1 projection
  |   |   +-- Conv2d(64, 128, 1x1, stride=2)
  |   |   +-- BatchNorm2d(128)
  |   +-- residual add + relu
  |
  +-- layer2.1: BasicBlock(128, 128)
  +-- layer3.0: BasicBlock(128, 256, stride=2)
  +-- layer3.1: BasicBlock(256, 256)
  +-- layer4.0: BasicBlock(256, 512, stride=2)
  +-- layer4.1: BasicBlock(512, 512)
  |
  +-- adaptive_avg_pool2d: (N, 512, 1, 1)                   <- NEW
  +-- fc: Linear(512, 1000)                                  <- EXISTING: matmul
```

### 1.2 Required Operators (6 Modules, 4 New + 2 Existing)

| # | Operator | Status | Kernel Type | Grid | Complexity |
|---|----------|--------|-------------|------|------------|
| 1 | **Conv2d (generalized)** | **NEW** | Stride + Padding + Dilation | 1D (per output pixel) | High |
| 2 | **BatchNorm2d** | **NEW** | Elementwise (eval) / Reduction (train) | 1D | Medium |
| 3 | **MaxPool2d** | **NEW** | Reduction over window | 1D (per output pixel) | Medium |
| 4 | **AdaptiveAvgPool2d** | **NEW** | Global reduction | 1D (per channel) | Low |
| 5 | ReLU | EXISTING | Elementwise | 1D | - |
| 6 | Add (residual) | EXISTING | Elementwise | 1D | - |
| 7 | Matmul (FC layer) | EXISTING | 2D tiled | 2D | - |

### 1.3 Conv2d Configurations in ResNet18

All convolution configurations, classified by parameter combination:

| Config ID | Layer | C_in | C_out | kH x kW | Stride | Padding | Notes |
|-----------|-------|------|-------|---------|--------|---------|-------|
| A | conv1 | 3 | 64 | 7x7 | 2 | 3 | Large kernel, stride=2 |
| B | layer1 conv | 64 | 64 | 3x3 | 1 | 1 | Standard 3x3, no downsample |
| C | layer2.0 conv1 | 64 | 128 | 3x3 | 2 | 1 | Stride=2 downsample |
| D | layer2.0 conv2 | 128 | 128 | 3x3 | 1 | 1 | Standard 3x3 |
| E | layer2.0 down | 64 | 128 | 1x1 | 2 | 0 | 1x1 projection |
| F | layer3.0 conv1 | 128 | 256 | 3x3 | 2 | 1 | Stride=2 downsample |
| G | layer3.0 conv2 | 256 | 256 | 3x3 | 1 | 1 | Standard 3x3 |
| H | layer3.0 down | 128 | 256 | 1x1 | 2 | 0 | 1x1 projection |
| I | layer4.0 conv1 | 256 | 512 | 3x3 | 2 | 1 | Stride=2 downsample |
| J | layer4.0 conv2 | 512 | 512 | 3x3 | 1 | 1 | Standard 3x3 |
| K | layer4.0 down | 256 | 512 | 1x1 | 2 | 0 | 1x1 projection |

Test priority: **A > C > E > B** (covers all parameter combinations; D/F/G/H/I/J/K are parameter variants of B/C/E).

---

## 2. Triton API Interfaces Required

### 2.1 API Coverage Analysis

#### Already Supported (sufficient for all operators)

| API | Used By | Usage |
|-----|---------|-------|
| `tl.program_id(axis)` | All kernels | Grid scheduling |
| `tl.arange(start, end)` | All kernels | Block offset generation |
| `tl.load(ptr, offsets, mask, other)` | All kernels | Gather from DRAM |
| `tl.store(ptr, offsets, values, mask)` | All kernels | Scatter to DRAM |
| `tl.zeros(shape, dtype)` | Conv2d, MaxPool2d | Accumulator init |
| `tl.full(shape, value, dtype)` | MaxPool2d | Init to -inf |
| `tl.sum(x, axis)` | Conv2d, AdaptiveAvgPool | Reduction |
| `tl.max(x, axis)` | MaxPool2d | Max reduction |
| `tl.maximum(x, y)` | ReLU, MaxPool2d | Elementwise max |
| `tl.where(cond, x, y)` | BatchNorm2d | Conditional select |
| `tl.sqrt(x)` | BatchNorm2d | sqrt(var + eps) |
| `tl.exp(x)` | - | Not needed for ResNet18 |
| `tl.dot(a, b)` | FC (matmul) | 2D matrix multiply |
| `tl.cdiv(x, y)` | All emulate wrappers | Grid size calculation |
| `tl.num_programs(axis)` | - | Optional bounds check |

#### Not Needed (no new API required)

ResNet18 convolution layers can be fully implemented with the existing emulator API surface. Key reason: all operations decompose into load/store/arithmetic/reduction patterns that the current `tl.*` stubs cover.

### 2.2 API Usage Patterns Per Operator

#### Conv2d (generalized with stride + padding)

```
tl.program_id(0)          # pid -> (n, oc, oh, ow) decomposition
tl.arange(0, BLOCK_CK)    # inner loop over C_in * kH * kW
tl.load(x_ptr, offsets, mask=mask, other=0.0)  # other=0.0 implements zero-padding
tl.load(w_ptr, offsets, mask=mask, other=0.0)
tl.sum(x_vals * w_vals, axis=0)                # dot product accumulation
tl.zeros((1,), dtype=tl.float32)               # accumulator init
tl.store(out_ptr, offsets, values)
```

Key insight: **Padding is implemented by mask + other=0.0** in `tl.load`. When `(ih, iw)` falls outside `[0, H)` or `[0, W)`, the mask sets that position to `other=0.0`, which is equivalent to zero-padding. No explicit padding array needed.

Stride is implemented by computing output coordinates as:
```
ih = oh * stride_h + kh_idx - pad_h
iw = ow * stride_w + kw_idx - pad_w
```

#### BatchNorm2d (eval mode)

```
tl.load(x_ptr, offsets, mask=mask)      # load feature values
tl.load(mean_ptr, channel_idx)          # running mean (scalar per channel)
tl.load(var_ptr, channel_idx)           # running variance
tl.load(gamma_ptr, channel_idx)         # scale parameter
tl.load(beta_ptr, channel_idx)          # bias parameter
tl.sqrt(var + eps)                      # denominator
tl.store(out_ptr, offsets, result, mask=mask)
```

#### MaxPool2d

```
tl.program_id(0)                        # pid -> (n, c, oh, ow)
tl.full((1,), float('-inf'), tl.float32) # init max accumulator to -inf
tl.load(x_ptr, offsets, mask=mask, other=float('-inf'))
tl.maximum(acc, loaded_value)            # running max over window
tl.store(out_ptr, offsets, values)
```

#### AdaptiveAvgPool2d (global)

```
tl.program_id(0)                    # pid -> (n, c)
tl.arange(0, BLOCK_HW)              # iterate over spatial
tl.load(x_ptr, offsets, mask=mask, other=0.0)
tl.sum(block_vals, axis=0)          # sum over H*W
# result / (H * W)                   # divide by spatial size
tl.store(out_ptr, offset, avg_val)
```

---

## 3. Development Scope

### 3.1 Module Development (4 New Operator Modules)

Each module follows the standard four-part structure (`emulators/test/<op>/__init__.py`):

---

#### Module 1: `conv2d_resnet/` — Generalized Conv2d

**Scope:** Extend existing `conv2d/` to support `stride`, `padding`, `dilation`, and `groups` (groups=1 for ResNet18).

**Kernel signature:**
```python
def conv2d_general_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C_in, H, W, C_out, kH, kW, H_out, W_out,
    stride_h, stride_w, pad_h, pad_w,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_CK: tl.constexpr,
):
```

**Key implementation details:**

1. **Padding via mask**: Instead of physically padding the input tensor, use the `other=0.0` parameter of `tl.load`:
   ```python
   ih = oh * stride_h + kh_idx - pad_h
   iw = ow * stride_w + kw_idx - pad_w
   in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
   mask_ck = mask_ck & in_bounds   # combined with BLOCK_CK boundary mask
   x_vals = tl.load(x_ptr, x_offsets, mask=mask_ck, other=0.0)
   ```

2. **Stride**: Output coordinate to input coordinate mapping:
   ```python
   ih = oh * stride_h + kh_idx - pad_h
   iw = ow * stride_w + kw_idx - pad_w
   ```

3. **Output size formula**:
   ```python
   H_out = (H + 2 * pad_h - kH) // stride_h + 1
   W_out = (W + 2 * pad_w - kW) // stride_w + 1
   ```

**Bug kernels (for feedback quality evaluation):**

| Bug ID | Description | Error Type |
|--------|-------------|------------|
| bug_pad_sign | `ih = oh * stride_h + kh_idx + pad_h` (pad sign wrong) | Numerical (silent) |
| bug_stride_formula | `ih = oh + kh_idx - pad_h` (forgot multiply stride) | Numerical (silent) |
| bug_bounds_check | Missing `in_bounds` mask → OOB without guard | EmulatorError (crash) |
| bug_output_size | `H_out = (H + 2*pad_h - kH) // stride_h` (missing +1) | Shape mismatch |

**Emulate wrapper input validation:**
- `x.ndim == 4`, `w.ndim == 4`
- `C_in` dimension consistency
- `H_out > 0` and `W_out > 0`
- `stride_h > 0`, `stride_w > 0`
- `pad_h >= 0`, `pad_w >= 0`

---

#### Module 2: `batchnorm2d/` — Batch Normalization

**Two modes:**

- **Eval mode** (inference, used for ResNet18 forward pass testing):
  ```
  y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
  ```
- **Training mode** (optional stretch goal):
  ```
  mean = mean(x, dim=[0,2,3])
  var  = var(x, dim=[0,2,3])
  y = gamma * (x - mean) / sqrt(var + eps) + beta
  ```

**Kernel signature (eval mode):**
```python
def batchnorm2d_eval_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N, C, H, W, eps,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
```

**Grid:** 1D, `grid_size = N * C * H * W / BLOCK_SIZE` (elementwise).

**Key detail:** Each element needs to know its channel index `c` to fetch `mean[c], var[c], gamma[c], beta[c]`. Channel index derivation from flat position:
```python
c = (flat_idx // (H * W)) % C
```

**Bug kernels:**

| Bug ID | Description | Error Type |
|--------|-------------|------------|
| bug_eps | Missing `+ eps` in denominator → division by zero | HAS_INF / NaN |
| bug_channel_idx | Wrong channel index decomposition | Numerical (silent) |

---

#### Module 3: `maxpool2d/` — Max Pooling

**Kernel signature:**
```python
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W, kH, kW, stride_h, stride_w, pad_h, pad_w,
    H_out, W_out,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_outn, stride_outc, stride_outh, stride_outw,
    BLOCK_KK: tl.constexpr,
):
```

**Grid:** 1D, `grid_size = N * C * H_out * W_out`, each program computes one output element.

**Key logic:**
```python
acc = tl.full((1,), float('-inf'), tl.float32)  # init to -inf
for kk_start in range(0, kH * kW, BLOCK_KK):
    # map flat kk -> (kh, kw)
    kh = kk // kW
    kw = kk % kW
    ih = oh * stride_h + kh - pad_h
    iw = ow * stride_w + kw - pad_w
    in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
    x_val = tl.load(x_ptr, offsets, mask=in_bounds, other=float('-inf'))
    acc = tl.maximum(acc, x_val)
```

**Bug kernels:**

| Bug ID | Description | Error Type |
|--------|-------------|------------|
| bug_init_zero | `tl.zeros` instead of `tl.full(-inf)` → wrong max for negative inputs | Numerical (silent) |
| bug_no_pad_mask | Missing bounds check → OOB | EmulatorError (crash) |

---

#### Module 4: `adaptive_avgpool2d/` — Adaptive Average Pooling

For ResNet18, this is always global average pooling: `(N, C, H, W) -> (N, C, 1, 1)`.

**Kernel signature:**
```python
def adaptive_avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_HW: tl.constexpr,
):
```

**Grid:** 1D, `grid_size = N * C`, each program computes one (n, c) output.

**Key logic:**
```python
pid = tl.program_id(0)
n = pid // C
c = pid % C
total = H * W
acc = tl.zeros((1,), dtype=tl.float32)
for hw_start in range(0, total, BLOCK_HW):
    offsets_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask = offsets_hw < total
    h_idx = offsets_hw // W
    w_idx = offsets_hw % W
    x_offsets = n * stride_xn + c * stride_xc + h_idx * stride_xh + w_idx * stride_xw
    vals = tl.load(x_ptr, x_offsets, mask=mask, other=0.0)
    acc = acc + tl.sum(vals, axis=0)
avg = acc / total
tl.store(out_ptr, pid, avg.reshape(1))
```

---

### 3.2 Integration Test: ResNet18 BasicBlock

Beyond single-operator tests, create an integration module to compose operators into a BasicBlock:

**File:** `emulators/test/resnet18_block/__init__.py`

```
BasicBlock(x, conv1_w, bn1_params, conv2_w, bn2_params, downsample=None):
    identity = x
    out = conv2d_general(x, conv1_w, stride, padding=1)
    out = batchnorm2d_eval(out, bn1_params)
    out = relu(out)
    out = conv2d_general(out, conv2_w, stride=1, padding=1)
    out = batchnorm2d_eval(out, bn2_params)
    if downsample:
        identity = conv2d_general(x, down_w, stride=2, padding=0)  # 1x1
        identity = batchnorm2d_eval(identity, down_bn_params)
    out = out + identity   # residual add
    out = relu(out)
    return out
```

**Reference:** `torch.nn.modules.resnet.BasicBlock` forward pass.

---

### 3.3 Implementation Order

```
Phase 1: Core operators (independent, can be developed in parallel)
  ├── Step 1.1: conv2d_resnet/     <- highest priority, most complex
  ├── Step 1.2: batchnorm2d/       <- independent
  ├── Step 1.3: maxpool2d/         <- independent
  └── Step 1.4: adaptive_avgpool2d/ <- simplest

Phase 2: Integration
  └── Step 2.1: resnet18_block/    <- depends on Phase 1

Phase 3: Full ResNet18 forward pass (optional stretch goal)
  └── Step 3.1: Chain all blocks from input to output
```

---

### 3.4 Fused Kernel (Optional Optimization)

If the architecture validation is successful, consider fused kernels:

| Fusion | Description | Benefit |
|--------|-------------|---------|
| Conv2d + BN + ReLU | Single kernel for conv → normalize → activate | Eliminates 2 DRAM roundtrips |
| BN + ReLU | Merge normalization and activation | Eliminates 1 DRAM roundtrip |
| Add + ReLU | Merge residual add and activation | Eliminates 1 DRAM roundtrip |

These are stretch goals — validate single operators first.

---

## 4. Verification Scenarios

### 4.1 Per-Operator Verification

#### Conv2d (generalized)

| Test ID | Scenario | Input Shape | Config | Verification |
|---------|----------|-------------|--------|--------------|
| C-1 | 3x3 conv, pad=1, stride=1 | (1, 64, 56, 56) | Config B | `torch.nn.functional.conv2d` |
| C-2 | 7x7 conv, pad=3, stride=2 | (1, 3, 224, 224) | Config A | `torch.nn.functional.conv2d` |
| C-3 | 1x1 conv, pad=0, stride=2 | (1, 64, 56, 56) | Config E | `torch.nn.functional.conv2d` |
| C-4 | 3x3 conv, pad=1, stride=2 | (1, 64, 56, 56) | Config C | `torch.nn.functional.conv2d` |
| C-5 | Non-BLOCK-aligned spatial | (1, 3, 13, 17) | 3x3/s1/p1 | Same |
| C-6 | Batch size > 1 | (4, 64, 56, 56) | Config B | Same |
| C-7 | No bias (bias=None) | (1, 64, 56, 56) | Config B | Same |
| C-8 | Minimum size (1x1 input) | (1, 1, 1, 1) | 1x1/s1/p0 | Same |
| C-9 | Bug: pad sign wrong | - | - | Expect numerical mismatch |
| C-10 | Bug: stride formula wrong | - | - | Expect numerical mismatch |
| C-11 | Bug: bounds check missing | - | - | Expect EmulatorError (OOB) |
| C-12 | Bug: output size off-by-one | - | - | Expect shape mismatch |
| C-13 | run_with_feedback (OOB bug) | - | - | Validate error dedup quality |

#### BatchNorm2d

| Test ID | Scenario | Details | Verification |
|---------|----------|---------|--------------|
| BN-1 | Eval mode, basic | (2, 64, 56, 56) | `torch.nn.BatchNorm2d(eval)` |
| BN-2 | Gamma=1, Beta=0 (identity-like) | All params set to identity | x_norm should be centered |
| BN-3 | Single channel | (1, 1, 4, 4) | Same |
| BN-4 | Zero variance (edge case) | var=0 for some channels | Verify eps protection |
| BN-5 | Bug: missing eps | - | Expect HAS_INF or NaN |
| BN-6 | Bug: wrong channel index | - | Expect numerical mismatch |

#### MaxPool2d

| Test ID | Scenario | Details | Verification |
|---------|----------|---------|--------------|
| MP-1 | 3x3 pool, stride=2, pad=1 | (1, 64, 112, 112) | `torch.nn.functional.max_pool2d` |
| MP-2 | 2x2 pool, stride=2, pad=0 | (1, 64, 56, 56) | Same |
| MP-3 | All-negative input | (1, 1, 4, 4) with x < 0 | Max should still be correct |
| MP-4 | Non-aligned spatial | (1, 64, 13, 17) | Same |
| MP-5 | Bug: init with zero | - | Expect failure on negative inputs |

#### AdaptiveAvgPool2d

| Test ID | Scenario | Details | Verification |
|---------|----------|---------|--------------|
| AAP-1 | Global pool basic | (1, 512, 7, 7) -> (1, 512, 1, 1) | `torch.nn.functional.adaptive_avg_pool2d` |
| AAP-2 | Batch > 1 | (4, 512, 7, 7) | Same |
| AAP-3 | 1x1 spatial (identity) | (1, 512, 1, 1) | Output == input |
| AAP-4 | Large spatial | (1, 64, 56, 56) | Same |

### 4.2 Integration Verification

| Test ID | Scenario | Details |
|---------|----------|---------|
| INT-1 | BasicBlock without downsample | layer1.0: (1,64,56,56) -> (1,64,56,56), weights from `torchvision.models.resnet18(pretrained=True)` |
| INT-2 | BasicBlock with downsample | layer2.0: (1,64,56,56) -> (1,128,28,28), with 1x1 projection |
| INT-3 | Two consecutive BasicBlocks | layer1.0 + layer1.1: validate intermediate shapes and final values |
| INT-4 | Full stem (conv1 + bn1 + relu + maxpool) | (1,3,224,224) -> (1,64,56,56) |
| INT-5 | Random weights, batch > 1 | Verify batch dimension independence |

### 4.3 Architecture Stress Tests

These scenarios specifically test the emulator framework's robustness:

| Test ID | What It Stresses | Scenario |
|---------|-----------------|----------|
| S-1 | **Large grid size** | conv1: grid = 1 * 64 * 112 * 112 = 802,816 programs — tests launch_kernel_1d scalability |
| S-2 | **Deep composition** | 5+ emulate_xxx calls chained — tests error propagation across operators |
| S-3 | **TraceLogger volume** | Enable trace on BasicBlock — tests log aggregation under high volume |
| S-4 | **run_with_feedback on compound** | run_with_feedback wrapping an entire BasicBlock — tests feedback quality for non-trivial pipelines |
| S-5 | **Bug in middle of chain** | Inject wrong conv weights in layer2 of a multi-layer chain — tests whether verify pinpoints the faulty layer |
| S-6 | **BLOCK_CK sensitivity** | Same kernel with BLOCK_CK = 16, 32, 64, 128, 256 — verify all produce identical results |

### 4.4 Numeric Precision Scenarios

| Test ID | Scenario | Tolerance |
|---------|----------|-----------|
| NP-1 | Conv2d with very small weights (~1e-6) | atol=1e-8 |
| NP-2 | Conv2d with large activations (~1e3) | rtol=1e-3 |
| NP-3 | BN with near-zero variance | Verify eps is sufficient |
| NP-4 | Accumulated error over BasicBlock chain | May need relaxed rtol=1e-2 |
| NP-5 | float32 vs float64 reference comparison | Quantify emulator precision gap |

---

## 5. Emulator Architecture Findings to Validate

Running ResNet18 through the emulator will stress-test several architectural aspects that simple operators don't exercise:

| # | Aspect | Expected Finding |
|---|--------|-----------------|
| 1 | **Grid scalability** | conv1 launches ~800K programs — is serial execution acceptably fast? |
| 2 | **Error dedup at scale** | If a conv2d bug triggers on every output pixel, does AggregatedEmulatorError deduplicate correctly? |
| 3 | **TraceLogger memory** | With 800K programs x 3 tl.load calls each, TraceLogger.logs can hit the 10K cap fast — is the cap appropriate? |
| 4 | **Operator composition pattern** | Is there a clean way to chain emulate_xxx calls, or do we need a "graph runner"? |
| 5 | **Feedback quality for deep errors** | When the bug is in conv2 but verify runs on the final output, is the feedback actionable for an LLM? |
| 6 | **keepdims=True behavior** | Does the emulator's `tl.sum(keepdims=True)` cause shape issues in deeper kernels? |

---

## 6. File Structure

```
emulators/test/
├── conv2d_resnet/
│   └── __init__.py          # conv2d_general_kernel + 4 bug kernels + emulate + reference + test
├── batchnorm2d/
│   └── __init__.py          # batchnorm2d_eval_kernel + 2 bug kernels + emulate + reference + test
├── maxpool2d/
│   └── __init__.py          # maxpool2d_kernel + 2 bug kernels + emulate + reference + test
├── adaptive_avgpool2d/
│   └── __init__.py          # adaptive_avgpool2d_kernel + emulate + reference + test
├── resnet18_block/
│   └── __init__.py          # BasicBlock composition + stem + integration tests
└── run_all_tests.py         # Add new modules to test runner
```

---

## 7. Success Criteria

- [ ] All 4 new operators pass correctness verification against PyTorch reference
- [ ] All Conv2d configurations (A-K) from Section 1.3 produce correct results
- [ ] Bug kernels trigger expected error types (crash / shape mismatch / numerical mismatch)
- [ ] run_with_feedback produces actionable LLM-readable feedback for each bug type
- [ ] BasicBlock integration test matches PyTorch output within rtol=1e-3
- [ ] Full stem (conv1+bn1+relu+maxpool) integration test passes
- [ ] No new API additions needed in `emulators/common/__init__.py`
- [ ] Execution time for single BasicBlock test < 60 seconds on CPU

---

## 8. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Serial execution too slow for large grids | Integration tests unacceptably slow | Use small spatial sizes (e.g., 14x14 instead of 224x224) for routine tests; full-size only in dedicated stress test |
| TraceLogger memory overflow | OOM or truncated traces | Adjust `_max_logs` cap dynamically, or disable trace for large-grid tests |
| float32 accumulation error in deep chains | Relaxed tolerance hides real bugs | Run reference in float64, compare with both tight and relaxed tolerances |
| `tl.sum(keepdims=True)` semantic difference | Shape bugs that wouldn't occur on real Triton | Document as known divergence; add explicit `.reshape()` after every reduction |
