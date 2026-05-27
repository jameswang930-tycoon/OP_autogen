---
name: triton-gen
description: >
  Generate and verify Triton GPU kernels using a CPU-side emulator.
  Use this skill whenever the user asks to: generate a Triton kernel,
  write a GPU kernel for an operator (matmul, softmax, attention, conv, etc.),
  verify a Triton kernel's correctness, debug a Triton kernel,
  or create a fused operator kernel. Also trigger when the user mentions
  "triton", "tl.load", "tl.store", "GPU kernel", or wants to test
  kernel correctness without a GPU. This skill handles the full closed loop:
  generate kernel -> execute on CPU emulator -> verify against reference ->
  if wrong, analyze trace and fix -> repeat until correct.
---

You are a Triton kernel generation and debugging expert. Your task is to **generate or fix** Triton kernels based on user input, and verify correctness using this project's CPU emulator.

User input: $ARGUMENTS

---

## Step 1: Determine Input Type

| Input Type | Detection Rules |
|------------|----------------|
| **Natural language** | Plain text describing an operator or formula |
| **PyTorch model** | `.pt`/`.pth`, `nn.Module`, `torch.nn`, `torchvision`, PyTorch code block |
| **ONNX model** | `.onnx`, `onnxruntime`, `onnx.` |
| **Baseline Triton kernel** | `@triton.jit`, `import triton`, `tl.program_id`, Triton code block |
| **Fixed shape info** | Model name from registry, or explicit `[B,C,H,W]` shapes |

Multiple types can co-occur. Explicit shapes always take priority.

**Scenario**: Generation (no file to fix) or Repair (file path / "fix/debug" keywords).

---

## Step 2: Extract Semantics by Input Type

### 2a: Natural Language → determine shapes, formula, reduction needs, grid dimension (1D elementwise / 2D matrix/conv)

### 2b: PyTorch Model → parse `forward()`, identify operators and shapes. Key mappings:
- `F.conv2d` → `conv2d_resnet`, `F.batch_norm` → `batchnorm2d`, `F.relu` → `relu`
- `F.max_pool2d` → `maxpool2d`, `F.adaptive_avg_pool2d` → `adaptive_avgpool2d`
- `F.linear` → `matmul` + `add`, `torch.matmul` → `matmul`

### 2c: ONNX Model → `onnx.load()` then extract nodes and shapes. Key mappings:
- `Conv` → `conv2d_resnet`, `BatchNormalization` → `batchnorm2d`, `Relu` → `relu`
- `MaxPool` → `maxpool2d`, `GlobalAveragePool` → `adaptive_avgpool2d`
- `MatMul` → `matmul`, `Gemm` → `matmul` + `add`, `Softmax` → `softmax`

### 2d: Baseline Triton → convert to emulator form:
- `import triton.language as tl` → `from common import tl`
- Remove `@triton.jit`
- `tl.load(ptr + offsets, mask=...)` → `tl.load(ptr, offsets, mask=...)`
- `kernel[grid](...)` → `launch_kernel_1d(kernel, ..., grid_size=N)`
- **keepdims gotcha**: emulator `tl.sum/max/min` keeps dims, real Triton does not

### 2e: Fixed Shape → read `models/shapes_registry.py` for model name. Use small spatial (8-32) for unit tests, real sizes for integration.

---

## Step 3: Generate Operator Module

Create `emulators/test/<op_name>/__init__.py` with 4-part structure:

1. **Kernel** — ONLY uses `tl.*` API. Data is 1D flat, offsets are linear indices, OOB must be masked.
2. **Emulate wrapper** — validate inputs → flatten → `launch_kernel_*` → reshape output
3. **Reference** — pure numpy/torch ground truth
4. **Test** — basic + edge cases

**Read `emulators/common/__init__.py`** for available `tl.*` APIs and their signatures. The source is the authoritative reference.

**Critical gotcha**: `tl.sum`/`tl.max`/`tl.min` use `keepdims=True` in the emulator. Add `.reshape()` after reduction if needed.

---

## Step 4: Run Verification

```bash
cd "/Users/wangshunxian/operator autogen/OP_autogen/emulators" && python -c "from test.<op_name> import test; test()"
```

If pass → register in `emulators/test/run_all_tests.py`. If model decomposition → continue to next operator.

---

## Step 5: Iteration Repair (max 5 rounds)

**Error Type A — EmulatorError** (crash with line number):
Fix the reported line directly. Common: `offsets OOB` → add mask; `Shape mismatch` → align store shapes; `Both must be 2D` → reshape before `tl.dot`.

**Error Type B — Shape Mismatch** (output shapes differ):
Check output size formula and grid_size calculation.

**Error Type C — Numerical Mismatch** (`max_abs_err`/`max_rel_err`):
- `HAS_NAN` → division by zero, log of negative
- `ALL_ZERO` → mask over-filtering or offsets all OOB
- No anomaly but values off → check stride/offset formulas, pid decoding

**Rules**: Smallest change per round. Re-run after every change. Record errors for emulator improvement.
