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

## Step 1: Determine the Scenario

Decide which path to take based on user input:

- **Generation path**: Input is an operator name or computation description (e.g. "layernorm", "gelu", "y = x * sigmoid(x)") -> Generate a complete operator module from scratch
- **Repair path**: Input contains a file path or keywords like "fix/debug/repair" -> Read the existing kernel, run the emulator, fix based on feedback

---

## Generation Path

### Step 1: Analyze Operator Semantics

Determine:
- Input/output tensor shapes and dtypes
- Mathematical formula
- Whether it needs reduction (softmax/rmsnorm) or is elementwise
- Grid dimension: elementwise -> 1D; matrix/convolution -> 2D

### Step 2: Create Operator Directory and File

Create `emulators/<op_name>/__init__.py` with the standard 4-part structure:

```python
"""
<OpName> Emulator: <one-line description>
===================================================
Kernel: <formula>
Grid:   <1D/2D>, <grid description>
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError, TraceLogger, run_with_feedback

# 1. Kernel — use ONLY tl.* API
def <op>_kernel(..., BLOCK_SIZE: tl.constexpr):
    ...

# 2. Emulator wrapper — validate inputs + flatten + launch + reshape output
def emulate_<op>(...) -> np.ndarray:
    ...

# 3. Reference — pure numpy/torch implementation
def reference_<op>(...):
    ...

# 4. Self-test
def test():
    ...

if __name__ == "__main__":
    test()
```

### Step 3: Run Verification

```bash
cd emulators && python -c "from <op_name> import test; test()"
```

If it fails, enter the iteration flow (see below).

### Step 4: Register in run_all_tests.py

Add an import and call in `emulators/run_all_tests.py`.

---

## Repair Path

### Step 1: Read and Understand the Current Kernel

Read the target file. Understand the kernel logic, emulate wrapper, and reference implementation.

### Step 2: Run Emulator to Get Feedback

```bash
cd emulators && python -c "from <op_name> import test; test()"
```

### Step 3: Analyze Errors and Fix

Enter the iteration flow.

---

## Iteration Repair Flow (shared by both paths)

Choose your strategy based on error type:

### Error Type A: EmulatorError (hard error, immediate crash)

Error message format: `[Triton Emulator Error] in tl.xxx(): ... at: L<line>: <source code>`

| Error | Meaning | Fix |
|-------|---------|-----|
| `offsets OOB, no mask` | load/store out-of-bounds without mask | Add a `mask` parameter to that `tl.load` call |
| `axis=N OOR for ndim=M` | reduction axis out of range | Check the `axis` parameter of `tl.sum`/`tl.max`/`tl.min` |
| `Shape mismatch` | values shape != offsets shape | Verify that `store` values and offsets shapes align |
| `Both must be 2D` | `tl.dot` input is not 2D | Ensure both inputs to `dot` are 2D tensors |

**Fix method**: Go directly to the reported line number and fix.

### Error Type B: Shape Mismatch (emulator vs reference output shapes differ)

Error message: `Shape mismatch: emulator (a,b) vs reference (c,d)`

**Fix method**: Check the output size formula (e.g. H_out = H - kH + 1) and grid_size calculation.

### Error Type C: Numerical Mismatch (value deviation)

Error message: `max_abs_err=X.XXe+XX, max_rel_err=Y.YYe+YY`

May include TraceLogger anomaly summary:
```
Trace anomalies (deduplicated):
L<line>: tl.<api>() <section>.<tensor> -> FLAG  (Nx across M pids)
```

**Fix method**:
1. If `HAS_NAN` present -> check for division by zero, log of negative, sqrt of negative
2. If `ALL_ZERO` anomaly -> check if mask is over-filtering or offsets are all OOB
3. If no anomalies but values are off -> check stride/offset formulas, pid decoding logic, loop boundaries

### Iteration Rules

1. **Must re-run tests** after every change to verify
2. **Maximum 5 rounds** — if still failing, stop and report current state with analysis
3. Each round: **make the smallest possible change**, don't fix multiple bugs at once

### Common Failure Patterns (Trace Symptom -> Cause -> Fix)

| Trace Symptom | Likely Cause | Fix |
|---------------|-------------|-----|
| `tl.store` values non-zero but output all-zero | Offset calculation wrong, writing to wrong positions | Check stride calculation and output offset formula |
| `tl.sum` output shape unexpected | keepdims=True adding extra dimension | Add `.reshape()` after reduction, or adjust downstream broadcast |
| `HAS_NAN` after `tl.exp` | Input too large, exp overflow | Subtract max before exp (numerical stability) |
| `ALL_ZERO` after `tl.maximum(x, 0)` | All inputs negative | Expected for ReLU; check if upstream computation is correct |
| `tl.dot` shape mismatch | Operands not 2D or inner dims don't match | Check reshape before dot |
| `MOSTLY_ZERO` appears broadly | Mask over-filtering or offsets mostly OOB | Check mask condition and offset range |
| `HAS_INF` after division | Division by zero or near-zero | Add epsilon to protect denominator |

---

## tl.* API Reference

### Scheduling
| API | Signature | Notes |
|-----|-----------|-------|
| `tl.program_id` | `(axis=0) -> int` | axis: 0/1/2 |
| `tl.num_programs` | `(axis=0) -> int` | Grid size on that axis |
| `tl.cdiv` | `(x, y) -> int` | Ceiling division |

### Memory
| API | Signature | Notes |
|-----|-----------|-------|
| `tl.load` | `(ptr, offsets, mask=None, other=0.0) -> xarray` | Also accepts OffsetPointer |
| `tl.store` | `(ptr, offsets, values, mask=None)` | Also accepts OffsetPointer |
| `tl.atomic_add` | `(base_ptr, offsets, values, mask=None)` | |

### Creation
| API | Signature | Notes |
|-----|-----------|-------|
| `tl.zeros` | `(shape, dtype=tl.float32) -> xarray` | |
| `tl.full` | `(shape, value, dtype=tl.float32) -> xarray` | |
| `tl.arange` | `(start, end) -> xarray` | |

### Math
| API | Signature | Notes |
|-----|-----------|-------|
| `tl.exp` / `tl.log` / `tl.log2` / `tl.sqrt` / `tl.abs` | `(x) -> xarray` | Unary functions |
| `tl.sigmoid` / `tl.tanh` | `(x) -> xarray` | |
| `tl.maximum` / `tl.minimum` | `(x, y) -> xarray` | Binary functions |
| `tl.where` | `(cond, x, y) -> xarray` | |
| `tl.clamp` | `(x, min_val, max_val) -> xarray` | |
| `tl.sum` / `tl.max` / `tl.min` | `(x, axis=0) -> xarray` | **keepdims=True** (differs from real Triton!) |
| `tl.dot` | `(a, b, allow_tf32=True) -> xarray` | 2D only |

### Type Aliases
| Alias | Maps To |
|-------|---------|
| `tl.float16` / `tl.float32` / `tl.float64` | numpy dtype |
| `tl.int8` / `tl.int16` / `tl.int32` / `tl.int64` | numpy dtype |
| `tl.constexpr` | `int` |

---

## Important Emulator Behaviors

1. **`tl.sum` / `tl.max` / `tl.min` use `keepdims=True`** — output shape retains the reduced dimension as size 1. Real Triton does NOT keepdims. Generated kernels must account for this: use `.reshape()` after reduction or adjust downstream broadcasting.

2. **Two pointer-passing conventions**:
   - **Pointer style**: kernel uses `ptr + offset` arithmetic -> wrap input arrays with `wrap_ptr()` before passing in
   - **Emulator style**: kernel uses `tl.load(base_array, offsets)` -> pass raw numpy arrays directly

3. **TraceLogger overhead**: Off by default. Only enable when debugging failed runs. Each `tl.*` call is logged when enabled.

4. **`tl.load` / `tl.store` accept multiple calling conventions**:
   - Emulator style: `tl.load(base_ptr_array, offsets_array, mask=...)`
   - Pointer style: `tl.load(offset_pointer, mask=...)` where offset_pointer = ptr + offsets

## Emulator Architecture Reference

```
emulators/
├── common/              # tl.* API stubs, xarray, TraceLogger, verify(), run_with_feedback()
│   └── __init__.py      # The ONLY file verification scripts need to import
├── add/                 # Individual operator reference emulators (for unit-testing
├── matmul/              #   the tl.* stubs — NOT called by generated kernels)
├── softmax/
├── relu/
├── rmsnorm/
├── addrmsnormgamma/
├── transpose/
├── reshape/
├── conv1d/
└── run_all_tests.py     # Full test suite entry point
```

## Critical Kernel Writing Constraints

1. **Kernel must ONLY use `tl.*` methods** — no raw numpy for computation logic (constructing offsets with np.array is OK)
2. **All data passed as 1D flat arrays** — shape info passed through scalar parameters
3. **Offsets are linear indices** — decompose multi-dimensional indices manually with `//` and `%`
4. **OOB access MUST be guarded with mask**: `mask = offsets < n_elements`
5. **Reduction axis is always 0** — because data within a block is a 1D vector
6. **BLOCK_SIZE must be declared with `tl.constexpr`** — it's a compile-time constant
7. **`tl.load` returns xarray** (numpy subclass) — supports normal arithmetic operations

---

## Complete Example: add Operator

```python
"""
Add Emulator: element-wise addition
=====================================
Kernel: out[i] = x[i] + y[i]
Grid:   1D, each program handles BLOCK_SIZE elements
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError

# ---- Kernel ----
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr, offsets, mask=mask)
    y = tl.load(y_ptr, offsets, mask=mask)
    output = x + y
    tl.store(output_ptr, offsets, output, mask=mask)

# ---- Emulator wrapper ----
def emulate_add(x: np.ndarray, y: np.ndarray, BLOCK_SIZE=1024) -> np.ndarray:
    if x.shape != y.shape:
        raise EmulatorError("add_kernel", f"Shape mismatch: x={x.shape}, y={y.shape}")
    n = x.size
    x_flat = x.ravel().astype(np.float32)
    y_flat = y.ravel().astype(np.float32)
    out_flat = np.zeros(n, dtype=np.float32)
    grid = tl.cdiv(n, BLOCK_SIZE)
    launch_kernel_1d(add_kernel, x_flat, y_flat, out_flat, n, BLOCK_SIZE, grid_size=grid)
    return out_flat.reshape(x.shape)

# ---- Reference ----
def reference_add(x, y):
    return (x + y).astype(np.float32)

# ---- Test ----
def test():
    print("=" * 60)
    print(" Add Emulator Test")
    print("=" * 60)
    x = np.random.randn(1024).astype(np.float32)
    y = np.random.randn(1024).astype(np.float32)
    out = emulate_add(x, y)
    ref = reference_add(x, y)
    verify(out, ref, "add_basic")
    # Unaligned size
    x2 = np.random.randn(100).astype(np.float32)
    y2 = np.random.randn(100).astype(np.float32)
    verify(emulate_add(x2, y2, BLOCK_SIZE=32), reference_add(x2, y2), "add_unaligned")
    print()

if __name__ == "__main__":
    test()
```

---

## Minimal Verification Script (End-to-End)

A complete, directly runnable minimal verification script showing the full flow from kernel definition to verify:

```python
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, launch_kernel_1d, verify, TraceLogger, wrap_ptr

# 1. Kernel code (as a string, for dynamic compilation)
KERNEL = r'''
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr, offs, mask=mask)
    y = tl.load(y_ptr, offs, mask=mask)
    tl.store(out_ptr, offs, x + y, mask=mask)
'''

# 2. Compile kernel
exec_env = {'tl': tl, 'np': np}
exec(KERNEL, exec_env)
kernel_fn = exec_env['add_kernel']

# 3. Prepare data and run
n = 256
x = np.random.randn(n).astype(np.float32)
y = np.random.randn(n).astype(np.float32)
out = np.zeros(n, dtype=np.float32)
BLOCK = 64

TraceLogger.enable()
launch_kernel_1d(kernel_fn, x, y, out, n, BLOCK, grid_size=(n + BLOCK - 1) // BLOCK)

# 4. Verify
ref = x + y
result = verify(out, ref, "add")

if not result["passed"]:
    TraceLogger.dump(pid_filter=(0, 0, 0))
TraceLogger.disable()

if result["passed"]:
    print("VERIFIED. Kernel is correct.")
    print(KERNEL)
```

## Workflow Checklist

- [ ] Kernel function uses ONLY `tl.*` API
- [ ] All `tl.load` calls have correct `mask` protection
- [ ] `emulate_xxx` validates input shape/dtype
- [ ] `reference_xxx` uses pure numpy or torch, result cast to float32
- [ ] `test()` covers at minimum: basic functionality, unaligned sizes, edge cases
- [ ] Tests pass: `cd emulators && python -c "from <op> import test; test()"`
