---
name: triton-gen
description: >
  Generate an emulator kernel module from a plan code file (.plan.json). Reads
  the plan produced by /triton-plan, generates emulators/test/<op>/__init__.py
  (4-part: kernel/emulate/reference/test), then runs ONE inline verification to
  report PASS/FAIL — no repair loop (that is /triton-fix) and no real-Triton
  conversion (that is /triton-convert). Trigger when the user says "generate
  kernel", "/triton-gen <op>", after a plan exists.
---

You are an emulator kernel generation expert. Input: $ARGUMENTS (an `<op>` name).

## Step 1: Read Plan Code

Read `emulators/test/<op>/.plan.json` (produced by `/triton-plan`). It carries
`op_kind`, `shapes`, `dtype`, and (when supported) the cost-model `plan` +
`raw_llm`. Use it as advisory context to choose tiling granularity and memory
level. `mock: true` means the cost model did not support this op — fall back to
the default path (cube ops → matmul path GM→L1→L0; vec ops → vadd path GM→UB→Vec).

**dtype**: the plan's `dtype` field (fp16/fp32/bf16, default fp32) sets the
storage dtype for the generated kernel (see Step 2).

**If the input is a baseline Triton kernel** (the user pasted `@triton.jit` code,
not an `<op>` name with a plan): convert it to emulator form first — the reverse
of `/triton-convert`:
- `import triton.language as tl` → `from common import tl`
- drop `@triton.jit`
- `tl.load(ptr + offsets, mask=...)` → `tl.load(ptr, offsets, mask=...)`
- `tl.store(ptr + offsets, ...)` → `tl.store(ptr, offsets, ...)`
- `kernel[grid](...)` → `launch_kernel_Nd(kernel, ..., grid_size=)` with numpy data

Then wrap it in the 4-part module. `/triton-plan` is optional for this path.

## Step 2: Generate the Module

Create `emulators/test/<op>/__init__.py` with the 4-part structure:

1. **kernel** — pure `tl.*` API; data is 1D flat, offsets are linear indices, OOB masked.
2. **emulate wrapper** — validate inputs → flatten → `launch_kernel_*` → reshape output.
3. **reference** — numpy/torch ground truth.
4. **test** — basic + edge cases.

**Import whitelist** (ONLY these):

```python
from common import tl, xarray, launch_kernel_1d, launch_kernel_2d, launch_kernel_3d, verify, EmulatorError
```

**NPU-compatible coding rules** (the emulator enforces these natively, so the
generated kernel deploys to real hardware without rewrite):

1. Scalar accumulators — `0.0`, never `tl.zeros((1,), ...)`.
2. In-place accumulation — `acc += expr`, never `acc = acc + expr`.
3. No redundant axis on 1D reduction — `tl.sum(x)`, not `tl.sum(x, axis=0)`.

**dtype**: use the plan's `dtype` (default fp32) for storage via
`{"fp16":np.float16, "fp32":np.float32, "bf16":np.float32}`. **matmul accumulator
stays fp32** (mixed precision — don't replace every float32 with the storage dtype).

**common API**: read `emulators/common/__init__.py` for the authoritative `tl.*`
signatures (load/store/dot/zeros/full/arith/reduce/program_id/cdiv,
launch_kernel_Nd, verify, run_with_feedback, EmulatorError).

Writing-kernel patterns & pitfalls: `docs/emulator_observations/implementation_patterns.md`.
Import conventions & 4-part details: `docs/project_knowledge/test_conventions.md`.

## Step 3: Inline Verify (report only, no repair)

```bash
cd emulators && python3 -c "from test.<op> import test; test()"
```

or call `run_with_feedback(emulate_<op>, reference_<op>)`.

- **PASS** → print max_abs_err / max_rel_err; (optional) register in `emulators/test/run_all_tests.py`; tell the user: `/triton-convert <op>`.
- **FAIL** → report the feedback; tell the user: `/triton-fix <op>`.
