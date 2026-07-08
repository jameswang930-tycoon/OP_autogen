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
`op`, `shapes`, `dtype`, `dsl`, and `raw_llm` (the simulator `--llm` full output).
Read `raw_llm` in depth for data flow (which engines), bottleneck (TIME BREAKDOWN +
CRITICAL PATH), and tiling (BANDWIDTH UTILIZATION regime / saturates_at). If
`raw_llm` is missing (mock/simulator failed), fall back to the default path (cube
→ matmul path GM→L1→L0; vec → vadd path GM→UB→Vec).

**Optional `retrieved_experience`**: if present (the memory module appended it via
`memory_cli.py inject <op>` after `/triton-plan`), it carries formatted historical
experience for this op class — read it as an extra generation reference (how similar
ops were tiled/parallelized, pitfalls hit). Absent ⇒ memory is off / empty; generate
exactly as without it. (See `docs/project_knowledge/plan_code_contract.md`.)

**dtype**: the plan's `dtype` (fp16/fp32/bf16, default fp32) sets storage dtype.

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
cd emulators && ../.venv/bin/python -c "from test.<op> import test; test()"
```

or call `run_with_feedback(emulate_<op>, reference_<op>)`.

- **PASS** → print max_abs_err / max_rel_err; (optional) register in `emulators/test/run_all_tests.py`; tell the user: `/triton-convert <op>`.
- **FAIL** → report the feedback; tell the user: `/triton-fix <op>`.
