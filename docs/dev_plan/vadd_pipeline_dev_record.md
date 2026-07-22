# Vadd End-to-End Pipeline Development Record

> Target: validate the full skill pipeline (triton-plan → triton-gen → triton-verify → triton-fix → triton-convert) on the **vadd** operator using the latest cost_emulator (real VecUnit numbers), running as a **serial skill-guided flow with no orchestrator agent**.

Date: 2026-06-25

---

## 1. Background

After splitting the single `triton-gen` skill into 5 single-responsibility skills, this record validates end-to-end that:

- the latest cost model (`costModel/cost_emulator` @ `5bbeeb3`) — where **VecUnit now uses real measured numbers** (peak 16000 GB/s, up from the 1500 placeholder) — produces a meaningful plan code;
- that plan code drives `triton-gen` to produce a correct vadd emulator;
- the whole chain `plan → gen → verify → (fix) → convert` runs serially, guided only by the skills (no orchestrator).

**Operator: vadd** — element-wise `out = x + scalar`. This matches the cost-model `vadd` DSL (`vadd(ub_c, ub_a, 1.0)` = vector + scalar). Note: the cost model has no dedicated vec+vec op, so `vadd` (vec+scalar) is the elementwise proxy — equivalent in compute cost, which is all that matters for the plan since the bottleneck is bandwidth.

---

## 2. Pipeline Walkthrough

### Stage 1 — /triton-plan

Input: `out = x + scalar`, N = 4096.

- Detected `op_kind = vadd`, `shapes = {N: 4096}`.
- Called `cost_planner.plan("vadd", {N:4096})` → `supported = True`.
- Dumped plan code to `emulators/test/vadd/.plan.json`.

**Plan code key metrics:**

| op | engine | duration | share | BW util | regime |
|----|--------|----------|-------|---------|--------|
| gm_to_ub | GM→UB | 8.27 ns | 26% | 990 / 1500 GB/s, 66% | **ramp** (8 KB < 12 KB sat) |
| vadd | VecUnit | 1.47 ns | 4.66% | 11130 / 16000 GB/s, 69% | ramp |
| ub_to_gm | UB→GM | 21.85 ns | **69%** | 750 / 750, 100% | saturated |

- `total_ns = 31.58`, **fully sequential** (RAW chain load → compute → store).
- **bottleneck = ub_to_gm (store, 69%)**; compute (vadd) is nearly free (4.66%).
- gm_to_ub sits on the bandwidth ramp (66% util) — its 8 KB tile is below the 12 KB saturation point.

### Stage 2 — /triton-gen

Read `.plan.json` and applied the plan's guidance:

- **vec path** (GM→UB→Vec→UB→GM) → `tl.load` → compute → `tl.store`, 1D grid.
- gm_to_ub was on the ramp → **set `BLOCK_SIZE = 8192` (16 KB tile)** to push the read above the 12 KB saturation point. *This is the concrete, plan-driven tiling decision — not a guess.*
- bottleneck is the store (unavoidable for an elementwise op); a single tile is sequential by RAW — no parallelism to exploit without multi-tile double-buffering.

Produced `emulators/test/vadd/__init__.py` (4-part: `vadd_kernel` / `emulate_vadd` / `reference_vadd` / `test`).

### Stage 3 — /triton-verify (inline in gen)

Ran `from test.vadd import test; test()`. **Result: 5/5 PASS** (max_abs = 0 — exact op):

```
[PASS] vadd_1d_default
[PASS] vadd_1d_scalar
[PASS] vadd_2d
[PASS] vadd_unaligned
[PASS] Correctly caught empty input error
```

### Stage 4 — /triton-fix

**Skipped** — verification passed on the first try; no errors to fix.

### Stage 5 — /triton-convert

Converted `__init__.py` → `triton_real.py` via the 5 mechanical rewrites (kernel compute logic unchanged):

1. import: `from common import …` → `import triton; import triton.language as tl` (+ torch)
2. `@triton.jit` decorator on the kernel
3. `tl.load(x_ptr + offsets, …)` (was `tl.load(x_ptr, offsets, …)`)
4. `tl.store(out_ptr + offsets, …)` (was `tl.store(out_ptr, offsets, …)`)
5. `vadd_kernel[grid](…)` with torch tensors (was `launch_kernel_1d(…)` with numpy)

NPU self-check: `grid = ceil(4096/8192) = 1`, far under the 65535 coreDim limit — no two-level tiling needed.

Static check confirmed all 5 rewrites and that the kernel compute logic (pid / offsets / mask / `out = x + scalar`) is byte-identical between the two files. Real-Triton run was not executed (no `triton` wheel in `.venv`).

---

## 3. Conclusion

- ✅ The latest cost model (real VecUnit numbers) produces a meaningful vadd plan code.
- ✅ The plan code drove `triton-gen`'s tiling decision (`BLOCK_SIZE = 8192` to escape the read-BW ramp) — concrete evidence the plan guidance is **actionable, not decorative**.
- ✅ The serial skill pipeline `plan → gen → verify → convert` ran end-to-end with no orchestrator; `fix` was not needed.
- ⚠️ Caveat: `convert`'s real-Triton run was skipped (no `triton` in `.venv`); only the static rewrite check passed.

## 4. Artifacts

- `emulators/test/vadd/.plan.json` — plan code (gitignored process artifact)
- `emulators/test/vadd/__init__.py` — emulator kernel (4-part)
- `emulators/test/vadd/triton_real.py` — real Triton kernel
