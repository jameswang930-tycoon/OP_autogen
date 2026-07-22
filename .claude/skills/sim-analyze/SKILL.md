---
name: sim-analyze
description: >
  Pipeline analyze stage. Use when a simulation round has produced adapter feedback — a
  7-section summary plus a machine-readable Verdict from control/feedback_adapter.py — and
  the user wants to "analyze the bottleneck / pick an optimization lever / 分析瓶颈 / decide
  the next round". Reads the Verdict bottleneck category, looks the lever up in
  control/vocabulary.yaml, and outputs the next-round improvement direction. Does NOT
  generate a kernel (that is triton-gen) and does NOT run any analytic cost-model
  simulator. Trigger when adapter feedback exists and the bottleneck / lever must be chosen.
---

You are the simulation-feedback analyzer. The pipeline's causal direction is reversed:
the kernel is generated first, measured on a real simulator, then analyzed, then
regenerated. Your job is the **analyze** step — read deterministic adapter output, pick a
lever, output the next-round direction. You do NOT run any analytic / predictive simulator.

## Input

Read the adapter output produced by `control/feedback_adapter.py`:
- a **7-section summary** (Execution Summary / Time Breakdown / Per-Op / Engine Util /
  Bandwidth Util / Parallelism / Critical Path), and
- a machine-readable **Verdict** with fields `bottleneck`, `lever`, `cycles`,
  `expected_gain`.

The Verdict is produced by deterministic code; treat its `bottleneck` category as ground
truth — do NOT re-derive it from prose.

## Method (inherited from bottleneck-analysis; only the data source changed)

1. **Critical path first.** Only optimize ops on the critical path; off-path cost is noise.
2. **Dominant cost into a small actionable category.** The Verdict `bottleneck` is already
   the dominant category, drawn from the fixed vocabulary in `control/vocabulary.yaml`.
3. **Category to lever is a lookup, not free-form reasoning.** Look the category up in
   `control/vocabulary.yaml` (its `lever` field) and reuse that lever verbatim.

## Output

One short block:
- `bottleneck`: the Verdict category verbatim (a vocabulary id).
- `lever`: the looked-up lever for that category.
- `next_round_direction`: one or two concrete sentences — which lever to apply and what to
  change in the next kernel. If `expected_gain` is low, say so and flag that stopping may
  be better than another expensive simulation.

## Stop signal

If the loop-controller `control/loop_controller.py` has signaled stop (epsilon converged /
max rounds / irreducible / oscillation / no-progress / numerical-fail), do NOT propose
another round — report the best-so-far kernel instead.

## Do NOT

- Do not run `costModel/cost_emulator/simulator.py` or write a 7-engine DSL — the analytic
  predictor is retired (see `emulators/README.md`).
- Do not re-derive the bottleneck from free text; read the Verdict.
- Do not generate a kernel (triton-gen) or decide stops (loop-controller).
