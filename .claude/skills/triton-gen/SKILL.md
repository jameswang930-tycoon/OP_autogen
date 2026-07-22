---
name: triton-gen
description: >
  Pipeline generate stage. Use when a plan or an adapter Verdict exists and a kernel must be
  generated ("生成算子" / generate kernel / turn the plan into a kernel). Produces a REAL
  Triton + extension kernel as a multi-segment module (kernel / reference / compare) ready
  to launch on the simulator — NOT an emulator-form kernel. Defaults to standard Triton and
  adds an extension primitive only when the Verdict bottleneck category requires it. Does
  NOT convert to emulator form and does NOT run a repair loop. Trigger for any kernel
  generation request once a plan / Verdict is available.
---

You are a Triton kernel generator. The pipeline is reversed: you generate a real Triton
kernel, it is measured on a real simulator, and the feedback drives the next round. Your
output is a **multi-segment module** (kernel / reference / compare) — not a bare kernel and
not an emulator-form kernel.

## Step 1: Read inputs

- The plan / op semantics: op name, shapes, dtype (fp16 / fp32 / bf16, default fp32).
- The adapter **Verdict** (from `control/feedback_adapter.py`) if a prior round ran: its
  `bottleneck` category tells you whether an extension primitive is warranted this round.
- `retrieved_experience` (injected by `memory_cli.py inject`): historical experience for
  this op class — read it as an extra generation reference (how similar ops were tiled or
  parallelized, which extension primitive helped, pitfalls). Absent means memory is off.

## Step 2: Extension usage rule (default to vanilla Triton)

Write **standard Triton** by default. Add an extension primitive **only** when the Verdict
`bottleneck` category explicitly calls for one. The category-to-primitive mapping lives in
the extension cheatsheet at `.claude/skills/extension-guide/`, indexed by bottleneck
category. Benefit: the baseline is always legal vanilla Triton; if you mis-apply an
extension, the worst case is correct-but-unoptimized code, not broken code.

If no Verdict exists (first round), generate plain vanilla Triton.

## Step 3: Generate the multi-segment module

Follow the launchable-unit template `LAUNCHABLE_TEMPLATE` in `control/launch_template.py`:
three segments — kernel / reference (numpy or torch gold standard) / compare harness. The
compare harness computes max_abs_err versus reference and emits the canonical raw_sim_output
(`correct`, `max_abs_err`, `cycles`, `pipeline`) so correctness and performance stay two
distinguishable signals.

Coding rules:
- Standard Triton syntax (`@triton.jit`, `import triton.language as tl`, pointer-plus-offset
  load / store). No emulator dialect, no `from common import tl`.
- dtype per the plan; matmul accumulator stays fp32 (mixed precision).

## Step 4: Pre-sim gate (cheap, before spending a simulation)

Run `control/presim_gate.py` on the generated kernel plus a shape_contract. If it reports
problems (syntax / shape / dtype), fix them here before launching — a wasted large-kernel
simulation is the most expensive mistake.

## Do NOT

- Do not generate emulator-form code (`from common import tl`, comma-form `tl.load`, etc.) —
  the emulator is retired (see `emulators/README.md`).
- Do not enter a repair loop; report PASS / FAIL and let the loop-controller decide.
