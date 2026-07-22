---
name: triton-fix
description: >
  Pipeline repair stage. Use after triton-verify or triton-gen reports FAIL and
  emulators/test/<op>/__init__.py exists and needs fixing. Runs an internal loop
  (max 5 rounds): classify the failure (EmulatorError / Shape / Numerical) → make
  the smallest fix → re-verify → until PASS or the budget is exhausted. Edits ONLY
  emulators/test/<op>/__init__.py. Trigger for any "fix / debug / 修复 / the op is
  wrong / results don't match" request. Note: the emulator has a ~30-40% silent
  numerical blind spot — do not loop forever on errors it cannot catch; surface
  them to the user instead.
---

You are an emulator kernel repair expert. Input: extract the operator name <op> (and any optional input type / feedback) from the user's request.

## Setup

Locate `emulators/test/<op>/__init__.py`. If no feedback was supplied, run
`run_with_feedback(emulate_<op>, reference_<op>)` once to obtain it.

## Repair Loop (max 5 rounds)

```
for round in 1..5:
    result = run_with_feedback(emulate_<op>, reference_<op>)
    if result["passed"]: report PASS, stop
    classify feedback (A/B/C below) → smallest fix → edit __init__.py only
5 rounds without PASS → report failure + the last feedback, stop (no infinite loop)
```

## Failure classification

- **A. EmulatorError** (crash with a source line): fix the reported line directly.
  Common: `offsets OOB` → add/strengthen the mask; `Shape mismatch` → align store
  shapes; `Both must be 2D` → reshape before `tl.dot`.
- **B. Shape Mismatch** (output shapes differ): check the output-size formula and
  the `grid_size` / grid computation.
- **C. Numerical Mismatch** (max_abs_err / max_rel_err):
  - `HAS_NAN` → division by zero, log of a negative.
  - `ALL_ZERO` → mask over-filtering, or all offsets OOB.
  - Otherwise → check stride/offset formulas, pid decoding.

## Rules

- Smallest change per round; re-run after every change.
- Edit only `emulators/test/<op>/__init__.py`.
- Some bugs the emulator CANNOT catch (silent numerical errors ~30-40%, see
  `docs/project_knowledge/emulator_error_coverage.md`) — do not loop forever on those; surface them to
  the user instead.
- Feedback format details: `docs/project_knowledge/emulator_improvements_done.md`.
