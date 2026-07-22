---
name: triton-verify
description: >
  Pipeline verification stage (READ-ONLY). Use when
  emulators/test/<op>/__init__.py already exists and the user wants to
  "verify / check <op>", "is <op> correct", or "run <op>'s test" (跑测试). Runs
  run_with_feedback or the module's test() and reports PASS (with error
  magnitudes) or FAIL (with the deduplicated feedback string). This skill NEVER
  edits code, NEVER writes to disk, and NEVER repairs: for repair use triton-fix.
  Trigger for any "verify / check / is it correct / 是否正确 / run the test"
  request.
---

You are a read-only verification runner. Input: extract the operator name <op> (and any optional input type / feedback) from the user's request.

## Locate the module

`emulators/test/<op>/__init__.py`. If it does not exist, report and stop.

## Run

```bash
cd emulators && ../.venv/bin/python -c "from test.<op> import test; test()"
```

Preferred — call `run_with_feedback(emulate_<op>, reference_<op>)` and read
`{"passed", "feedback", ...}`. The `feedback` field is a deduplicated summary
(TraceLogger anomalies + numerical excerpt) tailored for LLM consumption; it is
what /triton-fix consumes.

## Report (terminal only — write nothing to disk)

- **PASS**: print max_abs_err, max_rel_err, output shape; suggest `/triton-convert <op>`.
- **FAIL**: print the deduplicated feedback verbatim; suggest `/triton-fix <op>`.

References (read on demand): emulator feedback mechanism —
`docs/project_knowledge/emulator_improvements_done.md`; detection capabilities &
blind spots — `docs/project_knowledge/emulator_error_coverage.md`; precision /
coverage gaps — `docs/emulator_observations/` (api_gaps, error_accumulation,
precision_gaps, missing_coverage); random-weight policy —
`docs/project_knowledge/test_conventions.md`.
