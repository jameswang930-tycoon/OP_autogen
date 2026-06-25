---
name: triton-verify
description: >
  Read-only verification of an existing emulator module. Runs run_with_feedback
  or the module's test(), reports PASS (with error magnitudes) or FAIL (with the
  deduplicated feedback string). Does NOT modify code and does NOT repair — for
  repair use /triton-fix. Trigger when the user asks to "verify/check <op>",
  "is <op> correct", or "run <op>'s test".
---

You are a read-only verification runner. Input: $ARGUMENTS (<op> name or path).

## Locate the module

`emulators/test/<op>/__init__.py`. If it does not exist, report and stop.

## Run

```bash
cd emulators && python3 -c "from test.<op> import test; test()"
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
