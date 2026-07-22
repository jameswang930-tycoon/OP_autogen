# Emulator Improvements Done (2026-05-20)

Error feedback precision overhaul in `emulators/common/__init__.py`. Solved the problem where a single bug triggered massive duplicate error reports across hundreds of programs (e.g., ResNet conv2d with 288 programs).

## Changes

1. **EmulatorError structured line numbers** — added `source_line`/`source_file`/`source_code` attributes, auto-captured via `traceback.extract_stack()`
2. **AggregatedEmulatorError** — groups errors by `(line_number, API)` across multiple programs
3. **TraceLogger enhancements** — `_invocation_id` batch marker, `begin_invocation()`, `get_flags_summary_deduped()` deduplicated summary
4. **launch_kernel `collect_errors=True`** — runs all programs before aggregating and reporting
5. **verify() uses deduplicated summary** — replaced verbose per-entry listing
6. **run_with_feedback()** — one-step: enable TraceLogger → run → verify → produce concise feedback → disable
7. **conv1d/conv2d tests updated** — added `run_with_feedback` + `collect_errors` demo

## Results

- conv2d 288-program OOB error: from 288 duplicate reports → **1-line precise report**
- TraceLogger soft errors: from 2304 trace lines → **6-line deduplicated summary**
- All existing tests backward-compatible and passing

## Skills

The original single `triton-gen` skill was later split into 5 single-responsibility skills: `/triton-plan` (input→plan), `/triton-gen` (plan→emulator kernel), `/triton-verify` (read-only check), `/triton-fix` (repair loop), `/triton-convert` (→real triton). This document is referenced by `/triton-verify` and `/triton-fix` for the error-feedback format.
