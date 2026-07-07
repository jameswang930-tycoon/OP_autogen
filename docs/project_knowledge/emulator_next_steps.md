# Emulator Next Steps

## DualRunner Comparative Debugging (not yet implemented)

**Problem:** Current emulator pinpoints ~60-70% of common bugs. The remaining 30-40% are "silent numerical errors" (stride swap, pid decode errors, wrong index formulas) where the emulator can only report "value mismatch" without identifying root cause.

**Core idea:** Dual-track comparison at `tl.load/tl.store` boundaries — run both buggy kernel and reference kernel simultaneously, compare offsets per-program. The first divergence point is the root cause.

**Design doc:** `tmp/DESIGN_dual_runner.md`

## Cost Model (integrated via direct simulator call)

**Problem:** The full closed-loop goal is "correctness + performance". CallCapture data naturally supports performance analysis.

**Current status:** The cost model (`costModel/cost_emulator`, an external collaborator's work-in-progress, vendored as a subtree) is integrated via `/triton-plan`, which writes a cost_emulator DSL program from the op semantics, runs `cost_emulator/simulator.py` **directly** (`--verify` + `--llm --critical-path`), and dumps the simulator's `raw_llm` output verbatim as plan code to `.plan.json` = `{op, shapes, dtype, dsl, raw_llm}`. `/triton-gen` then reads `raw_llm` in depth to guide kernel generation. This is a read-only, zero-intrusion adapter — the cost model's internal DSL/bandwidth curves are NOT modified from this project. If the simulator call fails, `/triton-plan` writes a mock stub so downstream is not blocked.

## LLM Iteration Mode

Recommended: **structured JSON + natural language summary** dual output. JSON lets the LLM pinpoint exact line numbers; natural language helps the LLM understand "why it's wrong".
