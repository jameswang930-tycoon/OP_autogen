# Emulator Next Steps

## DualRunner Comparative Debugging (not yet implemented)

**Problem:** Current emulator pinpoints ~60-70% of common bugs. The remaining 30-40% are "silent numerical errors" (stride swap, pid decode errors, wrong index formulas) where the emulator can only report "value mismatch" without identifying root cause.

**Core idea:** Dual-track comparison at `tl.load/tl.store` boundaries — run both buggy kernel and reference kernel simultaneously, compare offsets per-program. The first divergence point is the root cause.

**Design doc:** `tmp/DESIGN_dual_runner.md`

## Cost Model (partially integrated)

**Problem:** The full closed-loop goal is "correctness + performance". CallCapture data naturally supports performance analysis.

**Current status:** A cost model (`costModel/cost_emulator`, an external collaborator's work-in-progress) is integrated loosely via `/triton-plan`, which calls `costModel/cost_planner.py` and dumps plan code to `.plan.json`. Only `vadd`/`matmul` are mapped; other ops get a mock plan stub. This is a read-only adapter — the cost model's internal DSL/bandwidth curves are NOT modified from this project.

**Extension points (future):**
- `CostModelAnalyzer.analyze_memory()` — total bytes, unique address count, access pattern classification
- `CostModelAnalyzer.analyze_compute()` — FLOPs estimation, arithmetic intensity, roofline comparison

## LLM Iteration Mode

Recommended: **structured JSON + natural language summary** dual output. JSON lets the LLM pinpoint exact line numbers; natural language helps the LLM understand "why it's wrong".
