# Emulator Next Steps

## DualRunner Comparative Debugging (not yet implemented)

**Problem:** Current emulator pinpoints ~60-70% of common bugs. The remaining 30-40% are "silent numerical errors" (stride swap, pid decode errors, wrong index formulas) where the emulator can only report "value mismatch" without identifying root cause.

**Core idea:** Dual-track comparison at `tl.load/tl.store` boundaries — run both buggy kernel and reference kernel simultaneously, compare offsets per-program. The first divergence point is the root cause.

**Design doc:** `docs/DESIGN_dual_runner.md` (in tmp/)

## Cost Model (not yet implemented)

**Problem:** The full closed-loop goal is "correctness + performance". CallCapture data naturally supports performance analysis.

**Extension points:**
- `CostModelAnalyzer.analyze_memory()` — total bytes, unique address count, access pattern classification
- `CostModelAnalyzer.analyze_compute()` — FLOPs estimation, arithmetic intensity, roofline comparison

## LLM Iteration Mode

Recommended: **structured JSON + natural language summary** dual output. JSON lets the LLM pinpoint exact line numbers; natural language helps the LLM understand "why it's wrong".
