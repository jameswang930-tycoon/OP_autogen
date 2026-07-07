# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Project Knowledge

These files contain project context. Read on demand (not auto-loaded). The **Owner skill** column shows which `/skill` owns each doc:

| File | Owner skill | When to Read |
|------|-------------|-------------|
| `docs/project_knowledge/project_overview.md` | public (landing) | Project structure, the 5-skill workflow, implemented operators |
| `docs/project_knowledge/environment_and_running.md` | public | Which Python to use (`.venv` vs system `python3`), how to run each component |
| `docs/project_knowledge/plan_code_contract.md` | /triton-plan + /triton-gen | `.plan.json` schema, the plan→gen handoff, cost-model entry point |
| `docs/project_knowledge/input_detection.md` | /triton-plan | Detecting input type, extracting op_kind + shapes |
| `docs/project_knowledge/test_conventions.md` | /triton-gen + /triton-verify | Import conventions (gen), weight/registration policy (verify) |
| `docs/project_knowledge/emulator_improvements_done.md` | /triton-verify + /triton-fix | Error-feedback format, dedup mechanism |
| `docs/project_knowledge/emulator_error_coverage.md` | /triton-verify + /triton-fix | Emulator detection capabilities & blind spots |
| `docs/project_knowledge/emulator_next_steps.md` | public (roadmap) | DualRunner / Cost Model / iteration mode roadmap |
| `docs/project_knowledge/emulator_to_triton_conversion.md` | /triton-convert | Converting emulator kernels to real Triton for NPU |
| `docs/emulator_observations/implementation_patterns.md` | /triton-gen | Writing-kernel patterns & pitfalls |
| `docs/emulator_observations/` (api_gaps, error_accumulation, precision_gaps, missing_coverage) | /triton-verify | Precision/coverage gaps, API differences |