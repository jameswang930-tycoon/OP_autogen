"""Canonical placeholder names for the dual-mode skill templates (T10).

The orchestrator (T11) injects EXACTLY these variable names; the skill bodies (T10) use
EXACTLY these as ``{{VAR}}``. Keeping them in one place makes the placeholder/injection
mismatch check (the easiest-to-introduce and hardest-to-spot bug) automatic.
"""

# triton-gen (llm_generate) template placeholders.
TRITON_GEN_PLACEHOLDERS = frozenset({
    "OP",
    "SHAPES",
    "DTYPE",
    "BASELINE_SRC",        # may be empty (no baseline / first round)
    "VERDICT_JSON",        # empty on the first round
    "FEEDBACK_SUMMARY",    # empty on the first round
    "RETRIEVED_EXPERIENCE",
    "EXTENSION_INDEX",
    "COMPILE_ERROR",       # empty unless the previous attempt failed to compile (T13-3)
    "OPTIMIZATION_HINT",   # 预留(V2):按瓶颈指向 optimization skill 的优化知识提示；空则降级（无内容）
})

# sim-analyze (llm_choose_lever) template placeholders.
SIM_ANALYZE_PLACEHOLDERS = frozenset({
    "VERDICT_JSON",
    "FEEDBACK_SUMMARY",
    "CANDIDATE_LEVERS",
})

# extension-guide is read as an index — no placeholders.
EXTENSION_GUIDE_PLACEHOLDERS = frozenset()
