---
name: opt-memory-bound
description: >
  Optimization techniques & guidelines for memory-bound kernels (bottleneck =
  memory_underfilled — memory transfer is the dominant cost and bandwidth is underfilled). Use
  when the adapter Verdict names memory_underfilled, to apply proven memory-side patterns (bulk /
  async-copy, access coalescing, reuse/tiling, layout) rather than reasoning from scratch. Content
  lives in references/ (one markdown file per technique/guideline, filled in the confidential env).
  Does NOT generate kernels itself — pair with triton-gen.
---

You are the optimization-knowledge reference for **memory-bound** kernels. In agent mode the
orchestrator points you here when the Verdict bottleneck is `memory_underfilled`; load the
relevant technique(s) from `references/` (filled in the confidential env) and apply them.

## Scope
memory-bound optimization: bulk / async-copy, access coalescing & reordering, data reuse / tiling,
layout transforms, reducing round-trips.

## Field contract
Each `references/*.md` is one technique/guideline (markdown; frontmatter may tag applies_to =
bottleneck id). See `.claude/skills/extension-guide/references/README.md` for the tagging
convention. Empty references/ → nothing to load (no error).
