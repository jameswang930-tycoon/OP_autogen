---
name: opt-stall-dependency
description: >
  Optimization techniques & guidelines for dependency-stall kernels (bottleneck =
  stall_dependency — the pipeline stalls waiting on dependencies). Use when the adapter Verdict
  names stall_dependency, to apply proven latency-hiding patterns (double buffering, pipeline
  overlap, dependency reduction) rather than reasoning from scratch. Content lives in references/
  (one markdown file per technique/guideline, filled in the confidential env). Does NOT generate
  kernels itself — pair with triton-gen.
---

You are the optimization-knowledge reference for **dependency-stall** kernels. In agent mode the
orchestrator points you here when the Verdict bottleneck is `stall_dependency`; load the relevant
technique(s) from `references/` (filled in the confidential env) and apply them.

## Scope
dependency-stall optimization: double buffering, pipeline overlap, dependency reduction /
reordering, prefetching to hide latency.

## Field contract
Each `references/*.md` is one technique/guideline (markdown; frontmatter may tag applies_to =
bottleneck id). See `.claude/skills/extension-guide/references/README.md` for the tagging
convention. Empty references/ → nothing to load (no error).
