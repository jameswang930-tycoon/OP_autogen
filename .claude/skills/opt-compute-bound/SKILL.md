---
name: opt-compute-bound
description: >
  Optimization techniques & guidelines for compute-bound kernels (bottleneck =
  compute_bound_at_peak — compute is the dominant cost). Use when the adapter Verdict names
  compute_bound_at_peak, to apply proven compute-side optimization patterns (operator strength,
  MAC/tensor primitive usage, instruction mix) rather than reasoning from scratch. Content lives
  in references/ (one markdown file per technique/guideline, filled in the confidential env).
  Does NOT generate kernels itself — pair with triton-gen.
---

You are the optimization-knowledge reference for **compute-bound** kernels. In agent mode the
orchestrator points you here when the Verdict bottleneck is `compute_bound_at_peak`; load the
relevant technique(s) from `references/` (filled in the confidential env) and apply them.

## Scope
compute-bound optimization: operator strength / arithmetic intensity, MAC or tensor primitive
selection, instruction mix, loop/order choices that raise compute throughput.

## Field contract
Each `references/*.md` is one technique/guideline (markdown; frontmatter may tag applies_to =
bottleneck id). See `.claude/skills/extension-guide/references/README.md` for the tagging
convention. Empty references/ → nothing to load (no error).
