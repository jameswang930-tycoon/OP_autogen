---
name: ext-reduction
description: >
  Reduction (归约) extension primitives — reduce_sum / reduce_max / reduce_min / reduce_mean
  (incl. int8 reduce). Use when the operator reduces one or more dimensions (a sum/max/min/mean
  over an axis). Loads only this category's primitives from references/ on demand. Does NOT
  generate kernels itself — pair with triton-gen; standard Triton stays the structural base.
---

You are the extension-primitive reference for **reduction** ops. In agent mode the orchestrator
points you here when the current operator is a reduction; load the relevant primitive from
`references/` (filled in the confidential env) and apply it as a local replacement on a
standard-Triton structural base.

## Scope
reduce_sum / reduce_max / reduce_min / reduce_mean (incl. int8 variants).

## Field contract
Each `references/*.yaml` is one primitive: `name` / `semantics` / `signature` / `category` /
`example` / `pitfalls` (+ optional `module` / `applies_to`). See
`.claude/skills/extension-guide/references/README.md`.
