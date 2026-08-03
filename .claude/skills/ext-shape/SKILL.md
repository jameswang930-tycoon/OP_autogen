---
name: ext-shape
description: >
  Shape-transform / slice / block-pointer extension primitives — reshape / transpose / slice /
  block_ptr. Use when the operator rearranges data layout, slices a view, or uses block
  pointers. Loads only this category's primitives from references/ on demand. Does NOT
  generate kernels itself — pair with triton-gen; standard Triton stays the structural base.
---

You are the extension-primitive reference for **shape-transform / slice / block-pointer** ops.
In agent mode the orchestrator points you here when the current operator reshapes, transposes,
slices, or uses block pointers; load the relevant primitive from `references/` (filled in the
confidential env) and apply it as a local replacement on a standard-Triton structural base.

## Scope
reshape / transpose / slice / block_ptr (and layout/conversion helpers).

## Field contract
Each `references/*.yaml` is one primitive: `name` / `semantics` / `signature` / `category` /
`example` / `pitfalls` (+ optional `module` / `applies_to`). See
`.claude/skills/extension-guide/references/README.md`.
