---
name: ext-activation
description: >
  Activation & elementwise-math extension primitives — softmax / gelu / relu / div / sqrt /
  exp. Use when the operator is an activation or a pointwise elementwise math op. Loads only
  this category's primitives from references/ on demand. Does NOT generate kernels itself —
  pair with triton-gen; standard Triton stays the structural base.
---

You are the extension-primitive reference for **activation / elementwise-math** ops. In agent
mode the orchestrator points you here when the current operator is an activation or elementwise
math op; load the relevant primitive from `references/` (filled in the confidential env) and
apply it as a local replacement on a standard-Triton structural base.

## Scope
softmax / gelu / relu / div / sqrt / exp (and similar pointwise math).

## Field contract
Each `references/*.yaml` is one primitive: `name` / `semantics` / `signature` / `category` /
`example` / `pitfalls` (+ optional `module` / `applies_to`). See
`.claude/skills/extension-guide/references/README.md`.
