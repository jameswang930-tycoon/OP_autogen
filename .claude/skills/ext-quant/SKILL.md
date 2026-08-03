---
name: ext-quant
description: >
  Quantization extension primitives — quant / dequant / cast for int8 / fp8. Use when the
  operator involves quantization or low-precision cast. Loads only this category's primitives
  from references/ on demand. Does NOT generate kernels itself — pair with triton-gen;
  standard Triton stays the structural base.
---

You are the extension-primitive reference for **quantization** ops. In agent mode the
orchestrator points you here when the current operator involves quantization or low-precision
cast; load the relevant primitive from `references/` (filled in the confidential env) and apply
it as a local replacement on a standard-Triton structural base.

## Scope
quant / dequant / cast (int8 / fp8 and related scaling primitives).

## Field contract
Each `references/*.yaml` is one primitive: `name` / `semantics` / `signature` / `category` /
`example` / `pitfalls` (+ optional `module` / `applies_to`). See
`.claude/skills/extension-guide/references/README.md`.
