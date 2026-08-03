---
name: ext-matmul
description: >
  Matrix-multiplication extension primitives — mmul / mmaddm / img2col. Use when the operator
  is a matmul or a convolution (img2col is convolution-only). Loads only this category's
  primitives from references/ on demand. Does NOT generate kernels itself — pair with
  triton-gen; standard Triton stays the structural base.
---

You are the extension-primitive reference for **matrix / convolution** ops. In agent mode the
orchestrator points you here when the current operator is a matmul or convolution; load the
relevant primitive from `references/` (filled in the confidential env) and apply it as a local
replacement on a standard-Triton structural base.

## Scope
mmul / mmaddm / img2col (img2col is convolution-only — do not apply to elementwise ops).

## Field contract
Each `references/*.yaml` is one primitive: `name` / `semantics` / `signature` / `category` /
`example` / `pitfalls` (+ optional `module` / `applies_to`). See
`.claude/skills/extension-guide/references/README.md`.
