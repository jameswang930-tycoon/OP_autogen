---
name: extension-guide
description: >
  Hardware-extension primitive cheatsheet, indexed by bottleneck category. Use when the
  triton-gen step or the user asks "which extension primitive / extension 速查 / 原语查询 /
  what primitive solves this bottleneck". The body holds a SHORT index (primitive name plus
  the bottleneck category it solves); detailed per-primitive entries live in references and
  are read on demand (progressive disclosure). Categories come from control/vocabulary.yaml.
  Does NOT generate kernels and does NOT analyze feedback.
---

You are the extension-primitive reference. The whole loop is bottleneck-driven, so
primitives are organized by the bottleneck category they solve (ids from
`control/vocabulary.yaml`). The body carries only a short index; load a detailed entry from
the references directory only when its category is hit.

## Short index (primitive to bottleneck category)

Real primitive entries are filled in the confidential environment (see
`.claude/skills/extension-guide/references/README.md`). Each YAML file under
`.claude/skills/extension-guide/references/` is one primitive; the index is the union of
those entries. Until they are filled, the only worked mapping is the sample:

| primitive (sample) | category |
|---|---|
| `sample_async_copy_template` | memory_underfilled |

Replace or augment with real primitives in the confidential env. Every entry's `category`
must be a vocabulary id — `control/check_extension_cheatsheet.py` enforces this.

## How to use

1. Read the adapter Verdict `bottleneck` category.
2. Find primitives whose `category` matches, and read the matching reference file.
3. Hand the primitive's signature, example, and pitfalls to triton-gen.

## Per-entry format

See the worked sample at `.claude/skills/extension-guide/references/sample_entry.yaml`:
fields are name / semantics (one line) / signature / category (vocabulary id) / example /
pitfalls.

## Do NOT

- Do not invent primitives or categories — only vocabulary ids are valid categories.
- Do not load the whole references directory into context; read on demand by category.
- Do not fabricate primitive signatures — if a real signature is unknown, leave it as a TODO.
