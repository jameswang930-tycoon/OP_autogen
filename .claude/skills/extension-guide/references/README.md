# extension-guide / references

One YAML file per extension primitive. The confidential-env GLM 4.7 fills the **real**
primitives here; the public branch ships only the worked sample (`sample_entry.yaml`).

## Required fields (every entry)

| field      | meaning                                                          |
|------------|------------------------------------------------------------------|
| name       | primitive name                                                   |
| semantics  | one-line semantics                                              |
| signature  | call signature (real signature; TODO if genuinely unknown). If the extension ships a `.pyi` stub / header, transcribe it verbatim |
| category   | bottleneck category id — MUST be in `control/vocabulary.yaml`   |
| example    | a COMPLETE, compilable minimal kernel (imports + `@triton.jit` + full fn), not a snippet |
| pitfalls   | list of common mistakes                                         |

## Optional fields (GLM52 — module attribution + applicable scene)

These are optional; the validator (`check_extension_cheatsheet.py`) ignores them, so existing
entries keep validating unchanged. Filling them lets the orchestrator present primitives better
and avoid hallucination / mis-use (GLM52 guide §2):

| field       | meaning                                                                                        |
|-------------|------------------------------------------------------------------------------------------------|
| module      | module the primitive lives in. The gen-prompt index renders the fully-qualified `module.name`, so the model does not guess the module (root cause of `tlext1.add`-style hallucinations). Absent → bare `name`. |
| applies_to  | list of operator kinds this primitive applies to (e.g. `[conv]` for img2col). The orchestrator's scene-based retrieval only surfaces a primitive for matching ops, so a conv-only primitive no longer pollutes element-wise candidates. Absent/empty → treated as general (never hidden on account of missing annotation; the index falls back to full if a scene query yields nothing). |

Evidence priority (GLM52 §2 lesson — "having data ≠ correct data"): the confidential
`api_inventory.txt` is a name index whose module attribution is unreliable; prefer
real-kernel usage > manual > inventory when filling `module`. Wrong examples actively teach the
model bad patterns, so leave `module`/`applies_to` unset rather than guess.

## Rules

- `category` must be a vocabulary id. `control/check_extension_cheatsheet.py` validates
  every entry and fails on an unknown category.
- Do not fabricate primitive semantics or signatures. If something is unknown in the
  confidential env, mark it `TODO` rather than inventing it.
- Confidential primitive names / signatures must not flow back to the public branch.

## Validation

```bash
.venv/bin/python -m control.check_extension_cheatsheet
```
