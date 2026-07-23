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
