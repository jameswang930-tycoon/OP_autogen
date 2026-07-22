# Plan-Code Contract (`.plan.json`)

`.plan.json` is the **single handoff** between the cost-model side (`/triton-plan`)
and the emulator side (`/triton-gen` / `/triton-verify` / `/triton-fix` /
`/triton-convert`). It is the only thing the emulator side consumes; it never reaches
into how the cost model plans (DSL construction, shape→byte conversion, tiling,
bottleneck math are all the cost-model side's business).

## Authoritative schema

Written by `/triton-plan` to `emulators/test/<op>/.plan.json`:

**Success** (simulator ran):
```json
{
  "op": "vadd",
  "shapes": {"N": 32768},
  "dtype": "fp16",
  "dsl": "alloc(gm_a,64.0KB) ... vadd(ub_c,ub_a,1.0) ub_to_gm(gm_c,ub_c)",
  "raw_llm": "<simulator --llm output, verbatim — 7 sections>"
}
```

**Failure** (simulator call failed or the DSL could not be written):
```json
{"mock": true, "op": "<op>", "shapes": {...}, "dtype": "<dtype>", "note": "simulator failed"}
```

## Field semantics

| Field | Meaning |
|---|---|
| `op` | operator kind (`matmul` / `vadd` / ...) — drives kernel choice |
| `shapes` | dimension dict (`matmul` → `{M,N,K}`, `vadd` → `{N}`, ...) |
| `dtype` | storage dtype: `fp16` / `fp32` / `bf16` (default `fp32`) |
| `dsl` | the cost_emulator program string handed to the simulator |
| `raw_llm` | simulator `--llm` full stdout, **verbatim** — `/triton-gen` reads it in depth for data flow (which engines), bottleneck (TIME BREAKDOWN + CRITICAL PATH), and tiling (BANDWIDTH UTILIZATION regime / `saturates_at`) |
| `mock` | present only on failure; signals `/triton-gen` to fall back to the default path (cube → matmul GM→L1→L0; vec → vadd GM→UB→Vec) |
| `retrieved_experience` | **optional** — present only when the memory module is integrated; formatted historical-experience text that `/triton-gen` may read as an extra generation reference. Absent ⇒ memory is off / empty, and `/triton-gen` behaves exactly as without memory. |

`/triton-gen` reads **only** `op` / `shapes` / `dtype` / `dsl` / `raw_llm` (+ `mock`,
+ optional `retrieved_experience`). It does **not** parse any structured "plan"
sub-object — the LLM reads `raw_llm` directly. `retrieved_experience` is appended to
an already-written `.plan.json` by the memory module's inject step (see
`docs/project_knowledge/memory_integration.md` §3); it is **not** produced by `/triton-plan`.

## How `raw_llm` is produced

`/triton-plan` writes the DSL by hand from the op semantics, then runs (from repo
root, with `.venv/bin/python` — see `environment_and_running.md`):

```bash
.venv/bin/python costModel/cost_emulator/simulator.py --verify "<dsl>"                  # memory-fit check first
.venv/bin/python costModel/cost_emulator/simulator.py --llm --critical-path "<dsl>"     # -> raw_llm
```

The DSL syntax, the seven-engine model, the size-dependent bandwidth curves, and how
to read each `--llm` section are documented **authoritatively** in the collaborator's
skill: **`costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md`**. That file is
the reference for anything beyond the field list above (engine table, memory
hierarchy, `--verify` capacity semantics, critical-path analysis).

## Historical note (schema drift)

The only existing sample, `emulators/test/vadd_fp16/.plan.json`, predates the
direct-simulator path: it carries extra fields (`supported`, `tile`, and a structured
`plan` object) left over from the now-removed `cost_planner.py` adapter. These extras
are **ignored** by `/triton-gen`. The authoritative schema is the 5-field form above,
and new `.plan.json` files must follow it.
