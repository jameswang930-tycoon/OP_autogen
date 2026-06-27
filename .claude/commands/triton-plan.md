---
name: triton-plan
description: >
  Write a cost_emulator DSL program from op semantics, run the simulator DIRECTLY
  (--verify + --llm --critical-path, no cost_planner wrapper), and dump raw_llm as
  plan code to .plan.json. Same call style as the collaborator's bottleneck-analysis
  skill. Only writes DSL + runs simulator + dumps raw_llm; does NOT interpret the
  output (that is triton-gen's job). Trigger when the user wants to "plan",
  "estimate cost", or before /triton-gen.
---

You are the cost-model planning adapter. User input: $ARGUMENTS

This skill writes a DSL program, runs the collaborator's simulator directly, and
dumps the raw output. It does NOT interpret the result.

## Step 1: Detect Input + Extract Semantics

Detect input type (NL / PyTorch / ONNX / baseline triton / fixed shape) and extract
`op_kind` + `shapes` + `dtype` (fp16/fp32/bf16, default fp32). Details:
`docs/project_knowledge/input_detection.md`.

## Step 2: Write the DSL program

Write a cost_emulator DSL program describing this op's data flow. DSL is the
collaborator's input format — see `costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md`
for the full engine table + syntax. Key points:

- **Engines**: `gm_to_ub` / `ub_to_gm` / `vadd`(VecUnit) / `gm_to_l1` / `l1_to_l0` / `matrixmul`(CubeUnit) / `l0_to_gm`
- **`alloc(name, size)`**: buffer sizes in KB, from `shapes × dtype bytes` (fp16/bf16=2B, fp32=4B). Units: B/KB/MB/GB.
- **vec ops** (elementwise/vadd): GM→UB load → VecUnit → UB→GM store
- **cube ops** (matmul): GM→L1 → L1→L0 → CubeUnit → L0→GM
- Example (vadd N=4096 fp16): `alloc(gm_a,8KB) alloc(gm_c,8KB) alloc(ub_a,16KB) alloc(ub_c,16KB) gm_to_ub(ub_a,gm_a) vadd(ub_c,ub_a,1.0) ub_to_gm(gm_c,ub_c)`

## Step 3: Run the simulator DIRECTLY (no cost_planner wrapper)

```bash
# COST_SIM_PYTHON must be Python 3.10+ (simulator uses str | None)
COST_SIM_PYTHON=.venv/bin/python .venv/bin/python costModel/cost_emulator/simulator.py --verify "<DSL>"
COST_SIM_PYTHON=.venv/bin/python .venv/bin/python costModel/cost_emulator/simulator.py --llm --critical-path "<DSL>"
```

`--llm` stdout = `raw_llm` (7 sections: execution summary / time breakdown / per-op /
engine util / bandwidth util / parallelism / critical path). Read-only call — never
modify `costModel/cost_emulator/` (collaborator's repo).

## Step 4: Dump plan code

Write `emulators/test/<op>/.plan.json` = `{op, shapes, dtype, dsl, raw_llm}` (raw_llm
verbatim from `--llm`; do NOT parse/summarize — triton-gen reads it directly). If the
simulator failed, dump a mock stub: `{mock:true, op, shapes, dtype, note:"simulator failed"}`.

References: `costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md` (DSL + engine
table), `docs/project_knowledge/input_detection.md` (input detection).
