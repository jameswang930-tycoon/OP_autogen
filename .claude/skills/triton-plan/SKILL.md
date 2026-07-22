---
name: triton-plan
description: >
  Pipeline stage 1 (cost-model planning). Use when the user wants to "plan",
  "estimate cost", or do bottleneck analysis (瓶颈分析) for an operator that has
  NO emulators/test/<op>/.plan.json yet. Input may be NL / PyTorch / ONNX /
  baseline triton / a fixed shape. This skill writes a cost_emulator DSL program,
  runs the simulator directly (--verify + --llm --critical-path), and dumps the
  raw_llm output verbatim to .plan.json. It ONLY plans: it does NOT interpret
  raw_llm (that is triton-gen's job) and does NOT generate a kernel. Trigger this
  for any "plan / estimate / bottleneck / 瓶颈分析 / 规划" request even if the
  user does not literally say "plan".
---

You are the cost-model planning adapter. Input: extract the operator name <op> (and any optional input type / feedback) from the user's request.

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

## Step 3: Run the simulator DIRECTLY

```bash
# simulator needs Python 3.10+ (uses str | None); system python3 is 3.7, so use .venv
.venv/bin/python costModel/cost_emulator/simulator.py --verify "<DSL>"
.venv/bin/python costModel/cost_emulator/simulator.py --llm --critical-path "<DSL>"
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
