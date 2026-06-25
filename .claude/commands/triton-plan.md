---
name: triton-plan
description: >
  Thin adapter layer. Parse input → extract op semantics (op_kind + shapes) →
  call the external cost model (costModel/cost_planner.py) → dump plan code to
  .plan.json. This skill ONLY adapts/calls/dumps; it does NOT interpret plan
  fields or decide how the plan guides generation (that is triton-gen's job).
  The cost model is an external collaborator's work-in-progress (currently only
  vadd fully mapped); ops it does not support get a mock plan stub and are not
  blocked. Trigger when the user wants to "plan", "estimate cost", or as the
  step before /triton-gen when a cost plan is desired.
---

You are the cost-model planning adapter. User input: $ARGUMENTS

This skill is intentionally thin: **detect input → extract semantics → call →
dump → stub**. Never interpret the plan or encode generation guidance here.

## Step 1: Detect Input Type

| Input Type | Detection Rule |
|------------|---------------|
| Natural language | Plain text operator description |
| PyTorch model | `.pt`/`.pth`, `nn.Module`, `torch.nn` |
| ONNX model | `.onnx`, `onnxruntime`, `onnx.` |
| Baseline Triton kernel | `@triton.jit`, `import triton`, `tl.program_id` |
| Fixed shape | Model name or `[B,C,H,W]` shape annotation |

Multiple types can coexist; explicit shape takes priority. Detection details +
shapes_registry: `docs/project_knowledge/input_detection.md`.

## Step 2: Extract Semantics → op_kind + shapes

Map the parsed input to an `op_kind` (e.g. matmul / vadd / conv2d / softmax ...)
and a `shapes` dict. Op→DSL mapping notes and shapes_registry live in
`input_detection.md`.

## Step 3: Call the External Cost Model + Dump Plan Code

```python
import sys, json, os
sys.path.insert(0, "costModel")
from cost_planner import plan

pc = plan(op_kind, shapes)   # op_kind currently supported: matmul, vadd; others stubbed
# COST_SIM_PYTHON must point to a Python 3.10+ interpreter (cost_emulator needs 3.10+)
```

- `pc["supported"] == True`: dump the **full** result (incl. `raw_llm`) verbatim.
  Do **NOT** summarize, interpret, or transform the fields — triton-gen reads
  them directly.
- `pc["supported"] == False` or the call raises: write a **mock plan** stub and do
  NOT block downstream:
  ```json
  {"mock": true, "op": "<op_kind>", "shapes": {...},
   "note": "cost model does not support this op; triton-gen uses the default path"}
  ```

This is a read-only call to the external cost model — never modify
`costModel/cost_emulator/` (it is the collaborator's repo, loosely coupled).

## Step 4: Write File + Hand Off

Create `emulators/test/<op>/` if needed, write the plan to
`emulators/test/<op>/.plan.json`, then tell the user: run `/triton-gen <op>`
to generate the emulator kernel.

References: `docs/project_knowledge/input_detection.md` (input detection +
shapes), `costModel/cost_planner.py` (plan() contract source).
