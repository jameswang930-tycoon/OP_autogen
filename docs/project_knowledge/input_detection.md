# Input Detection & Shapes

Reference for `/triton-plan`: how to detect the input type and extract `op_kind` + `shapes` before calling the cost model.

## Input Type Detection

| Input Type | Detection Rule | Processing Path |
|------------|---------------|-----------------|
| Natural language | Plain text operator description | Direct semantic analysis |
| PyTorch model | `.pt`/`.pth`, `nn.Module`, `torch.nn` | Extract operator semantics + shapes |
| ONNX model | `.onnx`, `onnxruntime`, `onnx.` | Parse the computation graph |
| Baseline Triton kernel | `@triton.jit`, `import triton`, `tl.program_id` | Convert to emulator-compatible form |
| Fixed shape | Model name or `[B,C,H,W]` shape annotation | Query `shapes_registry` |

Multiple input types can coexist; explicit shape takes priority.

## Common PyTorch / ONNX → op_kind mappings

- `F.conv2d` → `conv2d_resnet`, `F.batch_norm` → `batchnorm2d`, `F.relu` → `relu`
- `F.max_pool2d` → `maxpool2d`, `F.adaptive_avg_pool2d` → `adaptive_avgpool2d`
- `F.linear` → `matmul` + `add`, `torch.matmul` → `matmul`
- ONNX: `Conv` → `conv2d_resnet`, `MatMul` → `matmul`, `Gemm` → `matmul` + `add`, `Softmax` → `softmax`

## Fixed shapes

`models/shapes_registry.py` holds fixed shapes for model names (resnet18/34/50,
bert-base, gpt2). For unit tests use small spatial dims (8-32); for integration
tests use real sizes.

## Note on cost-model coverage

`/triton-plan` writes the cost_emulator DSL by hand from the op semantics (it is
not limited to a fixed op list) and runs `cost_emulator/simulator.py` directly.
Any op_kind whose data flow can be expressed in the seven-engine DSL
(`gm_to_ub`/`ub_to_gm`/`vadd`/`gm_to_l1`/`l1_to_l0`/`matrixmul`/`l0_to_gm`) is
covered; if the DSL cannot be written or the simulator call fails, `/triton-plan`
writes a mock stub so `/triton-gen` is not blocked. The cost model stays loosely
coupled (read-only, zero-intrusion).
