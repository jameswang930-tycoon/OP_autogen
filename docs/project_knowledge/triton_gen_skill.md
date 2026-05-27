# triton-gen Skill

The triton-gen skill (`.claude/commands/triton-gen.md`) supports 5 input types:

| Input Type | Detection Rule | Processing Path |
|------------|---------------|-----------------|
| Natural language | Plain text operator description | Step 2a: direct semantic analysis |
| PyTorch model | `.pt`/`.pth`, `nn.Module`, `torch.nn` | Step 2b: extract operator semantics + shapes |
| ONNX model | `.onnx`, `onnxruntime`, `onnx.` | Step 2c: parse computation graph |
| Baseline Triton kernel | `@triton.jit`, `import triton`, `tl.program_id` | Step 2d: convert to emulator-compatible form |
| Fixed shape | Model name or `[B,C,H,W]` shape annotation | Step 2e: query shapes_registry |

Multiple input types can coexist; explicit shape takes priority.

## Key Files

- `.claude/commands/triton-gen.md` — skill definition (~100 lines)
- `models/shapes_registry.py` — fixed shape registry (resnet18/34/50, bert-base, gpt2)
