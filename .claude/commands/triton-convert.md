---
name: triton-convert
description: >
  Convert a verified emulator kernel (emulators/test/<op>/__init__.py) into a
  real deployable Triton kernel (triton_real.py). Applies 5 mechanical rewrites
  — the kernel's compute logic stays identical — then runs an NPU constraint
  self-check. Trigger when the user says "convert to real triton", "上板", or
  "generate triton_real".
---

You are an emulator→real-Triton conversion expert. Input: $ARGUMENTS (<op> name).

Output: `emulators/test/<op>/triton_real.py`.

## Before converting: diff a closest example

Read one ground-truth example to align the before/after pattern:
- matmul-class ops → `emulators/test/matmul/triton_real.py`
- conv-class ops → `emulators/test/resnet18/triton_real.py`
- also available: `resnet34`, `mobilenetv3_small`.

## The 5 mechanical rewrites (kernel compute logic stays identical)

1. **import**: `from common import ...` → `import triton; import triton.language as tl` (+ `import torch`)
2. **decorator**: `def kernel` → `@triton.jit def kernel`
3. **load**: `tl.load(ptr, offsets, mask)` → `tl.load(ptr + offsets, mask, other=0.0)` (drop `.ravel()` / `.reshape()`)
4. **store**: `tl.store(ptr, offsets, vals, mask)` → `tl.store(ptr + offsets, vals, mask)`
5. **launch**: `launch_kernel_Nd(kernel, *flat, grid=)` → `kernel[grid](*tensors, BLOCK_*=...)` (numpy → torch tensors)

offset/mask/dot/accumulate/arange/program_id/cdiv logic is all unchanged.

## Missing kernels to add

Some ops done in numpy at the emulator level need real kernels: residual `add`
(element-wise), `linear` (per-output accumulate). See the "需要新增的 kernel"
table in `docs/project_knowledge/emulator_to_triton_conversion.md`.

## NPU constraint self-check (after rewrite)

- grid_size ≤ 65535 (Triton-Ascend coreDim limit). If exceeded, warn and suggest
  `TRITON_ALL_BLOCKS_PARALLEL=1` (correctness only) or two-level tiling.
- alignment: 32B (vector ops) / 512B (cube+vector fused).
- on-chip UB ≤ 192KB per aicore.

Two-level tiling rewrite is OUT OF SCOPE here — this skill only does the
mechanical rewrite + warn. Full conversion rules + NPU constraints:
`docs/project_knowledge/emulator_to_triton_conversion.md`.

## Verify (if torch + triton are available)

Run the `triton_real.py` test; otherwise do a static check (rewrite shape +
constraints only).
