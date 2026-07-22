# Project Overview

OP_autogen is a **Triton Language CPU Emulator** that enables the closed loop: **PyTorch → Triton kernel → Emulator (correctness + perf) → LLM feedback → iterate**.

The goal is to let LLMs get fast, precise correctness feedback when generating Triton kernels — without needing a GPU. A cost model (`costModel/cost_emulator`, vendored as a subtree) is integrated via `/triton-plan`, which runs the simulator directly to estimate cost/performance before hardware deployment.

## Directory Structure

```
OP_autogen/
├── emulators/
│   ├── common/                <- tl static class, xarray, PointerWrapper, launch_kernel, verify, run_with_feedback
│   └── test/                  <- all operators (basic + integration)
│       ├── add/               <- element-wise add
│       ├── matmul/            <- 2D tiled matrix multiplication
│       ├── transpose/         <- 2D matrix transpose
│       ├── reshape/           <- tensor reshape (metadata only)
│       ├── relu/              <- ReLU / Leaky ReLU
│       ├── softmax/           <- row-wise numerically stable softmax
│       ├── rmsnorm/           <- Root Mean Square Layer Normalization
│       ├── addrmsnormgamma/   <- fused: Add + RMSNorm + Gamma
│       ├── conv1d/            <- 1D convolution (stride + padding)
│       ├── conv2d/            <- 2D convolution (basic, no stride/padding)
│       ├── conv2d_resnet/     <- general Conv2d (stride + padding + bias)
│       ├── batchnorm2d/       <- BatchNorm2d (eval mode)
│       ├── maxpool2d/         <- MaxPool2d (stride + padding)
│       ├── adaptive_avgpool2d/<- global adaptive average pooling
│       ├── attention-relu/    <- scaled dot-product attention + ReLU
│       ├── gcn_spmm/          <- graph: sparse-dense matrix multiply
│       ├── gcn/               <- GCN integration test
│       ├── resnet18/          <- integration test + DEVELOPMENT_LOG.md
│       ├── resnet34/          <- integration test [3,4,6,3]
│       └── run_all_tests.py   <- run all operator self-tests
├── models/                    <- model files and shape registry (not in git)
├── docs/
│   ├── dev_plan/              <- development plans
│   ├── emulator_observations/ <- emulator observations (error, precision, API, patterns)
│   └── project_knowledge/     <- project knowledge (this directory)
├── .claude/commands/          <- 5 skills: triton-plan / triton-gen / triton-verify / triton-fix / triton-convert
└── README.md
```

## Architecture (4 layers)

1. **tl static class** (`emulators/common/__init__.py`) — numpy-stubbed Triton Language public API
2. **xarray** (numpy subclass) — tracks whether data resides in SRAM
3. **PointerWrapper / OffsetPointer** — simulates Triton pointer arithmetic and gather/scatter
4. **launch_kernel_1d/2d/3d** — simulates SPMD grid execution, serial per-program kernel invocation

## Operator Module Standard Structure (4-part)

Each operator directory contains:
1. `xxx_kernel()` — pure `tl.*` API Triton-style kernel
2. `emulate_xxx()` — wrapper: validate input → flatten → launch_kernel → reshape
3. `reference_xxx()` — numpy/torch ground truth implementation
4. `test()` — self-test: basic functionality + edge cases

## Implemented Operators

All operators are in `emulators/test/`. Each follows the 4-part structure (kernel/emulate/reference/test).

### Basic operators
add, matmul, transpose, reshape, relu, softmax, rmsnorm, addrmsnormgamma, conv1d, conv2d

### Integration test cases
- **conv2d_resnet** — general Conv2d (stride + padding + bias)
- **batchnorm2d** — BatchNorm2d (eval mode)
- **maxpool2d** — MaxPool2d (stride + padding)
- **adaptive_avgpool2d** — global average pooling
- **attention-relu** — scaled dot-product attention + ReLU
- **resnet18** — integration test (5 tests, all PASS)
- **resnet34** — integration test [3,4,6,3] (7 tests, all PASS)
- **gcn_spmm** — graph sparse matrix multiply (5 tests, all PASS)
- **gcn** — GCN integration: SpMM + matmul (3 tests, all PASS)

## Key Entry Points

- `emulators/test/run_all_tests.py` — run all operator self-tests
- `run_with_feedback()` — top-level LLM feedback interface

## Workflow (5 single-responsibility skills)

The kernel-generation pipeline is split into 5 skills, chained by **files on
disk** (no orchestrator — invoke each manually):

| Skill | Responsibility | Reads | Writes |
|-------|---------------|-------|--------|
| `/triton-plan` | input → DSL → run simulator directly | user input | `emulators/test/<op>/.plan.json` |
| `/triton-gen` | plan → emulator kernel (+ inline verify) | `.plan.json` | `emulators/test/<op>/__init__.py` |
| `/triton-verify` | read-only correctness check | `__init__.py` | (terminal only) |
| `/triton-fix` | repair loop (max 5 rounds) | `__init__.py` | `__init__.py` |
| `/triton-convert` | emulator → real Triton | `__init__.py` | `emulators/test/<op>/triton_real.py` |

Each skill's `.md` lists its own **References** (the docs it owns). Doc-to-skill
ownership is indexed in `CLAUDE.md` → Project Knowledge.

## Running & external interfaces

- **How to run anything** (which Python, simulator vs emulator commands): see
  `environment_and_running.md`. TL;DR — always use `.venv/bin/python` from the repo
  root; the system `python3` (3.7, no torch) cannot run this project.
- **The plan→gen handoff** (`.plan.json` schema, field semantics, the `raw_llm`
  contract): see `plan_code_contract.md`.
- **Cost-model deep reference** (DSL syntax, the seven-engine model, bandwidth
  curves, how to read `--llm` output): the collaborator's
  `costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md` is authoritative.
  `costModel/cost_emulator/` is a vendored subtree — read-only, never modified.
