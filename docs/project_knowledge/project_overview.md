# Project Overview

OP_autogen is a **Triton Language CPU Emulator** that enables the closed loop: **PyTorch → Triton kernel → Emulator (correctness + perf) → LLM feedback → iterate**.

The goal is to let LLMs get fast, precise correctness feedback when generating Triton kernels — without needing a GPU. Future work includes a cost model for performance estimation before hardware deployment.

## Directory Structure

```
OP_autogen/
├── emulators/
│   ├── common/                <- tl static class, xarray, PointerWrapper, launch_kernel, verify, run_with_feedback
│   ├── add/                   <- basic operators (4-part: kernel/emulate/reference/test)
│   ├── matmul/
│   ├── transpose/
│   ├── reshape/
│   ├── relu/
│   ├── softmax/
│   ├── rmsnorm/
│   ├── addrmsnormgamma/
│   ├── conv1d/
│   ├── conv2d/
│   ├── test/                  <- complex operator test cases
│   │   ├── conv1d/            <- 1D conv (stride + padding)
│   │   ├── attention-relu/    <- scaled dot-product attention + ReLU
│   │   ├── conv2d_resnet/     <- general Conv2d (stride + padding + bias)
│   │   ├── batchnorm2d/
│   │   ├── maxpool2d/
│   │   ├── adaptive_avgpool2d/
│   │   ├── gcn_spmm/          <- graph: sparse-dense matrix multiply
│   │   ├── gcn/               <- GCN integration test
│   │   ├── resnet18/          <- integration test + DEVELOPMENT_LOG.md
│   │   ├── resnet34/          <- integration test [3,4,6,3]
│   │   └── run_all_tests.py
│   └── run_all_tests.py
├── models/                    <- model files and shape registry (not in git)
├── docs/
│   ├── dev_plan/              <- development plans
│   ├── emulator_observations/ <- emulator observations (error, precision, API, patterns)
│   └── project_knowledge/     <- project knowledge (this directory)
├── .claude/commands/
│   └── triton-gen.md          <- triton-gen skill definition
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

### Basic operators
add, matmul, transpose, reshape, relu, softmax, rmsnorm, addrmsnormgamma, conv1d, conv2d

### Integration test cases
- **conv2d_resnet** — general Conv2d (stride + padding + bias)
- **batchnorm2d** — BatchNorm2d (eval mode)
- **maxpool2d** — MaxPool2d (stride + padding)
- **adaptive_avgpool2d** — global average pooling
- **conv1d** — 1D convolution (stride + padding)
- **attention-relu** — scaled dot-product attention + ReLU
- **resnet18** — integration test (5 tests, all PASS)
- **resnet34** — integration test [3,4,6,3] (7 tests, all PASS)
- **gcn_spmm** — graph sparse matrix multiply (5 tests, all PASS)
- **gcn** — GCN integration: SpMM + matmul (3 tests, all PASS)

## Key Entry Points

- `emulators/run_all_tests.py` — run all operator self-tests
- `run_with_feedback()` — top-level LLM feedback interface
- `/triton-gen` skill — Claude Code project-level command, supports NL/PyTorch/ONNX/baseline Triton/fixed shape inputs
