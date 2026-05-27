# Project Overview

OP_autogen 是一个 **Triton Language CPU Emulator**，目标是构建 **PyTorch → Triton kernel → Emulator (正确性+性能) → LLM 反馈 → 迭代** 的自动算子生成闭环。

让 LLM 在生成 Triton kernel 时无需 GPU 即可获得快速、精准的正确性反馈，实现高效迭代到正确版本。后续计划添加 cost model 预估性能时延，在上板前就能预测性能瓶颈。

## 目录结构

```
OP_autogen/
├── emulators/
│   ├── common/                ← tl 静态类、xarray、PointerWrapper、launch_kernel、verify、run_with_feedback 等
│   ├── add/                   ← 基础算子（四件套：kernel/emulate/reference/test）
│   ├── matmul/
│   ├── transpose/
│   ├── reshape/
│   ├── relu/
│   ├── softmax/
│   ├── rmsnorm/
│   ├── addrmsnormgamma/
│   ├── attention-relu/
│   ├── conv1d/
│   ├── conv2d/
│   ├── test/                  ← 新开发的复杂算子测试用例
│   │   ├── conv2d_resnet/     ← 通用 Conv2d（stride + padding + bias）
│   │   ├── batchnorm2d/
│   │   ├── maxpool2d/
│   │   ├── adaptive_avgpool2d/
│   │   ├── gcn_spmm/          ← 图算子：稀疏矩阵-稠密矩阵乘法
│   │   ├── gcn/               ← GCN 集成测试
│   │   ├── resnet18/          ← 集成测试 + DEVELOPMENT_LOG.md
│   │   ├── resnet34/          ← 集成测试 [3,4,6,3]
│   │   └── run_all_tests.py
│   └── run_all_tests.py
├── models/
│   ├── shapes_registry.py     ← 固定 shape 注册表
│   ├── gcn.py                 ← GCN PyTorch 源码
│   ├── resnet18-v1-7.onnx     ← ONNX 模型（不提交 git）
│   └── resnet34-v1-7.onnx     ← ONNX 模型（不提交 git）
├── docs/
│   ├── dev_plan/              ← 开发计划
│   ├── emulator_observations/ ← emulator 观察（误差、精度、API、实现模式）
│   └── project_knowledge/     ← 项目知识（本目录）
├── costModel/                 ← NPU 代价模型
├── .claude/commands/
│   └── triton-gen.md          ← triton-gen skill 定义
└── README.md
```

## 架构（4 层）

1. **tl 静态类**（`emulators/common/__init__.py`）— 用 numpy 打桩 Triton Language 全部公开 API
2. **xarray**（numpy 子类）— 追踪数据是否在 SRAM
3. **PointerWrapper / OffsetPointer** — 模拟 Triton 指针算术和 gather/scatter
4. **launch_kernel_1d/2d/3d** — 模拟 SPMD grid 执行，逐 program 串行调用 kernel

## 算子模块标准结构（四件套）

每个算子目录包含：
1. `xxx_kernel()` — 纯 `tl.*` API 的 Triton 风格 kernel
2. `emulate_xxx()` — 封装：输入验证 → flatten → launch_kernel → reshape
3. `reference_xxx()` — numpy/torch 的 ground truth 实现
4. `test()` — 自测：基本功能 + 边界条件

## 已实现算子

### 基础算子
add、matmul、transpose、reshape、relu、softmax、rmsnorm、addrmsnormgamma、attention-relu、conv1d、conv2d

### 测试用例
- **conv2d_resnet** — 通用 Conv2d（stride + padding + bias）
- **batchnorm2d** — BatchNorm2d (eval mode)
- **maxpool2d** — MaxPool2d（stride + padding）
- **adaptive_avgpool2d** — 全局平均池化
- **resnet18** — 集成测试（5 个测试全部 PASS）
- **resnet34** — 集成测试 [3,4,6,3]（7 个测试全部 PASS）
- **gcn_spmm** — 图稀疏矩阵乘法（5 个测试全部 PASS）
- **gcn** — GCN 集成测试：SpMM + matmul（3 个测试全部 PASS）

## 关键入口

- `emulators/run_all_tests.py` — 运行所有算子自测
- `run_with_feedback()` — LLM 反馈的顶层接口
- `/triton-gen` skill — Claude Code 项目级命令，支持 NL/PyTorch/ONNX/基线Triton/固定shape 五种输入
