# OP_autogen

Triton Language CPU Emulator -- 在 CPU 上以逐算子粒度模拟 Triton kernel 的执行，用于算子逻辑验证和自测。

> 核心模拟层在 [emulators/common/__init__.py](emulators/common/__init__.py)，阅读该文件的模块 docstring 可以快速理解整个仓库的设计思路。

## 项目结构

```
OP_autogen/
├── emulators/
│   ├── common/                # 公共基础设施（Triton API 打桩 + 验证工具）
│   │   └── __init__.py
│   └── test/                  # 所有算子（基础 + 集成测试）
│       ├── add/               # 逐元素加法
│       ├── matmul/            # 矩阵乘法 (2D tiled)
│       ├── transpose/         # 矩阵转置
│       ├── reshape/           # 张量 reshape
│       ├── relu/              # ReLU / Leaky ReLU
│       ├── softmax/           # 行级数值稳定 softmax
│       ├── rmsnorm/           # RMS Layer Normalization
│       ├── addrmsnormgamma/   # 融合：Add + RMSNorm + Gamma
│       ├── conv1d/            # 1D 卷积（stride + padding）
│       ├── conv2d/            # 2D 卷积（基础版）
│       ├── conv2d_resnet/     # 通用 Conv2d：stride + padding + bias
│       ├── batchnorm2d/       # BatchNorm2d (eval mode)
│       ├── maxpool2d/         # MaxPool2d：stride + padding
│       ├── adaptive_avgpool2d/# 全局自适应平均池化
│       ├── attention-relu/    # 缩放点积注意力 + ReLU
│       ├── gcn_spmm/          # 图算子：稀疏矩阵-稠密矩阵乘法
│       ├── gcn/               # GCN 集成测试
│       ├── resnet18/          # ResNet18 集成测试
│       ├── resnet34/          # ResNet34 集成测试 [3,4,6,3]
│       └── run_all_tests.py   # 全量自测入口
├── models/                    # 模型文件和 shape 注册表（不提交 git）
├── docs/
│   ├── dev_plan/              # 开发计划
│   ├── emulator_observations/ # emulator 观察（误差、精度、API、实现模式）
│   └── project_knowledge/     # 项目知识索引
├── .claude/
│   └── commands/
│       └── triton-gen.md      # triton-gen skill：自动算子生成/调试
├── CLAUDE.md                  # 编码规范
└── README.md
```

## 核心设计

[emulators/common/__init__.py](emulators/common/__init__.py) 提供 Triton Language 的 CPU 模拟层：

| 组件 | 说明 |
|---|---|
| `tl` | Triton API 打桩类，接口签名与真实 Triton 一致（`load`, `store`, `dot`, `sum`, `max`, `exp`, 原子操作等） |
| `xarray` | 带内存层级追踪的 ndarray（`in_fast_mem` 标记 SRAM/DRAM 状态） |
| `PointerWrapper` / `OffsetPointer` | 模拟 Triton 指针算术，支持 `ptr + offset` 语法 |
| `launch_kernel_1d/2d/3d` | kernel 启动器，模拟 1D/2D/3D grid 调度 |
| `verify()` | 输出 vs reference 数值对比，tolerance 可配 |
| `TraceLogger` | tl.\* 调用追踪，记录每个 API 的输入输出摘要，用于 debug |
| `EmulatorError` | 统一错误类型，含 API 名 + 详细信息 |
| `AggregatedEmulatorError` | 跨多个 program 聚合错误（用于 OOB 诊断） |
| `run_with_feedback()` | 包装 emulator + reference 执行，自动生成 LLM 可读的修复反馈 |

### 算子模块约定

每个算子目录下的 `__init__.py` 遵循统一模式：

1. **Triton-style Kernel** -- 纯粹的 kernel 函数，只使用 `tl.*` API
2. **`emulate_xxx()`** -- 封装函数，扁平化输入、启动 grid、reshape 输出
3. **`reference_xxx()`** -- NumPy / PyTorch 参考实现，用于对比验证
4. **`test()`** -- 自测函数，覆盖正常路径、边界条件、错误路径

## 运行测试

```bash
# 运行全部算子自测
python emulators/test/run_all_tests.py

# 单独运行某个算子自测（需在 emulators 目录下）
cd emulators && python -c "from test.add import test; test()"
cd emulators && python -c "from test.conv2d_resnet import test; test()"

# ResNet18 集成测试
cd emulators && python -c "from test.resnet18 import test; test()"
```

## 已支持的算子

所有算子位于 `emulators/test/`，每个遵循 4-part 结构（kernel/emulate/reference/test）。

### 基础算子

| 算子 | 说明 | Grid |
|---|---|---|
| `add` | 逐元素加法 `out = x + y` | 1D |
| `matmul` | 2D tiled 矩阵乘法 `C = A @ B` | 2D |
| `transpose` | 2D 矩阵转置 `out = x^T` | 2D |
| `reshape` | 张量形状变换（零拷贝） | 1D |
| `relu` | ReLU / Leaky ReLU 激活 | 1D |
| `softmax` | 行级数值稳定 softmax | 1D |
| `rmsnorm` | RMS Layer Normalization | 1D |
| `addrmsnormgamma` | 融合 Add + RMSNorm + Gamma | 1D |
| `attention-relu` | 缩放点积注意力 + ReLU | 2D |
| `conv1d` | 1D 卷积 | 1D |
| `conv2d` | 简单 2D 卷积（无 stride/padding） | 1D |

### 集成测试用例

| 算子 | 说明 | Grid | 设计文档 |
|---|---|---|---|
| `conv2d_resnet` | 通用 Conv2d：stride + padding + bias | 1D | [dev_plan](docs/dev_plan/resnet18_conv_dev_plan.md) |
| `batchnorm2d` | BatchNorm2d (eval mode) | 1D | - |
| `maxpool2d` | MaxPool2d：stride + padding | 1D | - |
| `adaptive_avgpool2d` | 全局自适应平均池化 `(N,C,H,W) -> (N,C,1,1)` | 1D | - |
| `resnet18` | 集成测试：Stem + BasicBlock + chain | - | [开发日志](emulators/test/resnet18/DEVELOPMENT_LOG.md) |
| `resnet34` | 集成测试 [3,4,6,3]：7 个测试全部 PASS | - | - |
| `gcn_spmm` | 图稀疏矩阵乘法（CSR 格式） | 1D | - |
| `gcn` | GCN 集成：SpMM + matmul | - | - |

### ResNet18 验证结果

| 测试 | 描述 | max_abs | max_rel |
|------|------|---------|---------|
| Stem | conv1(7x7,s2,p3)+bn+relu+maxpool(3x3,s2,p1) | 4.77e-07 | 1.14e-04 |
| Block (no down) | 64->64, 2x conv3x3 + residual | 2.38e-07 | 3.28e-05 |
| Block (downsample) | 64->128, conv3x3 s2 + 1x1 proj shortcut | 8.34e-07 | 1.82e-05 |
| Two blocks | layer1.0 + layer1.1 chained | 4.77e-07 | 1.65e-04 |
| Full chain | stem -> block -> avgpool | 1.19e-07 | 2.51e-06 |

详见 [emulators/test/resnet18/DEVELOPMENT_LOG.md](emulators/test/resnet18/DEVELOPMENT_LOG.md)

## triton-gen Skill

使用 `/triton-gen` 指令可以自动生成或修复 Triton kernel，支持 5 种输入类型：

- **自然语言** — 算子描述（如 "layernorm" 或 "y = x * sigmoid(x)"）
- **PyTorch 模型** — `nn.Module` 或 `.pt`/`.pth` 文件
- **ONNX 模型** — `.onnx` 文件，解析计算图提取算子语义
- **基线 Triton kernel** — `@triton.jit` 装饰的已有 kernel
- **固定 Shape** — 模型名（如 "resnet18"）或 `[B,C,H,W]` 标注

Skill 文件：[.claude/commands/triton-gen.md](.claude/commands/triton-gen.md)

## 重要设计约束

1. **`tl.sum` / `tl.max` / `tl.min` 使用 `keepdims=True`** -- 输出形状保留 reduced dimension，数值等价，且与 `tl.store` 的 `(1,)` offsets 天然对齐
2. **指针传递两种约定**：
   - Pointer style：kernel 用 `ptr + offset` → 调用前用 `wrap_ptr()` 包装
   - Emulator style：kernel 用 `tl.load(base_array, offsets)` → 直接传递 numpy 数组
3. **OOB 访问必须用 mask 守护**：`mask = offsets < n_elements`
4. **Reduction axis 始终为 0** -- block 内数据是 1D 向量

## 开发记录

- 2026-05-21：ResNet18 卷积层开发完成，验证了 emulator 对复杂算子的支撑能力 [详细日志](emulators/test/resnet18/DEVELOPMENT_LOG.md)
- 2026-05-27：ResNet34 集成测试 7/7 PASS，GCN 图算子（SpMM + matmul）3/3 PASS
- 2026-05-27：triton-gen skill 精简至 ~100 行，支持 5 种输入类型
- 2026-05-27：项目知识迁移至 `docs/project_knowledge/`，emulator 观察记录至 `docs/emulator_observations/`