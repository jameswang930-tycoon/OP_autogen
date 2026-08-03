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
├── perf_test/                 # NPU 性能微基准
│   └── 910B3/                 # Ascend 910B3 带宽/算力测试
│       ├── vecadd/            # vec core 五通路带宽 + 算力 (bench + plot)
│       └── matmul/            # cube core matmul 算力 (bench + plot)
├── docs/
│   ├── dev_plan/              # 开发计划
│   ├── emulator_observations/ # emulator 观察（误差、精度、API、实现模式）
│   └── project_knowledge/     # 项目知识索引
├── .claude/
│   └── commands/              # 5 个职责单一的 skill（见下「Skill 工作流」）
│       ├── triton-plan.md     # 输入 → cost model → plan code
│       ├── triton-gen.md      # plan → emulator kernel
│       ├── triton-verify.md   # 只读校验
│       ├── triton-fix.md      # 修复循环
│       └── triton-convert.md  # emulator → 真实 triton
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
| `mul` | 逐元素乘法 `out = x * y` | 1D |
| `matmul` | 2D tiled 矩阵乘法 `C = A @ B` | 2D |
| `transpose` | 2D 矩阵转置 `out = x^T` | 2D |
| `reshape` | 张量形状变换（零拷贝） | 1D |
| `relu` | ReLU / Leaky ReLU 激活 | 1D |
| `hardsigmoid` | Hardsigmoid `min(max(x+3,0),6)/6` | 1D |
| `hardswish` | Hardswish `x * min(max(x+3,0),6)/6` | 1D |
| `softmax` | 行级数值稳定 softmax | 1D |
| `rmsnorm` | RMS Layer Normalization | 1D |
| `addrmsnormgamma` | 融合 Add + RMSNorm + Gamma | 1D |
| `attention-relu` | 缩放点积注意力 + ReLU | 2D |
| `conv1d` | 1D 卷积 | 1D |
| `conv2d` | 简单 2D 卷积（无 stride/padding） | 1D |
| `conv2d_depthwise` | 深度可分离 2D 卷积（stride + padding + bias） | 1D |

### 集成测试用例

| 算子 | 说明 | Grid | 设计文档 |
|---|---|---|---|
| `conv2d_resnet` | 通用 Conv2d：stride + padding + bias | 1D | [dev_plan](docs/dev_plan/resnet18_conv_dev_plan.md) |
| `batchnorm2d` | BatchNorm2d (eval mode) | 1D | - |
| `maxpool2d` | MaxPool2d：stride + padding | 1D | - |
| `adaptive_avgpool2d` | 全局自适应平均池化 `(N,C,H,W) -> (N,C,1,1)` | 1D | - |
| `resnet18` | 集成测试：Stem + BasicBlock + chain | - | [开发日志](emulators/test/resnet18/DEVELOPMENT_LOG.md) |
| `resnet34` | 集成测试 [3,4,6,3]：7 个测试全部 PASS | - | - |
| `mobilenetv3_small` | 集成测试：完整 MobileNetV3-Small `[1,3,224,224] -> [1,1000]` | - | - |
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

## Skill 工作流

算子生成流水线拆成 5 个职责单一的 skill，靠文件传递产物（无 orchestrator，依次手动调用）：

| Skill | 职责 | 读 | 写 |
|-------|------|----|----|
| `/triton-plan` | 输入识别 + 语义提取 + 调外部 cost model | 用户输入（NL / PyTorch / ONNX / 基线 triton / shape） | `emulators/test/<op>/.plan.json` |
| `/triton-gen` | 读 plan → 生成 emulator kernel（+ 内联校验） | `.plan.json` | `emulators/test/<op>/__init__.py` |
| `/triton-verify` | 只读校验 | `__init__.py` | （仅终端输出） |
| `/triton-fix` | 修复循环（max 5 轮） | `__init__.py` | `__init__.py` |
| `/triton-convert` | emulator → 真实 Triton | `__init__.py` | `emulators/test/<op>/triton_real.py` |

Skill 文件在 [.claude/commands/](.claude/commands/)；各 skill 的文档归属见 [CLAUDE.md](CLAUDE.md) 的 Project Knowledge 表。

## 重要设计约束

1. **`tl.sum` / `tl.max` / `tl.min` 返回标量** -- 与真实 Triton 行为一致（`keepdims=False`），单 program 输出的累加器用 `0.0`，`tl.store` 支持标量 offset
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
- 2026-06-01：emulator 拉齐真实 Triton 行为（`keepdims=False`、标量累加器、标量 store），生成的 kernel 无需 NPU 编码适配
- 2026-06-25：triton-gen 拆分为 5 个职责单一 skill（plan/gen/verify/fix/convert），docs 按 skill 归属重组
- 2026-07-20：5 个 slash command 迁移为 Agent Skill（`.claude/skills/`，description 文件态前置条件治理触发），删除 `.claude/commands/`，CLAUDE.md 瘦身；验证链全绿（vadd_fp16 verify PASS + softmax 真触发实测）
- 2026-07-21：新增 `requirements.txt`（numpy + networkx）固化 emulator 依赖；`.venv` 不进 git，换机器用 `uv venv --python 3.13 && uv pip install -r requirements.txt` 重建
- 2026-07-30：GLM52 框架优化（P1–P5）提交——prompt 规则/数据分离、修 memory 两 bug（fingerprint 瓶颈错位 + retrieved_ids 硬编码）、extension 索引加模块归属+按场景检索、memory 未接线报 warning、文档卫生；冻结接缝 ext_distill/remote_dsl 零改动，pytest 191 passed；新增 `HANDOFF_GLM52.md`（保密环境交接/适配参照）
- 2026-08-01：orchestrator CLI `main()` 接通 memory——`store`/`log` 路径与 `output_dir` 对齐（`<out>/memory/`，store/log 同目录自动管 `best_cycles.json`），修复 `store` 默认 None 导致 retrieve/record 全程 no-op、`has_exp` 恒 False；真跑验证 round2 `has_exp=True`、`experience.json` 有内容、无 memory-disabled warning，pytest 191 passed
- 2026-08-03：V2 迭代（`local-adapt-v2` 分支，按 `docs/V2_IMPLEMENTATION_GUIDE.md`）——P1 choose_lever 重试+超时回退 vocabulary、P2 候选严格场景过滤（strict，治 softmax 误选）、P4 memory 三职责（best-so-far 迭代基准+失败原语避坑清单）、P3 triton-gen prompt 精简（96→77 行）、P5 agent+skill 化（`AgentBackend` 接口同 NgaBackend + gen prompt 双模式 `GEN_PROMPT_MODE` + extension-guide 按场景拆 5 个 ext-* skill）；冻结接缝零改动，pytest 224 passed。P5.4（真实 agent CLI/原语内容/multi-dir 读取）留环境侧