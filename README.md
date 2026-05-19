# OP_autogen

Triton Language CPU Emulator -- 在 CPU 上以逐算子粒度模拟 Triton kernel 的执行，用于算子逻辑验证和自测。

> 核心模拟层在 [emulators/common/__init__.py](emulators/common/__init__.py)，阅读该文件的模块 docstring 可以快速理解整个仓库的设计思路。

## 项目结构

```
OP_autogen/
├── emulators/
│   ├── common/                # 公共基础设施（Triton API 打桩 + 验证工具）
│   │   └── __init__.py
│   ├── add/                   # 逐元素加法 (element-wise add)
│   │   └── __init__.py
│   ├── matmul/                # 矩阵乘法 (2D tiled matmul)
│   │   └── __init__.py
│   ├── transpose/             # 矩阵转置 (2D transpose)
│   │   └── __init__.py
│   ├── reshape/               # 张量 reshape（无数据搬运，仅改元信息）
│   │   └── __init__.py
│   ├── relu/                  # ReLU / Leaky ReLU 激活
│   │   └── __init__.py
│   ├── softmax/               # 行级数值稳定 softmax
│   │   └── __init__.py
│   ├── rmsnorm/               # Root Mean Square Layer Normalization
│   │   └── __init__.py
│   ├── addrmsnormgamma/       # 融合算子：Add + RMSNorm + Gamma
│   │   └── __init__.py
│   ├── attention-relu/        # 缩放点积注意力 + ReLU 激活（替代 softmax）
│   │   └── __init__.py
│   └── run_all_tests.py       # 全量自测入口
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

### 算子模块约定

每个算子目录下的 `__init__.py` 遵循统一模式：

1. **Triton-style Kernel** -- 纯粹的 kernel 函数，只使用 `tl.*` API
2. **`emulate_xxx()`** -- 封装函数，扁平化输入、启动 grid、reshape 输出
3. **`reference_xxx()`** -- NumPy 参考实现，用于对比验证
4. **`test()`** -- 自测函数，覆盖正常路径、边界条件、错误路径

## 运行测试

```bash
# 运行全部算子自测
python emulators/run_all_tests.py

# 单独运行某个算子自测（需在 emulators 目录下）
cd emulators && python -c "from add import test; test()"
cd emulators && python -c "from matmul import test; test()"
```

## 已支持的算子

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
