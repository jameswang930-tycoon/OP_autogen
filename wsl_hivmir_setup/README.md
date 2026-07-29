# WSL2 HIVMIR 编译环境工作区

本目录包含在 Windows WSL2 下搭建 Ascend CANN + Bisheng 编译器环境的所有文件，
以及获取/解析 HIVM 中间表示的方案。

## 文件说明

| 文件 | 用途 |
|---|---|
| `SETUP_GUIDE.md` | ★ 完整安装指南：WSL2 → CANN → Bisheng → HIVM IR |
| `test_vec_add.mlir` | 测试用 MLIR 文件（VecAdd in HIVM dialect） |
| `hivmir_parser_v2.py` | (待实现) 适配真实 MLIR 格式的 HIVMIR 解析器 |

## 快速开始

1. 按 `SETUP_GUIDE.md` Step 1 安装 WSL2
2. 按 Step 2 安装 CANN
3. 在 WSL 中运行:
   ```bash
   bishengir-compile test_vec_add.mlir -o vec_add.o
   ```

## 关键认知

- **WSL2 可以跑 CANN 编译**：bishengir-compile 是纯 CPU 工具，不依赖 NPU 硬件
- **我们现有的 hivmir_analyzer.py 格式是错的**：假设的是简化格式（`hivm.gm_to_ub`），真实格式是 MLIR 标准语法（`hivm.hir.load`）
- **需要重写解析器**：alloc/size_kb/地址空间/操作映射全部要改
