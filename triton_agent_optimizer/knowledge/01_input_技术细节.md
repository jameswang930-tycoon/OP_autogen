# Input 阶段技术细节 — 输入算子文件 (kernel_op.py)

> 阶段 0/6: 输入 → 采集数据 → 分析瓶颈 → 制定策略 → 优化 → 验证
> 更新: 2026-08-14 · 对应代码: `main.py` / `analyzers/merge_single_file.py` / `analyzers/sweep_blocks.py`

---

## 1. 输入是什么

| 项 | 值 |
|---|---|
| 启动命令 | `python main.py input/<op> [--max-rounds 200] [--target 1.5] [--stub] [--fresh] [--resume] [--sweep-blocks]` |
| 核心输入 | **`input/<op>/kernel_op.py` 单文件**（每算子一份，17 个算子 = 17 份） |
| 兼容路径 | 旧式三文件目录（kernel/config/test）→ 启动时 `merge_single_file.merge()` 合并一次 |
| 产物位置 | `outputs/<op>/`（运行日志 `optimization.log` 追加写入） |

## 2. 文件结构（六个部分，缺一不可）

### 2.1 头注释 + imports
- 头注释声明**运算链语义**（如 `Y = GELU(X@W1+b1)@W2`）——这是 planner 的唯一语义信息来源，必须写清楚
- 固定 import 顺序：`os / sys / torch / torch_npu(必须先 import, 注册 NPU 后端) / triton / triton.language as tl`

### 2.2 场景 config 区
```python
M  = int(os.environ.get("MATMUL_M", 2048))   # 尺寸: env 可覆盖, 默认值必须与工业级基准对齐
DTYPE = torch.float32
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32       # 分块: sweep 扫描对象
```
- **尺寸默认值 = 对比场景定义**：必须与 `bench_910b3/bench_industrial.py _shapes()` 一致（matmul 2048³、conv2d N1C8H64W64K32R3S3P1、FA B1S2048H8D64 …），否则工业级对比不同场景、无效
- BLOCK 赋值格式：`BLOCK_M, BLOCK_N, BLOCK_K = a, b, c` 逗号连等（sweep `_read_current_block` 的解析前提，多行单赋值/带类型标注也可解析）
- 注释里标注"不传 num_warps/num_stages — triton-ascend 禁止 tune"

### 2.3 kernel 定义区（@triton.jit）
- 一个运算链 = 多个 kernel（matmul 链 3 个：`matmul_kernel`/`matmul_kernel2`/`bias_gelu_kernel`）
- **同一链中重复角色的 kernel 必须用不同函数名**（FC1 与 FC2 分开）——msprof 按名聚合，同名会被混合画像（A1 检查点）
- 参数结构：指针 + 运行时形状（M/N/K 等，非 constexpr）+ 分块（显式 `tl.constexpr`）
- 典型内容：`program_id` 分解 → `tl.arange` 偏移 → `tl.load(mask/other)` → `tl.dot` 或逐元素 → `tl.store`
- 注意：fp32 累加 `tl.zeros((BM,BN), dtype=tl.float32)` + `tl.dot(a, b, acc)` 三参形式

### 2.4 main() 分配区
```python
x  = (torch.rand(M, K, dtype=DTYPE, device="npu") - 0.5) * 0.1   # 小值输入, 避免 fp32 dot 溢出
z, h, y = torch.empty(...)
```
- **`\w+ = torch.randn|rand|empty|zeros|ones(` 单行分配 = Event 计时破 L2 重建的抓取对象**（正则匹配 `^\s{4,}\w+ = torch.(randn?|rand|empty|zeros|ones)(`）
- 派生张量（`F.unfold`/`reshape`/`pad` 产生）不在重建范围——conv2d 系破 L2 不完整（工作集小, 影响有限, 已知声明）

### 2.5 main() KERNEL_LOOP 循环（★核心约定）
```python
LOOP = int(os.environ.get("KERNEL_LOOP", "1"))
for _ in range(LOOP):
    <一次完整运算链的所有 kernel launch>
torch.npu.synchronize()
```
- **`for \w+ in range(LOOP):` 标准循环 = 所有计时注入的正则锚点**（verifier `_inject_event_timing`、measure_final_event `_inject` 都靠它定位循环体）
- 循环体 = 一次完整运算链调用（1 次 = 1 次端到端）；grid 计算在循环外
- 找不到该循环 → Event 测不出 → 方案 A 永不采纳

### 2.6 main() 正确性校验块
```python
if os.environ.get("MATMUL_VERIFY", "0") == "1":
    ...torch 参考实现 → abs_diff/rel_diff → print PASS/CHECK
```
- verify 每轮先跑它（`MATMUL_VERIFY=1`），不过门不进入计时
- 相对误差容限写在文件内（如 `rel_diff < 0.05`）
- 参考实现必须与头注释语义一致（这里 `F.gelu(approximate="tanh")` 对应 kernel 的 tanh-GELU）

## 3. 启动校验 (`main.validate_kernel_op`, 只警告不阻塞)

| 检查 | 缺失后果 |
|---|---|
| 缺 `KERNEL_LOOP` | verify 无法 ÷N 取单次端到端 → 加速比会错 |
| 缺 `MATMUL_VERIFY` | verify 每轮 FAIL（正确性未通过） |
| 缺 `if __name__ == "__main__":` | 无法直接运行/裸跑 |
| 无 `@triton.jit` kernel | 无优化对象 |
| 同名 kernel 调用 >1 次 | msprof 同名聚合 → deep 画像混合（形状不同时） |

## 4. 硬性技术约束（踩过的坑汇总）

| 约束 | 原因 | 违反后果 |
|---|---|---|
| kernel 体内**禁用** `triton.cdiv` | triton-ascend JIT 内联问题 | HIVM 编译失败, 采集不到 kernel 名 (conv1d 踩过) |
| 全局变量显式 `tl.constexpr` 传参 | HIVM 不支持闭包捕获 | 编译失败 |
| kernel 内 2D gather + padding 减法/除法取模派生索引 | HIVM root alloc 分析失败 | `hivm.hir.load op unsupported for finding the root alloc` (conv2d 踩过 → 改 host unfold) |
| UB 超 192KB（如 FA BLOCK_M=128） | 910B3 UB 容量 | `UB overflow: requires X bits while 1572864 available` (FA 踩过 → 回退 64) |
| 不传 `num_warps/num_stages` | triton-ascend 禁止 tune | 报错 |
| 复杂边界 mask gather（maxpool2d） | root alloc 分析失败 | 同上 → 改 host pad(-inf) 预处理 |

## 5. 与下游模块的耦合点

| 下游 | 依赖输入的什么 |
|---|---|
| `sweep_blocks` | config 区 BLOCK 赋值（`_read_current_block`/`_apply_block`）、尺寸常量（`_read_op_params`） |
| `verifier` | KERNEL_LOOP 循环锚点、MATMUL_VERIFY 块、分配行（破 L2 重建）、kernel 名（op_summary 目标） |
| `measure_final_event` | KERNEL_LOOP 循环锚点 + 分配行（严格单次/流水化注入） |
| `bench_910b3` | 尺寸默认值（场景对齐, 非代码依赖） |
| `planner` | 头注释语义 + 正确性参考实现（作为修改依据） |
| msprof | kernel 名（非 aclnn 前缀即目标 kernel；aclnn* 是框架 kernel 不算） |

## 6. 新增/修改算子检查清单

1. 头注释写清运算链语义
2. config 尺寸与 bench_910b3 `_shapes` 对齐
3. BLOCK 逗号连等格式
4. kernel 内无 `triton.cdiv`、无全局捕获、无 2D gather+padding、UB 预算 < 192KB
5. main 里标准 `for _ in range(LOOP):` 循环 + 循环前 torch 直接分配行
6. MATMUL_VERIFY 参考实现正确
7. 同名 kernel 不重复调用
8. 裸跑 `python3 kernel_op.py` 无错 → `MATMUL_VERIFY=1` 出 PASS
9. `main.py` 启动校验无警告
10. sweep 能读到 BLOCK（`python3 analyzers/sweep_blocks.py input/<op> --quick`）
