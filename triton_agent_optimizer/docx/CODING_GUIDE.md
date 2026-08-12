# Triton 3.4.0 编码与优化指导手册（Coder Agent 专用）

> 整合：豆包 AI 官方文档调研 + 项目实测验证
> 日期：2026-07-31
> 目标：Coder Agent 生成的代码必须通过全链路编译（Triton→TTIR→HIVM→AscendC→bisheng→msprof）

---

## 一、环境与全链路约束

```
Triton .py → ast_to_ttir() → TTIR MLIR → ttir_to_hivm.py → HIVM MLIR
→ hivm_to_ascendc.py → Ascend C .asc → CMake ASC + bisheng → demo
→ msprof op simulator → instr_exe.csv
```

**每层都可能失败，必须全部通过。**

| 层 | 检查点 | 失败后果 |
|---|---|---|
| Triton 3.4.0 | `ast_to_ttir()` 编译 | 直接崩溃 |
| TTIR→HIVM | 手写解析器匹配 | 0 ops，无诊断数据 |
| Ascend C | bisheng 编译 | msprof 无法重采 |
| msprof | simulator 运行 | 无 timing 数据 |

---

## 二、@triton.jit 正确用法（triton 3.4.0 官方 API）

### ❌ 错误写法（我们实测会崩）

```python
@triton.jit(num_warps=4)       # ❌ triton 3.4 报错：unexpected keyword argument
def kernel(...): ...

@triton.jit(num_stages=2)      # ❌ 同上
def kernel(...): ...
```

### ✅ 正确写法

```python
@triton.jit
def kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    ...
```

**`num_warps` 和 `num_stages` 不是 `@triton.jit` 的参数！** 它们在我们的 pipeline 中由 `GPUTarget` 编译选项设置（`options={"num_warps":4,"num_stages":1}`）。

### @triton.jit 合法参数（仅以下）

| 参数 | 用途 |
|------|------|
| `do_not_specialize` | 不对指定参数做类型特化 |
| `debug` | 调试模式 |

其他参数（version, repr, launch_metadata, noinline）基本不用。

---

## 三、可用 API（triton 3.4.0 验证通过）

### 内存操作
- `tl.load(ptr, mask=mask, other=0.0)` ✅
- `tl.store(ptr, val, mask=mask)` ✅
- `tl.arange(0, N)` — N 必须是 tl.constexpr ✅

### 归约
- `tl.sum(x, axis=0)`, `tl.max(x, axis=0)` ✅

### 数学
- `tl.math.exp`, `tl.math.sqrt`, `tl.math.rsqrt` ✅
- `tl.math.log`, `tl.math.log2`, `tl.math.exp2` ✅
- `tl.math.sin`, `tl.math.cos` ✅
- ⚠ `tl.math.erf` **triton-ascend 不可用**（编译失败）→ 激活一律用 `tl.math.tanh` 近似（与全部 playbook 一致）
- `x + y`, `x * y`, `x - y`, `x / y` ✅

### 张量操作（3.4.0 新增支持）
- `[:, None]` 广播 ✅
- `tl.zeros((M, N), dtype=tl.float32)` 2D ✅
- `tl.dot(a, b, acc)` 矩阵乘 ✅
- `tl.view(tensor, (M, N))` reshape ✅
- `tl.broadcast_to(tensor, (M, N))` ✅

### 禁止使用（TTIR→HIVM 转换失败）
- `tl.make_tensor_descriptor`, `tl.async_copy` (TMA，GPU专属)
- `tl.dot` 的 GPU 专属参数（`allow_tf32`, `matrix_instr_nonkdim`）
- `num_ctas > 1`（集群特性）
- `@tl.aggregate` 装饰器
- `@triton.autotune`（我们的 mock 环境不支持）

### 禁止使用（Python 语法不支持）
- `try/except`、`for i in some_list`（动态循环）
- `math.*`、`numpy.*`（必须用 `tl.math.*`）
- 字典、列表、类实例等动态结构

---

## 四、代码修改铁律

### 必须遵守（违反 = FAIL）

1. **不改函数名**：函数名必须和原 kernel 完全一致
2. **不改参数名、参数顺序、参数个数**
3. **不改数学公式**：RMSNorm 必须是 `x / rsqrt(mean(x^2)+eps) * w`，不能变成别的
4. **不新增 import**：不能加 `import torch`, `import numpy` 等
5. **不新增 @triton.autotune**：mock 环境不支持
6. **不改 @triton.jit 装饰器**：不加任何参数

### 允许的优化（真·优化）

1. **改 BLOCK_SIZE**：`256 → 512 → 1024 → 2048`（2的幂）
2. **融合相邻逐元素操作**：`x + y; z = x * w` → `z = (x + y) * w`
3. **消除冗余 Load**：同一指针被 load 两次 → 合并为一次
4. **调整计算顺序**：把循环不变量提到循环外
5. **添加变量缓存**：重复使用的表达式存到局部变量

### 不允许的"假优化"（会被检测出来 REVERT）

- 只改注释、空格、空行（no-op 检测）
- 加 dummy load/store（`tl.load(x, ...)`后不用结果）
- 改变量名但不改变执行逻辑
- 增加实际计算量的"优化"

---

## 五、常见错误及修复

| 错误 | 原因 | 修复 |
|------|------|------|
| `unexpected keyword argument 'num_warps'` | 给 @triton.jit 加了参数 | 删除，只保留 `@triton.jit` |
| `jit() got an unexpected keyword argument` | 同上 | 同上 |
| `object has no attribute '__name__'` | 加/删了 @triton.jit 或改了函数名 | 保持原函数名和装饰器 |
| `ModuleNotFoundError` | 新增了 import | 删除新 import |
| HIVM: 0 ops | 改的代码结构不能被 TTIR→HIVM 解析 | 保持 for-range 循环，避免动态 shape |
| `use of undeclared identifier` (bisheng) | Ascend C 代码用了不存在的 API | 只能用 DataCopy/Add/Mul/Exp 等已验证 API |
| `pointer<fp32> and int32 incompatible` | pid * 非constexpr参数 | 确保所有 stride/dim 参数是 tl.constexpr |

---

## 六、Tier 1 优化策略：算法结构

### 原则
算法决定数据流。改了算法后面所有层都要重来——必须最先确定。

### 可优化项

| 当前模式 | 优化方案 | 预期收益 |
|---------|---------|---------|
| 每个 program 处理 1 个元素 | 每个 program 处理 BLOCK_SIZE 个元素（已是最优） | — |
| 多次 tl.load 同一数据 | 一次 load，复用变量 | 减少 MTE2 |
| 中间变量未缓存 | 缓存重复计算到局部变量 | 减少 VecUnit ops |
| for 循环内做不变量计算 | 提到循环外 | 减少冗余计算 |

### 何时说"已最优"
只有同时满足以下 3 个条件：
1. num_ops ≤ 3（最基本的 load+compute+store）
2. RAW 依赖链 ≤ 1
3. bw_util > 90% 或已经是单 pass 算法

否则必须给出具体优化策略。

---

## 七、Tier 2 优化策略：算子融合

### 可融合模式
- 相邻 VecUnit 操作（无中间 store/load）
- `x + y; z = x * w` → `z = (x + y) * w`（消除中间变量）
- 多个 load 同一指针 → 合并为一次 load

### 不可融合
- 中间有 store 的（数据写到 GM 了，无法融合）
- 跨 tile 的（不同 program 处理的数据不同）

---

## 八、Tier 3 优化策略：分块参数

### BLOCK_SIZE 选择
- 256 → 512 → 1024 → 2048（必须是 2 的幂）
- 更大 = 更高带宽利用率，但更多 UB 占用
- 约束：BLOCK_SIZE × 4 bytes × N_buffers ≤ 192KB (UB)

### 验证
修改 BLOCK_SIZE 后，检查 HIVM ops 数量是否减少（更大的块→更少的 load/store 次数）
