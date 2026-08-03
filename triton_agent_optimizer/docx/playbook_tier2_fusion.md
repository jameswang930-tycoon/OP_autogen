# Triton 自动优化系统 Tier 2（Operator Fusion）优化策略指南
## 层级定位与优化目标
Tier 2 位于 Tier 1（算法结构优化）之后、Tier 3（分块参数与硬件配置优化）之前，属于**运算与访存的结构级合并优化**，核心目标是在不改变核心算法逻辑的前提下，识别并合并相邻的逐元素运算、冗余内存访问，减少全局内存读写次数、压缩算子总数、缩短 RAW 数据依赖链，提升计算密度与访存带宽利用率。

本层优化完全基于 HIVM ops 诊断数据（op 类型、依赖链、访存模式、管线通道）驱动。

> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md — WSL2 + Python3.9 + triton3.4.0
> - 仅用 `@triton.jit`，禁用 `@triton.autotune`
> - `num_warps`/`num_stages` 不在 `@triton.jit` 参数中
> - 所有修改必须通过 `ast_to_ttir()` → `ttir_to_hivm.py` → `bisheng` 三层编译

所有修改必须兼容 Triton 3.4.0 语法、可通过 TTIR→HIVM 转换、可被 bisheng 编译器正确编译。

---

## 一、可融合模式对照表
本层共三类核心可融合模式，可独立执行也可叠加组合，覆盖绝大多数逐元素访存与运算冗余场景。

| 融合模式 | 模式定义 | 典型触发场景 | 核心收益 |
|----------|----------|--------------|----------|
| 连续VecUnit逐元素融合 | 同一执行流中，多个连续的逐元素算术/数学运算，中间无全局内存写入、无分支跳转，可合并为单个融合向量运算单元 | 偏置加法+激活函数、多步逐元素四则运算、归一化后接仿射变换 | 消除中间临时变量，减少算子调度开销，缩短数据依赖链 |
| 同指针冗余Load融合 | 同一基址指针、相同偏移范围、相同掩码条件的多次全局加载操作，中间无对该地址的写入，可合并为单次加载后寄存器复用 | 同一输入被多个运算分支重复读取、计算链中重复引用同一源数据 | 减少全局内存访问次数，降低带宽占用，消除重复访存开销 |
| 串行算术链融合 | 由 RAW 依赖串联的多步算术运算链，中间无全局存储、无副作用操作，可整体折叠为单条融合运算流 | 多步多项式计算、归一化公式展开、连续类型转换 | 将长依赖链压缩为单步运算，消除串行等待开销，提升指令级并行度 |

> 组合规则：三类模式可叠加执行，通常先合并冗余 Load，再将 Load 后的连续算术链整体融合为单个 VecUnit 算子，收益最大化。

---

## 二、HIVM 诊断触发规则
所有诊断基于静态分析输出的 HIVM ops 数据，满足对应阈值即判定存在融合空间，必须执行融合优化。

### 2.1 分模式触发阈值
| 融合模式 | 触发条件（满足任意一条即执行） | 强制执行阈值 |
|----------|--------------------------------|--------------|
| 连续VecUnit逐元素融合 | 1. 同管线通道内，连续逐元素算术/math op 数量 ≥ 2<br>2. 相邻 op 之间无全局 store、原子操作、分支跳转 | 连续逐元素 op ≥ 3 且中间无 store |
| 同指针冗余Load融合 | 1. 同一基址指针、同 size_kb、同掩码的 load op 出现次数 ≥ 2<br>2. 两次 load 之间无对应地址的 store 操作 | 同条件 load 重复 ≥ 2 次 |
| 串行算术链融合 | 1. 纯逐元素运算构成的 RAW 依赖链长度 ≥ 3<br>2. 链中无任何内存写入、副作用操作 | RAW 链长度 ≥ 4 且全为算术 op |

### 2.2 辅助判定与最优标准

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断融合机会：
> - `per_op_statistics[].op_type` — 统计 VecUnit op 连续出现次数（≥2 可融合）
> - `per_op_statistics[].pipeline_channel` — 确认同管线（VECTOR）且中间无 MTE3（store）
> - `dependencies_summary.raw_chains` — RAW 链长度 ≥ 3 触发算术链融合
> - `per_op_statistics[].size_kb` — 同 size 同 memory_region 的 load op ≥ 2 触发 Load 合并
- **辅助排查条件**：单 kernel 内逐元素 op 总数 ≥ 3，且访存 op / 计算 op 比值 > 2 → 强制全量扫描融合机会
- **Tier 2 最优判定**：同时满足以下三条，可判定算子融合已达最优，允许进入 Tier 3
  1. 同条件全局 load 无重复（同指针同偏移仅出现 1 次）
  2. 连续逐元素运算最长链长度 ≤ 1
  3. 纯算术 RAW 依赖链长度 ≤ 1，无中间临时全局写入

> 前置约束：若 Tier 1 判定算法结构非最优，必须先完成 Tier 1 优化再执行 Tier 2 诊断，禁止跳过算法层直接做算子融合。

---

## 三、各模式 Before/After 代码示例
所有示例严格遵循 Triton 3.4.0 语法，仅使用 `@triton.jit` 装饰器，块尺寸参数标记 `tl.constexpr`，无 GPU 专属 API。

### 3.1 模式一：连续VecUnit逐元素融合
**场景**：偏置加法 + 缩放 + GELU 激活三步连续运算，中间无写回。

**Before（拆分写法）**
```python
@triton.jit
def fused_op_before(x_ptr, bias_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # 三步拆分运算，生成3个独立VecUnit op
    x_add = x + bias
    x_mul = x_add * 0.5
    x_gelu = 0.5 * x_mul * (1.0 + tl.erf(x_mul * 0.70710678118))
    
    tl.store(out_ptr + offsets, x_gelu, mask=mask)
```

**After（融合写法）**
```python
@triton.jit
def fused_op_after(x_ptr, bias_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    # 单表达式链式运算，编译器自动融合为1个VecUnit op
    x_fused = x + bias
    out = 0.5 * x_fused * (1.0 + tl.erf(x_fused * 0.35355339059))
    
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 3.2 模式二：同指针冗余Load融合
**场景**：同一输入数据被两次读取，分别用于不同计算分支，中间无修改。

**Before（重复Load写法）**
```python
@triton.jit
def redundant_load_before(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 第一次读取x用于统计量计算
    x1 = tl.load(x_ptr + base + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x1, axis=0)
    x_exp = tl.exp(x1 - row_max)
    
    # 第二次重复读取同一地址的x，完全冗余
    x2 = tl.load(x_ptr + base + offsets, mask=mask, other=-float('inf'))
    x_out = x2 * x_exp / tl.sum(x_exp, axis=0)
    
    tl.store(out_ptr + base + offsets, x_out, mask=mask)
```

**After（单次Load复用写法）**
```python
@triton.jit
def redundant_load_after(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 单次加载，寄存器全链路复用
    x = tl.load(x_ptr + base + offsets, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - row_max)
    x_out = x * x_exp / tl.sum(x_exp, axis=0)
    
    tl.store(out_ptr + base + offsets, x_out, mask=mask)
```

### 3.3 模式三：串行算术链融合
**场景**：归一化计算的多步算术串行链，连续的平方、求和、开方、除法运算。

**Before（拆分长链写法）**
```python
@triton.jit
def arith_chain_before(x_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    # 4步串行算术运算，RAW依赖链长度=4
    x_sq = x * x
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / n_cols
    var = mean_sq + eps
    inv_std = tl.math.rsqrt(var)
    out = x * inv_std
    
    tl.store(out_ptr + base + offsets, out, mask=mask)
```

**After（融合压缩写法）**
```python
@triton.jit
def arith_chain_after(x_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    # 归约后算术链整体融合，RAW依赖链压缩为1
    inv_std = tl.math.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    out = x * inv_std
    
    tl.store(out_ptr + base + offsets, out, mask=mask)
```

---

## 四、融合后 HIVM ops 预期变化
> **在我们的 pipeline 中验证融合效果**：
> 1. 融合后 `per_op_statistics` 中 VECTOR op 数量应减少
> 2. HIVM `gm_to_ub` op 数量应减少（Load 合并）
> 3. RAW 依赖链长度应缩短
> 4. 若 op 数量不变 = 融合失败（Coder 改了等于没改）→ REVERT

### 4.1 量化收益表
| 融合模式 | 优化前 op 数量 | 优化后 op 数量 | 算子压缩比例 | 依赖链变化 |
|----------|----------------|----------------|--------------|------------|
| 连续VecUnit逐元素融合 | N 个连续逐元素 op（N≥2） | 1 个融合 VecUnit op | 减少 N-1 个，压缩率 (N-1)/N | RAW 链长度从 N 缩短为 1 |
| 同指针冗余Load融合 | N 次同条件 load（N≥2） | 1 次 load | 减少 N-1 个，压缩率 (N-1)/N | 消除 N-1 条冗余访存依赖 |
| 串行算术链融合 | N 步串行算术 op（N≥3） | 1 条融合运算流 | 减少 N-1 个，压缩率 (N-1)/N | RAW 链长度从 N 缩短为 1 |

### 4.2 典型场景收益示例
- 偏置+缩放+激活（3个 arith/math op）→ 融合后 1 个 op，op 数量减少 67%
- 同一输入重复读取 2 次 + 3 步连续运算 → 融合后 load 减少 1 个、运算减少 2 个，总 op 数减少 50% 以上
- 附带收益：融合后单条指令计算密度提升，全局带宽利用率通常可提升 10%~30%

---

## 五、不可融合的边界场景
以下场景严格禁止融合，Agent 必须遵守边界规则，避免逻辑错误或性能回退。

1. **中间存在全局 Store 操作**
   两个运算之间存在对全局内存的写入操作，数据生命周期跨内存，无法跨 store 融合；即使写入地址不同，也禁止跨 store 合并运算流。

2. **跨 Tile / 跨循环迭代边界**
   分块循环内外的运算、不同循环迭代的运算，属于不同执行上下文，数据依赖不明确，禁止跨迭代融合。

3. **依赖链包含非逐元素操作**
   运算链中间插入归约（sum/max）、矩阵乘、原子操作、条件分支等非逐元素算子时，禁止跨算子融合；仅归约前后的独立逐元素链可分别融合。

4. **掩码 / 偏移范围不一致**
   待融合的多个操作对应的内存偏移范围、掩码条件不一致时，强行融合会导致边界越界或数值错误，禁止融合。

5. **存在副作用操作**
   运算之间包含 `tl.device_print`、`tl.device_assert` 等带副作用的调试操作，禁止跨越副作用操作融合。

6. **寄存器压力临界场景**
   融合后单块张量数据总 `size_kb` 超过寄存器/片上缓存阈值，会导致寄存器溢出、数据换入换出，反而性能下降。当单块运算张量数 ≥ 8 或总数据量 ≥ 64KB 时，需评估后再融合。

7. **数据类型存在精度风险**
   多步运算包含不同精度类型转换（如 FP32→FP16→FP32），融合可能改变精度与舍入行为，导致数值偏差，需保留显式类型转换步骤。

---

## 六、常见错误与修复方案
### 1. 掩码不一致导致边界错误
- **错误描述**：待融合的两个操作 mask 范围不同，强行融合后边界处出现越界读写或数值错误。
- **触发场景**：两个运算的分块偏移不同、边界条件不同。
- **修复方案**：融合前校验所有操作的偏移基址、掩码逻辑完全一致；若不一致，先统一边界处理逻辑，再执行融合。
- **校验方法**：对比 HIVM 中对应 load/store op 的 mask 表达式，完全等价才可融合。

### 2. 过度融合导致寄存器溢出
- **错误描述**：无限制合并大量运算，单周期驻留寄存器的张量过多，超出硬件寄存器上限，触发寄存器溢出到片上内存，性能反而下降。
- **触发场景**：一次性融合 ≥ 6 个输入张量的复杂运算、分块尺寸过大同时融合多算子。
- **修复方案**：通过 HIVM 的 `size_kb` 指标监控单块总数据量，超过阈值时拆分融合链，保留 2~3 级融合深度；优先融合访存开销最高的相邻运算。

### 3. 运算重排导致浮点精度偏差
- **错误描述**：融合时为了简化表达式调整了浮点运算顺序，改变了舍入行为，导致输出结果与原算法存在精度差异。
- **触发场景**：多步乘加、多项式计算融合时随意调整结合顺序。
- **修复方案**：严格保持原运算的运算顺序与结合性，禁止重新排列浮点运算；仅可合并完全等价的常量运算。
- **验证方法**：融合前后与参考实现做数值对比，FP32 下最大绝对误差不超过 1e-6。

### 4. 误融合跨迭代操作
- **错误描述**：将循环内的运算与循环外的运算、不同循环迭代的运算强行融合，破坏程序逻辑，导致结果错误。
- **触发场景**：优化分块循环时，将循环初始化、循环体、循环收尾的运算跨步骤合并。
- **修复方案**：仅融合同一控制流、同一循环迭代内的相邻操作；跨迭代的累积运算（如累加器）禁止与单次迭代运算融合。

### 5. 遗漏公共子表达式提取
- **错误描述**：仅合并了表层运算，未提取重复的公共子计算，融合收益未最大化。
- **触发场景**：两个分支重复计算相同的子表达式，仅做了外层融合。
- **修复方案**：融合同时识别重复的子计算逻辑，提取为公共变量，进一步减少运算量。

### 6. 融合后引入冗余类型转换
- **错误描述**：多步运算各自带类型转换，融合后出现连续的冗余 cast 操作，增加不必要开销。
- **触发场景**：多步运算混合 FP16/FP32 类型，融合后保留了中间转换步骤。
- **修复方案**：合并连续的同方向类型转换，统一为单次最终类型转换；中间计算优先使用高精度，输出时再做类型转换。

需要我补充融合收益的量化评估公式，或者针对特定算子（如 RMSNorm+Activation）的融合模板吗？