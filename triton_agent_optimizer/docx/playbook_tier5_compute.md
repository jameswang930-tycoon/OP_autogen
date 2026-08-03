# Triton 自动优化系统 Tier 5（Compute & Occupancy）优化策略指南
## 层级定位与前置约束
Tier 5 是全链路优化的最终指令级计算优化层，**必须在 Tier 1~Tier 4 全部完成并验证最优后执行**。本层不改变核心算法、算子融合结构、分块尺寸与访存模式，仅通过等价指令替换、计算融合、冗余变量消除等手段，适配 Ascend NPU 向量计算单元（VecUnit）硬件特性，提升计算单元占用率、降低单运算周期、减少寄存器冗余占用，榨取计算侧极限性能。

> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - num_warps/num_stages 由 GPUTarget 设置，Coder 不修改
> - **数学函数用 `tl.math.*` 命名空间**（我们的 ttir_to_hivm.py 以 `math.xxx` 为 key）
> - `tl.fma` 谨慎使用——直接写 `x*w+b` 让 bisheng 自动融合更安全

本手册严格适配环境约束：
- 前端：Triton 3.4.0 + Python 3.9，仅使用 `@triton.jit` 装饰器，禁用 `@triton.autotune`
- 中间链路：TTIR → HIVM 自定义转换，所有语法必须可解析
- 后端：CANN 9.0 + bisheng 编译器 + CMake 构建
- 验证：msprof op simulator 周期精确模拟
- 核心边界：`num_warps`、`num_stages` 不属于 `@triton.jit` 参数，仅通过调用配置传递

---

## 一、核心优化策略
按优先级从高到低分为三类，均为等价语义优化，不改变数值逻辑（仅浮点舍入次数存在可控微小差异）。

### 1. 原生数学指令替换
- **优化原理**：用硬件原生支持的专用数学指令，替代手动组合的等价运算，减少指令条数、降低运算延迟、减少寄存器占用。核心替换场景为**倒数平方根**：使用 `tl.math.rsqrt` 替代 `1.0 / tl.sqrt(x)`，将「开方+除法」两条指令合并为单条原生指令，运算周期减少 40%~60%。
- **适用场景**：归一化算子（RMSNorm、LayerNorm）中的标准差倒数计算、所有需要除以根号的运算场景。
- **扩展替换**：同类可替换模式还包括 `tl.exp2` / `tl.log2` 替代带常数缩放的 `tl.exp` / `tl.log`，进一步消除常数乘法开销。

### 2. 融合乘加（FMA）指令化
- **优化原理**：将独立的「乘法 + 加法」串行运算对，合并为单条融合乘加（Fused Multiply-Add, FMA）指令，仅做一次浮点舍入，节省一个指令周期，同时减少一次中间结果的寄存器占用。
- **适用场景**：偏置加法（`y = w*x + b`）、缩放平移、多项式计算、激活函数中的乘加组合等所有 `a*b + c` 模式的运算。
- **实现方式**：显式调用 `tl.fma(a, b, c)`（语义等价于 `a*b + c`），确保 bisheng 编译器稳定触发硬件 FMA 指令，避免编译器因浮点精度保守而不自动融合。

### 3. 冗余中间变量消除
- **优化原理**：消除计算链中无意义的中转中间变量，将串行的简单算术运算合并为连续表达式，缩短数据依赖链，提升指令级并行度，同时减少不必要的寄存器分配。
- **适用场景**：单步运算就赋值一个中间变量、无跨分支复用价值的串行计算链；仅做数据中转、无逻辑意义的临时变量。
- **边界约束**：关键统计量（如均值、方差、求和结果）、跨分支复用的变量必须保留，仅消除纯中转性质的冗余变量。

---

## 二、HIVM 诊断触发规则
所有优化动作由 HIVM ops 诊断数据驱动，满足对应阈值即执行对应优化。

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断计算优化机会：
> - `per_op_statistics[].op_type` — 统计 VECTOR 管线 op 数量（vadd/vmul/vdiv）
> - `per_op_statistics[].pipeline_channel` — 确认是 VECTOR 管线
> - `execution_summary.engine_usage_pct` — VecUnit 占比 > 60% 触发计算优化
> - 连续的 `vmul` + `vadd` RAW 链（无中间 store）→ FMA 候选

### 2.1 通用与专项触发阈值
| 触发类型 | 判定规则 | 执行动作 |
|----------|----------|----------|
| 核心触发 | VecUnit 类运算 op（算术、数学函数）占总 op 数量比例 **> 60%**，且全局带宽利用率 `bw_util < 70%` | 判定为计算瓶颈型负载，执行全量计算优化扫描 |
| 专项触发1 | HIVM 识别到 `div(sqrt(x), 1.0)` 等价运算模式，或存在手动组合的可替换数学运算 | 执行原生数学指令替换 |
| 专项触发2 | 存在连续的乘法 op + 加法 op，两者为 RAW 直连依赖、无其他分支、无中间存储 | 执行乘加 FMA 融合 |
| 专项触发3 | 单计算链内仅做数据中转的中间变量数量 ≥ 2，且无跨分支复用 | 执行冗余中间变量消除 |

### 2.2 Tier 5 最优判定标准
同时满足以下所有条件，判定计算与占用率层已达最优，全链路优化结束：
1. 所有数学运算均使用硬件原生指令，无手动组合的等价运算
2. 所有可融合的乘加对均已转化为 FMA 指令
3. 计算链无冗余中转变量，依赖链长度最短
4. VecUnit 计算单元占用率 ≥ 90%，无明显计算气泡
5. 单指令级调整无法带来正向性能收益

---

## 三、优化前后代码示例（Triton 3.4.0 语法）
所有示例严格遵循官方语法，仅使用标准 `@triton.jit` 装饰器，无 GPU 专属 API，可通过全链路编译。

### 3.1 原生 rsqrt 替代 1/sqrt
**典型场景**：RMSNorm 中的倒数平方根计算

**Before（手动组合写法）**
```python
@triton.jit
def rsqrt_before(x_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    var = tl.sum(x * x, axis=0) / n_cols
    # 两步运算：先开方，再求倒数，对应2条VecUnit指令
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = x * inv_std

    tl.store(out_ptr + base + offsets, out, mask=mask)
```

**After（原生指令写法）**
```python
@triton.jit
def rsqrt_after(x_ptr, out_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + base + offsets, mask=mask)
    var = tl.sum(x * x, axis=0) / n_cols
    # 单条原生倒数平方根指令，等价于1/sqrt，运算周期更短
    inv_std = tl.math.rsqrt(var + eps)  # 单条指令替代 1.0/tl.sqrt()，减少50% VecUnit ops
    out = x * inv_std

    tl.store(out_ptr + base + offsets, out, mask=mask)
```
> 说明：在我们的环境（triton 3.4.0 + 自定义 TTIR→HIVM 转换器）中，
> **使用 `tl.math.rsqrt`**（已验证通过编译和转换）。
> 不要用 `tl.math.rsqrt`——我们的转换器的 ARITH_TO_HIVM 映射以 `math.rsqrt` 为 key。

### 3.2 融合乘加（FMA）指令化（谨慎使用）
**典型场景**：线性层偏置加法、仿射变换

> **⚠️ 注意**：`tl.fma` 在我们的 TTIR→HIVM 转换器中**可能不被识别**。
> `ARITH_TO_HIVM` 映射表里没有 `fma` 条目，会生成未映射的 TTIR op。
> **推荐方案**：直接写 `x * w + b`，bisheng 编译器会自动融合为 FMA 指令。
> 仅当 HIVM 明确显示两条独立 VecUnit op（mul + add）且未自动融合时，才尝试显式 `tl.fma`。

**Before（拆分乘加写法）**
```python
@triton.jit
def fma_before(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 两步运算：先乘后加，对应2条VecUnit指令
    mul = x * w
    out = mul + b

    tl.store(out_ptr + offsets, out, mask=mask)
```

**After（显式FMA写法）**
```python
@triton.jit
def fma_after(x_ptr, w_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    # 单条融合乘加指令，等价于x*w + b，一次舍入更高效
    out = tl.fma(x, w, b)

    tl.store(out_ptr + offsets, out, mask=mask)
```

### 3.3 冗余中间变量消除
**典型场景**：串行多步逐元素运算，无复用的中转变量

**Before（冗余变量写法）**
```python
@triton.jit
def var_elim_before(x_ptr, out_ptr, scale, bias, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 3个无复用中转变量，对应3次独立指令调度
    x_scaled = x * scale
    x_biased = x_scaled + bias
    x_alpha = x_biased * alpha
    out = tl.erf(x_alpha)

    tl.store(out_ptr + offsets, out, mask=mask)
```

**After（变量消除写法）**
```python
@triton.jit
def var_elim_after(x_ptr, out_ptr, scale, bias, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 合并串行运算，消除中转变量，编译器自动做指令融合
    out = tl.erf((x * scale + bias) * alpha)  # 编译器自动融合 mul+add 为 FMA

    tl.store(out_ptr + offsets, out, mask=mask)
```
> 说明：仅消除纯中转、无复用的变量；关键统计量、跨分支复用变量必须保留，避免破坏逻辑可读性与调试性。

---

## 四、常见错误与修复方案
### 1. 指令替换导致数值精度超差
- **错误现象**：替换 `tl.math.rsqrt`、`tl.fma` 后，输出结果与原实现存在数值差异，超出业务精度阈值。
- **触发原因**：原生融合指令仅做一次浮点舍入，拆分运算为多次舍入，IEEE 浮点标准下存在固有微小差异；部分硬件快速近似指令精度更低。
- **修复方案**：
  1. 优先使用标准 IEEE 兼容指令（`tl.math.rsqrt`、`tl.fma`），禁止使用快速近似版本
  2. 精度敏感场景先做误差评估，FP32 下最大绝对误差应 ≤ 1e-6，超出则回退为拆分写法
  3. 归约累加、统计量计算等关键路径，优先保证精度，不强行替换
- **校验方法**：优化前后与参考实现做全量数值对比，误差在允许范围内再落地。

### 2. API 命名空间错误导致编译失败
- **错误现象**：`ast_to_ttir` 阶段报错，提示 `tl.math` 模块不存在或函数未定义。
- **触发原因**：误用 `tl.math.rsqrt` 等子命名空间 API，Triton 3.4.0 标准数学函数统一位于 `tl` 顶层命名空间，`tl.math` 为实验性别名，自定义 TTIR→HIVM 转换器可能不支持。
- **修复方案**：统一使用顶层标准 API，如 `tl.math.rsqrt`、`tl.fma`、`tl.exp`、`tl.log`，避免使用 `tl.math.*` 子命名空间，保证全链路编译兼容。

### 3. 过度消除变量导致寄存器溢出
- **错误现象**：消除中间变量后，性能不升反降，HIVM 出现寄存器溢出标记。
- **触发原因**：单计算链合并运算过多，同时驻留寄存器的张量数量超出硬件上限，触发寄存器溢出到片上缓存，引入额外访存开销。
- **修复方案**：
  1. 长计算链保留 2~3 级中间变量，平衡融合收益与寄存器压力
  2. 优先消除短链、无复用的中转变量，长计算链分步执行
  3. 结合 HIVM 寄存器占用指标调整，溢出时回退部分变量

### 4. 跨依赖强行 FMA 融合导致语义错误
- **错误现象**：融合后结果偏差较大，甚至出现数值溢出。
- **触发原因**：错误地将存在精度转换、多步依赖的运算强行合并为 FMA，改变了运算优先级与舍入次数，破坏了原语义。
- **修复方案**：
  1. 仅对纯 `a*b + c` 模式、无中间精度转换、无其他依赖的乘加对进行融合
  2. 禁止跨类型转换、跨归约、跨分支进行 FMA 融合
  3. 融合前后严格校验运算语义等价，禁止改变运算顺序

### 5. 消除关键变量导致调试困难
- **错误现象**：优化后出现数值错误，但无法定位具体计算步骤。
- **触发原因**：过度消除所有中间变量，包括关键统计节点、调试观测点，导致问题无法定位。
- **修复方案**：
  1. 仅消除纯中转、无逻辑意义的临时变量
  2. 均值、方差、求和、归一化系数等关键统计量必须保留独立变量
  3. 调试模式下可临时保留中间变量，定位问题后再做优化

### 6. 重复优化与上层层级冲突
- **错误现象**：优化后与 Tier 2 算子融合结果重复，无额外收益，甚至引入冗余。
- **触发原因**：Tier 2 已完成算子融合，Tier 5 重复做同层级优化，浪费优化轮次。
- **修复方案**：严格遵循优化层级顺序，Tier 5 仅做 Tier 2 未覆盖的指令级精细优化；Tier 2 已融合的算子不再重复处理，仅针对未融合的残余计算链优化。

需要我补充某个算子的完整 Tier 5 优化示例，或者计算单元占用率的评估公式吗？