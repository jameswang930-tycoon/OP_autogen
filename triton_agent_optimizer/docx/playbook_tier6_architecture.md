# Triton 自动优化系统 Tier 6（Pipeline & Toolchain）优化策略指南
## 层级定位与前置约束
Tier 6 是全链路优化的最终工具链适配层，**必须在 Tier 1~Tier 5 全部完成并验证通过后执行**。本层不改变算法逻辑、算子融合结构、分块参数、访存模式与计算指令，仅通过调整 Triton 代码的书写风格、语法子集与结构组织，适配自研的 TTIR→HIVM 转换器、AscendC 代码生成器与 msprof 性能模拟器，消除转换过程中的性能损耗、提升后端代码质量、保证性能统计的完整性与准确性。

> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - 全链路：ast_to_ttir() → ttir_to_hivm.py → hivm_to_ascendc.py → bisheng → msprof
> - `tl.math.*` 命名空间是我们的转换器支持的（以 `math.xxx` 为 key）
> - `tl.fma` 未映射——写 `x*w+b` 让 bisheng 自动融合
> - num_warps/num_stages 由 GPUTarget 管理，Coder 不修改

本手册严格适配自研全链路工具与环境约束：
- 前端：Triton 3.4.0 + Python 3.9，仅使用 `@triton.jit` 装饰器，禁用 `@triton.autotune`
- 编译链路：Triton Python → TTIR MLIR → 自研 HIVM 转换器 → 自研 AscendC 生成器 → bisheng 编译器
- 验证：msprof op simulator（CPU 周期精确 NPU 模拟器）
- 核心边界：`num_warps`、`num_stages` 不属于 `@triton.jit` 参数，仅通过调用配置传递

---

## 一、核心优化策略
### 1. TTIR→HIVM 转换损耗消减
目标是让 TTIR 算子与 HIVM 算子实现 1:1 原生映射

> **我们的转换器已知限制**：
> - `ARITH_TO_HIVM` 映射了 12 个 op（vadd/vsub/vmul/vdiv/vexp/vabs/vmax/vsqrt/vrelu/vtanh/load/store）
> - **未映射的 op 会被跳过**（SCALAR op），不影响 msprof 但不产生 HIVM 事件
> - `tl.dot` → `tt.dot` → 我们的转换器映射为 `matmul`（CubeUnit），但 AscendC 代码生成跳过 CubeUnit
> - 标量索引（`tensor[0]`）被跳过——这是预期行为，不是 bug，避免转换过程中插入冗余兼容 op、触发通用降级路径，最大化消除转换侧性能损失。

- **标准 API 优先原则**：仅使用转换器原生支持的核心语法子集，包括 `tl.load/store`、标准算术运算、`tl.sum/max/min` 归约、`tl.dot` 矩阵乘、`tl.arange/broadcast_to/transpose` 维度操作、`tl.fma/rsqrt` 等专用指令；禁止使用实验性 API、自定义聚合类型、高阶函数，避免触发未识别 op 降级。
> **注意**：我们的 `ttir_to_hivm.py` 的 `ARITH_TO_HIVM` 映射以 `math.xxx` 为 key，
> 所以 `tl.math.rsqrt`、`tl.math.exp` 等 **必须使用 `tl.math.*` 命名空间**（已验证）。
> 仅 `tl.fma` 例外——我们的转换器没有 fma 映射，建议写 `x*w+b` 让 bisheng 自动融合。
- **消除转换期冗余**：移除无意义的恒等运算、空 reshape、重复类型转换，这类节点在 TTIR 中存在但无实际计算价值，转换器会插入兼容搬运 op，凭空增加 HIVM 算子数量与调度开销。
- **统一运算表达范式**：同一种运算仅使用一种标准写法，如倒数平方根统一用 `tl.math.rsqrt`、乘加写 `x*w + b` 让编译器自动融合（我们的转换器没有 fma 映射），避免转换器对不同写法生成质量差异较大的 HIVM 代码。
- **降低 IR 结构复杂度**：保持计算图扁平，减少嵌套分支与深层表达式嵌套，降低转换器解析难度，避免因结构复杂触发通用降级路径。

### 2. AscendC 代码生成质量优化
目标是让自研生成器输出简洁、规整、可被 bisheng 编译器深度优化的 AscendC 代码，避免生成冗余循环、临时变量与未对齐访存。

- **循环结构标准化**：统一使用静态 `for-range` 循环，生成器可直接识别并自动插入软件流水线、循环展开等硬件优化；非标准循环会生成朴素跳转代码，bisheng 无法做深度指令调度。
- **访存对齐显式化**：通过 `tl.multiple_of` 标注指针对齐属性，生成器可直接输出对齐的内存访问指令，触发 bisheng 的突发传输优化。
- **计算单元清晰映射**：矩阵运算用标准 `tl.dot`（映射 CUBE 单元）、逐元素运算用标准向量 op（映射 VECTOR 单元）、内存操作用标准 `load/store`（映射 MTE 单元），便于生成器直接映射到对应硬件指令集。
- **减少动态分支**：边界逻辑优先使用向量化 mask，避免多层 `if-else` 分支，生成器可输出无分支的向量掩码指令，提升流水线效率。

---

## 二、诊断触发规则
所有优化基于编译链路各阶段的输出数据驱动，满足以下阈值即触发对应优化。

### 2.1 核心触发规则
1. **代码膨胀判定（强制触发）**
   - 核心指标：**TTIR 字符数（chars）环比增长 ≥ 10%，但 HIVM 有效 op 数量无减少甚至增加**
   - 判定结论：TTIR 存在大量冗余节点、无效表达式与嵌套结构，属于无效代码膨胀，必须精简代码结构、消除冗余表达式
   - 辅助阈值：单 op 对应 TTIR 字符数 > 500 → IR 过于冗长，存在优化空间

2. **转换损耗判定**
   - 转换损耗率 = HIVM op 总数 / TTIR op 总数
   - 损耗率 > 1.1 → 轻度损耗，存在少量兼容冗余 op
   - 损耗率 > 1.2 → 重度损耗，已触发通用降级路径，必须立即优化
   - 硬触发：HIVM 中出现 `fallback_generic`、`unknown_op` 等降级标记 → 立即替换为标准 API 实现

3. **生成质量判定**
   - AscendC 生成阶段输出未识别 IR 节点、强制降级、动态分支告警 → 对应调整代码写法
   - 生成的 AscendC 代码行数 / HIVM op 数 > 20 → 代码过于冗余，存在精简空间

4. **模拟器完整性判定**
   - msprof 输出缺失 CUBE / VECTOR / MTE 任意一类管线事件 → 时序统计失真，必须调整代码映射方式

### 2.2 Tier 6 最优判定标准
同时满足以下所有条件，判定工具链适配层已达最优，全链路优化结束：
1. 转换损耗率 ≤ 1.05，几乎无转换开销
2. 无任何降级 op 与生成告警
3. TTIR 结构精简，无无效代码膨胀
4. msprof 输出 CUBE、VECTOR、MTE 三类管线事件完整，周期统计有效

---

## 三、代码风格与 IR 强制规范
所有 Triton 代码必须遵循以下规范，确保转换器与生成器最优解析，避免降级与损耗。

### 3.1 循环规范：强制使用静态 for-range
- 所有循环必须使用 `for i in range(start, end, step)` 静态范围循环，`start`/`end`/`step` 必须均为 `tl.constexpr` 编译期常量。
- 禁止使用 `while` 循环、动态边界循环、`break`/`continue` 跳出逻辑；自研转换器对静态 `for-range` 支持最完善，可自动做循环展开、软件流水线优化。
- 循环体保持扁平，嵌套循环不得超过 2 层，且内层外层均必须为静态 `for-range`。

### 3.2 形状规范：禁止动态 shape
- 所有分块尺寸、张量维度、循环边界必须声明为 `tl.constexpr`，编译期完全可确定。
- 禁止使用运行时动态参数作为 `tl.reshape`、`tl.transpose` 的维度参数，所有维度变换必须静态可推导。
- 禁止依赖运行时输入的动态 shape 分支，所有尺寸分支必须在编译期通过 constexpr 特化完成。
- 指针 stride 优先使用常量表达式，避免运行时变量作为步长。

### 3.3 IR 扁平化要求
- 条件分支层级不超过 2 层，优先使用 `tl.where` 向量化掩码替代多层 `if-else`。
- 计算表达式嵌套不超过 3 层，过长表达式拆分为 2~3 个有意义的中间变量，平衡 IR 复杂度与可读性。
- 禁止过度内联自定义辅助函数，避免单 kernel 的 IR 节点爆炸、结构混乱。
- 归约操作仅嵌套 1 层算术运算，禁止归约内部嵌套复杂函数调用。

### 3.4 显式化原则
- 所有类型转换必须显式使用 `.to(dtype)`，禁止依赖隐式类型转换（如 int 转 float、FP16 转 FP32），避免转换器插入冗余 cast op。
- 所有张量广播必须显式调用 `tl.broadcast_to`，禁止依赖形状不匹配的隐式广播。
- 所有指针运算采用「基址 + 偏移」的显式格式，避免复杂指针表达式导致转换器解析错误。

---

## 四、与 msprof 模拟器的配合规范
msprof op simulator 基于硬件管线事件统计周期，**必须同时产生 CUBE、VECTOR、MTE 三类管线事件，才能输出完整、精确的 timing 数据**，缺少任意一类都会导致周期统计缺失、性能评估失真。

### 4.1 三类管线事件对应关系
| 管线单元 | 对应运算类型 | Triton 标准 API | 统计作用 |
|----------|--------------|----------------|----------|
| CUBE | 矩阵乘、张量核心运算 | `tl.dot` | 矩阵乘法计算周期统计 |
| VECTOR | 逐元素算术、数学函数、归约 | `tl.add/mul/fma`、`tl.math.rsqrt/exp`、`tl.sum/max` | 向量计算周期统计 |
| MTE | 全局内存读写、数据搬运 | `tl.load`、`tl.store`（映射为 HIVM `gm_to_ub`/`ub_to_gm`）| 内存传输周期统计 |

### 4.2 模拟器适配规则
1. **标准 API 强制映射**：所有核心运算必须使用上表中的标准 API 实现，禁止用逐元素循环模拟矩阵乘、用标量运算模拟归约，否则生成器无法识别对应硬件单元，导致事件缺失。
2. **单 kernel 功能聚焦**：一个 kernel 内核心运算类型不超过 2 类，避免混合过多不同单元的运算导致模拟器调度统计混乱。
3. **分块匹配硬件粒度**：分块尺寸匹配 CUBE 单元最优计算粒度（16/32 的整数倍），访存对齐匹配 MTE 突发传输粒度（128 字节对齐），确保模拟器周期计算与真实硬件一致。
4. **消除无效运算**：禁止插入无副作用的空运算、恒等运算，避免产生无效管线事件，干扰周期统计。

> **在我们的 pipeline 中验证 Tier 6 效果**：
> 1. TTIR chars 减少但 HIVM ops 不增加 = 代码精简有效
> 2. AscendC Build OK + msprof 有 event = 工具链兼容
> 3. HIVM ops 数量不变但 msprof timing 变好 = 生成的 AscendC 代码质量提升
> 4. TTIR chars 增大但 HIVM ops 不变 = 代码膨胀 → 需要精简

### 4.3 事件完整性校验流程
1. 每轮优化后运行 msprof，检查输出中是否包含三类事件的独立周期统计
2. 若缺失某类事件：回溯对应运算的实现方式，替换为标准 API，确保生成器可正确映射
3. 若某类事件占比异常（如纯计算算子 MTE 事件占比过高）：检查是否存在转换期冗余访存，对应精简 IR

---

## 五、典型场景 Before/After 代码示例
所有示例语义完全等价，仅调整书写风格适配工具链，不改变算法与计算逻辑。

### 示例1：循环写法优化（while → 静态 for-range）
**Before（易触发降级的 while 循环）**
```python
@triton.jit
def loop_while_before(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = 0
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    # 动态边界while循环，转换器无法做流水线优化
    while i < n:
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) + i
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += x
        i += BLOCK_SIZE
    tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE), acc)
```

**After（适配工具链的静态 for-range）**
```python
@triton.jit
def loop_for_after(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr, NUM_ITER: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    base = pid * BLOCK_SIZE
    # 编译期确定迭代次数，转换器可自动生成流水线/展开代码
    for i in range(NUM_ITER):
        offsets = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        acc += x
    tl.store(out_ptr + base + tl.arange(0, BLOCK_SIZE), acc)
```

### 示例2：动态 shape → 全静态维度优化
**Before（动态维度，触发通用降级）**
```python
@triton.jit
def dynamic_shape_before(x_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < M
    # 转置维度运行时确定，转换器无法静态解析
    x = tl.load(x_ptr + rows[:, None] * N + tl.arange(0, 16)[None, :], mask=mask[:, None])
    x_t = tl.transpose(x)
    tl.store(out_ptr + tl.arange(0, 16)[:, None] * M + rows[None, :], x_t, mask=mask[None, :])
```

**After（全静态维度，原生支持）**
```python
@triton.jit
def static_shape_after(x_ptr, out_ptr, M, N,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask_m = rows < M
    mask_n = cols < N
    # 所有维度均为constexpr，转换器可静态解析转置操作
    x = tl.load(x_ptr + rows[:, None] * N + cols[None, :],
                mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    x_t = tl.transpose(x)
    tl.store(out_ptr + cols[:, None] * M + rows[None, :], x_t,
             mask=mask_n[:, None] & mask_m[None, :])
```

### 示例3：隐式广播 → 显式广播优化
**Before（隐式广播，转换插入冗余 op）**
```python
@triton.jit
def implicit_broadcast_before(x_ptr, scale, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 标量与向量隐式广播，转换器可能插入独立broadcast op
    out = x * scale
    tl.store(out_ptr + offsets, out, mask=mask)
```

**After（显式广播，转换零损耗）**
```python
@triton.jit
def explicit_broadcast_after(x_ptr, scale, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # 显式广播，转换器直接识别，无冗余op生成
    scale_vec = tl.broadcast_to(scale, (BLOCK_SIZE,))
    out = x * scale_vec
    tl.store(out_ptr + offsets, out, mask=mask)
```

---

## 六、常见错误与修复方案
### 1. while 循环导致转换降级
- **现象**：HIVM 出现循环跳转标记，转换损耗率 > 1.2，性能比 for-range 版本低 30% 以上
- **原因**：自研转换器对 while 循环支持不完善，无法做流水线和展开优化，生成通用跳转代码
- **修复**：全部替换为静态 `for-range` 循环，迭代次数通过 `tl.constexpr` 参数传入

### 2. 动态 shape 触发通用降级路径
- **现象**：msprof 中全是 VECTOR 通用事件，无专用指令优化，性能远低于预期
- **原因**：运行时动态维度无法被转换器静态解析，退化为通用标量运算，无法映射到专用硬件指令
- **修复**：所有维度、分块、步长全部声明为 `tl.constexpr`，确保编译期完全可确定

### 3. 隐式类型转换导致转换损耗过高
- **现象**：HIVM 中出现大量冗余 cast op，转换损耗率 > 1.1
- **原因**：Triton 前端自动插入隐式类型转换，转换器无法合并，生成独立算子
- **修复**：所有类型转换显式使用 `.to()`，保持运算过程中类型一致，减少不必要的转换

### 4. IR 嵌套过深导致解析错误
- **现象**：TTIR→HIVM 转换失败，或生成的 AscendC 代码逻辑错误
- **原因**：表达式嵌套超过转换器支持的深度，解析时出现逻辑遗漏或栈溢出
- **修复**：拆分复杂表达式为 2~3 级中间变量，保持 IR 结构扁平，嵌套不超过 3 层

### 5. 非标准实现导致 msprof 事件缺失
- **现象**：矩阵乘算子 msprof 输出缺少 CUBE 事件，只有 VECTOR 事件，周期统计严重失真
- **原因**：用逐元素循环模拟矩阵乘，未使用标准 `tl.dot`，生成器无法识别为 CUBE 运算
- **修复**：矩阵运算必须使用标准 `tl.dot` API，确保映射到 CUBE 单元，获得准确周期统计

### 6. 代码冗余膨胀导致编译效率下降
- **现象**：TTIR 字符数大幅增加但 HIVM op 数不变，编译时间变长，性能无提升
- **原因**：过度内联、冗余中间变量、无意义运算导致 TTIR 膨胀，转换器处理效率下降
- **修复**：精简代码，消除恒等运算和无意义中转变量，保持计算逻辑简洁清晰

需要我补充工具链各阶段的自动校验脚本模板，或者针对某类算子的 Tier 6 适配最佳实践吗？