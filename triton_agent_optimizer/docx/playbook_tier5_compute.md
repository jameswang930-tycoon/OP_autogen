# Triton 优化 Tier 5（计算占用）策略指南 — 针对 triton-ascend 910B3

> 本层在 Tier1~4 之后：**指令级计算优化**（向量化/原生指令/ILP/冲突/标量），**不改算法/融合/分块/访存模式**。
> 昇腾计算优化的核心：**引导编译器做对向量化加载、原生数学指令、循环展开**，消灭标量降级和计算空等。
>
> **★环境铁律（triton-ascend）**：
> - `num_warps`/`num_stages` **禁止**传；不用 `@triton.autotune`
> - **数学函数用 `tl.math.*`**（`tl.math.tanh`/`tl.math.rsqrt` 已验证可行）；**别用 `tl.erf`**
> - **mask/指针别用 int64/div/mod**（会触发标量降级 → 性能崩）
> - 累加器/归约 fp32；热路径**别用 `tensor.item()`**（CPU-NPU 同步）
> - 优先级：**正确性 > 泛化性 > 性能**

---

## 一、诊断触发规则（v4 字段 → 动作）

看 `07_tier5_fields.txt`（含全局摘要）：

| v4 字段 | 触发 | 动作 |
|---|---|---|
| `task.pipes_us.aic_scalar_time_us` / `engine_utilization.scalar` | 高（标量拖累） | 消除标量降级（情况A） |
| `compute.vector_fops` / `vec` 利用率 | 低 | 向量化加载（情况B） |
| 数学运算慢（cube 已满） | 用了 1/sqrt 等手动组合 | 原生指令（情况C） |
| `compute.cube_fp16_ratio` | 低 | 精度（回 Tier1 A，本层引用） |
| `conflict.bank_cflt_ratio` | >4~5% | 调整访问/swizzle（情况F） |
| 性能不升反降 | 寄存器溢出 | 控制展开/ILP（情况E） |

### ★决策流程图
```
scalar_time/ratio 高 ?
  ├─ mask/指针用了 int64/div/mod → 改 int32/消除 (情况A)
  └─ 逐元素标量加载 → 向量化 (情况B)
数学运算慢 ?
  ├─ 1/sqrt → tl.math.rsqrt; erf → tanh (情况C)
  └─ mul+add 没自动融合 → 直接 x*w+b (情况D)
bank_cflt_ratio >4% → 访问调整 (情况F)
性能不升反降/寄存器溢出 → 控制展开/ILP (情况E)
否则 → 晋升 Tier6
```

---

## 情况A：消除标量降级（int64 mask / div-mod 指针）★最容易被忽略

**触发**：`scalar_time`/`scalar` 占比高，但算法/分块都合理。
**收益**：消除标量降级后向量化执行 → 显著提速（昇腾上 mask 用 int64 会把整个 load 降级成标量循环）。
**怎么查**：搜 "triton-ascend scalar degradation int64 mask" / "triton div mod pointer scalar"。

### ❌ 问题示例代码（mask/指针 int64 或 div-mod → 标量降级）
```python
# ❌ offs 默认 int64 → mask `offs < cols` 触发标量比较; 或指针里 div/mod
offs = tl.arange(0, BLOCK)                        # int64
mask = offs < n_elements                          # ← int64 比较 → 标量降级
x = tl.load(x_ptr + offs, mask=mask, ...)
# ❌ 指针里 % 运算: 如 bias_gelu 的 offs % N → 非结构化标量寻址
b = tl.load(bias_ptr + (offs % N), mask=mask, ...)   # % → 标量逐元素寻址
```
**出现的问题**：昇腾对 int64 比较 / div-mod 寻址会把 load/store 降级成**逐元素标量循环**（不再走 NDA 向量指令），带宽和吞吐崩。

### ✅ 修改后正确代码（int32 + 消除 div-mod）
```python
# ✅ offs 显式 int32; mask 用 int32 比较 → 向量化
offs = tl.arange(0, BLOCK).to(tl.int32)
mask = offs < n_elements
x = tl.load(x_ptr + offs, mask=mask, ...)
# ✅ 避免指针里的 %: 把 bias 按列广播改成 2D 索引 (offs_n 直接乘 stride)
b = tl.load(bias_ptr + offs_n, mask=offs_n < N, ...)   # 没有 %, 连续寻址
```
**约束/坑**：`tl.arange` 默认 int32（我们 bias_gelu 的 `offs % N` 是风险点，改 2D 索引）；mask 用 int32。

---

## 情况B：向量化加载（逐元素标量 load → 一次向量 load）

**触发**：`vector_fops`/`vec` 利用率低，带宽远低于峰值（Memory Throughput 低）。
**收益**：向量加载（128-bit）vs 标量逐元素，带宽利用率差 5~10 倍。
**怎么查**：搜 "triton vectorized load scalar load bandwidth"。

### ❌ 问题示例代码（逐元素标量加载）
```python
# ❌ 循环里逐个元素 load → 每次 1 个 float, 无向量化
for i in range(BLOCK):
    val = tl.load(x_ptr + offsets[i])   # 标量 load × BLOCK 次 → 带宽极低
    ...
```
**出现的问题**：逐元素标量 load 一次只搬 4 字节，不走向量加载指令 → 带宽利用率 ~10%。

### ✅ 修改后正确代码（一次向量 load）
```python
# ✅ 一次 load 整个 BLOCK 向量 → 编译器生成 128-bit 向量加载
offs = tl.arange(0, BLOCK).to(tl.int32)
mask = offs < n_elements
x = tl.load(x_ptr + offs, mask=mask, other=0.0)   # 向量 load, 连续 → 向量指令
```
**约束/坑**：offs 连续、对齐；2D 时最快维匹配布局（tier4 情况A）。

---

## 情况C：原生数学指令（rsqrt / tanh，别用手动组合）

**触发**：数学运算多步手动组合（如 `1.0 / tl.sqrt(x)`），`scalar_time` 高。
**收益**：rsqrt 一条指令替代"开方+除法"两条，周期降 40~60%。
**怎么查**：搜 "triton tl.math.rsqrt native instruction"。

### ❌ 问题示例代码（手动组合 + erf 不支持）
```python
# ❌ 两步: 先开方再除法 → 2 条指令
inv_std = 1.0 / tl.sqrt(var + eps)
# ❌ tl.erf 在 triton-ascend 可能不支持 → 编译失败
out = tl.erf(x)
```
**出现的问题**：手动组合多 1 条指令；`tl.erf` 不确定支持 → 编译风险。

### ✅ 修改后正确代码（原生指令）
```python
inv_std = tl.math.rsqrt(var + eps)        # ✅ 单条原生指令 (与我们 tl.math.tanh 同命名空间)
# 激活用 tanh 近似 (已验证可行):
cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845 * (val + 0.044715 * val*val*val)))
```
**约束/坑**：`tl.math.*` 命名空间（`tl.math.rsqrt`/`tl.math.tanh` 验证可行）；fp32 计算；精度差 >1e-6 回退。

---

## 情况D：FMA（mul+add 自动融合）

**触发**：HIVM 里 `vmul` + `vadd` 独立两条，未自动融合。
**收益**：FMA 一条指令（1 次舍入），省 1 条。
**怎么查**：搜 "triton fma automatic fusion mul add"。

### ❌ 问题示例代码（显式 tl.fma 可能不支持）
```python
# ❌ 显式 tl.fma 在我们的转换器可能不识别 (ARITH_TO_HIVM 无 fma 条目)
out = tl.fma(x, w, b)   # → 未映射 TTIR op → 编译失败
```
**出现的问题**：`tl.fma` 的 TTIR 映射缺失 → 编译失败；或强制 FMA 改变了舍入语义。

### ✅ 修改后正确代码（直接 x*w+b 让编译器自动融合）
```python
out = x * w + b    # ✅ bisheng 自动融合为 FMA (1 次舍入)
```
**约束/坑**：直接写 `x*w+b` 最稳；仅当 HIVM 明确显示两条独立且未融合时才考虑显式干预。

---

## 情况E：寄存器溢出 / ILP（过度展开的代价）

**触发**：性能不升反降，或编译报寄存器溢出（spill）。
**收益**：控制展开/中间变量，平衡 ILP 与寄存器压力。
**怎么查**：搜 "triton loop unroll register spill" / "triton ILP instruction scheduling"。

### ❌ 问题示例代码（过度展开 → 寄存器溢出）
```python
# ❌ 手动展开 8 次 K 循环 → 寄存器暴涨 → spill 到 GM → 更慢
for k in range(0, K, BLOCK_K):    # 编译器展开过多 → VGPR/寄存器溢出
    acc = tl.dot(tl.load(a_ptrs + k*...), tl.load(b_ptrs + k*...), acc)
```
**出现的问题**：过度展开（unroll）导致寄存器溢出，数据被写回慢速内存 → 性能不升反降（web：AMD 都专门修过 unroll 导致 spill 的问题）。

### ✅ 修改后正确代码（让编译器平衡展开，控制中间变量）
```python
# ✅ 保持循环原样 (编译器自动决定展开); 中间变量只留 2~3 级
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BLOCK_K; b_ptrs += BLOCK_K
# 需要强展开时用 tl.static_range(0, K, BLOCK_K) (提示编译器, 但监控寄存器)
```
**约束/坑**：展开 vs 寄存器是权衡；spill 就回退；中间变量控制 2~3 级。

---

## 情况F：bank 冲突（UB 访问冲突）

**触发**：`conflict.bank_cflt_ratio` >4~5%。
**收益**：消除冲突后访存效率提升。
**怎么查**：搜 "triton bank conflict shared memory swizzle"。

### ❌ 问题示例（同一 bank 同时访问）
```python
# ❌ 多个核/线程同时读同一 bank 的 UB 地址 → N-way 冲突 → 带宽掉到 1/N
# 表现为 conflict.bank_cflt_ratio 高
```
**出现的问题**：UB 有多个 bank，若并发访问落在同一 bank → 冲突 → 吞吐掉到 1/N。

### ✅ 修改后正确代码（访问顺序调整 / swizzle）
```python
# ✅ 调整访问顺序或分块 (swizzle 消除 bank 冲突, 见 tier3 §三 / tier4 §C)
# 编译器也自动 swizzle layout 最小化冲突 (triton-ascend 有相关优化)
# 手动时: 错开 bank 索引 (让并发访问落在不同 bank)
```
**约束/坑**：先看 `conflict.bank_cflt_ratio` 是否真高；多数情况编译器已自动处理，别手动硬调。

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| int64 mask/指针 div-mod | 标量降级性能崩 | int32 + 消除 div-mod |
| 逐元素标量 load | 带宽 ~10% | 一次向量 load |
| 手动 1/sqrt | 多 1 指令 | tl.math.rsqrt |
| tl.erf | 编译失败 | tl.math.tanh |
| 显式 tl.fma | 编译失败 | 直接 x*w+b |
| 过度展开 | 寄存器溢出更慢 | 控制展开/中间变量 |
| tensor.item() 热路径 | CPU-NPU 同步 | 避免 |
| 归约不升精度 | 精度崩 | 累加器/归约 fp32 |
