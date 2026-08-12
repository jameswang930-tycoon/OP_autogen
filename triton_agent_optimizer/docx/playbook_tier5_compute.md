# Triton 优化 Tier 5（计算占用）策略指南 — 针对 triton-ascend 910B3

> ## ★★速查卡（★先读这个；你实际能读全文，但速查卡在最前面=最快定位，决策先靠它，细节看情况A~J）★★
>
> **工作流（3 步，别跳）**：
> 1. 读 `07_tier5_fields/tier5_fields.txt` + `planner_context.json` → 找 **aic_scalar_time / scalar / vector_fops / vec / bank_cflt_ratio / wait_ratio / cube 耗时 / bottleneck_type**
> 2. 照下表定位"情况"→ 看情况 A~J 代码
> 3. 输出 changes[]：old_code 逐字复制；拿不准不改（宁缺勿错）
>
> **字段 → 情况速查表（★主决策表）**：
> | 字段/现象 | 情况 | 动作（一句话） |
> |---|---|---|
> | scalar_time 高 + 指针含 div/mod 或显式 int64 | A | 消除 div-mod、int32 索引（bias_gelu 的 `offs % N` 是风险点） |
> | vec 利用率低 + 带宽低 | B | 一次向量 load（offs 连续+对齐） |
> | 数学多步手动组合（1/sqrt） | C | `tl.math.rsqrt` / `tl.math.tanh`（别用 tl.erf） |
> | mul+add 未融合 | D | 直接写 `x*w+b`（编译器自动 FMA；别用 tl.fma） |
> | 性能不升反降 / 寄存器溢出 | E | 控制展开/中间变量（2~3 级） |
> | bank_cflt_ratio >4% | F | swizzle/访问顺序（tier3 §三/tier4 §C）；多数已自动处理 |
> | 逐元素 x/s 除法 / 链式除法喂归约 | G | 倒数算一次 + 乘法（inv_N 在 host 算好） |
> | vec 高 cube 低 + 外积累加（conv 现状） | H | `tl.dot` 或分块 2D 累加（★别物化 3D 中间量） |
> | bank_cflt>4% 且尾轴非 32B/512B 倍数 | I | 尾轴对齐 + 显式 pad + mask |
> | cube 已满（compute_bound）且 scalar/conflict 不高 | J | ★停手：查 cube_fp16_ratio→回 Tier1 精度，否则停止/晋升 |
> | 上面全不触发 | — | 计算已优 → promote Tier6（附 evidence） |
>
> **★我们算子对照（本层能做什么）**：
> - `input/matmul`：bias_gelu 的 `offs % N` → 改 2D 索引（情况A）；除法检查（情况G）
> - `input/rms_norm` / `layernorm`：**rsqrt 已用**（`tl.math.rsqrt`，无 `/sqrt` 残留）→ 只剩 `/N → *inv_N`（情况G）；softmax 是 `e/s`（每行一次倒数提升，情况G）
> - `input/conv2d` / `conv_bias_relu`：外积累加 → tl.dot（情况H，配合 Tier1-G 的 im2col）
> - `input/attention_mlp`：scalar 检查（情况A）；exp/tanh 已用 math 命名空间（正确样例）
> - 其余：查 scalar_time / bank_cflt（A/F）
>
> **★字段位置提醒**：`vector_fops` / `vec` / `cube_fp16_ratio` / `bottleneck_type` **不在 07_tier5_fields.txt**（那里只含 cube_us/scalar_us/scalar/fixpipe/cube_ratio/conflict/wait）——这些在 `07_tier5_fields/planner_context.json` 的 `kernels[].deep` 里。看到 07 文件里"无数据"别误判采集失败，去 context json 找。
>
> **禁止（违反=失败）**：改算法/融合/分块/访存（跨层）、tl.erf、tl.fma、tensor.item() 热路径、累加器非 fp32、int64 索引、跨层"假装优化"。

---

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
| `conflict.bank_cflt_ratio` | >4~5% | 调整访问/swizzle + 尾轴对齐（情况F/I） |
| 性能不升反降 | 寄存器溢出 | 控制展开/ILP（情况E） |
| `aic_scalar_time` 高且代码含逐元素 `x/s` 除法 | 标量除法拖累 | 倒数+乘法（情况G） |
| `vec` 高 `cube` 低 + 外积累加（conv2d 现状） | 3D 广播展开 / 没上 cube | 分块 2D 累加或 tl.dot（情况H） |
| `bank_cflt_ratio`>4% 且尾轴非 32B/512B 倍数 | 自动 padding / bank 冲突 | 尾轴对齐 + mask（情况I） |
| `cube` 已满（compute_bound）且 scalar/conflict 不高 | 计算已到硬件上限 | 停手/晋升 Tier1 精度或停止（情况J） |

### ★决策流程图
```
scalar_time/ratio 高 ?
  ├─ 指针 div/mod / int64 索引 → 改 int32/消除 (情况A)
  ├─ 逐元素标量加载 → 向量化 (情况B)
  └─ 逐元素 x/s 除法 → 倒数+乘法 (情况G)
数学运算慢 ?
  ├─ 1/sqrt → tl.math.rsqrt; erf → tanh (情况C)
  ├─ mul+add 没自动融合 → 直接 x*w+b (情况D)
  └─ vec 高 cube 低 + 外积累加 → tl.dot / 分块 2D 累加 (情况H)
bank_cflt_ratio >4% → 访问调整/尾轴对齐 (情况F/I)
性能不升反降/寄存器溢出 → 控制展开/ILP (情况E)
cube 已满且 scalar/conflict 不高 → 停手/晋升 (情况J)
否则 → 晋升 Tier6
```

---

## 情况A：消除标量降级（int64 索引 / div-mod 指针）★最容易被忽略

**触发**：`scalar_time`/`scalar` 占比高，但算法/分块都合理。
**收益**：消除标量降级后向量化执行 → 显著提速（昇腾上指针里的 div/mod、显式 int64 索引算术会把整个 load 降级成标量循环）。
**怎么查**：搜 "triton-ascend scalar degradation int64 mask" / "triton div mod pointer scalar"。

### ❌ 问题示例代码（指针里 div-mod / 显式 int64 索引 → 标量降级）
```python
# ❌ 指针里 % 运算: 如 bias_gelu 的 offs % N → 非结构化标量寻址 (逐元素标量 load)
offs = tl.arange(0, BLOCK)                        # tl.arange 默认 int32
b = tl.load(bias_ptr + (offs % N), mask=offs < N, ...)   # % → 标量逐元素寻址
# ❌ 显式 int64 索引 (如 .to(tl.int64)): 64-bit 索引算术 + 比较 → 标量降级
offs64 = tl.arange(0, BLOCK).to(tl.int64)
x = tl.load(x_ptr + offs64, mask=offs64 < n_elements, ...)   # ← int64 比较/寻址 → 标量降级
```
**出现的问题**：昇腾对 div/mod 寻址、显式 int64 索引算术会把 load/store 降级成**逐元素标量循环**（不再走 NDA 向量指令），带宽和吞吐崩。（注意：`tl.arange` 默认就是 int32，普通 int32 mask 比较是向量化的，不是问题所在。）

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

## 情况G：标量除法 → 乘法倒数（div → mul）★逐元素除法拖累

**触发**：`aic_scalar_time`/`scalar` 高，代码里有逐元素 `x / s`（s 是标量/广播常量），或链式除法喂归约。
**收益**：昇腾 fp32 除法**没有单指令**（硬件无 div 指令，编译器展开成多条序列指令）；逐元素除法直接把指令数放大。改成"一次倒数（标量）+ 逐元素乘法"后，热路径只剩原生向量乘法。GPU 侧实测链式除法喂归约会 PTX 爆炸（div 指令 2080→8 条）；昇腾同理会放大标量/向量指令数。
**怎么查**：搜 "triton division reciprocal multiply faster" / "ascend fp32 div instruction sequence"。

### ❌ 问题示例代码（逐元素除法 / 链式除法喂归约）
```python
# ❌ 逐元素除法: 每个元素一条 fp32 div (昇腾无单指令 → 序列指令, 慢)
y = x / scale
# ❌ 链式除法喂归约: 两次逐元素 div 再归约 → 除法指令加倍 + 编译器冗余展开
y2 = x / s
w = y2 / s
denom = tl.sum(w, axis=0)
```
**出现的问题**：昇腾 fp32 除法是序列指令，逐元素除法把"一次倒数+乘法"变成 N 次序列展开；链式除法喂归约还会触发编译器对归约的过度展开（GPU 端就是 2080 条 div 的典型案例）。`scalar_time`/指令数虚高。

### ✅ 修改后正确代码（倒数提升成标量，逐元素只乘）
```python
# ✅ 倒数算一次 (标量, loop 外/不变量), 逐元素只做乘法 (原生向量指令)
inv_scale = 1.0 / scale
y = x * inv_scale
# ✅ 链式同理: 倒数只算一次, 全部用乘法
r = 1.0 / s
y2 = x * r
w = y2 * r
denom = tl.sum(w, axis=0)
# ✅ rms_norm 的 mean(x²)/N 同理: 除 N → 乘 inv_N
ms = tl.sum(x * x, axis=0) * inv_N        # inv_N = 1.0 / N 在 kernel 外算好
```
**约束/坑**：精度差异 ~1 ulp（已 CPU 验证：逐元素 div vs 倒数乘法 rel≈8.6e-8、链式归约 rel≈2.1e-7、`sum/N` vs `sum*inv_N` 全等，fp32 可接受）；**仅当除数是标量/loop 不变量时收益最大**（每元素不同除数不能换）；`inv_N` 这类 kernel 外常量在 host 端算好。

---

## 情况H：外积累加别物化 3D 中间张量（广播展开 → 分块 2D 累加 / tl.dot）

**触发**：`engine_utilization.vec` 高、`cube` 低，代码是 `acc += wv[:, None] * xv[None, :]` 这类外积，或 `tl.sum(x[:, :, None] * w[None, :, :], axis=1)` 3D 广播归约（conv2d/conv_bias_relu 现状，见 tier4 案例3）。
**收益**：一次性 3D 广播展开的中间量在 UB 直接溢出（BLOCK=64 时 64³×4B=1MB >> 192KB）；改分块 2D 累加后峰值中间量降到 `[BLOCK_M, BLOCK_N]`（16KB）。能上 cube 一律 `tl.dot`（Tier6-D）。
**怎么查**：搜 "triton broadcast expansion reduce intermediate memory" / "triton outer product accumulation vector".

### ❌ 问题示例代码（一次性 3D 广播展开再归约）
```python
# ❌ y[m,n] = Σ_k x[m,k]*w[k,n] 用 3D 广播一次展开:
#    x[:,:,None]*w[None,:,:] → [M,K,N] 中间量
y = tl.sum(x[:, :, None] * w[None, :, :], axis=1)
#    BLOCK_M=BLOCK_K=BLOCK_N=64: 64³×4B = 1024KB fp32 >> UB 192KB
```
**出现的问题**：`[M,K,N]` fp32 中间量 = `M·K·N×4B`。64³ 就是 1MB，远超 UB 192KB → 编译 `ub overflow` 或退化成逐元素慢跑；即使分块到 `BK=16`，`[64,16,64]×4B=256KB` 仍超 UB。已 CPU 实测三种写法等价但中间量天差地别。

### ✅ 修改后正确代码（上 cube 或分块 2D 累加）
```python
# ✅ 正确1 (首选, 上 cube): tl.dot → 编译器走 Cube mmad, 无 3D 中间量 (见 Tier6-D)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)    # [BM, BK]
    b = tl.load(b_ptrs, mask=..., other=0.0)    # [BK, BN]
    acc = tl.dot(a, b, acc)                     # 累加器 fp32
# ✅ 正确2 (纯 vector 兜底): 逐 kk 的 2D 外积步进, 峰值中间量 = [BM, BN]
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for kk in range(BLOCK_K):
    acc += xb[:, kk, None] * wb[kk, None, :]    # 每步 [BM,BN]×4B = 16KB ✓ (双缓冲 ÷2 仍够)
```
**约束/坑**：纯 vector 的外积累加指令多（BLOCK_K 步），是 tl.dot 的兜底不是首选；能上 cube 一定 `tl.dot`；`acc` 累加器必须 fp32；BLOCK 16 倍数；已 CPU 验证（3D 展开/分块 2D/参考 matmul 三者 rel<3e-7 一致，性能数据需真机 msprof 确认）。

---

## 情况I：bank 冲突/自动 padding 的落地手段（尾轴 32B/512B 对齐）

**触发**：`conflict.bank_cflt_ratio` >4%，或 triton-ascend 编译期自动 padding 告警（尾轴太短，如 shape `(2048,3)`/`(2048,1)`）。
**收益**：尾轴对齐避免编译器自动 padding（官方明确警告自动 padding 会显著降性能）；对齐到 32B(VV)/512B(CV) 的倍数让 UB 访问更整，bank 冲突概率下降。
**怎么查**：搜 "ascend avoid ub bank conflict padding" / "triton-ascend tail axis 512B alignment".

### ❌ 问题示例代码（尾轴非 32B/512B 倍数 → 自动 padding）
```python
# ❌ 尾轴(innermost 维)元素数太小/非对齐倍数:
#    triton-ascend 对 VV 要求尾轴 ≥32B、CV(含 cube) 要求 ≥512B
#    shape (2048, 3) fp32 → 尾轴 12B, 编译器自动 padding → 冗余搬运/计算
BLOCK_N = 3      # fp32: 3×4=12B, 非 32B 倍数
# ❌ 逐元素 kernel 同理: 每行只剩 1 个元素 → (2048,1) 型自动 pad
```
**出现的问题**：尾轴不足 32B/512B 时 triton-ascend 自动 padding，产生冗余搬运和计算；UB bank 并发访问落在同一 bank（bankgroup 不合理）时带宽掉到 1/N——两者都是 `bank_cflt_ratio` 高/性能掉的表现。

### ✅ 修改后正确代码（尾轴对齐 + 显式 pad + mask）
```python
# ✅ 尾轴对齐到 32B(VV)/512B(CV) 倍数:
#    fp32: VV ≥ 8 元素(32B); CV ≥ 128 元素(512B)   → BLOCK_N=128 ✓
#    fp16: VV ≥ 16 元素(32B); CV ≥ 256 元素(512B)  → BLOCK_N=256 ✓
BLOCK_N = 128     # fp32, 512B 对齐 (配合 tier3 §二-A 的 L0C 约束)
# ✅ 逻辑 shape 主动 pad 到对齐倍数 + mask, 避免编译器自动 padding:
#    host 端把 N=3 的权重 pad 成 16(32B 对齐), kernel 内 mask 掉尾部 (见 情况B)
offs = tl.arange(0, BLOCK_N)
mask = offs < N
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
```
**约束/坑**：Triton 里**不能像 Ascend C 那样给 UB 手动 +2 padding**（UB 布局由编译器决定）——用户侧能做的是：①尾轴对齐（上面）；②innermost stride=1（tier4 情况A）；③让编译器 swizzle；④真遇到 vector 转置类 bank 冲突，考虑 host 端预转置（tier4 案例1/2）。已 CPU 验证索引数学（pad+mask 取回 rel=0），性能收益需真机 msprof 确认。

---

## 情况J：cube 已满（compute_bound）→ 停手/晋升，别再试指令级改动

**触发**：`aic_cube_time`/`cube_ratio` 高（cube 忙满），且 `scalar`/`conflict` 不高；`roofline.bottleneck_type=compute_bound`。
**含义**：cube 已 100% 忙 = 计算到硬件上限，指令级改动（ILP/展开/除法替换）在 cube 时间上无效——cube 不关心标量/向量那几拍，只在乎矩阵 MAC。
**怎么查**：搜 "compute bound matmul no benefit instruction tuning"（tier3 §二-C 同款结论）。

### ❌ 错误动作（cube 满了还在试指令级优化）
```python
# ❌ cube 已满还去调 ILP/展开/倒数替换 → 轮次白费 (cube 时间由 MAC 决定)
for k in range(0, K, BLOCK_K):
    acc = tl.dot(a, b, acc)      # cube MAC 已占满, 改循环写法不省 cube 时间
```
**出现的问题**：cube 时间由矩阵 MAC 决定，指令级微调（标量/向量那几拍）不改变 cube 占用 → 改了也白改，还消耗优化轮次（与 tier3 §二-C "compute_bound 别调分块"、tier4 "rms_norm 17 轮无果" 同类）。

### ✅ 正确动作（收尾）
```python
# ✅ 查 cube_fp16_ratio: 该用 fp16 没用 → 回 Tier1 (精度层), 这是唯一能再省 cube 时间的
#    (cube fp16 吞吐是 fp32 的 ~2×, 但改动精度必须 Tier1 决策, 本层只指路)
# ✅ 否则 → 本算子计算已最优, 停止/晋升 (别在本层空转)
```
**约束/坑**：cube 已满时唯一能省 cube 时间的是"降精度/降指令量"（Tier1）；本层只负责判断"该停手"，与 tier3 §二-C、tier4 实战教训（rms_norm 17 轮无果）是同一铁律。需真机 msprof 确认 cube_ratio 确实满、且非误报。

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| 指针 div-mod / 显式 int64 索引 | 标量降级性能崩 | 消除 div-mod、int32 索引（情况A） |
| 逐元素标量 load | 带宽 ~10% | 一次向量 load（情况B） |
| 手动 1/sqrt | 多 1 指令 | tl.math.rsqrt（情况C） |
| tl.erf | 编译失败 | tl.math.tanh（情况C） |
| 显式 tl.fma | 编译失败 | 直接 x*w+b（情况D） |
| 过度展开 | 寄存器溢出更慢 | 控制展开/中间变量（情况E） |
| tensor.item() 热路径 | CPU-NPU 同步 | 避免 |
| 归约不升精度 | 精度崩 | 累加器/归约 fp32 |
| 逐元素 x/s 除法 / 链式除法喂归约 | 标量/指令数虚高 | 倒数+乘法，倒数算一次（情况G） |
| 外积累加 3D 广播展开 | UB 溢出 / 极慢 | tl.dot 或分块 2D 累加（情况H） |
| 尾轴非 32B/512B 倍数 | 自动 padding / bank 冲突 | 尾轴对齐 + 显式 pad + mask（情况I） |
| cube 已满还在调指令 | 轮次白费 | 查 cube_fp16_ratio→Tier1，否则停手（情况J） |
