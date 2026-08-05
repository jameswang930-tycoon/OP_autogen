# Triton 优化 Tier 2（算子融合）策略指南 — 针对 triton-ascend 910B3

> 本层在 Tier1（算法）之后：**合并相邻算子，消除中间 GM 往返**，不改变算法逻辑。
> 本层**只做融合**，**不改算法选择**（Tier1）、**不调分块**（Tier3）。
>
> **★环境铁律（triton-ascend，违反必报错）**：
> - `num_warps` / `num_stages` **禁止**传给 kernel；不用 `@triton.autotune`
> - 激活用 **`tl.math.tanh`**，不用 `tl.erf`
> - 融合后单块数据量 ≤ UB 192KB（超了 `ub overflow`）
> - **epilogue 融合的 N 维要 32B 对齐**（fp16 时 BLOCK_N×2 是 32 的倍数；否则 UB 越界写）

---

## 一、诊断触发规则（v4 字段 → 融合决策）

看 `07_tier2_fields.txt`（含全局摘要 + 融合字段）：

| v4 字段 | 触发 | 融合决策 |
|---|---|---|
| `summary.num_kernels` | >1 且各 kernel 是 matmul→逐元素→matmul 链 | 把逐元素并进 matmul epilogue（情况A/B） |
| `summary.api_overhead_total_us` | 大（launch 开销占端到端比例高） | 合并 kernel 减 launch |
| **全局摘要** `roofline.bottleneck_type` | `memory_bound` 且中间张量大 | 消除中间 GM 往返（情况A/B） |
| `kernels[].task.pipes_us` | 中间 kernel 耗时大且是纯搬运/逐元素 | 值得融合 |
| `main_mem_read/write_gb_s` | 高但算力利用率低 | 中间量在 GM 来回 → 融合 |

### ★决策流程图
```
多 kernel 且中间有 GM 往返 ?
  ├─ matmul 后跟 独立 bias/激活 → 并进 epilogue (情况A)
  ├─ 大 kernel 后跟 独立 add(残差) → 并进 epilogue (情况B)
  ├─ 同一张量被读多次 → 单次 load 复用 (情况C)
  └─ attention 内 scale/mask/softmax 是独立步骤 → 并入 QK^T/softmax (情况F)
memory_bound 但算力利用率低 ?
  ├─ 有隐式格式转换(GEMM 输出格式≠下游输入) → 消除转换 (情况G)
  └─ 中间量无法消除 ? → 检查融合后 UB/对齐/寄存器 (情况D/E)
无可融合 → 晋升 Tier3
```

---

## 情况A：激活/偏置并入 matmul epilogue（★省中间 Z 的 GM 往返，我们 MLP/attention 最相关）

**触发**：matmul 后跟独立逐元素 kernel（我们的 MLP：`fc1 → bias_gelu(独立) → fc2`），`main_mem` 流量大。
**收益**：省中间 Z 的写+读 GM（Z[2048²] fp32 = 16MB 写 + 16MB 读 = 32MB 无谓流量）。
**怎么查**：搜 "triton matmul epilogue bias gelu fused" / "fused matmul epilogue UB alignment"。

### ❌ 问题示例代码（独立 bias_gelu kernel，中间 Z 落 GM）
```python
# ❌ fc1 把结果 Z 写回 GM, bias_gelu 再读回来
matmul_kernel[g](x, w1, z, ...)          # ① Z = X@W1 → 写 GM [M,H] (16MB)
bias_gelu_kernel[g](z, b1, h, ...)       # ② 读 Z(16MB) → GELU → 写 H → GM
matmul_kernel[g](h, w2, y, ...)          # ③ 读 H
# Z 的 32MB 无谓 GM 往返, memory_bound 下浪费大
```
**出现的问题**：Z 是中间结果，写 GM 再读回 = 2 次无谓传输（memory_bound 时占瓶颈）。3 个 kernel → 2 次 launch 开销。

### ✅ 修改后正确代码（bias+gelu 并进 fc1 的 epilogue）
```python
@triton.jit
def fc1_gelu_kernel(a_ptr, w_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=..., other=0.0)
        w = tl.load(w_ptrs, mask=..., other=0.0)
        acc = tl.dot(a, w, acc)                       # matmul
        a_ptrs += BLOCK_K; w_ptrs += BLOCK_K
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = acc + bias[None, :]                         # ✅ bias 在 epilogue
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845 * (val + 0.044715 * val*val*val)))
    y = val * cdf                                     # ✅ GELU 在 epilogue
    tl.store(c_ptrs, y, mask=...)                     # 直接写最终结果, Z 不进 GM
```
**约束/坑**：
- **N 维 32B 对齐**（web 实测：epilogue 融合时 N 不满足 32B 对齐 → UB 越界写）。fp16 时 `BLOCK_N×2` 是 32 倍数；fp32 时 `BLOCK_N×4`
- 融合后单块 = acc(L0C) + bias + val + y（UB/L0 压力↑）→ 监控 UB，超了减小 BLOCK 或拆
- 我们的 `mlp_gelu_kernel` 已是这个模式（正确样例）

---

## 情况B：残差并入前一 kernel epilogue（省 O 的往返）

**触发**：大 matmul 后跟独立 `add` kernel（我们 attention 的 `Out = Z + O` 是独立 `add_kernel`）。
**收益**：省 Z/O 的一次 GM 往返。
**怎么查**：搜 "triton residual add fused epilogue"。

### ❌ 问题示例代码（独立 add kernel）
```python
matmul_kernel[g](y, w2, z, ...)    # Z = Y@W2 → GM
add_kernel[g](z, o, out, ...)      # 读 Z + 读 O → out → GM   (残差多一次往返)
```
**出现的问题**：残差 `+O` 单独一个 kernel，Z 和 O 都要从 GM 读回再写 out = 多 2 次 GM 传输 + 1 次 launch。

### ✅ 修改后正确代码（残差并入 Z 的 epilogue）
```python
@triton.jit
def fc2_residual_kernel(a_ptr, w_ptr, res_ptr, c_ptr, M, N, K, ...):
    ...
    acc = tl.dot(a, w, acc)       # Z = Y@W2
    c_ptrs = c_ptr + (offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn)
    res = tl.load(res_ptr + 同样的偏移, mask=...)          # ✅ 残差 O 直接在 UB
    tl.store(c_ptrs, acc + res, mask=...)                  # ✅ Z+O 一次写
```
**约束/坑**：残差张量 O 必须与 acc 同 tile 对齐（同 offs_m/offs_n）；`res` load 与 `acc` 的 tile 形状一致。

---

## 情况C：冗余 Load 融合（同一张量读多次）

**触发**：同一指针同一偏移被 load ≥2 次（中间无 store）。
**收益**：减 GM 读次数。
**怎么查**：搜 "triton redundant load register reuse"。

### ❌ 问题示例代码（x 读两次）
```python
x1 = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))   # 第一次
row_max = tl.max(x1, axis=0)
x_exp = tl.exp(x1 - row_max)
x2 = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))   # ← BUG: 重复读
out = x2 * x_exp / tl.sum(x_exp, axis=0)
```
**出现的问题**：同一地址 load 两次 = 2 次 GM 读（或 L1 命中但多一次指令），冗余。

### ✅ 修改后正确代码
```python
x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))   # 读一次, 寄存器复用
row_max = tl.max(x, axis=0)
x_exp = tl.exp(x - row_max)
out = x * x_exp / tl.sum(x_exp, axis=0)
```
**约束/坑**：仅当两次 load 之间无对 x 的写；mask/偏移一致。

---

## 情况D：算术链融合（精度风险）

**触发**：RAW 链上多步纯算术（平方→求和→开方→除法），中间无 store。
**收益**：依赖链压缩，减 op 数。
**怎么查**：搜 "triton fused arithmetic chain precision"。

### ❌ 问题示例代码（融合时改了浮点顺序 → 精度偏差）
```python
# ❌ 为"简化"重排了运算顺序
inv_std = tl.math.rsqrt(tl.sum(x*x, axis=0)/n_cols + eps)    # 先平方和再开方, 顺序对
out = x * inv_std
# 但若改成: inv_std = tl.math.rsqrt(tl.sum((x/n_cols)**2, axis=0) ...) ← 改顺序, 精度变
```
**出现的问题**：融合时**重排浮点运算**会改变舍入 → 与参考差 >1e-6，`MATCH_VERIFY` 失败。融合只允许"合并表达式"，**禁止改运算顺序/结合性**。

### ✅ 修改后正确代码（保序融合）
```python
# ✅ 只合并中间变量, 严格保持原运算顺序
x_sq = x * x
inv_std = tl.math.rsqrt(tl.sum(x_sq, axis=0) / n_cols + eps)   # 顺序不变
out = x * inv_std
```
**约束/坑**：FP32 下融合前后最大绝对误差 ≤ 1e-6；归约（sum/max）不参与逐元素融合。

---

## 情况E：过度融合（寄存器/UB 溢出）

**触发**：想融合更多，但融合后单块张量太多/太大。
**怎么查**：搜 "triton fused kernel register spill UB"。

### ❌ 问题示例代码（融合过多输入 → 寄存器溢出）
```python
# ❌ 一个 kernel 塞 6+ 个输入张量 (a,b,c,d,e,f) + 计算 → 寄存器/L0C 溢出 → 性能反而降
y = f(a,b,c,d,e,f, acc, res, ...)    # 寄存器爆 → spill 到 GM → 更慢
```
**出现的问题**：融合收益有上限——**融合过多输入导致寄存器/L0C 溢出**（web：PyTorch 模板 epilogue 融合在 XPU 因寄存器溢出无法提速）。单块张量数 ≥6 或数据量 ≥64KB 时评估后再融。

### ✅ 修改后正确代码（控制融合深度）
```python
# ✅ 只融合访存开销最高的相邻 2~3 级 (bias+激活), 不贪多
# 溢出信号: 编译报 register spill / L0C overflow → 减小 BLOCK 或拆融合链
```
**约束/坑**：融合深度 2~3 级；监控 `l0c` 占用（BLOCK_M×BLOCK_N×4 ≤ 128KB）；溢出就拆。

---

## 情况F：scale/mask 并入 attention（融合 attention 内的小算子）

**触发**：attention 的 `scale`（÷√dim）或 `mask` 是独立 kernel 或单独乘法步骤；我们 `attention_scores_kernel` 已把 scale 乘进 acc（这是正确样例）。
**收益**：省 scale/mask 的额外 GM 读写 + launch；企业级融合 attention（GEMM+scale+mask+softmax）实测 **3.9×**。
**怎么查**：搜 "triton fused attention scale mask softmax" / "flash attention causal mask before max"。

### ❌ 问题示例代码（mask 在 max 之后 → 归一化被污染）
```python
# ❌ mask 加在 exp 之后, max 还是看到被 mask 位置的值
S_tile = tl.dot(Q_tile, K_tile.T) * scale
m_new = tl.maximum(m_row, tl.max(S_tile, axis=1))   # ← BUG: max 没先 mask
S_tile = tl.where(mask, S_tile, float("-inf"))       # mask 在 max 之后 → 白 mask
P_tile = tl.exp(S_tile - m_new)
```
**出现的问题**：`tl.max` 在 mask 前 → 无效(未来/边界)位置的大值污染 `m_new`，整个 softmax 归一化偏移（flash 最常见 bug 之一）。scale 若也是独立 kernel → 多一次 GM 往返。

### ✅ 修改后正确代码（mask 先于 max，scale 并进 acc）
```python
# ✅ scale 乘进 acc (我们 attention_scores 已做): s = acc * scale 再 store
# ✅ mask 必须先于 max:
S_tile = tl.dot(Q_tile, K_tile.T) * scale
S_tile = tl.where(mask, S_tile, float("-inf"))       # ✅ 先 mask
m_new = tl.maximum(m_row, tl.max(S_tile, axis=1))    # max 只看到有效位置
P_tile = tl.exp(S_tile - m_new)
```
**约束/坑**：`mask` 用 `-inf` 不是 0（0 的 exp=1 混进 sum）；K 越界 load `other=-inf`、V 越界 `other=0`。

---

## 情况G：消除隐式格式转换（Ascend 特有，memory_bound + 算力利用率低）

**触发**：`main_mem` 流量大但 `compute_utilization` 低，怀疑 **GEMM 输出格式（NC1HWC0）与下游算子的 ND 输入不匹配 → 隐式格式转换 kernel 白白跑**。真实案例：attention 35ms→11ms（3.2×），主因就是消除 GEMM↔Softmax 的隐式格式转换。
**收益**：消除隐式转换的额外 GM 搬运（可达 2~3×）。
**怎么查**：搜 "ascend NC1HWC0 format conversion implicit kernel" / "triton-ascend layout conversion fusion"。

### ❌ 问题示例（融合了但格式不匹配 → 隐式转换仍跑）
```python
# ❌ 融合了算子, 但 GEMM 输出是 NC1HWC0 布局, 下游读它当 ND → 编译器插隐式转换 kernel
#    表现为: op 数没少多少, main_mem 流量还大, 算力利用率低
```
**出现的问题**：Ascend 上 GEMM 输出常用 NC1HWC0（5D）布局，若下游逐元素算子按 ND（2D）读，编译器会插一个**隐式格式转换 kernel**（额外 GM 读写）→ 融合白做。
### ✅ 修改后正确代码（保持布局一致 / 融合中访问）
```python
# ✅ 融合进同一 kernel 内, 在 UB 里按 GEMM 输出布局直接做后续逐元素, 不落 GM 再转
# 判断: 融合后 op 数应减少, main_mem 流量应降 — 若没降, 查是否有隐式转换
```
**约束/坑**：Ascend 特有；看 `main_mem` 里是否有额外转换搬运；融合尽量在**同一 kernel** 内完成，避免跨 kernel 格式不匹配。

---

## 不可融合边界（必须遵守）

1. **中间有 GM store**（数据跨内存生命周期）→ 不跨 store 融合
2. **跨 tile/循环迭代**（不同执行上下文）→ 不跨迭代融合
3. **链里有归约/原子/条件分支**（非逐元素）→ 只融合归约前后的独立逐元素链
4. **mask/偏移不一致** → 先统一边界再融合
5. **有副作用**（`tl.device_print`/`tl.device_assert`）→ 不跨副作用
6. **寄存器/UB 临界**（单块 ≥64KB 或张量 ≥6）→ 评估后拆
7. **精度风险**（FP16/FP32 混转）→ 保留显式类型转换

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| 独立中间 kernel 落 GM | memory_bound 还慢 | 并进前一 matmul epilogue（情况A/B） |
| epilogue N 未 32B 对齐 | UB 越界写/垃圾 | 对齐 BLOCK_N（fp16×2/fp32×4 是 32 倍数） |
| 融合改浮点顺序 | 精度差 >1e-6 | 严格保序，只合并表达式 |
| 过度融合 | 寄存器溢出/更慢 | 融合深度 2~3 级，监控 L0C |
| 用 tl.erf | 编译失败 | 换 tl.math.tanh |
| 跨迭代/跨 store 融合 | 结果错 | 遵守不可融合边界 |
