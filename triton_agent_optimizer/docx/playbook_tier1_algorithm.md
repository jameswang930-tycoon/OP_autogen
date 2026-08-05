# Triton 优化 Tier 1（算法结构）策略指南 — 针对 triton-ascend 910B3

> 本层是**最高优先级，最先做**：选对算法 / 精度 / 结构。算法一变，后续 Tier2(融合)/Tier3(分块) 全要重置重做。
> 本层**只改算法结构 / 精度 / kernel 重组**，**不调分块细节**（Tier3）、**不做逐元素融合**（Tier2）。
>
> **★环境铁律（triton-ascend，违反必报错）**：
> - `num_warps` / `num_stages` **禁止**传给 kernel（自动管理）
> - 不用 `@triton.autotune` / `@libentry`；只用裸 `@triton.jit`
> - **累加器/归约必须 fp32**（triton 不自动提升 fp16 归约精度）
> - 激活**别用 `tl.erf`**（triton-ascend 支持不确定）——用 `tl.math.tanh` 近似
> - 分块 16 倍数；`tl.dot(a, b, acc)` 带 fp32 acc

---

## 一、诊断触发规则（v4 字段 → 算法决策）

看 `07_tier1_fields.txt`（含**全局摘要** + 本层字段）：

| v4 字段 | 触发 | 算法决策 |
|---|---|---|
| `roofline.compute_utilization` | **低 (<0.3)** 且非 memory | 算法选错 → 换算法 |
| `compute.cube_fp16_ratio` | 低 且 compute_bound | fp16 计算 + fp32 累加（情况A） |
| `roofline.bottleneck_type` | `memory_bound` 且算术强度低 | 冗余访存（如 S 中间量）→ flash（情况B） |
| `summary.num_kernels` / `api_overhead_total_us` | 多同结构 matmul / launch 大 | QKV 三合一 / persistent（情况C/F） |
| `roofline.arithmetic_intensity` | 明显低于平衡点(≈86) | 算法/访存结构问题 |

### ★决策流程图
```
compute_bound 且 cube_fp16_ratio 低 ? → fp16 (情况A)
memory_bound 且 巨大中间张量(S[seq²]) ? → flash 省 S (情况B)
多个同结构小 matmul 串行 ? → 合并一个 GEMM (情况C)
归约类算子多遍扫数据 ? → online/单遍 (情况D)
compute_utilization 极低但算法看着对 ? → 分块/融合问题, 回 Tier2/3
否则 → 算法层已最优, 晋升 Tier2
```

---

## 情况A：fp16 计算 + fp32 累加（compute_bound，我们 MLP/attention 最相关）

**触发**：`bottleneck_type=compute_bound`（comp≥0.8）且 `cube_fp16_ratio` 低。
**收益**：fp16 cube 313 vs fp32 74 TFLOPS → 计算瓶颈下 ~4×。
**怎么查**：搜 "triton matmul fp16 fp32 accumulate" / "triton fp16 reduction precision"。

### ❌ 问题示例代码（fp16 累加 bug）
```python
# ① config
DTYPE = torch.float16

# ② kernel — ❌ acc 也用了 fp16
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)   # ← BUG: 累加器该 fp32
    for k in range(0, K, BLOCK_K):
        ...
        acc = tl.dot(a, b, acc)      # fp16 累加 → K=2048 次累加精度损失大
    tl.store(c_ptrs, acc, ...)
```
**出现的问题**：
- fp16 尾数仅 10 位，2048 次 `tl.dot` 累加后相对误差 ~1e-2+，大矩阵结果错（`MATMUL_VERIFY` 会 CHECK/FAIL）
- triton **不自动**把 fp16 归约提升到 fp32（PyTorch 都专门修过这个问题）——你不显式 fp32，它就 fp16 累加

### ✅ 修改后正确代码
```python
    # ② kernel 内只改这一处: acc 用 fp32 (fp16 输入自动提升到 fp32 计算)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # ✅ 累加器 fp32
    for k in range(0, K, BLOCK_K):
        ...
        acc = tl.dot(a, b, acc)      # fp16 输入 + fp32 累加 = 标准做法
    tl.store(c_ptrs, acc.to(DTYPE), ...)   # 写回时降精度, 不要提前降
```
**约束/坑**：
- 累加器、归约、`m/l` 统计量**一律 fp32**；只有输入和最终存储才 fp16
- kernel 内归约（`tl.sum` 等）如果输入是 fp16，**先 `.to(tl.float32)` 再归约**
- 激活（tanh）对 fp16 输入敏感：kernel 内 `val.to(tl.float32)` 算激活再存
- 验证 `MATMUL_VERIFY=1`，误差阈值放宽到 ~1e-2（fp16 输入 + fp32 累加 vs torch fp16 参考）

---

## 情况B：Flash Attention（省 S 中间量，memory_bound 的 attention）

**触发**：`bottleneck_type=memory_bound` 且我们 attention 的 `S=Q@K^T`=[seq,seq]=2048² 大中间量（写 16MB + 读 16MB = 32MB 无谓流量）。
**收益**：省 S 的 GM 往返 → attention 端到端 2~4×。
**怎么查**：搜 "triton flash attention fwd kernel" / "online softmax matmul"。

### ❌ 问题示例代码（在线 softmax 合并公式 bug）
```python
# 核心思路: 不写 S 到 GM, 在线合并行 max/sum 直接累加进 O
m_row = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
l_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for n_tile in range(0, seq, BLOCK_N):
    S_tile = tl.dot(Q_tile, K_tile.T) * scale
    m_new = tl.maximum(m_row, tl.max(S_tile, axis=1))
    P_tile = tl.exp(S_tile - m_new)
    acc += tl.dot(P_tile, V_tile)          # ← BUG1: 没乘 alpha=exp(m_row-m_new) 重标度
    l_row += tl.sum(P_tile, axis=1)        # ← BUG2: l_row 也没重标度
    m_row = m_new
O = acc / l_row[:, None]
```
**出现的问题**：
- **B1 漏重标度**：`m_new > m_row` 时，之前累加的 `acc` 和 `l_row` 是按旧 max 的 scale，直接 `+=` 会让输出**偏向后面处理的 KV 块** → 结果错。这是 flash 最常见的 bug，`corr = exp(m_row - m_new)` 必须同时乘到 acc 和 l_row（只乘一个也错）
- **B2 数值溢出**：`exp(m_row - m_new)` 当 max 变化大时可能 overflow/underflow

### ✅ 修改后正确代码
```python
m_row = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
l_row = tl.zeros((BLOCK_M,), dtype=tl.float32)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for n_tile in range(0, seq, BLOCK_N):
    S_tile = tl.dot(Q_tile, K_tile.T) * scale     # S 在 UB, 不进 GM
    m_new = tl.maximum(m_row, tl.max(S_tile, axis=1))
    alpha = tl.exp(m_row - m_new)                 # ✅ 重标度因子
    P_tile = tl.exp(S_tile - m_new)
    acc = acc * alpha[:, None] + tl.dot(P_tile, V_tile)        # ✅ acc 和 l 都要乘 alpha
    l_row = l_row * alpha + tl.sum(P_tile, axis=1)
    m_row = m_new
O = acc / l_row[:, None]
```
**约束/坑**：
- `m_row/l_row/acc` **必须 fp32**（fp16 的 max 上限 65504，长序列会饱和）
- **causal/边界 mask 必须在 `tl.max` 之前**（先 mask 再求 max，否则无效位置污染归一化）
- K 越界 load 的 `other` 用 **`-inf`**（不是 0——0 会贡献进 sum）
- 这是算法级大改，必须 `MATCH_VERIFY=1` 数值校验；fp32 参考下 online 版可能差 ~1e-5（顺序不同），可接受

---

## 情况C：QKV 三合一（多个同结构 matmul → 一个 GEMM）

**触发**：`num_kernels` 里多个同结构 matmul（我们 attention 的 Q/K/V 三个 `X@W`），launch_count 高。
**收益**：3 launch → 1，X 只 load 一次。
**怎么查**：搜 "triton attention fused QKV projection" / "qkv stride view bug"。

### ❌ 问题示例代码（QKV 拼接 stride 错位 bug）
```python
# 主机端拼 W → 一次 matmul
W_qkv = torch.cat([Wq, Wk, Wv], dim=1)          # [dim, 3*dim]
qkv = matmul_kernel[g](x, W_qkv, qkv_out, seq, 3*dim, dim, ...)   # [seq, 3*dim]
# Q/K/V 切片视图
Q = qkv[:, :dim]      # ← 这是 [seq, dim] 视图, 但列 stride = 3*dim (不是 dim!)
# 后续 kernel 若假设 Q 连续 (stride=dim) → 读错位置 → 垃圾值/NaN
```
**出现的问题**：`Q = qkv[:, :dim]` 是 strided 视图（列步长 `3*dim` 而非 `dim`）。如果下游 kernel 硬编码 `stride=dim` 或 `.view()`，就读错内存 → 垃圾值。**stride 假设错误是融合投影最常见的 bug**（vLLM/sglang 一堆修这个的 PR）。

### ✅ 修改后正确代码（两种方案）
```python
# 方案1 (推荐, 零拷贝): 显式传 qkv 的列 stride 给下游 kernel
#   下游 kernel 的 b_ptrs 用 stride_bn = 3*dim (不是 dim), 读 Q 的 [seq, dim] 视图
matmul_kernel(qkv_ptr, ...)   # 通过 stride 参数指定从 qkv 里读 Q/K/V

# 方案2 (简单, 多一次拷贝): 主机端 .contiguous() 切出独立张量
Q = qkv[:, :dim].contiguous(); K = qkv[:, dim:2*dim].contiguous(); V = qkv[:, 2*dim:].contiguous()
```
**约束/坑**：`3*dim` 编译期常量；方案1 要改下游 kernel 的 stride 参数（改对，不然又是 stride bug）；方案2 多一次 GM 拷贝但最稳。

---

## 情况D：Online Softmax（归约类算子，我们 softmax_kernel 已是单遍）

**触发**：softmax/norm 多遍扫数据（max 一遍 + exp/sum 一遍 + 写回）。
**收益**：1.2~2.8×（长序列）。
**怎么查**：搜 "triton online softmax single pass"。

### ❌ 问题示例代码（没减 max → 溢出）
```python
@triton.jit
def softmax_bug(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row*cols + offs, mask=mask, other=0.0)   # ← BUG: other 该 -inf
    e = tl.exp(x)                       # ← BUG: 没减 max → 大值溢出
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + row*cols + offs, e / denom, mask=mask)
```
**出现的问题**：x 值大（如 100）时 `exp(100)` 溢出 → `nan/inf`；且没减 max 数值极不稳定。mask 的 `other=0` 也会污染（0 的 exp=1 混进 sum）。

### ✅ 修改后正确代码
```python
@triton.jit
def softmax_fixed(x_ptr, y_ptr, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, BLOCK)
    mask = offs < cols
    x = tl.load(x_ptr + row*cols + offs, mask=mask, other=float("-inf"))   # ✅ -inf
    m = tl.max(x, axis=0)              # ✅ 先减 max (数值稳定)
    e = tl.exp(x - m)
    denom = tl.sum(e, axis=0)
    tl.store(y_ptr + row*cols + offs, e / denom, mask=mask)
```
**约束/坑**：`other=float("-inf")` 保证 exp(-inf)=0 不污染；`m/denom` 保持 fp32。

---

## 情况E：Split-K（超大 K 分解）

**触发**：K 非常大（>4096）且 `compute_utilization` 低。
**收益**：K 并行到多 program。
**怎么查**：搜 "triton matmul split k parallel reduce"。

### ❌ 问题示例代码（部分和覆盖 bug）
```python
# 每个 program 算部分 K 的 acc, 直接 store → 覆盖, 只留最后一块
for k in range(k_start, k_end, BLOCK_K):
    ...
    acc += tl.dot(a, b)
tl.store(c_ptr + offs, acc, ...)    # ← BUG: 不同 program 写同一位置, 覆盖
```
**出现的问题**：split-k 的多个 program 各算一份部分和，直接 store 会互相覆盖 → 只留最后一块，结果错。

### ✅ 修改后正确代码（归约）
```python
# 方案1: 第二 kernel 归约 (每个 program 输出部分和到临时, 再求和)
# 方案2: 原子加 (tl.atomic_add, 精度低些)
# 方案3: tl.sum 归约 (把部分和沿 K 维 reduce)
```
**约束/坑**：K 不够大别用（归约开销倒挂）；我们 attention K=2048 一般不需要。

---

## 情况F：Persistent Kernel（小 grid + 高 launch 开销）

**触发**：`num_kernels` 少、`api_overhead_total_us` 大。
**收益**：减 launch 次数。
**怎么查**：搜 "triton persistent kernel matmul"。

### ❌ 问题示例代码（tile_id 越界 bug）
```python
# persistent program 循环处理 tile_id, 但没判越界
for tile_id in range(pid, num_tiles, num_programs):
    ...   # ← BUG: 若 num_tiles 不是 num_programs 整数倍, 部分 program 空转/越界
```
**出现的问题**：`num_tiles % num_programs != 0` 时，最后的 program 循环到越界 tile → 读越界。

### ✅ 修改后正确代码
```python
for tile_id in range(pid, num_tiles, num_programs):
    if tile_id >= num_tiles:
        break    # ✅ 越界保护
    ...
```
**约束/坑**：我们 2048³ grid=1024，launch 开销一般不大，慎用。

---

## 通用算法替换原则

1. **先判瓶颈类型再选算法**：compute_bound → 精度/算法效率；memory_bound → 省访存（online/flash/合并）。
2. **算法级改动必须 `MATCH_VERIFY=1` 数值校验**，尤其 softmax/flash 的合并公式（online 版 vs torch fp32 参考可能差 ~1e-5，顺序不同可接受）。
3. **GPU 版代码适配规则**：1D grid（`pid//grid_n`）、`tl.dot(a,b,acc)` 带 fp32 acc、去掉 num_warps/autotune、分块 16 倍数、激活用 tanh。
4. **算法改完后续层全要重做**（回 Tier2/3）。

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| acc/归约 用 fp16 | 大矩阵精度崩 | 全用 fp32；只有输入/存储 fp16 |
| flash 漏 rescale | 输出偏向后面的块 | `acc=acc*alpha + P@V`, `l=l*alpha+sum(P)` 都要 |
| flash mask 在 max 后 | 归一化被无效位污染 | 先 mask 再 max |
| K 越界 other=0 | sum 被污染 | other 用 `-inf` |
| softmax 没减 max | exp 溢出 nan | 减 max + other=-inf |
| QKV 拼接 stride 错 | 读错位置垃圾值 | 显式 stride 或 .contiguous() |
| 用 tl.erf | 编译失败 | 换 tl.math.tanh |
| 算法改完不重调后续 | 收益被掩盖 | 回 Tier2/3 重做 |
