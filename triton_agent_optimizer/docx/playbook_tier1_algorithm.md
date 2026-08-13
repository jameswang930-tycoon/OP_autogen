# Triton 优化 Tier 1（算法结构）策略指南 — 针对 triton-ascend 910B3

> ## ★★速查卡（★先读这个；你实际能读全文，但速查卡在最前面=最快定位，决策先靠它，细节看情况A~J）★★
>
> **工作流（3 步，别跳）**：
> 1. 读 `07_tier1_fields/tier1_fields.txt` 和 `07_tier1_fields/planner_context.json`（每 kernel 全量数据）→ 找 **bottleneck_type / compute_utilization / cube_fp16_ratio / num_kernels / total_ns**
> 2. 照下表定位"情况"→ 看情况 A~G 的 ❌/✅ 代码
> 3. 输出 changes[]：**old_code 必须从你读到的 kernel_op.py 里逐字复制**（不是凭记忆）；拿不准就不改（宁缺勿错，输出 promote 或空 changes+说明）
>
> **字段 → 情况速查表（★主决策表）**：
> | 字段/现象 | 情况 | 动作（一句话） |
> |---|---|---|
> | compute_bound 且 cube_fp16_ratio 低 | A | DTYPE 改 fp16（值域大→bf16），acc/归约保持 fp32 |
> | memory_bound 且 attention 物化 S[seq²] 中间量 | B | 改 online softmax（S 不进 GM） |
> | 因果 attention 且 KV 循环全量 loop | B-剪枝 | KV 循环加因果上界 `kv_hi`（改 loop 上界） |
> | 多个同结构 matmul 串行（如 QKV 三连） | C | 合并一个 GEMM（★注意 stride 视图） |
> | softmax/norm 多遍扫数据 | D | 单遍：减 max + `other=-inf` + fp32 |
> | conv 用 `wv[:,None]*xv[None,:]` 外积 | G | 改软件 im2col + `tl.dot` 走 cube（★conv2d 最优先） |
> | K>4096 且 compute 低 | E | split-k（我们算子一般不需要） |
> | grid 小 + launch 开销大 | F | persistent kernel（一般不需要） |
> | compute_utilization<0.3 但算法看着对 | — | 回 Tier2/3（是分块/融合问题，不是算法） |
> | 上面全不触发 | — | 算法已最优 → promote Tier2（★必填 promote_evidence） |
>
> **★我们算子对照（本层能做什么，别自己发明）**：
> - `input/matmul`（两层 MLP，3 kernel）：fp32 若 compute_bound → 降 fp16（情况A）
> - `input/attention_mlp`（9 kernel）：QKV 三合一（情况C）；已融合 scale（正确样例）
> - `input/flash_attention`：已 flash + **块内 causal mask（★KV 循环未剪枝, 全量 32 块）**；fp16 输入 + fp32 输出 + fp32 参考（★fp16 正确样例）→ **B-剪枝（KV 循环加因果上界）是本层直接可做优化**
> - `input/layernorm`：**两遍扫 + 二次 load X**（代码注释已标 Tier1 可单遍合并）→ online 单遍（一次 load 同时累 sum/sum_sq）（情况D）
> - `input/conv2d` / `conv_bias_relu`：**外积不上 cube，慢 12×** → im2col（情况G，本层最优先）
> - `input/softmax` / `rms_norm` / `rms_norm_residual`：已是单遍（正确样例）
> - `input/matmul_relu` / `matmul_transpose`：fp32 matmul → compute_bound 可降 fp16（情况A）
> - `input/sigmoid` / `vector_add` / `fused_add_mul`：无算法空间 → 直接 promote
>
> **禁止（违反=失败）**：num_warps/num_stages 传参、@triton.autotune、tl.erf、`input_precision="tf32"`、acc/归约用 fp16、非 16 倍数分块、**跨层改动**（Tier1 只改算法/精度/kernel 重组；改 BLOCK 是 Tier3、融合是 Tier2）。
> **★验证命令（改完必跑）**：`MATMUL_VERIFY=1`（不是 MATMUL_VERIFY），kernel 必须输出 `result check: PASS` 才算过；fp16/bf16 输入 + fp32 累加时阈值 ~1e-2，**参考侧升 fp32 计算**（torch 参考 `.float()` 后再算，避免 fp16 参考误差干扰判定）。
> **★离层前必读**：文末「结构层优化执行教学」——**结构层（走 cube/剪枝/单遍）必须先做完再做参数层**：教学1 因果剪枝（情况B 扩展）、教学2 im2col 走 cube（情况G）、教学3 结构改动标准流程（校验→sweep 重扫→验收）；promote 前对照 8 项检查清单逐项打勾。

---

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

### ★算子 → 最优算法选择表（Tier1 第一步：先选对算法，再谈参数）

> 判据：先看 `bottleneck_type`（compute vs memory）。**compute → 提精度/算法效率**；**memory → 省访存**（im2col 复用输入、flash 省 S、online 单遍）。

| 算子族 | 规范最优算法 | 关键点 | 对应 |
|---|---|---|---|
| matmul / GEMM | 分块 GEMM（tiled `tl.dot`） | fp32 累加；BLOCK 16 倍数 | 情况A/C |
| **卷积 conv2d** | **im2col implicit GEMM**（软件收集 patch → `tl.dot` 走 cube） | 直接卷积+向量外积 = **不用 cube**，实测慢 PyTorch 12× | **情况G** |
| attention (Q@K^T+softmax+P@V) | Flash Attention（online softmax 融合单 kernel） | 省 S 中间量 [seq²]；rescale 别漏 | 情况B |
| softmax / norm | online 单遍归约 | 减 max + `other=-inf`；m/l/acc 全 fp32 | 情况D |
| 逐元素 (bias/relu/gelu/残差) | 向量化逐元素 kernel | 大 BLOCK 连续访存；别多次单独 launch | Tier2 融合 |
| 大 K matmul (K>4096) | split-k | 部分和 atomic_add/归约 | 情况E |
| 小 grid + 高 launch | persistent kernel | 越界保护 | 情况F |

---



## 情况A：fp16 计算 + fp32 累加（compute_bound，我们 MLP/attention 最相关）

**触发**：`bottleneck_type=compute_bound`（comp≥0.8）且 `cube_fp16_ratio` 低。
**收益**：fp16 cube 峰值约为 fp32 的 2~4×（910B3 标称 fp16 294.9 TFLOPS vs fp32 73.7 TFLOPS，实际加速倍数需真机 msprof 确认）→ 计算瓶颈下显著。
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
- 验证 `MATMUL_VERIFY=1`，误差阈值放宽到 ~1e-2；**参考侧升 fp32 计算**（fp16 输入下 torch 参考 `.float()` 后再算，避免 fp16 参考误差干扰判定——flash_attention 已按此模式改）

### 精度策略扩展：fp16 vs bf16 vs tf32（Ascend 910B3 视角，别只用 fp16 一招）

**触发**：compute_bound 决定降精度时，选哪种取决于**值域 vs 精度**。
**怎么查**：搜 "bf16 vs fp16 dynamic range precision" / "tf32 tensor core precision"。

| 精度 | 尾数位 | 指数位 | 适合 | Ascend 可用性 |
|---|---|---|---|---|
| fp16 | 10 位 | 5 位 | 值域正常 (±65504 内) | ✅（我们默认） |
| bf16 | **7 位** | **8 位** | 值域大/训练易溢出 | ✅（同 fp16 路径，动态范围同 fp32） |
| tf32 | 10 位 | 8 位 | fp32 输入的快速近似 | ⚠ triton-ascend **未验证**，别用 |

**关键点**：
- **值域溢出比精度损失更致命**：fp16 最大值 65504，softmax 前 Q@K^T·scale 或大激活可能超 → 用 bf16（指数位 8，动态范围同 fp32）
- **精度敏感（attention/softmax）→ 用 fp16**（尾数 10 位 > bf16 的 7 位）；**大动态范围（norm/大激活）→ 用 bf16**
- 无论 fp16/bf16，**累加器一律 fp32**（tl.dot 的 acc / 归约 / m·l 统计量）
- ⚠ `tl.dot(..., input_precision="tf32")` 是 NVIDIA Triton 参数，**triton-ascend 不一定支持**——不要靠它，降精度就走 DTYPE（张量建 fp16/bf16 + fp32 累加）

### ❌ 问题示例代码（误用 tf32 参数 / bf16 全精度损失）
```python
acc = tl.dot(a, b, acc, input_precision="tf32")   # ❌ triton-ascend 未必支持该参数, 编译可能报错
# 或 bf16 做 attention: 尾数 7 位, Q@K^T 累加 2048 次误差 ~1e-1 → 结果 CHECK/FAIL
```

### ✅ 修改后正确代码（按瓶颈类型选 DTYPE + fp32 累加）
```python
# config: compute_bound 且值域正常 → fp16 (尾数更精细)
DTYPE = torch.float16
# config: 值域大/softmax 前易溢出 → bf16 (动态范围同 fp32)
DTYPE = torch.bfloat16

# kernel: 不管哪种, 累加器/归约/统计量全 fp32, 只有输入和最终存储降精度
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, K, BLOCK_K):
    acc = tl.dot(a, b, acc)          # fp16/bf16 输入 + fp32 累加
tl.store(c_ptrs, acc.to(DTYPE), ...) # 写回才降精度
```
**约束/坑**：选精度 = 值域优先（溢出→bf16）→ 精度优先（attention→fp16）；别用 `input_precision`（Ascend 未验证）；改 DTYPE 后 `MATMUL_VERIFY=1` 重验，阈值 ~1e-2。

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
- 这是算法级大改，必须 `MATMUL_VERIFY=1` 数值校验；fp32 参考下 online 版可能差 ~1e-5（顺序不同），可接受

### 因果剪枝：KV 循环只跑到 query 块末位（省 ~50% key 块，纯赚）

**触发**：因果 attention 且当前 kernel 对**所有** key 块都 loop + mask（我们 flash_attn 现在就是：`for start in range(0, seq, BLOCK_N)` 永远 32 块）。
**收益**：seq=2048、BLOCK_M=64 时 key 块从 1024 → ~528（**省 ~48%**）；每块都省一次 K 加载 + dot + mask。
**怎么查**：搜 "flash attention causal skip kv blocks diagonal bound"。

### ❌ 问题示例代码（因果仍全量 loop，白算对角线上方块）
```python
for start in range(0, seq, BLOCK_N):      # ❌ 无因果上界: 每个 query 块都处理全部 key 块
    offs_n = start + tl.arange(0, BLOCK_N)
    kk = tl.load(...);  vv = tl.load(...)
    s = tl.dot(q, kk) * scale
    causal = offs_n[None, :] <= offs_m[:, None]   # mask 全在内部, 上三角全白算
    s = tl.where(causal, s, float("-inf"))
    ...
```
**出现的问题**：因果时 key n > query m 的块对结果无贡献（被 mask 成 -inf），但**仍付出完整的 K 加载 + dot + softmax**。seq=2048 有一半 key 块纯浪费。

### ✅ 修改后正确代码（KV 循环加因果上界）
```python
# query 块末位 = m_block*BLOCK_M + BLOCK_M - 1; 只处理 kv_start ≤ 它的 key 块
kv_hi = min(seq, m_block * BLOCK_M + BLOCK_M)    # ✅ 因果上界 (含对角块, 它部分有贡献)
for start in range(0, kv_hi, BLOCK_N):           # ✅ 对角线上方的 key 块直接跳过
    offs_n = start + tl.arange(0, BLOCK_N)
    kk = tl.load(...);  vv = tl.load(...)
    s = tl.dot(q, kk) * scale
    causal = offs_n[None, :] <= offs_m[:, None]  # 对角块内部仍需 mask
    s = tl.where(causal, s, float("-inf"))
    ...
```
**约束/坑**：
- `kv_hi` 必须是**动态上界**（每 query 块不同），不是全局 `seq`；对角块（`start ≤ kv_hi` 的最后一个）**仍要 mask**（部分 key ≤ query）
- 只适用于因果（非因果 attention 不能加）
- ★当前 `input/flash_attention` 的 KV 循环**仍是全量 loop**（块内 mask）——**本层直接可做**；改后 `MATMUL_VERIFY=1` 重验 + 真机 msprof 确认收益
- 这是我们 flash_attn 的**直接可做优化**（改 loop 上界一行 + 用 m_block 算 kv_hi）

### ★完整改造教学（从当前 input/flash_attention/kernel_op.py 出发，三步）

**改造目标**：把 KV 循环从"永远扫全部 32 块"改为"每 query 块只扫到对角块"。kernel 已定义了
`m_block = pid % num_m`（第几个 query 块）和 `offs_m = m_block*BLOCK_M + tl.arange(0, BLOCK_M)`——
`kv_hi` 直接由它们算出，不需要新变量。

**改前（当前代码，kernel 内 KV 循环）**：
```python
    # 循环前已有: offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    for start in range(0, seq, BLOCK_N):          # ❌ 无因果上界: 每个 query 块扫全部 key 块
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq
        k_ptrs = k_ptr + head * (dim * seq) + offs_k[:, None] * seq + offs_n[None, :]
        kk = tl.load(k_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        s = tl.dot(q, kk) * scale
        causal = offs_n[None, :] <= offs_m[:, None]   # 上三角块全被 mask 成 -inf, 但已付出完整计算
        s = tl.where(causal, s, float("-inf"))
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        p = tl.exp(s - m_curr[:, None]).to(tl.float16)
        alpha = tl.exp(m_i - m_curr)
        l_i = alpha * l_i + tl.sum(p.to(tl.float32), axis=1)
        v_ptrs = v_ptr + head * (seq * dim) + offs_n[:, None] * dim + offs_k[None, :]
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vv)
        m_i = m_curr
```

**改后（只动两处：循环前加 kv_hi 定义 + 循环上界；循环体一行不改）**：
```python
    # ① 循环前加一行: 本 query 块允许的最大 key 下标 = 块末位 (含对角块, 它部分有贡献)
    kv_hi = min(seq, m_block * BLOCK_M + BLOCK_M)
    # ② 循环上界 seq → kv_hi: 对角线上方的 key 块直接跳过 (不再 mask 后白算)
    for start in range(0, kv_hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < seq
        k_ptrs = k_ptr + head * (dim * seq) + offs_k[:, None] * seq + offs_n[None, :]
        kk = tl.load(k_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        s = tl.dot(q, kk) * scale
        causal = offs_n[None, :] <= offs_m[:, None]   # ✅ 对角块内部仍需 mask (部分 key ≤ query)
        s = tl.where(causal, s, float("-inf"))
        m_curr = tl.maximum(tl.max(s, axis=1), m_i)
        p = tl.exp(s - m_curr[:, None]).to(tl.float16)
        alpha = tl.exp(m_i - m_curr)
        l_i = alpha * l_i + tl.sum(p.to(tl.float32), axis=1)
        v_ptrs = v_ptr + head * (seq * dim) + offs_n[:, None] * dim + offs_k[None, :]
        vv = tl.load(v_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p, vv)
        m_i = m_curr
```

**为什么循环体一行都不用改**：剪枝只跳过"整块都无效"的 key 块；对角块内部依然有有效 key
（key ≤ query），原有 `causal` mask 逻辑继续正确工作。`n_mask`（offs_n < seq）保留不动——
kv_hi 已 ≤ seq，但保留它防御 kv_hi 取到 seq 时的尾部越界。

**改造后必须做的验证**（缺一不可）：
1. `MATMUL_VERIFY=1` 跑一遍 → `result check: PASS`（结果应与剪枝前**逐位一致**——数学上剪枝
   只跳过被 mask 成全 -inf 的块，softmax 分母不受影响；若 PASS 判定用相对误差，数值应完全不变）
2. 结构变化 → **sweep 会自动重扫**（调度器 round1 + 每个 tier3 round 都跑）→ 重扫后 BLOCK 可能
   变化（循环变短后更大 BLOCK 更划算），以 sweep 结果为准
3. 真机确认收益：`KERNEL_LOOP=30 python3 input/flash_attention/kernel_op.py` 前后对比端到端，
   或跑 `feedback/remeasure_best.py --op flash_attention` 看工业级口径数字

**常见错**：
- `kv_hi = m_block * BLOCK_M + BLOCK_M` 忘加 `min(seq, ...)` → 最后一个 query 块越界读
- 用全局 `seq` 当上界（没剪）→ 白改
- 把对角块的 `causal` mask 删了 → 对角块无效 key 污染 max/softmax（结果错）

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

## 情况G：Conv im2col implicit GEMM（conv 走 cube，我们 conv2d 最相关）

**触发**：conv2d 类算子（我们 input/conv2d、conv_bias_relu），当前用**向量外积** `acc += wv[:,None]*xv[None,:]` → 没用 cube（矩阵单元）。
**收益**：`tl.dot` 走 cube。**原理差距**：fp32 向量单元 vs cube 吞吐差 ~50×；实测我们 conv2d 484us vs PyTorch 40.6us（**12× 慢**）——根因就是这个算法。im2col 后接近 PyTorch。
**怎么查**：搜 "triton convolution implicit gemm im2col"（官方教程用 TMA 硬件 im2col，我们**没有 TMA**，用下面的软件 im2col）。

### ❌ 问题示例代码（直接卷积 + 向量外积，不用 cube）
```python
# 每 program 算 [BLOCK_K 通道 × BLOCK_OW 空间], c/r/s 三重循环外积累加
for c in range(C):
    for r in range(R):
        for s in range(S):
            xv = tl.load(x_ptr + n*C*H*W + c*H*W + ih*W + iw, mask=valid, other=0.0)  # [BLOCK_OW]
            wv = tl.load(w_ptr + offs_k*C*R*S + c*R*S + r*S + s, mask=k_mask, other=0.0)  # [BLOCK_K]
            acc += wv[:, None] * xv[None, :]        # ❌ 向量外积: 走 AIC 向量单元, 不上 cube
```
**出现的问题**：`wv[:,None] * xv[None,:]` 是外积（向量单元），C*R*S 次循环 = 算力只用了向量单元的一小部分 → 我们 conv2d 慢 PyTorch 12×。**外积 ≠ tl.dot**——这是 conv 最常见的算法级错误。

### ✅ 修改后正确代码（软件 im2col → tl.dot 走 cube）
```python
# 核心: 把 conv 重写成 GEMM。
#   M_GEMM = 输出空间 (n*OH*OW), N_GEMM = 输出通道 K, K_GEMM = C*R*S (滤波tap×输入通道)
#   每 program 一次 tl.dot: [K, CRS] @ [CRS, OW] → [K, OW]  (BLOCK_CRS = next_pow2(C*R*S))
@triton.jit
def conv2d_im2col_kernel(x_ptr, w_ptr, y_ptr,
                         N, H, W, K, OH, OW,
                         BLOCK_K: tl.constexpr, BLOCK_OW: tl.constexpr, BLOCK_CRS: tl.constexpr,
                         C: tl.constexpr, R: tl.constexpr, S: tl.constexpr, PAD: tl.constexpr):
    pid = tl.program_id(axis=0)
    total_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % total_ow
    tmp = pid // total_ow
    oh = tmp % OH
    n = tmp // OH

    offs_k = tl.arange(0, BLOCK_K)
    offs_ow = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    offs_crs = tl.arange(0, BLOCK_CRS)          # ≥C*R*S 的 2 幂 (72 → 128)
    c = offs_crs // (R * S); r = (offs_crs % (R * S)) // S; s = offs_crs % S
    ih = oh + r - PAD                            # [CRS]
    iw = offs_ow[None, :] + s[:, None] - PAD     # [CRS, OW]

    # 软件 im2col: patch[crs, ow] = X[n, c, ih, iw]; padding/越界 → 0
    valid = (offs_crs[:, None] < C * R * S) & (ih[:, None] >= 0) & (ih[:, None] < H) \
            & (iw >= 0) & (iw < W)
    patch = tl.load(x_ptr + n * C * H * W + c[:, None] * H * W + ih[:, None] * W + iw,
                    mask=valid, other=0.0)        # [CRS, OW]  innermost(ow) stride=1 连续

    # W 拍平 [K, C*R*S]: wtile[k, crs] = W[k, c, r, s]
    wtile = tl.load(w_ptr + offs_k[:, None] * (C * R * S) + offs_crs[None, :],
                    mask=(offs_crs[None, :] < C * R * S) & (offs_k[:, None] < K),
                    other=0.0)                    # [K, CRS]

    acc = tl.zeros((BLOCK_K, BLOCK_OW), dtype=tl.float32)
    acc = tl.dot(wtile, patch, acc)               # [K,CRS]@[CRS,OW]→[K,OW], ★三参 acc 与我们所有 kernel 一致
    y_ptrs = y_ptr + n * K * OH * OW + offs_k[:, None] * OH * OW + oh * OW + offs_ow[None, :]
    tl.store(y_ptrs, acc, mask=(offs_k[:, None] < K) & (offs_ow[None, :] < OW))
```
**约束/坑**：
- **BLOCK_CRS 必须 ≥C*R*S 的 2 幂**（tl.arange 要求）；72→128，多出的 tap 用 mask 置 0（0×patch=0，安全）
- **L0 核验**（我们 C=8 R=S=3 K=32 OW=64, BLOCK_K=32, BLOCK_CRS=128, BLOCK_OW=64, fp32）：L0A=32×128×4=**16KB**✓ L0B=128×64×4=**32KB**✓ L0C=32×64×4=**8KB**✓ UB=56KB✓
- **innermost 连续**：patch 的最后一维是 ow（X 内存 stride=1）→ DMA 连续突发✓
- **已用 `tl.dot(a,b,acc)` 三参 + fp32 acc**（与我们所有 kernel 一致，避免二参兼容风险）
- **已用 torch conv2d 参考验证索引数学**：max_err=1.5e-5（本场景）
- conv_bias_relu 的 conv2d_kernel 同款替换；grid 不变 `(N*OH*ceil(OW/BLOCK_OW),)`
- ⚠ 具体加速倍数需真机 msprof 确认（原理上从"向量外积不上 cube"变"cube"，但小卷积可能访存受限）

---

## 通用算法替换原则

1. **先判瓶颈类型再选算法**：compute_bound → 精度/算法效率；memory_bound → 省访存（online/flash/合并）。
2. **算法级改动必须 `MATMUL_VERIFY=1` 数值校验**，尤其 softmax/flash 的合并公式（online 版 vs torch fp32 参考可能差 ~1e-5，顺序不同可接受）。
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
| conv 用向量外积不上 cube | 慢 PyTorch 12× | im2col + `tl.dot` 走 cube（情况G） |
| 因果 attention 全量 loop key 块 | 上三角白算 ~50% | KV 循环加因果上界 `kv_hi`（情况B 扩展） |
| 用 `input_precision="tf32"` | triton-ascend 未必支持 | 降精度走 DTYPE (fp16/bf16) + fp32 累加 |
| 用 tl.erf | 编译失败 | 换 tl.math.tanh |
| 算法改完不重调后续 | 收益被掩盖 | 回 Tier2/3 重做 |

---

## ★结构层优化执行教学（Tier1 离层前必做，2026-08-13）

> **这节教"怎么做"**：Tier1 的优化分两类——**结构层**（换算法/走 cube/剪枝，收益 2~30×）和
> **参数层**（BLOCK/精度微调，收益 10~30%）。**结构层必须最先做完**，否则参数层优化全白费。
> 下面按"当前算子上实际怎么改"教结构层三件事，每件都有完整做法，不要跳过。

### 教学1：因果剪枝（KV 循环加动态上界）— 完整做法见「情况B 扩展」

因果 attention 的 kernel 里，只要 KV 循环是 `for start in range(0, seq, BLOCK_N):`（全量扫），
就按情况B 扩展的三步改：循环前加 `kv_hi = min(seq, m_block * BLOCK_M + BLOCK_M)` → 循环上界
改成 `range(0, kv_hi, BLOCK_N)` → 循环体一行不动（对角块内部 causal mask 保留）。
**改完必须**：`MATMUL_VERIFY=1` 跑 PASS + 确认 sweep 重扫后的 BLOCK。

### 教学2：卷积走 cube（软件 im2col + tl.dot）— 完整做法见「情况G」

conv 类 kernel 的判断标准：**kernel 里有没有 `tl.dot`**。没有 = 在 vector 上模拟，算力只剩
cube 的 1/4~1/10，**任何其他层的优化都救不回来**。做法（情况G 有完整可替换代码）：
1. 重写成 GEMM 视角：`M=输出空间, N=输出通道, K=C*R*S`
2. `BLOCK_CRS = next_pow2(C*R*S)`（tl.arange 要 2 幂；本例 C=8,R=S=3 → 128）
3. 软件 im2col：`patch[crs, ow] = X[n, c, ih, iw]`，padding/越界置 0（`mask=valid, other=0.0`）
4. 权重拍平 `wtile[k, crs]`，一次 `tl.dot(wtile, patch, acc)` 走 cube
5. 按情况G 的 L0 核验公式核对 BLOCK（L0A=BLOCK_K×BLOCK_CRS×4 ≤ 64KB 等）
6. conv_bias_relu 同款替换 + bias/relu 留在 epilogue（回 Tier2 再融合）

### 教学3：结构改动后的标准流程（每次结构层改动后照做）

```
① 改代码（只改 kernel 区，不动 main/验证块）
② MATMUL_VERIFY=1 数值校验 → 必须 result check: PASS
③ 交给下一轮迭代：sweep 会在 round1/每个 tier3 round 自动重扫 BLOCK
   （结构变化后 BLOCK 最优解可能变，以重扫结果为准，不要沿用旧 BLOCK）
④ 每层做完 / 优化结束时跑 feedback/acceptance_report.py 看验收比值
   （验收 = 工业级最优 ÷ 我们最优 Event，两端同口径；比值偏低先查结构层
     有没有漏做——不是盲目加轮数调参数）
```

### Tier1 离层检查清单（promote 前逐项打勾，缺任何一项都先做掉再走）

```
[ ] 有矩阵运算的 kernel（matmul/conv/attention）→ 有 tl.dot
    （conv 必须 im2col 走 cube，禁止 vector 外积模拟 —— 教学2 + 情况G）
[ ] 因果 attention → KV 循环有 kv_hi 动态上界（不是全量 loop + 块内 mask —— 教学1 + 情况B 扩展）
[ ] online softmax → acc/l_row 都乘 alpha 重标度；mask 先于 tl.max；K 越界 other=-inf
[ ] 归约类（rms_norm/softmax）→ 已单遍（一次 load 同时累 sum/sum_sq）
[ ] 多同结构 matmul（QKV/MLP 两段）→ 已评估 QKV 三合一/融合（回 Tier2）
[ ] 算法级改动已 MATMUL_VERIFY=1 数值校验（教学3 流程②）
[ ] 结构改动后 sweep 已重扫（BLOCK 以重扫为准，教学3 流程③）
[ ] 优化结束已跑 acceptance_report.py，验收偏低时回查结构层（教学3 流程④）
```
