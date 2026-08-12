# Triton 优化 Tier 2（算子融合）策略指南 — 针对 triton-ascend 910B3

> ## ★★速查卡（★先读这个；你实际能读全文，但速查卡在最前面=最快定位，决策先靠它，细节看情况A~J）★★
>
> **工作流（3 步，别跳）**：
> 1. 读 `07_tier2_fields/tier2_fields.txt` + `07_tier2_fields/planner_context.json` → 找 **num_kernels / api_overhead_total_us / main_mem 带宽 / bottleneck_type**
> 2. (有 08_fusion/fusion_analysis.json 就 `cat` 它) 照下表定位"情况"→ 看情况 A~G 代码
> 3. 输出 changes[]：**old_code 从你读到的 kernel_op.py 逐字复制**；拿不准不改（宁缺勿错）
>
> **字段 → 情况速查表（★主决策表）**：
> | 字段/现象 | 情况 | 动作（一句话） |
> |---|---|---|
> | matmul 后跟独立 bias/激活 kernel | A | 并进 matmul epilogue（bias+tanh-GELU 在 store 前算） |
> | matmul 后跟独立 add（残差） | B | 残差 O 在 UB 里直接 load 相加（不进 GM） |
> | 同一指针同一偏移 load ≥2 次 | C | 读一次，寄存器复用 |
> | RAW 链多步纯算术（平方→求和→开方） | D | 合并表达式（★禁止改运算顺序） |
> | 想融合但单块张量 ≥6 / ≥64KB | E | 控制融合深度 2~3 级（防寄存器/UB 溢出） |
> | attention 的 scale/mask 是独立步骤 | F | 并进 QK^T（scale 乘进 acc；★mask 必须先于 max） |
> | main_mem 大但算力利用率低（融合后仍高） | G | 查 GEMM 输出格式（NC1HWC0）与下游不匹配的隐式转换 |
> | fusion_analysis 的 raw_deps 有 matmul→逐元素 链 | A/B | 优先融合占比最大的中间 kernel |
> | 融合前中间 kernel <2×launch 开销 | — | 评估是否值得（省的是 GM+launch，不是算术量） |
> | 上面全不触发 | — | 无可融合 → promote Tier3（★必填 promote_evidence） |
>
> **★我们算子对照（本层能做什么，别自己发明）**：
> - `input/matmul`（两层 MLP）：**FC1 → 独立 bias_gelu → FC2，3 分离 kernel** → 把 bias+GELU 并进 FC1 epilogue（情况A，★本层最优先，文档有完整重构示例）
> - `input/matmul_relu`：matmul → 独立 relu → 并进 epilogue（情况A）
> - `input/attention_mlp`（9 kernel）：残差 add_kernel 并进 FC2（情况B）；mlp_gelu 已融合（正确样例）
> - `input/conv_bias_relu`（3 kernel）：bias/relu 独立 kernel → 并进 conv epilogue（情况A，conv 改 im2col 后在 epilogue 加）
> - `input/rms_norm_residual`：残差加并入（情况B）
> - 单 kernel 算子（flash_attention/sigmoid/vector_add/softmax/rms_norm/layernorm/fused_add_mul/conv2d）：无融合空间 → 直接 promote
>
> **禁止（违反=失败）**：跨 store/跨迭代融合、融合时改浮点顺序、过度融合（寄存器/L0C 溢出）、num_warps/autotune、tl.erf、跨层改动（Tier2 只做融合；改算法是 Tier1、改 BLOCK 是 Tier3）。

---

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

### ★怎么读我们的 HIVM 融合分析（`08_fusion/fusion_analysis.json`）

> Tier2 每轮框架会编译 kernel → HIVM MLIR → LLM 分析依赖，产出这份文件。**这是融合决策的直接依据，先读它再动手。**

```json
{
  "op_count": 12,                    // HIVM 里的算子数 (含各 kernel 内步骤)
  "raw_deps":  [{"from": "matmul", "to": "bias", "type": "RAW"}, ...],
  "war_deps":  [...],                // 写后读 (buffer 复用冲突)
  "waw_deps":  [...],                // 写后写 (同一输出写两次)
  "fusion_candidates": [{"ops": ["fc1", "bias_gelu"], "reason": "RAW 链上连续逐元素"}]
}
```

| 键 | 含义 | 融合决策 |
|---|---|---|
| `raw_deps` | **读后写**：op2 读 op1 的输出 → 天然可融合 | **RAW 链上的连续逐元素 op → 并进前一 kernel epilogue**（情况A/B） |
| `war_deps` | 写后读：缓冲被复用 → 读写顺序冲突 | 融合时**换新 buffer**（不要复用被 WAR 的缓冲） |
| `waw_deps` | 写后写：同一输出写两次 | 合并成一次写 或 重命名输出 |
| `fusion_candidates` | LLM 建议的融合候选列表 | 挑**占比最大的中间 kernel** 先融（收益最大） |

**判断标准**：看 `raw_deps` 里 `from` 是 matmul/cube、`to` 是逐元素（vector）的链 → 融合进 epilogue 几乎必赚；`to` 是另一个 cube/matmul → 用上面的**成本模型**评估再融。

**注意**：`fusion_candidates` 是 LLM 的初步建议，**还要自己核一遍**（LLM 可能建议跨 store/跨迭代的非法融合）——对照下面的"不可融合边界"过滤。

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
- ★内部一致性提醒：`attention_mlp` 的 `mlp_gelu_kernel` 已是融合模式（正确样例）；但 **`input/matmul` 这个算子本身没融合**（`matmul_kernel(FC1) → 独立 bias_gelu_kernel → matmul_kernel2(FC2)` 3 个分离 kernel）——它是本层**最直接的融合目标**，别被"已是正确样例"误导而跳过

### 完整重构示例：input/matmul 的 FC1 + bias_gelu 融合（★按我们的真实代码改）

**改前**（input/matmul/kernel_op.py，3 个分离 kernel）：
```python
# ① FC1: Z = X@W1
matmul_kernel[grid1](x, w1, z, M, HIDDEN, K, ..., BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
# ② bias_gelu: H = GELU(Z + b1)  ← 独立 kernel, Z 落 GM 又读回
bias_gelu_kernel[grid_g](z, b1, h, M * HIDDEN, HIDDEN, BLOCK_SIZE=BLOCK_SIZE)
# ③ FC2: Y = H@W2
matmul_kernel2[grid2](h, w2, y, M, N, HIDDEN, ..., BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK)
```
**改后**（FC1 的 epilogue 直接算 bias+GELU，删掉 bias_gelu_kernel）：
```python
@triton.jit
def fc1_gelu_kernel(a_ptr, w_ptr, b_ptr, c_ptr, M, N, K, ...,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...  # 与 matmul_kernel 完全相同的 K 循环, acc fp32
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    val = acc + bias[None, :]                                # ✅ bias 在 epilogue
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608028654 * (val + 0.044715 * val * val * val)))
    y = val * cdf                                            # ✅ GELU 在 epilogue (tanh 近似)
    tl.store(c_ptrs, y, mask=c_mask)                         # 直接写 H, Z 不进 GM
```
**配套改动**（融合不只是改 kernel）：
- **删** `bias_gelu_kernel` 定义 + main() 里的 launch
- **grid**：`grid1` 不变（FC1 的 tile 网格）；**删 `grid_g`**；`h` 张量保留（FC2 输入）
- **main()**：`fc1_gelu_kernel[grid1](x, w1, b1, h, ...)` 替代 ①②
- **verify**：`MATMUL_VERIFY=1` 里参考 `h_ref = F.gelu(torch.matmul(x,w1)+b1, approximate="tanh")`（已有，无需改）

**约束/坑**：
- **tanh-GELU 公式已 CPU 验证** = torch `F.gelu(approximate="tanh")`（max_err=1.5e-07）
- 融合后 BLOCK 不变也合法：FC1 单块 = acc(L0C) + bias + val + y（UB 压力↑，2048³ BLOCK 64³ 实测安全；超了就减 BLOCK）
- `bias` 是 [BLOCK_N] 向量 load（offs_n），`val = acc + bias[None,:]` 广播
- 结果 kernel 数 3→2，中间 Z 不再写/读 GM（省 32MB 无谓流量）

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
**出现的问题**：融合时**重排浮点运算**会改变舍入 → 与参考差 >1e-6，`MATMUL_VERIFY` 失败。融合只允许"合并表达式"，**禁止改运算顺序/结合性**。

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

## 融合收益/成本模型（★先算值不值得，再决定融不融）

> 不是所有相邻算子都值得融合。先估成本再动手，避免**过度融合**（情况E）和**无效融合**（白融还没收益）。

**Ascend 量级**：kernel launch 开销 ~5-20us；HBM 读写 ~快一个数量级于片上。判断公式：
```
中间 kernel 耗时 > 2 × launch 开销（~10-40us）  → 融合值得（省 GM 往返 + launch）
中间 kernel 是纯逐元素（<10us）                 → 融合几乎必然值得（launch 占大头）
两个大 matmul 之间夹小逐元素                     → 值得（epilogue 吸收"几乎免费"）
两个都 compute_bound 的大 kernel 硬融            → 可能倒挂（占用崩，性能反降）
```

| 场景 | 是否融合 | 依据 |
|---|---|---|
| matmul 后跟 bias/激活/残差（逐元素） | ✅ 几乎必融 | epilogue 吸收"几乎免费"；GEMM+Bias+ReLU 实测 **28~39%** 提速 |
| 小/中算子叠小逐元素 | ✅ 强烈建议 | launch 开销占比高，实测 1.5~3.13× |
| 中间 kernel 耗时 < 2×launch | ⚠ 评估 | 收益可能 < 噪声地板 |
| 两个都 compute_bound 的大 matmul | ⚠ 别硬融 | 资源压力崩占用，收益倒挂 |
| 中间张量 > UB（无法留在片上） | ❌ 不融 | 必须落 GM 再读 |
| 融合后寄存器/L0C 溢出 | ❌ 拆 | 溢出 → spill 到 GM → 更慢 |
| 依赖全量输出（softmax 行归约） | ❌ 除非 online | 需要整行 sum 才能算 → 只有 flash 的 online 才可融 |

**★实测警示**：有案例 GELU 并进 GEMM epilogue 端到端只 **1.03×**（< 噪声地板 1.05×）——因为 erf 计算成本没消失，只是从独立 kernel 搬进 epilogue。**融合前先算**：省的是 GM 流量 + launch，**算的不是算术量**；若中间 kernel 本来就不碰 GM（寄存器里就有），融合收益≈0。

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
| 无效融合（中间不碰 GM） | 收益≈0（如 GELU 只 1.03×） | 先算成本模型：省的是 GM+launch，不是算术量 |
| 两个 compute_bound 大 kernel 硬融 | 占用崩，性能倒挂 | 只融 compute+memory 组合 |
| 不读 fusion_analysis 就动手 | 融错/漏融 | 先读 raw_deps/fusion_candidates（见上） |
| 用 tl.erf | 编译失败 | 换 tl.math.tanh |
| 跨迭代/跨 store 融合 | 结果错 | 遵守不可融合边界 |
