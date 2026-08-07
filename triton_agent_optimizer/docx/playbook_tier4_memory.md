# Triton 优化 Tier 4（内存访问）策略指南 — 针对 triton-ascend 910B3

> 本层在 Tier1/2/3 之后：**优化访存模式**（连续/对齐/L2复用/流水线），**不改算法、不融合、不调分块**。
> 昇腾 80~90% 的性能瓶颈是**内存访问效率**，不是算力。目标：数据尽量在 UB/L2 流转，少碰 HBM，每次访问"满载而归"。
>
> **★环境铁律（triton-ascend）**：
> - `num_warps`/`num_stages` **禁止**传（双缓冲默认开，自动管理）
> - 连续访问优先：DMA 跨步访问搬整条 cache line 只用一点 = "运空气"
> - **128-bit（16B）对齐**：指针/维度对齐是性能基石
> - 分块 16 倍数（Cube）；UB ≤ 192KB
>
> **★实战教训（rms_norm 17 轮无果根因）**：
> - 若 `main_mem` 带宽已接近峰值（`memory_utilization` ≥80%）→ **访存层已到上限**，别再折腾流水线/合并
> - 此时该算子大概率已最优（或瓶颈在别处），直接晋升/停止；**别在本层反复试 3 轮**

---

## 一、诊断触发规则（v4 字段 → 动作）

看 `07_tier4_fields.txt`（含全局摘要）：

| v4 字段 | 触发 | 动作 |
|---|---|---|
| `bandwidth_gb_s.main_mem_read/write` | 高但算力利用率低 → memory_bound | 减 GM 流量（连续/对齐/L2） |
| `bandwidth_gb_s.gm_to_ub` / `ub_to_gm` | 低（搬运效率差） | 连续访问 + 对齐（情况A/B） |
| `l2_hit_rate` | 低 | L2 复用（swizzle/访问顺序/权重预排）（情况C） |
| `task.pipes_us.aic_mte2/mte3_time` | 高（搬运耗时长） | 减搬运量 + 流水线（情况D） |
| 对齐警告 / `main_mem` 流量异常 | 非 16B 对齐 | 对齐 + padding（情况B） |

### ★决策流程图
```
main_mem 带宽接近峰值(高) → 减 GM 流量
  ├─ 有跨步/非连续访问 → 连续化 (情况A)
  ├─ 指针/维度未对齐 → 128-bit 对齐 + pad (情况B)
  ├─ l2_hit_rate 低 → swizzle/访问顺序/权重预排 (情况C)
  └─ mte 忙但 cube 空等 → 流水线/UB 控制 (情况D)
否则 → 晋升 Tier5
```

---

## 情况A：连续访存（strided → contiguous，Ascend 收益最大）

**触发**：`gm_to_ub`/`main_mem_read` 带宽低（比峰值差很多），有跨步/转置访问。
**收益**：Ascend 实测连续 980 GB/s(85%) vs 跨步 120 GB/s(10%) → **5~8×**。
**怎么查**：搜 "triton memory coalescing contiguous access" / "昇腾 strided access bandwidth"。

### ❌ 问题示例代码（跨步访问，DMA 运空气）
```python
# ❌ 按列读矩阵: 地址间隔 = N 个元素 (非连续), DMA 搬整条 cache line 只用 1 个
col = tl.load(mat_ptr + row_offsets * N + col_idx, mask=mask)   # 跨步访问 → 带宽 ~120GB/s
```
**出现的问题**：DMA 每次搬运 32B+ 的 cache line，跨步访问只用到其中 1 个元素 → 其余都是"运空气" → 带宽利用率掉到 ~10%。

### ✅ 修改后正确代码（连续访问，最快维匹配内存布局）
```python
# ✅ 连续读整块行: 最快变化的维度 offs_k 匹配内存布局 (row-major 时 k 是连续维)
a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
#   即: offs_k[None,:] 是最快维 → 连续地址 → DMA 一次搬满
# 逐元素 kernel: 扁平化 1D (numel), offs = pid*BLOCK + tl.arange(0, BLOCK) → 全连续
```
**约束/坑**：最快变化的维度必须匹配内存布局；扁平化 1D 时保证 `offs` 连续；我们 matmul 已是连续访问（正确样例）。

---

## 情况B：128-bit（16B）对齐 + Padding

**触发**：对齐警告，或 `main_mem` 流量比预期大（拆分事务）。
**收益**：对齐后避免多次拆分事务，带宽提升。
**怎么查**：搜 "ascend 128-bit alignment memory" / "triton memory alignment padding"。

### ❌ 问题示例代码（N 非 16B 倍数 → 拆分事务）
```python
# ❌ fp16 下 N=10 (20 字节), 非 16B 倍数 → 硬件发起多次拆分事务, 带宽浪费
#    或指针首地址非 16B 对齐 → 抛异常/多次事务
N = 10    # fp16: N×2 = 20 字节, 非 16 倍数
```
**出现的问题**：昇腾要求 16B(128-bit) 对齐。列数 N 或指针不满足 → 硬件拆分多次事务（每个事务只搬部分）→ 带宽骤降甚至异常。

### ✅ 修改后正确代码（Pad 到 16B 倍数）
```python
# ✅ 维度 pad 到 16B 倍数
#   fp16: N×2 是 16 的倍数 → N 是 8 的倍数 (pad N=10 → 16)
#   fp32: N×4 是 16 的倍数 → N 是 4 的倍数
#   指针: 确保 tensor 首地址 16B 对齐 (torch 一般已对齐; 别用非对齐视图)
BLOCK_N = 16    # fp16, 16×2=32B = 16B 的倍数
```
**约束/坑**：BLOCK 16 倍数（Cube）；pad 只影响尾部（mask 处理）；`tl.constexpr` 让编译器知道对齐。

---

## 情况C：L2 复用提升（访问顺序 / 权重预排）

**触发**：`l2_hit_rate` 低、`main_mem_read` 高（同一 B 块被反复从 HBM 搬）。
**收益**：L2 复用优化大 GEMM >10%；权重预排 +5~15%。
**怎么查**：搜 "triton L2 reuse grouped ordering weight preshuffle"。

### ❌ 问题示例代码（row-major 顺序 → B 块被挤出 L2）
```python
# ❌ row-major: 处理完一行所有 tile 再下一行 → 下一行的 B 块已被挤出 L2, 每行重新从 HBM 搬
pid_m = pid // grid_n
pid_n = pid % grid_n
```
**出现的问题**：默认 row-major 调度，处理第 2 行时矩阵 B 的 tile 已被挤出 L2 → 重复从 HBM 读 → `main_mem_read` 高、`l2_hit_rate` 低。

### ✅ 修改后正确代码（grouped swizzle / 权重预排）
```python
# ✅ 方案1: grouped swizzle (完整代码在 tier3 §三) → 相邻 program 复用 L2 里的 B 块
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_SIZE_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
pid_m = first_pid_m + (pid % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m

# ✅ 方案2 (权重预排, 不改 kernel): 主机端把 W 排成 [K//BK, N//BN, BK, BN] 布局
#   让每个 tile 连续加载, +5~15% 访存效率
```
**约束/坑**：`GROUP_SIZE_M` 8/16；swizzle 属于 Tier3（本层引用）；权重预排是 host 端一次重排（推理场景划算，训练每轮要重排）。

---

## 情况D：双缓冲/流水线隐藏访存延迟 + UB 控制

**触发**：`aic_mte2/mte3_time` 高但 `cube` 空等（compute bubble）；或 UB 溢出。
**收益**：流水线让搬运与计算重叠，隐藏 HBM 延迟（Ascend 默认双缓冲，但需循环结构清晰）。
**怎么查**：搜 "triton double buffering pipelining load compute overlap" / "昇腾 UB double buffer"。

### ❌ 问题示例代码（load 与 compute 强依赖 → 编译器没法流水线）
```python
# ❌ 循环内 load 结果立刻被同一个表达式用, 且没有清晰的"load 早于使用"结构
for k in range(0, K, BLOCK_K):
    val = tl.load(a_ptrs) * factor + tl.load(b_ptrs)   # load→compute 紧耦合
    acc += tl.dot(val, ...)                             # 每轮等 load 完才能算
# 编译器无法预取下一轮数据 → 计算空等搬运
```
**出现的问题**：搬运和计算串行（每次等 load 完成才算），没有流水线 → `mte` 忙但 `cube` 空等。UB 太大也会让双缓冲没空间。

### ✅ 修改后正确代码（保持循环结构清晰让编译器流水线）
```python
# ✅ load 独立成清晰步骤, 循环结构规整 → 编译器识别双/三缓冲 (Ascend 默认)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)   # load (MTE2)
    b = tl.load(b_ptrs, mask=..., other=0.0)   # load
    acc = tl.dot(a, b, acc)                    # compute (cube) — 与下一轮 load 重叠
    a_ptrs += BLOCK_K * stride_ak              # 常量步进 → 编译器预取
    b_ptrs += BLOCK_K * stride_bk
# 约束: UB ≤ 192KB/n_bufs (双缓冲时 ÷2); 计算逻辑与加载逻辑避免强依赖
```
**约束/坑**：triton-ascend 默认开双缓冲，不用手动 `num_stages`；保持循环清晰；UB 别塞满（留双缓冲空间）。

---

## 情况E：小传输合并（零散小 load → 一次大 load）

**触发**：`gm_to_ub` 里多个地址连续的 <1KB 小传输。
**收益**：合并后总线调度开销降，吞吐提升。
**怎么查**：搜 "triton merge small transfers coalesced load"。

### ❌ 问题示例代码（多次小 load）
```python
# ❌ 3 次独立小 load (地址连续) → 3 次总线请求
bias  = tl.load(params_ptr + 0)
scale = tl.load(params_ptr + 1)
shift = tl.load(params_ptr + 2)
```
**出现的问题**：地址连续的多个小传输，每次单独发总线请求 → 调度开销高。

### ✅ 修改后正确代码（一次大 load，寄存器切片）
```python
params = tl.load(params_ptr + tl.arange(0, 3))   # ✅ 一次连续 load
bias, scale, shift = params[0], params[1], params[2]
```
**约束/坑**：仅当地址连续、mask 一致才合并；合并不连续地址会读到错数据。

---

## 实战案例（真实 910B3 排障）：跨步访问 → 88× 差距与修复

> **来源**：attention_mlp / flash_attention / conv2d 优化后 vs PyTorch 出现几十倍差距。
> **排查结论**：测量链路先核对无误（warmup×3 + 一次 msprof 内 KERNEL_LOOP=30 次 ÷30；PyTorch 侧 warmup×3 + Event 30 次 forward ÷30，两者对齐）→ 差距真实 → 根因全是 **DMA 跨步访问**（情况A 的真实实例）。
>
> **判断铁律**：`tl.load` 的 tile 若**最后一维（innermost）stride ≠ 1** → 必跨步，必慢。对照工作正常的 matmul：`b_ptrs = b_ptr + (offs_k[:,None]*stride_bk + offs_n[None,:]*stride_bn)`，`stride_bn=1`。

### 案例1：Q@K^T 指针转置 → 88× 慢（attention_mlp.attention_scores_kernel）

**触发**：`gm_to_ub` 带宽低到 ~1%，attention_scores_kernel 占端到端 99%。
**现象**：attention_mlp 端到端 **158,823us** vs PyTorch **1,793us**（88×），其中 attention_scores 就 **156,218us**。
**根因**：K^T 用指针直接转置，tile 最后一维 n 的 stride = `dim` = **2048 元素（8KB）** → DMA 搬 32B+ 整条 cache line 只用 1 个 float，其余"运空气"。

### ❌ 问题示例代码
```python
# ❌ 指针直接转置: K^T[k,n] = K[n,k], tile 最后一维 n stride = dim = 2048 (8KB)
k_ptrs = k_ptr + (offs_n[None, :] * dim + offs_k[:, None])   # 每行只取 1 个, 运空气
```
**出现的问题**：inner-loop 每次读 K 都是 8KB 跨步 → 带宽利用率 ~1%。数值正确但慢 88×，且看不见（matmul 长得一样却很快，因为 B 矩阵 `stride_bn=1`）。

### ✅ 修改后正确代码（host 端预转置）
```python
# host (main): 预分配 k_t, 每轮 K 更新后转置 (真实部署 K 每轮变, 必须付)
k_t = torch.empty(dim, seq, dtype=DTYPE, device=npu)   # 分配放 loop 外, 只一次
for _ in range(LOOP):
    ...
    k_t.copy_(k.T)                                     # k.T 是 view, copy_ 写入预分配 buffer
    attention_scores_kernel[g_scores](q, k_t, s, seq, dim, scale, BLOCK_*=...)

# kernel: K_t[k,n] = k_ptr + k*seq + n, innermost(n) stride=1 → 连续突发 ✓
k_ptrs = k_ptr + (offs_k[:, None] * seq + offs_n[None, :])
for k in range(0, dim, BLOCK_K):
    ...
    k_ptrs += BLOCK_K * seq        # ★布局变了, 递增也要乘 seq (前进一个 k-block)
```
**约束/坑**：
- 转置 kernel 每轮 ~100us（16MB fp32），是**诚实成本**；PyTorch/CANN matmul 原生支持 transB 不用转置 → 这是剩余差距
- `k_t` 分配必须放 loop 外，否则每轮分配计入测量
- 只转置被跨步访问的矩阵，其余保持原布局
- 已用 torch CPU 模拟核对索引数学：`max_err=1.9e-6`（vs 参考 S=Q@K^T·scale）

---

### 案例2：Flash-Attention 每头 K^T 加载（flash_attn_mha_kernel，含"缩 stride ≠ 消除"教训）

**触发**：`gm_to_ub` 低；flash_attention 端到端 **25,785us** vs PyTorch **1,012us**（25×）。
**根因**：布局 `[seq, nh, dim]` 时，K 块 tile `(BLOCK_K, BLOCK_N)` 最后一维 n 的 stride = `nh*dim` = 512 元素（2KB）。

### ❌ 问题示例代码（含第一版"不彻底修复"的坑）
```python
# ❌ 第一版: 只把布局换成 [nh, seq, dim], tile 最后一维 n 的 stride 缩到 dim=64 (256B)
#    教训: 缩小 stride ≠ 消除 stride — innermost 仍非 1, DMA 依然不能突发
k_ptrs = k_ptr + head * (seq * dim) + offs_n[None, :] * dim + offs_k[:, None]
```

### ✅ 修改后正确代码（K 预转置成每头 K^T）
```python
# host: 只转置 K;  Q/V 用 [nh,seq,dim] 天然连续, 不动
q_t = q.permute(1, 0, 2).contiguous()   # [nh, seq, dim]
k_t = k.permute(1, 2, 0).contiguous()   # [nh, dim, seq]  ← 每头 K^T
v_t = v.permute(1, 0, 2).contiguous()   # [nh, seq, dim]

# kernel: K_t[h,k,n] = head*(dim*seq) + k*seq + n, innermost(n) stride=1 ✓
k_ptrs = k_ptr + head * (dim * seq) + offs_k[:, None] * seq + offs_n[None, :]
```
**约束/坑**：
- 转置是 host 一次性（在测量 loop 外），不占每轮；只有 K 需要转置
- verify 段 permute 必须跟着改：`k_ref = k_t.permute(2, 0, 1)` 还原 `[seq, nh, dim]`
- 已核对索引数学：`rel=1.0e-7`（vs 逐头 torch 因果注意力参考）

---

### 案例3：Conv2D 权重跨步重读（conv2d / conv_bias_relu，已定位、待算法重构）

**触发**：conv2d **484us** vs PyTorch **40.6us**（12×）；conv_bias_relu **306us** vs **89.2us**（3.4×）。
**根因（两层）**：
1. **权重跨步重读**：c/r/s 三重循环内 `offs_k * C*R*S`，stride=72 元素（288B），72 次重复 → 9KB 权重被 64 program × 72 次跨步读
2. **没用 Cube**：`acc += wv[:,None] * xv[None,:]` 是向量外积（AIC 向量单元），没走 `tl.dot`（Cube 矩阵单元）→ 算力用不上

### ❌ 问题示例代码
```python
for c in range(C):
    for r in range(R):
        for s in range(S):
            xv = tl.load(x_ptr + n*C*H*W + c*H*W + ih*W + iw, mask=valid, other=0.0)  # [BLOCK_OW] 连续✓
            wv = tl.load(w_ptr + offs_k*C*R*S + c*R*S + r*S + s, mask=k_mask, other=0.0)  # [BLOCK_K] stride=72 跨步✗
            acc += wv[:, None] * xv[None, :]        # 向量外积, 无 tl.dot → 不上 Cube
```

### ✅ 修复方向（★未定稿，实现并验证后更新本节）
- **im2col**：host 端把输入 patch 重排成 `[C*R*S, BLOCK_OW]` 连续块 + 权重 `[BLOCK_K, C*R*S]` 连续块，kernel 内 `acc = tl.dot(W, X_patch)` 走 Cube（对应 Tier1 算法层）
- 或**权重预排**：host 把 W 排成 `[C*R*S, K]` 连续，让每个 `wv` 变连续 `[BLOCK_K]` 读（消除跨步，但仍是向量外积，治标不治本）
**约束/坑**：**未验证的修复不写进"✅正确代码"** —— 本案例只记录诊断与方向；遇到同款先确认算子是否真的用上了 Cube（`tl.dot`），再看访存。

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| 跨步访问 | 带宽 ~10% | 连续化（最快维匹配布局） |
| 转置 matmul 用指针直接转置（Q@K^T 等） | 带宽 ~1%，慢 88× | host 端预转置 K，kernel 内 innermost stride=1（见实战案例1/2） |
| 只缩 stride 不消除（256B/512B 仍跨步） | 只改善几倍 | 预转置让 innermost=1，别停在"缩小" |
| 非 16B 对齐 | 拆分事务/异常 | pad N 到 16B 倍数 |
| row-major 无 L2 复用 | l2_hit_rate 低 | swizzle / 权重预排 |
| load 与 compute 强耦合 | cube 空等 | 循环结构清晰让编译器流水线 |
| 合并不连续小传输 | 读错数据 | 仅连续地址才合并 |
| UB 塞满 | 双缓冲没空间 | UB ≤ 192KB/n_bufs |
