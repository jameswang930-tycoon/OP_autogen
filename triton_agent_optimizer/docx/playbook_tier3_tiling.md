# Triton 优化 Tier 3（分块配置）策略指南 — 针对 triton-ascend 910B3

> ## ★★速查卡（★先读这个；你实际能读全文，但速查卡在最前面=最快定位，决策先靠它，细节看情况A~J）★★
>
> **工作流（3 步，别跳）**：
> 1. ★本层第一动作是看 `09_tier3_sweep/sweep_result.json`（框架已枚举全部 L0 合法 BLOCK 并真机实测）——**有实测数据就照数据，别手工猜 BLOCK**
> 2. 读 `07_tier3_fields/tier3_fields.txt` 找 **block_dim / mte1 / mte2 / cube / bottleneck_type**
> 3. 无 sweep 数据才照下表手工判断 → 输出 changes[]（old_code 逐字复制）
>
> **字段 → 动作速查表（★主决策表）**：
> | 字段/现象 | 动作（一句话） |
> |---|---|
> | sweep_result.json 存在 | ★用它的 best 块（别猜新值；sweep 已是全链路实测最优） |
> | mte2 高 / cube 低 / l0a/l0b_read 低 | 增 BLOCK_M/N（传输瓶颈搬更大块） |
> | mte1 高 | 增 BLOCK_K |
> | block_dim < 40 | 减 BLOCK_M/N（核没吃满） |
> | memory_bound 且 mem_util≥0.8 | ★别调块（带宽已到平台）→ promote |
> | compute_bound | ★不调块 → promote Tier1/5（附 evidence） |
> | compute_utilization < 0.3 | 回 Tier1（是算法问题，不是分块） |
> | 上面全不触发 | 分块已最优 → promote（附 evidence） |
>
> **★硬约束（改 BLOCK 前必算，违反=编译失败或设备崩）**：
> - **全部 2 的幂**（16/32/64/128/256/512/1024；非 2 幂 tl.dot/tl.arange 会报错/降级）；L0A/B = `BM×BK×dtype ≤ 64KB`；L0C = `BM×BN×dtype ≤ 128KB`；UB ≤ 192KB
> - **★代码对 L0A/B/L0C/UB 一律留 ×0.9 安全余量**（贴边界候选会 OOM 打崩设备 "575:NPU function error" 并污染设备）——sweep 的合法候选已自动排除贴界块，别手工选贴界值
> - fp32 安全示例：128×128×64（L0A/B=32KB✓ L0C=64KB✓）；fp16 可 ×2
> - 行宽对齐：fp32 BLOCK_N∈{128,256}（512B），fp16 BLOCK_N∈{256,512}
> - ★别跳最大合法块（寄存器 spill 断崖）——sweep 实测找拐点
>
> **★我们算子对照（sweep 会扫的）**：
> - matmul 族（matmul/attention_mlp/flash_attention/matmul_relu/matmul_transpose）：扫 (BM,BN,BK)
> - conv 族（conv2d/conv_bias_relu）：扫 (BLOCK_K,BLOCK_OW)，★BLOCK_K ≥ K_OUT(32)
> - flash_attention：BK ≤ 头维 64；grid = ceil(seq/BM)×nheads
> - rms_norm/layernorm/sigmoid/vector_add/fused_add_mul/softmax：无自由分块参数 → sweep 跳过，无 BLOCK 可改 → 直接 promote
>
> **禁止（违反=失败）**：改 DTYPE/算法/融合（跨层）、num_warps/num_stages/autotune、非 16 倍数、超 L0/UB、compute_bound 还调块、**手工改一个新 BLOCK 覆盖 sweep 实测最优**。

---

> 本层**只调分块/调度参数**（BLOCK_M/N/K、BLOCK_SIZE、group swizzle、sub-block、mask），
> **不改算法、不融合、不改精度**（Tier1/2/5）。违反 = 越层，fail。
>
> **★环境铁律（triton-ascend，违反必报错）**：
> - `num_warps` / `num_stages` **禁止**传给 kernel（传了报 `please do not tune args`）
> - `@triton.autotune` 可用 `configs=[]` 触发后端自动分块生成，但当前 pipeline 使用自定义 sweep（sweep_blocks.py）做全面确定性扫描，**外部 tuner 不要传显式 configs list**
> - **分块必须 16 倍数**（Cube MMA 16×16 基础粒度；非 16 倍数掉到 ~10% 算力）；**★优先 2 的幂**（16/32/64/128/...，sweep 只枚举 2 幂，非 2 幂 tl.dot/tl.arange 可能报错/降级）
> - `BLOCK_*` 必须 `tl.constexpr`
> - L0 上限：L0A/B=64KB、L0C=128KB；UB=192KB → **超过就 `ub overflow` 编译失败**

> **★速查表（每次动 BLOCK 前先看这个）**：
> **瓶颈 → 块动作（★传输是瓶颈 → 搬运更大块）**：
> - `mte2` 高（GM→L1 是瓶颈）或 `cube` 低或 `l0a/l0b_read` 低 → **搬运瓶颈 → 增 BLOCK_M/N**（一次多搬点、减少搬运次数）
> - `mte1` 高（L1→L0A/B 是瓶颈）→ **增 BLOCK_K**（K 方向一次多搬，提高 L0 数据复用）
> - `memory_bound` 且带宽已接近峰值（mem_util≥0.8）→ **别调块**（已到带宽平台，怎么调都是噪声）
> - `compute_bound`（cube 已满）→ **不调块**，promote
>
> **硬件最大块（★超过 = `ub overflow` 编译失败；必须先算再选）**：
> | 上限 | 公式 | fp32 最大示例 | fp16 最大示例 |
> |---|---|---|---|
> | L0A | `BM×BK×dtype≤64KB` | 128×128 (64KB满) | 256×128 (64KB满) |
> | L0B | `BN×BK×dtype≤64KB` | 同 L0A | 同 L0A |
> | L0C | `BM×BN×dtype≤128KB` | 128×128 (64KB) / 256×128 (128KB满) | 256×256 (128KB满) |
> | UB(逐元素) | `BLOCK×dtype×bufs≤192KB` | 16384×4×3=192KB满 | 8192×2×3=48KB |
> **改块流程**：① 判断瓶颈在哪个搬运引擎 → ② 按上表先算最大允许值 → ③ 取"不超限 + 接近带宽平台"的块 → ④ 全部保持 16 倍数
>
> **★实战教训（rms_norm 17 轮无果根因）**：
> - 若 `bottleneck_type = memory_bound` 且 `memory_utilization` 已 ≥80%（带宽接近峰值）→ **别调 BLOCK**，
>   分块已到记忆体带宽上限，怎么调都是噪声（±1% 波动）
> - 此时应 `promote_to` 回算法层(Tier1)确认算法，或直接晋升；**绝不在本层反复调 BLOCK 空转 3 轮**

---

## 一、诊断触发规则（v4 字段 → 动作）+ 决策流程图

| v4 字段 | 触发 | 动作 |
|---|---|---|
| `task.block_dim` | **< 40**（核没吃满） | 减 BLOCK_M/N（§二-B） |
| `engine_utilization.mte1` | 高（L1→L0 搬运瓶颈） | 增 BLOCK_K（§二-A） |
| `engine_utilization.mte2` | 高（GM→L1 瓶颈） | 增 BLOCK_M/N（§二-A） |
| `engine_utilization.cube` | 低（cube 没满） | 增 BLOCK_M/N（§二-A） |
| `bandwidth_gb_s.l0a/l0b_read` | 低 | 增 BLOCK_M/N/K（§二-A）+ swizzle（§三） |
| `roofline.bottleneck_type` | `compute_bound` | **不调分块**，promote Tier1/5（§二-C） |
| `roofline.compute_utilization` | 极低(<0.3) | **回 Tier1 查算法**，不是分块 |

```
compute_utilization<0.3 → 回 Tier1;  compute_bound → promote Tier1/5
block_dim<40 → 减 BLOCK (§二-B);  mte1高 → 增K;  mte2高/cube低/l0a低 → 增MN (§二-A)
memory_bound → 增 tile + swizzle (§三);  否则 promote
```

---

## §二-A：BLOCK_M/N/K 调优（matmul 核心）

**触发**：mte1/mte2 高 或 cube 低 或 l0a/l0b 低（传输/利用率瓶颈）。
**怎么查**：搜 "triton matmul tiling L0C overflow" / "triton BLOCK_K ub overflow"。

### ❌ 问题示例代码（盲目增大 BLOCK → L0 溢出）
```python
# ① config — ❌ fp32 下 BLOCK_M=256, BLOCK_K=128 超 L0A
BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 128
# L0A = 256×128×4 = 128KB > 64KB(上限) → 编译报 ub overflow / L0 overflow
```
**出现的问题**：只想着"增大分块提复用"，没算 L0A/L0B/L0C 上限。fp32 时 `BLOCK_M×BLOCK_K×4 ≤ 64KB`、`BLOCK_N×BLOCK_K×4 ≤ 64KB`、`BLOCK_M×BLOCK_N×4 ≤ 128KB`，超了就编译失败。

### ✅ 修改后正确代码（先算约束再选值）
```python
# fp32: 保证 L0A/B ≤ 64KB, L0C ≤ 128KB
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
#   L0A=128×64×4=32KB ✓  L0B=32KB ✓  L0C=128×128×4=64KB ✓
# 想更大: 512×64×64 (L0C=512×64×4=128KB 满) 或 128×128×128 (L0A=64KB 满)
# fp16: BK 可更大 (×2 字节), 如 256×128×128 (L0A=64KB 满)
```
**约束/坑**：分块 vs 带宽是**饱和曲线**——小分块程序开销主导（带宽≈分块翻倍），大分块记忆体带宽封顶（变平）。还在翻倍 = 没到平台可继续增；变平 = 到平台，再增无用。**BLOCK 必须 16 倍数**。

### 寄存器溢出悬崖（tile 过大性能崩塌，★为什么数据驱动是唯一可靠手段）

**触发**：只想着"越大越好"，BLOCK_M/N 过大 → 累加器 `BLOCK_M×BLOCK_N` 个 fp32 寄存器超出硬件可容纳 → **spill 到慢速内存，性能断崖**。
**怎么查**：搜 "triton register spilling matmul large block"。

### ❌ 问题示例代码（tile 过大 → 崩塌）
```python
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32   # fp32: acc=128×128=16384 个寄存器
# 某 GPU 实测: 32×32×32=2.77 TFLOPS → 128×128×32=0.238 TFLOPS (崩 10×+)
# 原因: 累加器寄存器超限 → spill; 且 grid 缩到 16×16=256, 占用下降
```
**出现的问题**：L0C 没超（128×128×4=64KB < 128KB），但**寄存器/L0C 压力 + 占用下降**双重打击 → 性能比小 tile 差一个数量级。**约束表只保证"编译能过"，不保证"性能好"**——这就是必须实测（sweep）而非手工推算的原因。

### ✅ 修改后正确代码（饱和曲线的实测段选块）
```python
# 实测饱和点: 从小块起, 每轮 sweep 记录 ns, 找到"还在降"→"变平"的拐点
#   拐点前: 带宽≈分块×2 (程序开销主导); 拐点后: 带宽封顶 (变平, 再增无用)
# 不要一次性跳到最大合法块 (可能跨过拐点掉进寄存器悬崖)
```
**约束/坑**：①**先算 L0/UB 上限**（编译能过）②再**实测找拐点**（性能好）；③Ascend 额外注意 **行宽 512B 对齐**（见下）；④别用 `BLOCK_M=BLOCK_N` 对称假设——实测常是非对称更好。

### ★Ascend 512B 行宽对齐（带宽利用率）

**触发**：算子 innermost 行宽不是 512B 整数倍 → 带宽利用率上不去。
**怎么查**：搜 "ascend operator row width 512B alignment bandwidth"。

| 精度 | 512B = 元素数 | 推荐 BLOCK_N 行宽 |
|---|---|---|
| fp32 | 128 元素 | BLOCK_N ∈ {128, 256}（128×4=512B✓） |
| fp16/bf16 | 256 元素 | BLOCK_N ∈ {256, 512}（256×2=512B✓） |

**约束/坑**：行宽 = 每行 innermost 连续元素数（matmul 的 BLOCK_N）。**配合** §二-A 的 L0C 约束 + 饱和曲线实测一起选；512B 对齐是带宽前提，不是唯一因素。

---

## §二-B：block_dim < 40 → 缩小 BLOCK 增并行

**触发**：`block_dim` < 40（20 AI Core），grid 太小核闲着。
**怎么查**：搜 "triton grid too small block dimension"。

### ❌ 问题示例代码（减 BLOCK 减到非 16 倍数）
```python
# ① config — ❌ 为增大 grid 把 BLOCK 减成 100
BLOCK_M, BLOCK_N, BLOCK_K = 100, 128, 64   # 100 不是 16 倍数 → cube 掉到 ~10% 算力
```
**出现的问题**：目的对（减 BLOCK 让 grid 变大），但 `100` 非 16 倍数 → Cube MMA 无法触发 → 性能反而崩。

### ✅ 修改后正确代码
```python
# grid = M/BM × N/BN, 减 BM 增 grid, 但保持 16 倍数
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64     # M=N=2048: grid=16×16=256 (核吃满)
# 或更小: 64×128×64 (grid=32×16=512), 保持 2 的幂
```
**约束/坑**：减到够用就行（block_dim≥40）；太小 grid 超大 → 调度开销反升（见 §六 SUB_BLOCK）。

---

## §二-C：计算瓶颈（compute_bound）→ 不调分块

**触发**：`bottleneck_type=compute_bound`（comp≥0.8 且 mem<0.5）→ cube 已满。
**怎么查**：搜 "compute bound matmul tiling no benefit"。

### ❌ 问题示例（错误动作：compute_bound 下还调 BLOCK）
```python
# ❌ compute_bound 时: cube 已 100% 满, 调 BLOCK_K 64→128 徒劳
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64   # 改 128, 128, 128 → L0 可能溢出 + 无收益
# 加速比不变 (cube 已满, 分块对吞吐无帮助)
```
**出现的问题**：compute_bound 下分块已到收益上限，继续调 BLOCK 浪费轮次，甚至引发 L0 溢出。**分块不是瓶颈**。
### ✅ 正确动作（不调分块，promote）
```python
# promote=true, promote_to=1 (算法: fp16 计算 / cube_fp16_ratio 低)
# 或 promote_to=5 (计算占用: 冲突/标量拖累)
# 铁律: 计算瓶颈下禁止乱调 BLOCK 假装优化
```
**约束/坑**：`compute_utilization` 低但算力已满 → 算法选错（Tier1），不是分块。

---

## §二-D：逐元素/softmax 的 BLOCK_SIZE（vector 引擎）

**触发**：`engine_utilization.vec` 低 或 grid 太大（调度开销）。
**怎么查**：搜 "triton BLOCK_SIZE ub overflow elementwise"。

### ❌ 问题示例代码（BLOCK_SIZE 太大 → UB 溢出）
```python
# ① config — ❌ bias_gelu 3 缓冲, BLOCK_SIZE 32768 超 UB
BLOCK_SIZE = 32768
# bias_gelu 读 x + 读 bias + 写 h = 3 缓冲 → 32768×4×3 = 384KB > 192KB → ub overflow
```
**出现的问题**：逐元素 kernel 的 UB 占用 = `BLOCK_SIZE × 字节 × n_bufs`。3 缓冲的 bias_gelu 上限 16384（×4×3=192KB 满），32768 直接溢出。

### ✅ 修改后正确代码
```python
BLOCK_SIZE = 8192     # 8192×4×3 = 96KB < 192KB (安全)
# 或 BLOCK_SIZE = 16384 (×4×3=192KB 极限, 可能报错→fail-fast)
# softmax 特例: softmax_kernel 每行一个 program, BLOCK_S ≥ seq (我们 seq=2048 → BLOCK_S=2048)
```
**约束/坑**：BLOCK_SIZE 16 倍数；想更大用 §六 SUB_BLOCK 内层分片。

---

## §三：★Group Launch / Swizzle（GROUP_SIZE_M，大 GEMM 最高收益）

**触发**：`l0a/l0b_read_gb_s` 低、`main_mem_read` 高（GM 反复搬同一 B 块）。
**收益**：vLLM 实测 1.17~1.98×。
**怎么查**：搜 "triton matmul group pid swizzle GROUP_SIZE_M"。

### ❌ 问题示例代码（swizzle 漏边缘处理 → 越界/错 tile）
```python
    pid = tl.program_id(axis=0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    num_pid_in_group = GROUP_SIZE_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    pid_m = first_pid_m + (pid % GROUP_SIZE_M)     # ← BUG: 没算 group_size_m
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
```
**出现的问题**：`grid_m` 不是 `GROUP_SIZE_M` 整数倍时（如 grid_m=5, GROUP_SIZE_M=8），最后组的 `pid_m = first_pid_m + (pid % 8)` 会**越界到 grid_m 之外** → 读越界/错 tile。

### ✅ 修改后正确代码（min 处理边缘）
```python
    num_pid_in_group = GROUP_SIZE_M * grid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)   # ✅ 最后不满一组
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```
**完整改动清单**：① config 加 `GROUP_M=8`；② kernel 签名加 `GROUP_SIZE_M: tl.constexpr`；③ 调用传 `GROUP_SIZE_M=GROUP_M`；④ pid 解码换 swizzle。
**约束/坑**：GROUP_SIZE_M=1 退化为 row-major（安全兜底）；`GROUP_SIZE_M` 是 constexpr。

### swizzle 的 Ascend 注意（★收益需真机实测）

- **原理跨架构有效**（L2 复用是通用层次结构），但 **1.17~1.98× 是 vLLM 在 GPU 上实测**，Ascend 910B3 的具体收益**未在我们框架内实测**——改动后用 msprof 对比，收益 < 噪声地板（1.05×）就回退
- **triton-ascend 有 Swizzle2D 变体**（比通用 GROUP_SIZE_M 更适配昇腾的 L2/UB 结构）——若 `GROUP_SIZE_M` 收益不明显，搜 "triton-ascend-case-matmul-swizzle2d" 换 Swizzle2D
- **依赖程序 id 按序 launch**（Triton 调度实现细节，非硬件保证）——实测确认生效再保留
- **配合 §二-A 饱和曲线**：swizzle 只在搬运是瓶颈时有用；compute_bound 下别加（徒增指令）

---

---

## §四：Cube 16×16 对齐

**触发**：`cube_instr_number` 异常低 / cube 利用率上不去但 block_dim≥40。
**怎么查**：搜 "triton tl.dot 16 alignment cube"。

### ❌ 问题示例代码（非 16 倍数）
```python
# ① config — ❌ BLOCK 非 16 倍数
BLOCK_M, BLOCK_N, BLOCK_K = 100, 96, 40
# 编译器要做布局转换(NC1HWC0 重排) → cube 利用率掉到 ~10%
```
**出现的问题**：Cube MMA 要求 16×16 基础粒度，非 16 倍数触发布局转换 + 冗余搬运 → 算力崩。

### ✅ 修改后正确代码
```python
BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32    # 2 的幂 (sweep 枚举范围)
# 或 128, 128, 64
```
**约束/坑**：BLOCK_M/N/K 全部 16 倍数；**优先 2 的幂**（64/128/256...，80/96/112 这类 sweep 不枚举、可能编译降级）。

---

## §五：EVEN_K 快路径（K % BLOCK_K == 0 免 mask）

**触发**：K 整除 BLOCK_K 但 K 循环仍带边界 mask（浪费）。
**怎么查**：搜 "triton EVEN_K no mask fast path"。

### ❌ 问题示例代码（`if EVEN_K` 不是 constexpr）
```python
# ② kernel — ❌ EVEN_K 是运行时 bool, 不是 tl.constexpr
EVEN_K = (K % BLOCK_K == 0)          # K 是 runtime 参数
for k in range(0, K, BLOCK_K):
    if EVEN_K:                        # 运行时判断 → 两支都编译, mask 没省掉
        a = tl.load(a_ptrs)
    else:
        a = tl.load(a_ptrs, mask=offs_k[None,:] < K-k, other=0.0)
```
**出现的问题**：`EVEN_K` 若不是 constexpr，`if` 在运行时判断、**两支都生成**，省不了 mask 开销。

### ✅ 修改后正确代码
```python
# ① config: EVEN_K 由调用处按 K 算好传入 (constexpr)
# ② kernel 签名加: EVEN_K: tl.constexpr
# ③ 调用: EVEN_K=(K % BLOCK_K == 0)
# ② kernel:
    for k in range(0, K, BLOCK_K):
        if EVEN_K:                    # ✅ constexpr 分支, 编译期消掉另一支
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None,:] < (K-k), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:,None] < (K-k), other=0.0)
        acc = tl.dot(a, b, acc)
```
**约束/坑**：`if EVEN_K:` 是 constexpr 分支，编译器只保留一支，零运行时开销。

---

## §六：两层分块（SUB_BLOCK）— UB 溢出/调度开销

**触发**：编译报 `ub overflow`，或 grid 比核数大太多调度慢。
**怎么查**：搜 "triton SUB_BLOCK tiling ub overflow"（43µs→7µs 案例：BLOCK 匹配核数 + SUB_BLOCK 控 UB）。

### ❌ 问题示例代码（整数除法漏余数）
```python
# ② kernel — ❌ num_sub 用整数除法, BLOCK 不是 SUB_BLOCK 整数倍时漏处理
num_sub = BLOCK_SIZE // BLOCK_SIZE_SUB        # 假设整除 → 若有余数, 尾部元素没处理
for i in range(num_sub):
    offs = base + i*BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
    ...
```
**出现的问题**：`BLOCK_SIZE=25000, BLOCK_SIZE_SUB=10000` 时，`25000//10000=2`，剩 5000 元素没处理 → 尾部结果错。

### ✅ 修改后正确代码（cdiv 向上取整 + mask）
```python
num_sub = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)   # ✅ 向上取整
for i in range(num_sub):
    offs = base + i*BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
    mask = offs < n_elements                     # ✅ 尾部 mask 兜底
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x*2.0, mask=mask)
```
**约束/坑**：`BLOCK_SIZE_SUB×dtype ≤ 192KB/n_bufs`（double buffer 时 ÷2）；外层 BLOCK 决定 grid（匹配核数）。

---

## §七：care_padding=False（去 mask 默认填充）

**触发**：masked load 造成 MTE2 与 vector 的依赖，`vec` 占比低。
**怎么查**：搜 "triton care_padding=False"。

### ❌ 问题示例代码（误用：填充值被后续用到）
```python
# ❌ 若 mask 外的填充值后续被读 → care_padding=False 会读到垃圾
x = tl.load(x_ptr + offs, mask=mask, care_padding=False)   # 未填充区域=未定义
y = tl.where(use_other, x, other_val)    # ← 若 use_other 覆盖 x 的填充区, 这里读到垃圾
```
**出现的问题**：`care_padding=False` 跳过默认置零，**未填充区域的值未定义**。如果后续计算读那些位置 → 垃圾值。

### ✅ 修改后正确代码（确认安全才用）
```python
# ✅ 仅当 mask 未覆盖部分不影响结果时用 (如: 结果只写 mask 内的位置)
x = tl.load(x_ptr + offs, mask=mask, care_padding=False)   # 提速
tl.store(out_ptr + offs, x * 2.0, mask=mask)               # store 也 mask → 垃圾不外泄
# ⚠ triton-ascend 若版本不支持该参数 → 去掉 (宁缺勿错, fail-fast)
```
**约束/坑**：安全条件 = 未填充值绝不参与后续结果；否则保留默认 padding。

---

## §八：★分块实测 sweep（本层优先手段，数据驱动不是猜）

> **本层第一动作是跑 sweep，不是手工推 BLOCK。** §二-A 到 §七 的手工判断只在 sweep 不可用时兜底。
> 框架自动跑 `sweep_blocks.sweep()`：**程序化枚举全部 L0 合法 (BM,BN,BK) 候选（全部 2 的幂、L0/UB 带 ×0.9 余量）→ 单进程 torch.npu.Event 实测每 config warmup3+计时10 取平均 → 最优写回 kernel_op.py**。
> ★sweep 在 **`outputs/<op>/best_kernel.py`（历史最优）** 上扫（不在当前 kernel/源文件上）；写回的是 **round_dir 副本 kernel_op.py**（不碰 `input/<op>` 源文件），sweep 后 current_kernel 指向该副本——**别去读 input 源文件看 BLOCK**，那是最初的值。

**sweep 跑什么**：
- matmul 族（matmul/attention_mlp/flash_attention/**matmul_relu/matmul_transpose**）：枚举 `(BM,BN,BK)`，约束 = L0A/B≤64KB、L0C≤128KB、UB≤192KB（全部 ×0.9 余量）、**全部 2 的幂**、grid∈[16,3000]
- conv 族（conv2d/conv_bias_relu）：枚举 `(BLOCK_K,BLOCK_OW)`，**BLOCK_K ≥ K_OUT**（否则只算一半通道，见 tier4 案例3）
- **flash_attention 特殊**：BK 上限 = 头维 dim（BK>64 无意义，K 循环就 64 长）；★`bk_min = ceil_pow2(dim)`（BK<dim 只算部分头维 → 分数/输出数值错且计时偏快可能被误选）；grid = ceil(seq/BM)×nheads
- rms_norm/layernorm/rms_norm_residual/sigmoid/vector_add/fused_add_mul/softmax：**无自由分块参数**（行级/逐元素）→ sweep 跳过，无 BLOCK 可改 → 直接 promote

**怎么读 `round_dir/09_tier3_sweep/sweep_result.json`**：
```json
{"available": true, "vars": ["BLOCK_M","BLOCK_N","BLOCK_K"],
 "configs": [{"block":[64,64,64], "ns": 123456, "speedup": 1.2, "is_current": true}, ...],
 "best": {"block":[128,128,64], "ns": 100000, ...}}
```
- `configs[]` 按 ns 升序，`[0]` = 最优；`is_current` 标当前块；**`speedup` = 相对「当前块」的加速比**（`cur_ns ÷ 该候选 ns`，当前块恒 = 1.0）——**比大小看 ns**，别把 speedup 当相对 round1 基线的累计加速比
- **决策**：最优明显快于当前 → `changes[]` 直接采用最优块（数据就是答案，别猜新值）；当前已接近最优/无增益 → 分块已到位，promote 下一层（附 evidence）

**★多 kernel 共享一个 BLOCK 的协调**（attention_mlp 9 kernel / MLP 3 kernel）：
- 多个 kernel 共用 `BLOCK_M/N/K`，**单 kernel 最优 ≠ 全链路最优**（matmul 爱大 tile、softmax 爱小 tile）
- **sweep 测的是全链路**（runner 一次跑完所有 kernel 计时）——它已经帮你协调了多 kernel 的取舍，**信任 sweep 的全链路结果，别手工改成单 kernel 最优**
- 手工调 BLOCK 时同理：改动后必须**整算子 verify**（不是只看一个 kernel 的 ns）

**约束/坑**：
- sweep 写回最优块到当前 kernel，planner 决策基于实测数据；**coder 不要手工改一个新 BLOCK 覆盖掉 sweep 结果**（除非有明确理由，比如 sweep 报错）
- sweep 用 torch.npu.Event 计时（非 msprof）——相对排序可靠，绝对 ns 别用于跨轮对比
- sweep 报错/无候选 → 回退 LLM + §二-A/B 手工兜底（不是崩）

---

## 八、禁忌 + 收益 + 常见错误

**禁忌**：禁 num_warps/autotune/非 16 倍数/超 L0·UB/计算瓶颈乱调/算法·融合·精度（越层）。
**收益**：BLOCK_K 增 10~40%；BLOCK_M/N 增 10~30%；**swizzle 15~100%**；EVEN_K 5~15%；care_padding 5~15%。

| 错误 | 现象 | 修复 |
|---|---|---|
| BLOCK 超 L0/UB | `ub overflow` 编译失败 | 先算 L0A/B≤64KB、L0C≤128KB、UB≤192KB |
| tile 过大越过饱和拐点 | 寄存器 spill，性能断崖（实测 10×+） | sweep 找实测拐点，别跳最大合法块 |
| 非 16 倍数 | cube 利用率 ~10% | BLOCK 改 16 倍数 |
| 行宽非 512B 倍数 | 带宽利用率低 | BLOCK_N 对齐（fp32:128, fp16/bf16:256） |
| 绕过 sweep 手工改 BLOCK | 覆盖实测最优 / 猜错 | 优先 sweep 数据；手工只在 sweep 不可用时 |
| 多 kernel 改成单 kernel 最优块 | 全链路变慢 | 信任 sweep 全链路结果；改后整算子 verify |
| swizzle 漏 min | 越界/错 tile | `group_size_m = min(grid_m-first_pid_m, GROUP_SIZE_M)` |
| swizzle 收益未实测就保留 | 白加指令 | msprof 对比，<1.05× 就回退；可试 Swizzle2D |
| EVEN_K 非 constexpr | mask 没省掉 | 调用处算好传 constexpr |
| SUB_BLOCK 整数除 | 尾部漏处理 | `tl.cdiv` + 尾部 mask |
| care_padding 误用 | 垃圾值 | 仅未填充值不影响结果时用 |
| grid < 40 | 核没吃满 | 减 BLOCK_M/N 或减 BLOCK_SIZE |
| compute_bound 还调分块 | 浪费轮次 | promote Tier1/5 |
