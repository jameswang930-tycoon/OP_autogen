# Triton 优化 Tier 3（分块配置）策略指南 — 针对 triton-ascend 910B3

> 本层**只调分块/调度参数**（BLOCK_M/N/K、BLOCK_SIZE、group swizzle、sub-block、mask），
> **不改算法、不融合、不改精度**（Tier1/2/5）。违反 = 越层，fail。
>
> **★环境铁律（triton-ascend，违反必报错）**：
> - `num_warps` / `num_stages` **禁止**传给 kernel（传了报 `please do not tune args`）
> - `@triton.autotune` 可用 `configs=[]` 触发后端自动分块生成，但当前 pipeline 使用自定义 sweep（sweep_blocks.py）做全面确定性扫描，**外部 tuner 不要传显式 configs list**
> - **分块必须 16 倍数**（Cube MMA 16×16 基础粒度；非 16 倍数掉到 ~10% 算力）
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
# 或更小: 96×128×64 (grid≈21×16=336), 但 96=16×6 仍是 16 倍数
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
BLOCK_M, BLOCK_N, BLOCK_K = 96, 96, 32    # 都是 16 倍数 (96=16×6, 32=16×2)
# 或 128, 128, 64
```
**约束/坑**：BLOCK_M/N/K 全部 16 倍数（64/80/96/112/128...）。

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

## 八、禁忌 + 收益 + 常见错误

**禁忌**：禁 num_warps/autotune/非 16 倍数/超 L0·UB/计算瓶颈乱调/算法·融合·精度（越层）。
**收益**：BLOCK_K 增 10~40%；BLOCK_M/N 增 10~30%；**swizzle 15~100%**；EVEN_K 5~15%；care_padding 5~15%。

| 错误 | 现象 | 修复 |
|---|---|---|
| BLOCK 超 L0/UB | `ub overflow` 编译失败 | 先算 L0A/B≤64KB、L0C≤128KB、UB≤192KB |
| 非 16 倍数 | cube 利用率 ~10% | BLOCK 改 16 倍数 |
| swizzle 漏 min | 越界/错 tile | `group_size_m = min(grid_m-first_pid_m, GROUP_SIZE_M)` |
| EVEN_K 非 constexpr | mask 没省掉 | 调用处算好传 constexpr |
| SUB_BLOCK 整数除 | 尾部漏处理 | `tl.cdiv` + 尾部 mask |
| care_padding 误用 | 垃圾值 | 仅未填充值不影响结果时用 |
| grid < 40 | 核没吃满 | 减 BLOCK_M/N 或减 BLOCK_SIZE |
| compute_bound 还调分块 | 浪费轮次 | promote Tier1/5 |
