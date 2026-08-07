# Triton 优化 Tier 6（910B3 架构专属）策略指南 — 针对 triton-ascend 910B3

> 本层是**最后兜底**：Tier1~5 都做完后，只剩**架构级信号**（引擎失衡/阻塞/冲突/核数不匹配）没解决。
> 本层**不新增优化方向**，而是把架构信号映射到具体该回哪一层做，或做最后的代码风格适配。
>
> **★环境铁律（triton-ascend）**：
> - `num_warps`/`num_stages`/`multibuffer`/`unit_flag` 等是**编译器自动管理**，kernel 代码里**不能设置**（只能看信号，别硬调）
> - 矩阵运算**必须用 `tl.dot`**（否则不触发 Cube → 掉到 ~10% 算力）
> - 数学函数用 `tl.math.*`（`tl.math.tanh`/`tl.math.rsqrt`）；别用 `tl.erf`
> - 循环用静态 `for-range`；维度全 `tl.constexpr`；避免动态 shape/while

---

## 一、诊断触发规则（v4 架构字段 → 动作）

看 `07_tier6_fields.txt`（含全局摘要）：

| v4 字段 | 触发 | 含义 | 动作 |
|---|---|---|---|
| `conflict.wait_ratio` | 高（vec 被阻塞） | 计算在等数据 | 回 Tier4（流水线/load 独立） |
| `conflict.mte_cflt_ratio` | 高 | MTE 搬运冲突 | 回 Tier4/3（传输/分块） |
| `conflict.bank_cflt_ratio` | >4% | UB bank 冲突 | 回 Tier5 F（swizzle/访问） |
| `engine_utilization.cube` | 低但任务是 matmul | cube 没用上 | 检查是否用 `tl.dot`（情况D） |
| `engine_utilization` 整体 | cube/vec 严重失衡 | 结构问题 | 回 Tier2（融合平衡引擎） |
| `task.task_type` | 非 cube 但该算 matmul | 走错引擎 | 用 `tl.dot`（情况D） |
| `task.block_dim` | 远小于 40 | 核没吃满 | 回 Tier3（分块） |
| `task.block_dim` / `api_overhead` | grid 远大于物理核数 / launch 开销大 | 调度/核启动开销 | grid 固定核数 + 核内 stride 循环（情况G） |
| 多核耗时不均 / 计算拖尾 | 尾核空转 | 负载不均衡 | stride 切分（情况H） |
| `aic_cube_wait_ratio` / `wait_ratio` | cube/vec 被阻塞但各 pipe 利用率都不高 | 跨引擎流水气泡 | 循环分块 + 双缓冲 UB 预算（情况I） |
| 纯 vector 算子 `task_type=AIV` | 却按 cube 核数定 grid | vector 核少用一半 | 按引擎选核数 40/20（情况J） |

### ★决策流程图
```
wait_ratio 高 / mte_cflt 高 → 回 Tier4 (流水线/传输) 或 Tier3 (分块)
bank/bankgroup_cflt >4% → 回 Tier5 F (swizzle/访问)
cube 利用率低 且 任务是 matmul ?
  ├─ 没用 tl.dot (vector 模拟) → 用 tl.dot (情况D)
  └─ 用了 tl.dot 但 cube 低 → 回 Tier3 (分块/16倍数) 或 Tier1 (算法)
cube/vec 严重失衡 → 回 Tier2 (融合平衡)
grid >> 核数 / launch 开销大 → grid 固定核数 + 核内 stride 循环 (情况G)
多核耗时不均 / 尾核空转 → stride 切分 (情况H)
cube/vec 被阻塞但各 pipe 利用率都不高 (流水线气泡) → K 循环分块 + 双缓冲 UB 预算 (情况I)
纯 vector 算子 (rms_norm/softmax/bias_gelu) 没吃满 vector 核 → 按引擎选核数 40/20 (情况J)
代码风格有 while/动态shape/非math命名空间 → 适配 (情况F)
全无 → Tier6 已最优, 停止
```

---

## 情况A：wait_ratio 高（vec 被阻塞等数据）

**触发**：`conflict.wait_ratio` 高（vector 大量 cycle 在等数据/上一条完成）。
**含义**：计算单元在空等（搬运没和计算重叠，或依赖链串行）。
**怎么查**：搜 "ascend aiv_vec_wait_ratio optimization pipeline overlap"。

### ❌ 问题示例代码（load→compute 强耦合，计算空等搬运）
```python
# ❌ 循环里 load 结果立即被算, 且没有让编译器预取的结构 → vec 等 MTE
for k in range(0, K, BLOCK_K):
    acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)   # load 完才算 → 每轮等搬运
    a_ptrs += BLOCK_K; b_ptrs += BLOCK_K
# msprof: aiv_vec_wait_ratio 高, 计算有大量空等
```
**出现的问题**：计算与搬运串行（等 load 完成才算）→ vector 单元大量 cycle 空等，`wait_ratio` 高。

### ✅ 修改后正确代码（load 独立成步骤，让编译器流水线重叠）
```python
# ✅ load 独立清晰步骤, 常量步进 → 编译器自动双缓冲/流水线 (Ascend 默认)
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=..., other=0.0)   # MTE 预取
    b = tl.load(b_ptrs, mask=..., other=0.0)
    acc = tl.dot(a, b, acc)                    # 与下一轮 load 重叠
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
```
**约束/坑**：这是 Tier4 的核心（本层只指路）；若结构已最优仍高 → 看是否有跨 kernel 的串行依赖（回 Tier2 融合）。

---

## 情况B：mte_cflt 高（MTE 搬运冲突）

**触发**：`conflict.mte_cflt_ratio` 高（vector 指令被 MTE 搬运阻塞）。
**含义**：搬运与计算抢资源，或搬运指令本身冲突。
**怎么查**：搜 "ascend aiv_vec_mte_cflt_ratio" / "triton mte conflict double buffer"。

### ❌ 问题示例代码（大量小搬运并发 → MTE 冲突）
```python
# ❌ 循环里多次小 load (每次几 KB) → MTE 指令密集, 互相冲突
for i in range(many):
    val = tl.load(x_ptr + tiny_offsets[i], ...)   # 多个小搬运
```
**出现的问题**：多个小 MTE 搬运并发 → MTE 冲突，`mte_cflt` 高。

### ✅ 修改后正确代码（大块连续搬运，减少 MTE 指令数）
```python
# ✅ 一次大连续 load (向量) → 减少 MTE 指令数, 冲突少
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int32)
mask = offs < n_elements
x = tl.load(x_ptr + offs, mask=mask, other=0.0)   # 一次大搬运
```
**约束/坑**：本质是 Tier4（连续/合并搬运）；本层指路。UB ≤192KB 留双缓冲空间。

---

## 情况C：bank/bankgroup 冲突

**触发**：`conflict.bank_cflt_ratio` 或 `bankgroup_cflt_ratio` >4%。
**含义**：并发访问同一 UB bank（bank）或同一 bank group（bankgroup，block stride 不合理）。
**怎么查**：搜 "ascend aiv_vec_bank_cflt_ratio swizzle"。

### ❌ 问题示例代码（并发访问同 bank 的 UB 地址）
```python
# ❌ 多个并发操作访问同一 bank 的 UB → N-way 冲突 → 吞吐 1/N
#   表现为 conflict.bank_cflt_ratio / bankgroup_cflt_ratio 高
```
**出现的问题**：UB 多 bank，并发落在同一 bank → 冲突 → 带宽掉 1/N（bankgroup = block stride 设置不合理）。

### ✅ 修改后正确代码（访问顺序/swizzle，回 Tier3/5）
```python
# ✅ 调整访问顺序或分块 (swizzle 消除冲突, 见 tier3 §三 / tier5 F)
# 编译器 triton-ascend 已自动 swizzle layout 最小化冲突; 手动时错开 bank 索引
```
**约束/坑**：先确认 `bank_cflt_ratio` 真高；多数已自动处理，别手动硬调。

---

## 情况D：cube 利用率低但该算 matmul（用 vector 模拟了）

**触发**：`engine_utilization.cube` 极低 或 `task.task_type` 非 cube，但算的是矩阵乘。
**收益**：用 `tl.dot` 触发 Cube → 从 ~10% 到接近峰值。
**怎么查**：搜 "triton tl.dot cube mmad" / "ascend cube utilization low vector emulation"。

### ❌ 问题示例代码（用逐元素循环模拟矩阵乘 → 不触发 Cube）
```python
# ❌ 用向量循环模拟 matmul → 只在 vector 跑, cube 闲置
for i in range(BLOCK_M):
    for j in range(BLOCK_N):
        acc = 0.0
        for k in range(K):
            acc += tl.load(a_ptr + ...) * tl.load(b_ptr + ...)   # 标量/向量模拟
        tl.store(c_ptr + ...)
# → task_type 不是 cube, cube_utilization ~0, 性能 ~10%
```
**出现的问题**：没用 `tl.dot` → 编译器不生成 Cube 的 mmad 指令 → 矩阵乘在 vector 上慢跑（~10% 算力），cube 闲置。

### ✅ 修改后正确代码（用 tl.dot 触发 Cube）
```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=..., other=0.0)
        b = tl.load(b_ptrs, mask=..., other=0.0)
        acc = tl.dot(a, b, acc)          # ✅ 触发 Cube mmad
    tl.store(c_ptrs, acc, mask=...)
```
**约束/坑**：`tl.dot` 输入维度 16 倍数（Cube 粒度）；累加器 fp32；我们 kernel 已用 `tl.dot`（正确样例）。

---

## 情况E：cube/vec 引擎严重失衡

**触发**：`engine_utilization` 里 cube 和 vec 严重失衡（一个 ~100% 一个 ~10%）。
**含义**：算子结构让一个引擎空转（如巨大 gelu 全在 vector，cube 闲着；或反之）。
**怎么查**：搜 "ascend cube vector engine balance fusion"。

### ❌ 问题示例代码（matmul 和 大 elementwise 分离 → 引擎轮流空转）
```python
# ❌ matmul(cube) → 巨大独立 gelu(vector) → matmul(cube): 引擎轮流空等
matmul_kernel[g](x, w1, z, ...)        # cube 忙, vector 闲
gelu_kernel[g](z, h, ...)              # vector 忙, cube 闲
matmul_kernel[g](h, w2, y, ...)        # cube 忙, vector 闲
```
**出现的问题**：cube 和 vector 串行轮流用，一个忙时另一个空转 → 引擎利用率失衡。

### ✅ 修改后正确代码（融合平衡引擎，回 Tier2）
```python
# ✅ 把 gelu 并进 matmul epilogue (tier2 情况A) → cube 算完直接 vector 处理, 引擎更均衡
#   或把小的 vector 工作合并到 cube kernel 的 epilogue
# 铁律: 引擎失衡的解法在 Tier2(融合)/Tier1(算法), 不是本层硬调
```
**约束/坑**：这是结构问题（Tier2 融合解决）；本层只诊断指路。

---

## 情况F：代码风格适配（后端最优解析）

**触发**：编译告警/降级（while 循环、动态 shape、非 `tl.math` 命名空间）。
**含义**：代码写法让后端解析差（不能流水线/展开/映射原生指令）。
**怎么查**：搜 "triton-ascend while loop static range migration"。

### ❌ 问题示例代码（while + 非 math 命名空间 + 隐式广播）
```python
# ❌ while 循环 → 后端无法流水线; tl.erf → 不支持; 隐式广播
i = 0
while i < n:                      # 动态循环, 后端只能朴素跳转
    ...
    out = tl.erf(x)               # 可能不支持
```
**出现的问题**：while 循环后端没法做流水线/展开；`tl.erf` 不支持；隐式广播插入冗余 op。

### ✅ 修改后正确代码（静态 for-range + tl.math 命名空间 + 显式）
```python
# ✅ 静态 for-range (迭代次数 constexpr) → 后端可流水线/展开
for i in range(NUM_ITER):                    # NUM_ITER: tl.constexpr
    ...
    out = x * 0.5 * (1.0 + tl.math.tanh(...))   # ✅ tl.math.* 命名空间
# 维度全 tl.constexpr; 类型转换显式 .to(); 不用动态 shape
```
**约束/坑**：循环边界、分块、步长全 `tl.constexpr`；数学函数 `tl.math.*`；显式 `.to()`；避免 while/动态 shape。

---

## 情况G：grid 调度 — grid 固定物理核数 + 核内 stride 循环（launch/调度开销）

**触发**：`task.block_dim` 远大于物理核数（20 cube + 40 vector），或 kernel 耗时很小但总耗时高（launch/调度开销占比大）。
**含义**：grid=每 tile 一个 program 时，超过物理核数的部分要**多轮派发**，每轮都有核启动/初始化头开销；算子越小（rms_norm/softmax/bias_gelu）占比越大。GPU 习惯 launch 巨大 grid，910B3 物理核少，应固定核数 + 核内循环，一次派发。
**怎么查**：搜 "triton-ascend grid fixed core number stride loop" / "TRITON_ALL_BLOCKS_PARALLEL"。

### ❌ 问题示例代码（grid = 每 tile 一个 program → 多轮派发）
```python
# ❌ grid = 总 tile 数, 每 program 只处理 1 个 tile
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
# 2048² fp32 BLOCK=64 → grid=1024, 但物理只有 20~40 核 → 多轮批派发,
#   每轮核启动/初始化头开销; 小算子时 launch 开销可能 > kernel 本体
pid = tl.program_id(axis=0)
pid_m = pid // grid_n
pid_n = pid % grid_n
```
**出现的问题**：超过物理核数的 program 按批派发，批间有核启动延迟；小算子时 launch/调度开销占比高，`task.block_dim` 显示 grid 远超核数。

### ✅ 修改后正确代码（grid 固定核数 + 核内 stride 循环）
```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)                # core id: 0~NUM_CORES-1
    num_cores = tl.num_programs(axis=0)        # = grid 固定核数 (20/40)
    grid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = tl.cdiv(M, BLOCK_M) * grid_n
    for tile in range(pid, num_tiles, num_cores):     # ✅ 每核 stride 循环 (一次派发, 步骤均衡)
        pid_m = tile // grid_n
        pid_n = tile % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
# launch: matmul_kernel[(NUM_CORE,)](...)    NUM_CORE=20 (有 tl.dot→cube 核数); 纯 vec 用 40
# 已 CPU 验证: 512³ BLOCK=64 时 64 个 tile 全覆盖、每核 3~4 个、max|C-ref|=4.8e-7;
#    tile 总数不能被核数整除时 max-min ≤ 1 (比连续切分更均衡, 见情况H)
```
**约束/坑**：grid 与核数匹配**按算子类型选**（有 `tl.dot` 用 cube 核数 20，纯 vector 用 vector 核数 40，见情况J）；`for tile in range(pid, num_tiles, num_cores)` 是运行时边界的循环，**迭代之间无数据依赖**才能安全分发（本 kernel 每 tile 独立）；grid 极大（>65535，`coreDim` 上限 UINT16_MAX）时设 `TRITON_ALL_BLOCKS_PARALLEL=1`（按序 launch 才生效，需实测）；**launch 开销/派发收益需真机 msprof 确认**（看调度/启动耗时是否下降）。

---

## 情况H：核间负载均衡 — stride 切分 vs 连续切分（尾核空转/计算拖尾）

**触发**：`task.block_dim` 用满但总耗时偏高，`aic_time`/`aiv_time` 与 Task Duration 差距大；多核耗时不均、计算拖尾。
**含义**：tile 总数不能被核数整除时，**连续切分**（每核领一段）让靠后的核分不到 → 尾核空转；**stride 切分**（轮流领）把余数摊到前几个核，全局最均衡（max-min ≤ 1）。
**怎么查**：搜 "ascend 核间负载均衡 尾核 计算拖尾" / "triton stride loop load balance"。

### ❌ 问题示例代码（连续切分 → 尾核空转）
```python
# ❌ 每核领一段连续 tile: 最后一段分不完的 tile 只给前几个核, 后面的核空转
per_core = triton.cdiv(num_tiles, NUM_CORES)
start = pid * per_core
end = min(start + per_core, num_tiles)
for tile in range(start, end):        # pid 越大领到的 tile 越少
    ...
# 例: 25 tile / 20 核 → 核0~11 各 2, 核12 各 1, 核13~19 共 7 核空转
```
**出现的问题**：连续切分让 tile 余数集中在前几个核，靠后的核空转 → 整核耗时=最忙核的耗时，空转核拖低吞吐；tile 数接近核数且不整除时最严重。

### ✅ 修改后正确代码（stride 切分 → 全局均衡）
```python
# ✅ 每核轮流领 tile (round-robin): 余数摊到前几个核, 所有核都忙
for tile in range(pid, num_tiles, NUM_CORES):
    ...
# 例: 25 tile / 20 核 → 核0~4 各 2, 核5~19 各 1 → 20 核全忙, max-min=1
# 已 CPU 验证: (25,20)→连续空闲7核 vs stride空闲0核; (257,20)→连续max-min=3 vs stride=1;
#   (2048,40)→连续max-min=32 vs stride=1; stride 恒 max-min≤1、全覆盖无重复
```
**约束/坑**：stride 切分的代价是每核的 tile 不相邻（L2 局部性可能变差，matmul 的 A/B 块复用需 group swizzle 配合，见 tier3 §三）；tile 数整除核数时两种切分相同（如 1000/40）；**收益需真机 msprof 确认**（看各核耗时分布 / Task Duration）。常与情况G（固定核数 grid）组合使用。

---

## 情况I：跨引擎流水线 & 双缓冲 UB 预算（cube/vec/mte2/mte3 重叠）

**触发**：`aic_cube_wait_ratio` 或 `aiv_vec_wait_ratio` 高，但单独看每个 pipe 利用率都不高（无主导瓶颈）→ 流水线气泡。
**含义**：910B3 每核 = 1 cube + 2 vector，MTE2(GM→L1/UB)、MTE1(L1→L0A/B)、cube、MTE3(UB→GM) 指令队列**物理上独立可并行**；编译器 multi-buffer（默认开）只在「循环分块 + 无数据依赖 + UB 放得下」时才生效，把 load(MTE2) 与 cube/vec 计算、store(MTE3) 重叠成流水线。
**怎么查**：搜 "ascend double buffer ping-pong MTE2 MTE3 overlap" / "triton-ascend multi-buffer UB"。

### ❌ 问题示例代码（单 pass 无循环 → multi-buffer 不生效 → 等待）
```python
# ❌ 单 pass: load→dot→store 一次完成, 没有 K 循环 → 没有"下一轮"可预取 → 流水线气泡
a = tl.load(a_ptrs)
b = tl.load(b_ptrs)
acc = tl.dot(a, b)
tl.store(c_ptrs, acc)
# msprof: aic_cube_wait_ratio / aiv_vec_wait_ratio 高, mte2 忙完 cube 才动
```
**出现的问题**：单 pass 没有多轮 load，编译器无法双缓冲；搬运(M)与计算(cube)串行 → wait_ratio 高，跨引擎重叠为 0。

### ✅ 修改后正确代码（K 循环分块 + 独立 load 步骤 + UB 留双缓冲空间）
```python
# ✅ K 循环分块 + load 独立步骤 → 编译器自动 multi-buffer (默认开), 隐藏搬运延迟
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)   # MTE2
    b = tl.load(b_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)   # MTE2
    acc = tl.dot(a, b, acc)                                          # cube (与下一轮 load 重叠)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk
# epilogue (bias/gelu/softmax) 并进同一 kernel → cube 结果经 FIX→UB→vector→MTE3, 与下一 tile 的 cube 重叠
# 约束: UB 双缓冲预算 = (BM×BK + BK×BN)×dtype×n_bufs ≤ 192KB
#   fp32 128×128×64 → (128·64+64·128)×4×2 = 128KB ✓ (可双缓冲)
#   fp32 256×256×64 → 256KB ✗ (溢出, 编译器会关 multi-buffer 或报错)
#   已 CPU 验证: UB 预算算术; tanh-GELU epilogue 公式 vs torch F.gelu(approximate="tanh") rel=7.8e-8
```
**约束/坑**：multi-buffer 默认开，但**数据依赖/同步、单 pass、UB 不够**时会关；`care_padding=False` 可去掉 masked load 的 vector 填零依赖（见 tier3 §七）；UB 双缓冲=可用 UB 减半（192KB→~96KB/缓冲），分块要按此预算回 Tier3；**跨引擎重叠收益需真机 msprof 确认**（wait_ratio 是否下降、MTE2 与 CUBE/VEC 重叠比例是否 >30%）。

---

## 情况J：纯 vector 算子的引擎/核数选择（AIV vs AI_CORE 启动开销）

**触发**：纯 vector 算子（rms_norm/softmax/bias_gelu，`task.task_type=AIV`）却按 cube 核数（20）定 grid，或 MIX 启动混入 cube 启动开销。
**含义**：910B3 = 20 cube 核 + 40 vector 核。**纯 vector 算子（无 `tl.dot`）不需要 cube 核**：按 cube 核数定 grid 会少用一半 vector 核；若以 MIX 方式启动，cube 核无计算指令也要付核启动/初始化头开销。
**怎么查**：搜 "triton-ascend pure vector core number" / "ascend AIV AIC 核数 混合启动头开销"。

### ❌ 问题示例代码（纯 vector 按 cube 核数定 grid）
```python
# ❌ rms_norm 是纯 vector 算子 (没有 tl.dot), 却按 20 (cube 核数) 定 grid
NUM_CORES = 20            # 从 matmul 抄来的, 但 rms_norm 没有 tl.dot!
grid = (NUM_CORES,)
# → 40 个 vector 核只用 20 个, 剩一半空转; 若 MIX 启动还多付 cube 核启动开销
```

### ✅ 修改后正确代码（按算子类型选核数）
```python
# ✅ 按算子引擎类型选 grid 目标核数:
#   纯 vector (rms_norm / softmax / bias_gelu):         NUM_CORES = 40
#   CV 融合 (matmul / flash_attention / MLP 有 tl.dot): NUM_CORES = 20
grid = (NUM_CORES,)       # + 核内 stride 循环 (情况G)
# 已 CPU 验证: 选择逻辑 has_dot→20 / 纯vec→40 正确; 启动头开销收益需真机 msprof 确认
```
**约束/坑**：triton-ascend 的引擎归属由编译器按代码决定（有 `tl.dot` → cube），**不能手动指定 task_type**；本层能改的是**按算子类型选 grid 目标核数**（纯 vec 用 40 才吃满 vector 核）；纯 vec 算子若 `task.task_type` 显示 MIX 且 cube 有启动开销 → 检查是否混进意外的大 `tl.dot`/矩阵计算，否则回 Tier2 拆分；**收益需真机 msprof 确认**。

---

## 常见错误与修复

| 错误 | 现象 | 修复 |
|---|---|---|
| vector 模拟 matmul | cube 利用率 ~0 | 用 `tl.dot` |
| load→compute 强耦合 | wait_ratio 高 | load 独立步骤让编译器流水线（回 Tier4） |
| 小搬运密集 | mte_cflt 高 | 大连续 load（回 Tier4） |
| 引擎失衡 | cube/vec 一满一闲 | 回 Tier2 融合平衡 |
| while/动态 shape | 后端降级 | 静态 for-range + constexpr |
| tl.erf | 编译失败 | tl.math.tanh |
| 想设 num_warps/multibuffer | 报错/无效 | 编译器自动管理，别硬调 |
| grid=每 tile 一 program 远超核数 | launch/调度开销大 | grid 固定核数 + 核内 stride 循环（情况G） |
| 连续切分 tile | 尾核空转/计算拖尾 | stride 切分（情况H） |
| 单 pass 无 K 循环 | multi-buffer 关, wait_ratio 高 | K 循环分块 + 留双缓冲 UB 空间（情况I） |
| 纯 vec 算子按 cube 核数定 grid | vector 核少用一半 | 按引擎选核数 40/20（情况J） |
