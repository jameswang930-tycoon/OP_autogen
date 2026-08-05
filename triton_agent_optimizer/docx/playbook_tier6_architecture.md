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

### ★决策流程图
```
wait_ratio 高 / mte_cflt 高 → 回 Tier4 (流水线/传输) 或 Tier3 (分块)
bank/bankgroup_cflt >4% → 回 Tier5 F (swizzle/访问)
cube 利用率低 且 任务是 matmul ?
  ├─ 没用 tl.dot (vector 模拟) → 用 tl.dot (情况D)
  └─ 用了 tl.dot 但 cube 低 → 回 Tier3 (分块/16倍数) 或 Tier1 (算法)
cube/vec 严重失衡 → 回 Tier2 (融合平衡)
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
