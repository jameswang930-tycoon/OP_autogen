# Triton 自动优化系统 Tier 4（Memory Access）优化策略指南
## 层级定位与前置约束
Tier 4 位于 Tier 1（算法结构）、Tier 2（算子融合）、Tier 3（分块与编译配置）全部完成并验证通过后执行，属于**微架构级内存访问模式专项优化**。本层不改变核心算法逻辑、算子融合结构、分块尺寸，仅通过调整访存指令的组织方式，适配 Ascend NPU 内存控制器与 CANN 9.0 总线特性，消除冗余传输、降低调度开销、提升带宽利用率。

> **环境约束（Coder Agent 必读）**：同 CODING_GUIDE.md
> - WSL2 + Python3.9 + triton3.4.0，仅用 @triton.jit
> - 所有修改通过 ast_to_ttir() → ttir_to_hivm.py → bisheng 三层编译
> - num_warps/num_stages 由 GPUTarget 设置，Coder 不修改
> - 标量索引（params[0]）在 TTIR→HIVM 中会被跳过（SCALAR op），不影响 msprof

本手册严格适配环境约束：
- 前端：Triton 3.4.0 + Python 3.9，仅使用 `@triton.jit` 装饰器，禁用 `@triton.autotune`
- 中间链路：TTIR → HIVM 自定义转换，所有语法必须可解析
- 后端：CANN 9.0 + bisheng 编译器 + CMake 构建
- 验证：msprof op simulator 周期精确模拟
- 核心边界：`num_warps`、`num_stages` 不属于 `@triton.jit` 参数，仅通过调用配置传递

---

## 一、核心优化策略
按优先级从高到低分为三类，优先执行收益最高的冗余加载消除，再做传输合并与访问模式连续化。

### 1. 冗余全局 Load 消除
- **优化原理**：识别同一基址指针、相同偏移范围、相同掩码条件的多次全局加载操作，若两次加载之间无对应地址的写入操作，则合并为单次加载后寄存器复用，消除重复的内存总线请求。
- **适用场景**：跨计算分支的重复读取（如归约分支与逐元素分支分别读取同一输入）、多输出算子重复读取源数据、历史迭代残留的冗余加载。
- **与 Tier 2 的边界区分**：Tier 2 聚焦运算链融合附带消除冗余 Load；Tier 4 专门处理跨非连续运算、跨分支的独立冗余 Load，不改变运算逻辑本身。

### 2. 零散小传输合并
- **优化原理**：将地址空间连续、尺寸较小的多次独立 load/store 操作，合并为一次大尺寸连续传输，再在寄存器内做切片拆分，减少内存控制器的请求次数与调度开销。
- **适用场景**：多组小尺寸权重/偏置的分散加载、分块边界的零散补零传输、多通道独立的标量参数读写。
- **收益逻辑**：NPU 内存控制器对大尺寸连续突发请求的调度效率远高于多次小请求，合并后可显著降低总线调度开销。

### 3. 非连续访存连续化
- **优化原理**：将跨步、转置、非连续的内存访问，通过调整分块内访问顺序、寄存器内数据重排，转换为连续内存访问，匹配内存控制器的突发传输模式。
- **适用场景**：列优先矩阵访问、跨通道跨步读取、转置操作中的非连续读写。
- **核心约束**：仅调整访存顺序与寄存器重排，不改变全局数据布局，不引入任何额外全局写入。

---

## 二、HIVM 诊断触发规则
所有优化完全由 HIVM ops 诊断数据驱动，满足对应阈值即强制执行对应优化。

> **适配我们的 bottleneck_diagnoser**：
> Agent 从 `merged_report.json` 获取以下指标判断内存优化机会：
> - `per_op_statistics[].op_type` — 统计 `gm_to_ub` (load) 和 `ub_to_gm` (store) 数量
> - 同 `src`/`size_kb`/`memory_region` 的 load op ≥ 2 → 触发冗余 Load 消除
> - `per_op_statistics[].size_kb` — 连续小 size (< 1KB) 的 load/store ≥ 4 → 触发传输合并
> - `per_op_statistics[].pipeline_channel` — 确认 MTE2/MTE3 管线占用

### 2.1 分策略触发阈值
| 优化策略 | 触发条件（满足任意一条即执行） | 强制执行阈值 |
|----------|--------------------------------|--------------|
| 冗余全局 Load 消除 | 1. 同一基址指针、同 size_kb、同掩码表达式的 load op 出现次数 ≥ 2<br>2. 两次 load 之间无对应地址的 store/原子操作<br>3. 冗余访存占总访存流量 ≥ 10% | **同一指针被 load > 1 次且中间无 store** |
| 零散小传输合并 | 1. 单 kernel 内 ≤ 32 字节的零散 load/store 数量 ≥ 4 个<br>2. 多个小传输的地址空间连续、无间隔<br>3. 小传输总占比 ≥ 15% | 连续小传输 ≥ 4 个 |
| 非连续访存连续化 | 1. 元素级步长 > 1 的非连续访存 op 占总访存 op ≥ 30%<br>2. 带宽利用率 `bw_util < 80%` 且无计算瓶颈<br>3. 存在转置/跨步访问标记 | 跨步访存占比 ≥ 30% |

### 2.2 Tier 4 最优判定标准
同时满足以下所有条件，判定内存访问层已达最优，全链路优化结束：
1. 同条件全局 load 无重复（同指针同偏移同掩码仅出现 1 次）
2. 主计算路径所有访存均为连续访问，跨步访存占比 < 5%
3. 零散小传输占总访存比例 < 5%
4. 带宽利用率稳定在 **90%~95%** 区间（mock 环境下用 SATURATION_PARAMS 估算值代替），无明显访存瓶颈

---

## 三、各策略 Before/After 代码示例
所有示例严格遵循 Triton 3.4.0 语法，仅使用 `@triton.jit` 装饰器，块参数标记 `tl.constexpr`，无 GPU 专属 API，可通过全链路编译。

### 3.1 策略一：冗余全局 Load 消除
**场景**：同一行输入被归约分支和逐元素分支分别读取，中间无写入，属于跨分支冗余加载。

**Before（冗余加载写法）**
```python
@triton.jit
def redundant_load_before(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 第一次加载：供给归约分支计算均值
    x_sum = tl.load(x_ptr + base + offsets, mask=mask)
    mean = tl.sum(x_sum, axis=0) / n_cols

    # 第二次加载：同一地址同一掩码，供给逐元素分支，完全冗余
    x_val = tl.load(x_ptr + base + offsets, mask=mask)
    out = x_val - mean

    tl.store(out_ptr + base + offsets, out, mask=mask)
```

**After（单次加载复用写法）**
```python
@triton.jit
def redundant_load_after(x_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    base = row_idx * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 单次全局加载，寄存器同时供给归约和逐元素分支
    x = tl.load(x_ptr + base + offsets, mask=mask)
    mean = tl.sum(x, axis=0) / n_cols
    out = x - mean

    tl.store(out_ptr + base + offsets, out, mask=mask)
```

### 3.2 策略二：零散小传输合并
**场景**：分别加载偏置、缩放、偏移三个连续存储的标量参数，分散为 3 次独立小传输，合并为单次大传输后寄存器切片。

**Before（分散小传输写法）**
```python
@triton.jit
def small_transfer_before(x_ptr, params_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 3次独立小传输，地址连续，总线调度开销高
    bias = tl.load(params_ptr + 0)
    scale = tl.load(params_ptr + 1)
    shift = tl.load(params_ptr + 2)

    out = x * scale + bias + shift
    tl.store(out_ptr + offsets, out, mask=mask)
```

**After（合并大传输写法）**
```python
@triton.jit
def small_transfer_after(x_ptr, params_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    # 1次连续大传输，寄存器内切片取值，减少总线请求次数
    # 一次性加载3个连续参数，寄存器内切片（标量索引会被TTIR→HIVM跳过，不影响msprof）
    params = tl.load(params_ptr + tl.arange(0, 3))
    bias = params[0]
    scale = params[1]
    shift = params[2]

    out = x * scale + bias + shift
    tl.store(out_ptr + offsets, out, mask=mask)
```

### 3.3 策略三：非连续访存连续化
**场景**：按列读取矩阵时跨步访问，内存效率低；调整为分块连续读取行数据，寄存器内完成转置提取。

**Before（跨步非连续写法）**
```python
@triton.jit
def stride_access_before(mat_ptr, out_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    col_idx = tl.program_id(0)
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < M

    # 跨步访问：地址间隔为N个元素，内存控制器无法突发传输
    col = tl.load(mat_ptr + row_offsets * N + col_idx, mask=mask)
    tl.store(out_ptr + col_idx * BLOCK_SIZE + row_offsets, col, mask=mask)
```

**After（连续访问+寄存器重排写法）**
```python
@triton.jit
def stride_access_after(mat_ptr, out_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)

    # 连续读取一块行数据，全局访存完全连续，匹配突发传输
    block = tl.load(
        mat_ptr + rows[:, None] * N + cols[None, :],
        mask=(rows[:, None] < M) & (cols[None, :] < N),
        other=0.0
    )
    # 寄存器内转置，无额外全局访存开销
    col_block = tl.transpose(block)

    tl.store(
        out_ptr + cols[:, None] * M + rows[None, :],
        col_block,
        mask=(cols[:, None] < N) & (rows[None, :] < M)
    )
```

---

## 四、优化后 HIVM ops 预期变化
| 优化策略 | 优化前访存 op 数量 | 优化后访存 op 数量 | 算子压缩比例 | 附带收益 |
|----------|--------------------|--------------------|--------------|----------|
| 冗余全局 Load 消除 | N 次同条件 load（N≥2） | 1 次 load | 减少 N-1 个 load op，压缩率 (N-1)/N | 带宽利用率提升 5%~20%，减少 RAW 依赖节点 |
| 零散小传输合并 | N 个连续小传输（N≥4） | 1 个大传输 | 减少 N-1 个访存 op，压缩率 (N-1)/N | 总线调度开销降低 30%~60%，访存吞吐显著提升 |
| 非连续访存连续化 | N 个跨步访存 op | N 个连续访存 op | op 数量不变，访问模式质变 | 带宽利用率提升 15%~40%，访存延迟降低 20%~50% |

> **在我们的 pipeline 中验证 Tier 4 效果**：
> 1. `gm_to_ub` (load) op 数量应减少（冗余消除/传输合并）
> 2. `size_kb` 应增大（小传输合并为大传输）
> 3. `bw_utilization` 应提升
> 4. 若 HIVM ops 数量不变 → 优化无效 → REVERT

### 典型场景整体收益
- 存在 2 次冗余加载 + 4 个零散小传输：总访存 op 减少 5 个，带宽利用率提升 15%~30%
- 高比例跨步访问场景：带宽利用率从 60% 提升至 90% 以上，整体性能提升 30%~50%

---

## 五、常见错误与修复方案
### 1. 误合并存在中间写入的 Load
- **错误现象**：合并后输出结果错误，数值与原逻辑不一致。
- **触发原因**：两次 load 之间存在对同一地址的 store 操作（存在 RAW/WAR 依赖），强行合并导致读取到旧数据。
- **修复方案**：合并前必须校验两次 load 之间无对应地址的 store/原子操作；存在依赖时禁止合并，严格遵循 HIVM 依赖链分析结果。
- **校验方法**：检查 HIVM 依赖链，两次 load 之间无写后读、写后写依赖，才可执行合并。

### 2. 合并后掩码范围不一致
- **错误现象**：边界处出现越界读取，输出边缘数值错误。
- **触发原因**：待合并的多个 load 掩码条件不同，强行合并后使用统一掩码，导致越界或读取无效数据。
- **修复方案**：合并前严格校验掩码表达式完全等价；若掩码范围不同，取并集统一掩码后内部再做分支处理，禁止直接合并不等价掩码的 load。

### 3. 合并不连续地址的小传输
- **错误现象**：合并后读取到错误数据，或触发未对齐访问告警。
- **触发原因**：待合并的小传输地址不连续、存在间隔，强行合并导致读取到无关地址的数据。
- **修复方案**：仅合并地址空间连续、无间隔的小传输；地址不连续时保持独立加载，禁止强行合并。

### 4. 连续化优化引入额外全局写入
- **错误现象**：op 数量反而增加，性能不升反降。
- **触发原因**：为了实现连续访存，引入了中间全局内存临时存储，反而增加了访存总开销。
- **修复方案**：连续化优化必须仅在寄存器内完成数据重排，禁止引入任何额外的全局 store；若无法纯寄存器完成，则放弃该优化。

### 5. 跨步计算错误导致访存越界
- **错误现象**：msprof 模拟出现地址非法错误，边界数据异常。
- **触发原因**：调整访存顺序时，stride 与偏移量计算错误，超出数组边界。
- **修复方案**：所有地址计算保留原始基址+偏移的逻辑，仅调整访问顺序；优化后必须验证边界掩码与原逻辑完全一致。

### 6. 过度合并导致寄存器溢出
- **错误现象**：合并大传输后，寄存器占用激增，出现溢出，性能不升反降。
- **触发原因**：一次性合并过多小传输，单条加载数据量过大，超出硬件寄存器容量。
- **修复方案**：单次合并总数据量不超过 64KB，超过时拆分为 2~3 次传输；优先合并地址最连续、调度开销最高的一组。

需要我补充针对特定算子（如矩阵乘、LayerNorm）的 Tier 4 专项优化模板，或者访存优化的收益量化计算公式吗？
---

## ★matmul 专属改码示例（纯 triton, 910B3 验证可行）

### Swizzle（grouped ordering）— 提升 L2 复用, 经典 >10% 优化

当前我们的 kernel 用简单 row-major（`pid_m = pid // grid_n`）。改成 **grouped swizzle**（官方 03 教程方法，纯 triton，triton-ascend 可用）：

**before（当前, 低 L2 复用）:**
```python
    pid = tl.program_id(axis=0)
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // grid_n
    pid_n = pid % grid_n
```

**after（swizzle, L2 复用更高）:**
```python
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    GROUP_SIZE_M = 8                       # 每个 group 多少行 M-tile (超参, 可 4/8/16)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

**改哪**: kernel 函数开头的 pid 计算部分（② kernel 区）。`GROUP_SIZE_M` 可放 ① config 区。
**判定**: Tier4 `l2_hit_rate` 低 或 `main_mem_read_bw` 高 → 优先加 swizzle。
**注意**: 只改 pid 计算, 不改 offs_m/offs_n 的用法, 不改数学。

