# msprof 性能数据字段完整参考

> 环境：Ascend 910B3 (Atlas A2 训练系列, NpuArch 2201) / CANN 8.5.1
> 列名已按昇腾官方文档核实；A2 系字段前缀规则：**cube 用 `aic_`，vector 用 `aiv_`**（推理产品统一 `aic_`）。
> 最后更新：2026-08-04

---

## 0. 总览：两个工具，两种输出

| 工具 | 输出目录 | 覆盖级别 | 何时用 |
|---|---|---|---|
| `msprof op` | `OPPROF_{时间戳}_XXX/` | 单算子（kernel 级 + 全通路） | ★主源：真实带宽 / L2 / cube / 引擎利用率 |
| 通用 `msprof` | `PROF_XXX/mindstudio_profiler_output/` | 任务级（每 kernel 一行 + 统计） | op_summary 每 op 耗时 / 核数 / 多算子 / launch |

主流程只用这两个（hivm/sim 已弃用主流程，按需保留）。

---

## 一、`msprof op` 产出的 8 个 CSV

> 默认全量采集（**不要指定 `--aic-metrics`**，8.5.1 指定会限制/报错）。
> 每个 CSV 内含基础列：`block_id` / `sub_block_id` / `aic_time(us)` / `aiv_time(us)` / `aic_total_cycles` / `aiv_total_cycles`。

### 1. OpBasicInfo.csv — 算子基础信息（每 kernel 一行）

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `Task Type` | 任务类型：`AI_CORE`(cube) / `AIV`(vector) / `AI_CPU` / `MIX` | 判断算子跑在哪个引擎 |
| `Task ID` / `Task Start Time(us)` | 任务 ID / 开始时间 | 关联其他 CSV |
| `Task Duration(us)` | **任务总耗时**（含调度+执行+结束响应） | 端到端耗时基准 |
| `Task Wait(us)` | 上个任务结束→本任务开始间隔 | 调度/launch 开销 |
| `Block Dim` / `Sub Block Dim` | 切分 block 数（核数）/ 子 block 数 | tiling：核数 < 硬件核数说明没吃满 |
| `Op Name` / `Op Type` | 算子名 / 算子类型 | 定位算子 |
| `Input Shape(s)` / `Input Data Type(s)` | 输入形状 / 数据类型 | **算搬运块字节数 = 元素数×dtype字节** |
| `Output Shape(s)` / `Output Data Type(s)` | 输出形状 / 数据类型 | 同上 |
| `aicore_time(us)` | AI Core 理论执行时间（所有 block 同时调度） | 理论耗时下界 |
| `aiv_time(us)` | AI Vector Core 理论执行时间 | 同上 |
| `aic_total_cycles` / `aiv_total_cycles` | AI Core / AI Vector 总 cycle | 算每 cycle 耗时 |
| `Current Freq(MHz)` | 当前频率 | roofline 校准 |

### 2. PipeUtilization.csv — 计算/搬运单元耗时及占比 ★核心

> 每 pipe 有 `*_time(us)` 和 `*_ratio`（该指令 cycle 占 total cycle 比）。
> **这些 per-pipe 耗时 = 每通路的搬运时间**，配合 shape 算出的字节数 → 每通路带宽。

| 字段 | 含义（对应搬运通路） | 优化策略 |
|---|---|---|
| `aic_cube_time(us)` / `aic_cube_ratio` | cube 矩阵指令（fp16/s16 MAC） | 算法层：算力利用 |
| `aiv_vec_time(us)` / `aiv_vec_ratio` | vector 向量指令 | 算法层：向量占比过高说明 cube 没吃重 |
| `aic_mte1_time(us)` / `aic_mte1_ratio` | **MTE1：L1→L0A/L0B** | 分块：L0A/B 带宽瓶颈 |
| `aic_mte2_time(us)` / `aic_mte2_ratio` + `aiv_mte2_*` | **MTE2：GM→AICORE（读）** | 访存：GM 读带宽瓶颈 |
| `aic_mte3_time(us)` / `aic_mte3_ratio` + `aiv_mte3_*` | **MTE3：AICORE→GM（写）** | 访存：GM 写带宽瓶颈 |
| `aic_scalar_time(us)` / `aic_scalar_ratio` + `aiv_scalar_*` | 标量指令 | 计算：标量拖后腿 |
| `aic_fixpipe_time(us)` / `aic_fixpipe_ratio` | **fixpipe：L0C→L1** | 访存：累加结果搬运 |
| `aic_icache_miss_rate` | 指令 cache 缺失率 | 架构：指令取指 |

**诊断判据（官网调优指南）**：
- `aiv_vec_ratio` < 10% → vector 流水没充分利用
- `aiv_mte2_time` ≈ 总 Duration → MTE2 搬运 bound（带宽瓶颈，基本调优到头）
- 最长 pipe 耗时决定瓶颈层级：`mte2≈mte3≈GM带宽` → memory_bound；`cube` 满 → compute_bound

### 3. ArithmeticUtilization.csv — cube/vector 指令耗时和占比

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `aic_cube_fops` | **cube 浮点运算数（FLOPs）** | roofline 算力分子 |
| `aic_cube_ratio` / `aic_cube_fp16_ratio` / `aic_cube_int8_ratio` | cube 占比 / fp16 / int8 细分 | 确认用的什么精度 |
| `aic_cube_total_instr_number` / `fp_instr_number` / `int_instr_number` | cube 指令条数 / fp / int | 冗余计算判断 |
| `aiv_vec_fops` | **vector 浮点运算数（FLOPs）** | roofline 算力分子（vector 部分） |
| `aiv_vec_ratio` / `aiv_vec_fp32/fp16/int32/int16/misc_ratio` | vec 占比及细分 | 向量精度利用 |
| `aic_total_cycles` / `aiv_total_cycles` | 总周期 | 与耗时换算 |

### 4. Memory.csv — 内存读写带宽速率（GB/s）★真实带宽

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `aic_main_mem_read_bw` / `aic_main_mem_write_bw` | **主存(GM)读/写带宽** | roofline 访存分子（对 1.8TB/s 峰值） |
| `aic_l1_read_bw` / `aic_l1_write_bw` | L1 读/写带宽 | L1 带宽瓶颈 |
| `aiv_gm_to_ub_bw` | **GM→UB 带宽（MTE2 load）** | 搬运通路 |
| `aiv_ub_to_gm_bw` | **UB→GM 带宽（MTE3 store）** | 搬运通路 |
| —（无列） | **L2 无独立带宽列**（A2 系），命中率看 L2Cache.csv | — |

> 单位注意：部分版本为 MB/s，parser 按量级（≥1e4）自动转 GB/s。

### 5. MemoryL0.csv — L0A/L0B/L0C 读写带宽

| 字段 | 含义 |
|---|---|
| `aic_l0a_read_bw` / `aic_l0a_write_bw` | L0A 读/写（A 矩阵进 cube） |
| `aic_l0b_read_bw` / `aic_l0b_write_bw` | L0B 读/写（B 矩阵进 cube） |
| `l0c_read_bw_cube` / `l0c_write_bw_cube` | cube 读/写 L0C（累加结果） |

### 6. MemoryUB.csv — mte/vector/scalar 的 UB 读写带宽

| 字段 | 含义 | 备注 |
|---|---|---|
| `aiv_ub_read_bw_vector` / `aiv_ub_write_bw_vector` | vector 从/向 UB 带宽 | A2 系 |
| `aiv_ub_read_bw_scalar` / `aiv_ub_write_bw_scalar` | scalar 从/向 UB 带宽 | A2 系 |
| `ub_read_bw_mte` / `ub_write_bw_mte` | mte 的 UB 带宽 | **仅推理产品**，910B3 合法缺 |

### 7. L2Cache.csv — L2 Cache 命中率

| 字段 | 含义 |
|---|---|
| `aic_total_hit_rate(%)` / `aic_read_hit_rate(%)` / `aic_write_hit_rate(%)` | 总/读/写命中率（百分数，归一化到 0~1） |
| `aic_write_cache_hit` / `aic_write_cache_miss_allocate` / `aic_read_cache_hit` / `aic_read_cache_miss_allocate` | 命中/缺失计数 |

> L2 命中率低 → 数据复用差 → 调分块/流水让 A 块在 L2 复用。

### 8. ResourceConflictRatio.csv — 资源冲突占比 ★UB bank 冲突

| 字段 | 含义 | 调优阈值 |
|---|---|---|
| `aiv_vec_bank_cflt_ratio` | **bank 冲突**（操作数读写指针地址不合理） | >4% 提示，加 padding(32B) |
| `aiv_vec_bankgroup_cflt_ratio` | **bankgroup 冲突**（block_stride 不合理） | >4% 提示，调 repeatStride/blockStride |
| `aiv_vec_total_cflt_ratio` | vec 总冲突占比 | **>5% 触发优化流程** |
| `aiv_vec_resc_cflt_ratio` | 执行单元资源冲突 | — |
| `aiv_vec_mte_cflt_ratio` | mte 冲突 | — |
| `aiv_vec_wait_ratio` / `aic_cube_wait_ratio` | vec/cube 被阻塞占比 | 流水重叠不足 |

---

## 二、通用 `msprof` 产出的文件

> 输出在 `PROF_XXX/mindstudio_profiler_output/`，文件名带时间戳（`op_summary_1.csv` 等）。
> 8.5.1 用 `--ai-core=on`（`--aic-metrics` 退出 255 不支持）。

### 9. op_summary_*.csv — 每 kernel 一行（★per-op 全信息）

> **这个文件把 op 身份 + shape/dtype + 每 pipe 耗时合在一起** → 可算 per-op 每通路带宽。

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `Op Name` / `Op Type` | 算子名 / 算子类型 | 定位 |
| `Task Type` | `AI_CORE`/`AIV`/`AI_CPU`/`MIX` 引擎归属 | 判断引擎 |
| `Task Duration(us)` / `Task Start Time(us)` / `Task Wait(us)` | 耗时 / 开始 / 调度间隔 | 端到端 + launch 开销 |
| `Block Dim` | 核数（<硬件核数说明没吃满） | tiling |
| `Input Shape(s)` / `Input Data Type(s)` | 输入形状 / 类型 | **搬运块字节数 = 元素数×dtype字节** |
| `Output Shape(s)` / `Output Data Type(s)` | 输出形状 / 类型 | 同上 |
| `aicore_time(us)` | AI Core 理论执行时间 | 理论下界 |
| `aic_time(us)` / `aiv_time(us)` | cube/vector 理论执行时间（`--task-time=l1 --aic-mode=task-based` 生成，8.5.1 `--ai-core=on` 可能没有 → 合法缺） | 引擎时间 |
| `Total Cycles` | 总 cycle | 换算 |
| `aic_mac/scalar/mte1/mte2/fixpipe` + `aiv_vec/scalar/mte2/mte3` 时间列 | **每 pipe 耗时** | **per-op 每通路带宽 = 字节数 / pipe耗时** |
| `*_ratio` | 各 pipe 利用率 | 引擎利用率 |

### 10. op_statistic_*.csv — 算子调用次数及耗时（按类型聚合）

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `OP Type` | 算子类型 | — |
| `Count` | 调用次数 | — |
| `Total Time(us)` | 总耗时 | 找总耗时长的算子类型 |
| `Avg Time(us)` / `Min Time(us)` / `Max Time(us)` | 平均/最小/最大 | 波动（max>>avg → 调用异常） |
| `Core Type` | AI_CORE / AI_VECTOR_CORE / AI_CPU | — |
| `Ratio(%)` | 耗时占比 | **>40% 需优化；占比过大可能有重复编译** |
| `Device_id` / `Model Name` | 设备 / 模型 | — |

### 11. task_time_*.csv — Task Scheduler 任务调度信息

任务调度队列 / 多线程时间线数据（详细字段见文档 5.7.3.12–13 节）。

### 12. api_statistic_*.csv — CANN 层 API 执行耗时（★launch 开销）

| 字段 | 含义 | 优化用途 |
|---|---|---|
| `Device_id` | 设备 ID | — |
| `Level` | API 级别：AscendCL / Runtime / Node / Model / HCCL | — |
| `API Name` | API 名（如 aclrtLaunchKernel） | launch 开销定位 |
| `Time(us)` | API 总耗时（降序） | 前 N 个高耗时 API |
| `Count` | 调用次数 | — |
| `Avg(us)` / `Min(us)` / `Max(us)` | 平均/最小/最大 | 稳定性 |
| `Variance` | 方差 | 是否某次调用异常慢 |

> launch 开销大（api 耗时接近 kernel 耗时）→ 算子粒度小 → 融合/加大算子。

### 13. l2_cache_*.csv — 任务级 L2 Cache

| 字段 | 含义 |
|---|---|
| `Stream Id` / `Task Id` | 流 / 任务（Task Id 可关联 task_time） |
| `Hit Rate` | AI Core 命中 L2 占比 |
| `Victim Rate` | victim 占比 |
| `Op Name` | 算子名 |

### 14. msprof_*.json — timeline（chrome trace 兼容）

| 字段 | 含义 |
|---|---|
| `Title` | API 名称 |
| `Start` | 起始时间（与 chrome trace 对齐） |
| `Wall Duration` | API 调用耗时 |
| `Self Time` | API 自身执行时间 |
| `Mode` | `ACL_OP`(单算子) / `ACL_MODEL`(模型) / `ACL_RTS`(运行时) |

---

## 三、per-op 每通路带宽（op_summary 推导）

**原理**：op_summary 每行同时有 shape/dtype 和每 pipe 耗时 → `带宽 = 搬运字节数 ÷ 对应pipe耗时`。

标准 tiled matmul 的体积估算（每元素每通路搬一次的近似，**标 est**）：

| 通路 | 字节数 | pipe 耗时 |
|---|---|---|
| MTE2 读 (GM→L1/UB) | `(M×K + K×N) × dtype字节` | `aic_mte2_time` |
| MTE1 (L1→L0A/L0B) | `(M×K + K×N) × dtype字节` | `aic_mte1_time` |
| 写回 (L0C/UB→GM) | `M×N × dtype字节` | `aic_mte3_time` 或 `aiv_mte3_time` 或 `aic_fixpipe_time` |
| cube 计算 | `2×M×N×K` MAC | `aic_cube_time` |

> ⚠️ 估算偏差来源：tiling 复用（double-buffer / 多 pass L1）会让实际流量高于此下界；多输入逗号分隔的 shape 字符串会被误当一个高维张量（用 dtype 张量数兜底拆分）。结果要标 est，别当精确值。

---

## 四、字段 → 优化策略映射（按优化大顺序 Tier 1→6）

> 每个字段标「来源」；「用途/判据」是看它干嘛、怎么判断该不该动。判据里的峰值见第六节。

### Tier 1 算法结构 — 先定"用对算法、算力用对没"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `aic_cube_fops`（cube浮点运算数） | msprof op / ArithmeticUtilization.csv | 算力分子：cube 实际 FLOPs，对 294.9TFLOPS 看利用 |
| `aiv_vec_fops`（向量浮点运算数） | msprof op / ArithmeticUtilization.csv | vector FLOPs |
| `aic_cube_ratio`（cube指令周期占比） | msprof op / PipeUtilization.csv | cube 利用率低 → 算法不行 |
| `aiv_vec_ratio`（向量指令周期占比） | msprof op / PipeUtilization.csv | 高 + cube 低 → 该用 matmul 却走了逐元素 → 换算法 |

### Tier 2 算子融合 — 再看"中间存储要不要消"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `num_kernels`（去重算子数） | 通用 msprof / op_summary.csv | >1 → 有融合机会 |
| `Task Type`（引擎归属） | 通用 msprof / op_summary.csv | AI_CORE/AIV/MIX → 串行判断 |
| `Time(us)`+`Count`（API总耗时/次数） | 通用 msprof / api_statistic.csv | API 耗时≈kernel 耗时 → launch 开销大 → 融合 |
| `Ratio(%)`（算子类型耗时占比） | 通用 msprof / op_statistic.csv | 占比大的类型优先优化 |

### Tier 3 分块 & 启动配置 — 然后调"核吃满没、L0A/B 够不够"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `Block Dim`（核数） | msprof op / OpBasicInfo.csv 或 通用 msprof / op_summary.csv | < 40（vector核）→ tile 太小/并行不足 |
| `aic_mte1_time(us)`/`aic_mte1_ratio`（L1→L0A/B 搬运） | msprof op / PipeUtilization.csv | 占比高 → L0A/B 搬运瓶颈 → 调 BLOCK_K |
| `aic_l0a_read_bw`/`aic_l0b_read_bw`（L0A/L0B读带宽） | msprof op / MemoryL0.csv | 接近饱和 → 减搬运量/换通路 |

### Tier 4 内存访问 — 再调"带宽用满没、L2 命中"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `aic_main_mem_read_bw`/`aic_main_mem_write_bw`（GM读/写带宽） | msprof op / Memory.csv | 接近 1.8TB/s → memory_bound → 减数据量/升精度 |
| `aic_total_hit_rate(%)`（L2总命中率） | msprof op / L2Cache.csv | 低 → 数据复用差 → 调分块驻留 L2 |
| `aic_mte2_time(us)`（GM→核 读）、`aic_mte3_time(us)`（核→GM 写） | msprof op / PipeUtilization.csv | ≈总耗时 → 读/写是瓶颈 |
| `aiv_gm_to_ub_bw`/`aiv_ub_to_gm_bw`（GM→UB/UB→GM 带宽） | msprof op / Memory.csv | 搬运通路饱和 → 合并小传输/double buffer |

### Tier 5 计算 & 占用 — 再调"指令冲突、标量拖累"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `aic_cube_time(us)`（cube耗时） | msprof op / PipeUtilization.csv | 满 → compute_bound |
| `aic_scalar_time(us)`（标量指令耗时） | msprof op / PipeUtilization.csv | 高 → 减地址计算/循环展开 |
| `aiv_vec_bank_cflt_ratio`（bank冲突） | msprof op / ResourceConflictRatio.csv | >4% → padding(32B) |
| `aiv_vec_bankgroup_cflt_ratio`（bankgroup冲突） | msprof op / ResourceConflictRatio.csv | >4% → 调 repeatStride/blockStride |
| `aiv_vec_total_cflt_ratio`（vec总冲突） | msprof op / ResourceConflictRatio.csv | >5% → 触发冲突优化流程 |

### Tier 6 910B3 架构 — 最后调"取指、阻塞、引擎分配"

| 字段（中文） | 来源 | 用途/判据 |
|---|---|---|
| `aic_icache_miss_rate`（指令cache缺失率） | msprof op / PipeUtilization.csv | 高 → 指令体量大 → 代码紧凑化 |
| `aiv_vec_wait_ratio`（vector被阻塞占比） | msprof op / ResourceConflictRatio.csv | 高 → double buffer / 加深流水重叠 |
| `aic_cube_wait_ratio`（cube被阻塞占比） | msprof op / ResourceConflictRatio.csv | 高 → 流水重叠不足 |
| `Task Type`（kernel归属引擎） | 通用 msprof / op_summary.csv | 引擎分配不均 → 切 pipeline / 调 grid |

---

## 五、全部字段来源总表（msprof / msprof op → 文件 → 列名）

> 上面各层用到的字段，全部来自这两个工具。下表是**完整来源清单**：产出工具 → 文件 → 原始列名。
> `msprof op` → `OPPROF_*/`；通用 `msprof` → `mindstudio_profiler_output/`（文件名带时间戳）。

### 5.1 `msprof op` 产出（单算子 kernel 级 + 全通路）★主源

**OpBasicInfo.csv** — 算子基础信息

| 字段 | 原始列名 |
|---|---|
| 引擎归属 | `Task Type` |
| 总耗时 | `Task Duration(us)` |
| 核数 | `Block Dim` |
| 算子名/类型 | `Op Name` / `Op Type` |
| 输入形状/类型 | `Input Shape(s)` / `Input Data Type(s)` |
| 输出形状/类型 | `Output Shape(s)` / `Output Data Type(s)` |
| AI Core/Vector 理论时间 | `aicore_time(us)` / `aiv_time(us)` |
| 总周期 | `aic_total_cycles` / `aiv_total_cycles` |
| 频率 | `Current Freq(MHz)` |

**PipeUtilization.csv** — 计算/搬运单元耗时及占比

| 字段 | 原始列名 |
|---|---|
| cube 指令耗时/占比 | `aic_cube_time(us)` / `aic_cube_ratio` |
| 向量指令耗时/占比 | `aiv_vec_time(us)` / `aiv_vec_ratio` |
| L1→L0A/L0B (MTE1) | `aic_mte1_time(us)` / `aic_mte1_ratio` |
| GM→核 读 (MTE2) | `aic_mte2_time(us)` / `aic_mte2_ratio` / `aiv_mte2_time(us)` |
| 核→GM 写 (MTE3) | `aic_mte3_time(us)` / `aic_mte3_ratio` / `aiv_mte3_time(us)` |
| 标量指令 | `aic_scalar_time(us)` / `aic_scalar_ratio` / `aiv_scalar_time(us)` |
| L0C→L1 (fixpipe) | `aic_fixpipe_time(us)` / `aic_fixpipe_ratio` |
| 指令cache缺失率 | `aic_icache_miss_rate` |
| 基础时间/周期 | `aic_time(us)` / `aiv_time(us)` / `aic_total_cycles` / `aiv_total_cycles` |

**ArithmeticUtilization.csv** — 计算量及指令占比

| 字段 | 原始列名 |
|---|---|
| cube FLOPs / 占比 | `aic_cube_fops` / `aic_cube_ratio` |
| cube fp16/int8 占比 | `aic_cube_fp16_ratio` / `aic_cube_int8_ratio` |
| cube 指令条数 | `aic_cube_total_instr_number` / `aic_cube_fp_instr_number` / `aic_cube_int_instr_number` |
| vector FLOPs / 占比 | `aiv_vec_fops` / `aiv_vec_ratio` |
| vector 精度细分 | `aiv_vec_fp32_ratio` / `aiv_vec_fp16_ratio` / `aiv_vec_int32_ratio` / `aiv_vec_int16_ratio` / `aiv_vec_misc_ratio` |
| 总周期 | `aic_total_cycles` / `aiv_total_cycles` |

**Memory.csv** — 内存读写带宽速率(GB/s)

| 字段 | 原始列名 |
|---|---|
| GM 读/写带宽 | `aic_main_mem_read_bw` / `aic_main_mem_write_bw` |
| L1 读/写带宽 | `aic_l1_read_bw` / `aic_l1_write_bw` |
| GM→UB 带宽 (MTE2) | `aiv_gm_to_ub_bw` |
| UB→GM 带宽 (MTE3) | `aiv_ub_to_gm_bw` |

**MemoryL0.csv** — L0A/L0B/L0C 带宽

| 字段 | 原始列名 |
|---|---|
| L0A 读/写 | `aic_l0a_read_bw` / `aic_l0a_write_bw` |
| L0B 读/写 | `aic_l0b_read_bw` / `aic_l0b_write_bw` |
| L0C 读/写 (cube) | `l0c_read_bw_cube` / `l0c_write_bw_cube` |

**MemoryUB.csv** — UB 读写带宽

| 字段 | 原始列名 |
|---|---|
| vector 的 UB 读/写 | `aiv_ub_read_bw_vector` / `aiv_ub_write_bw_vector` |
| scalar 的 UB 读/写 | `aiv_ub_read_bw_scalar` / `aiv_ub_write_bw_scalar` |

**L2Cache.csv** — L2 命中率

| 字段 | 原始列名 |
|---|---|
| 总/读/写命中率 | `aic_total_hit_rate(%)` / `aic_read_hit_rate(%)` / `aic_write_hit_rate(%)` |

**ResourceConflictRatio.csv** — 资源冲突占比

| 字段 | 原始列名 |
|---|---|
| bank 冲突 | `aiv_vec_bank_cflt_ratio` |
| bankgroup 冲突 | `aiv_vec_bankgroup_cflt_ratio` |
| vec 总冲突 | `aiv_vec_total_cflt_ratio` |
| 资源冲突 | `aiv_vec_resc_cflt_ratio` |
| mte 冲突 | `aiv_vec_mte_cflt_ratio` |
| vec/cube 被阻塞 | `aiv_vec_wait_ratio` / `aic_cube_wait_ratio` |

### 5.2 通用 `msprof` 产出（任务级）

**op_summary_*.csv** — 每 kernel 一行（★per-op 全信息）

| 字段 | 原始列名 |
|---|---|
| 算子名/类型 | `Op Name` / `Op Type` |
| 引擎归属 | `Task Type` |
| 总耗时/开始/调度间隔 | `Task Duration(us)` / `Task Start Time(us)` / `Task Wait(us)` |
| 核数 | `Block Dim` |
| 输入/输出形状 | `Input Shape(s)` / `Output Shape(s)` |
| 输入/输出类型 | `Input Data Type(s)` / `Output Data Type(s)` |
| 理论执行时间 | `aicore_time(us)` / `aic_time(us)` / `aiv_time(us)` |
| 总周期 | `Total Cycles` |
| 每 pipe 耗时 (per-op 搬运带宽用) | `aic_mac_time(us)` / `aic_scalar_time(us)` / `aic_mte1_time(us)` / `aic_mte2_time(us)` / `aic_fixpipe_time(us)` / `aiv_vec_time(us)` / `aiv_scalar_time(us)` / `aiv_mte2_time(us)` / `aiv_mte3_time(us)` |

**op_statistic_*.csv** — 算子调用统计

| 字段 | 原始列名 |
|---|---|
| 算子类型 | `OP Type` |
| 调用次数 | `Count` |
| 总耗时 | `Total Time(us)` |
| 平均/最小/最大 | `Avg Time(us)` / `Min Time(us)` / `Max Time(us)` |
| 核类型 | `Core Type` |
| 耗时占比 | `Ratio(%)` |
| 设备/模型 | `Device_id` / `Model Name` |

**api_statistic_*.csv** — CANN 层 API 耗时

| 字段 | 原始列名 |
|---|---|
| API 级别 | `Level` |
| API 名 | `API Name` |
| 总耗时/次数 | `Time(us)` / `Count` |
| 平均/最小/最大 | `Avg(us)` / `Min(us)` / `Max(us)` |
| 方差 | `Variance` |

**l2_cache_*.csv** — 任务级 L2

| 字段 | 原始列名 |
|---|---|
| 流/任务 | `Stream Id` / `Task Id` |
| 命中率 | `Hit Rate` |
| victim 率 | `Victim Rate` |
| 算子名 | `Op Name` |

**task_time_*.csv** — Task Scheduler 调度信息（字段见文档 5.7.3.12–13 节）

**msprof_*.json** — timeline（chrome trace 兼容）

| 字段 | 原始列名 |
|---|---|
| API 名 | `Title` |
| 起始时间 | `Start` |
| 调用耗时 | `Wall Duration` |
| 自身执行时间 | `Self Time` |
| 模式 | `Mode` |

---

## 六、910B3 理论峰值（roofline 基准）

| 指标 | 值 | 备注 |
|---|---|---|
| GM 带宽 | 1.8 TB/s | 对 `main_mem_*_bw` |
| cube 算力 | 294.9 TFLOPS (fp16) | 20核 × 16³ FMA × 1.8GHz |
| UB | 192 KB | — |
| L1 | 512 KB | — |
| L0A / L0B / L0C | 64 / 64 / 128 KB | — |
| L2 | 192 MB | — |

---

## 七、命令速查

```bash
# msprof op（★主源，默认全量 8 CSV，别指定 --aic-metrics）
msprof op --kernel-name=matmul_kernel --warm-up=10 --output=./board_prof python3 test_matmul.py

# 通用 msprof（8.5.1 用 --ai-core=on）
msprof --output=./task_prof --application="python3 test_matmul.py" --ai-core=on

# 字段校验（区分"列名不匹配 BUG" vs "源无此字段 合法缺"）
python3 analyzers/check_fields.py board.json task.json diagnosis.json
```

## 八、资料来源

- CANN 工具使用（8.5.0 alpha002）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/devaids/optool/atlasopdev_16_00851.html>
- Memory / MemoryL0 / MemoryUB：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1alpha002/devaids/optool/atlasopdev_16_0096.html>
- ResourceConflictRatio（8.5 商用）：<https://www.hiascend.com:6066/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0100.html>
- ArithmeticUtilization（8.5 商用）：<https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0093.html>
- 获取性能数据（op_summary）：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0008.html>
- op_statistic：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/devaids/Profiling/atlasprofiling_16_0068.html>
- api_statistic：<https://www.hiascend.com:6066/document/detail/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0063.html>
- L2Cache：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/opdev/optool/atlasopdev_16_0103.html>
- msopprof 模式性能数据：<https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_simulator_user_guide.md>
- Triton-Ascend 性能分析：<https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/debug_guide/profiling.html>
