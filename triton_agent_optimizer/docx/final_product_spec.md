# 最终产物规格 — diagnosis.json（完整字段 + 来源）

> 最终整合产物：`input/matmul/e2e_run/06_diagnosis/diagnosis.json`
> 生成：`run_optimize.sh` → 通用 msprof(骨架) + 逐 kernel msprof op(deep) → `integrate.py` 按 kernel 名合并
> 规则详见 `aggregation_rules.md`；列名核实见 `msprof_fields_reference.md`；缺字段核对见 `field_extraction_checklist.md`

---

## 0. 三个文件的分工

| 文件 | 来源 | 角色 |
|---|---|---|
| **`diagnosis.json`** | integrate.py | **★最终产物（看这个）** |
| `task.json` | 通用 msprof → 骨架 | 输入①：kernel 槽位 + 每 kernel 耗时/形状/引擎/launch/L2 |
| `board_<i>.json` | msprof op → 每 kernel 深层 | 输入②：每 kernel 真实带宽/引擎/算力/冲突/roofline |

---

## 1. 完整结构（每个字段标来源）

```
diagnosis.json
├── meta
│   ├── source = "msprof (generic) + msprof op per-kernel"
│   ├── num_kernels / filled_kernels
│   └── inputs = {task, boards}      ← 记录用哪几个输入合成
├── summary                                  ← 全部来自 通用msprof (task.json)
│   ├── num_kernels           ← op_summary 去重非框架 Op Name 数（优化目标数）
│   ├── num_kernels_total     ← op_summary 全部 distinct 数（含框架）
│   ├── total_ns              ← op_summary `Task Duration(us)` 最大 ×1000
│   ├── num_cores             ← op_summary `Block Dim` 最大
│   ├── api_overhead_total_us ← api_statistic `Time(us)` 求和（launch 开销）
│   ├── l2_hit_rate           ← l2_cache `Hit Rate`（>1 归一化 0~1）
│   └── filled_kernels        ← 有几个 kernel 的 deep 被 msprof op 填了
│
├── kernels[]                 ← 每个 = 一个优化目标 kernel（非框架 aclnn*）
│   ├── kernel_name           ← op_summary `Op Name`
│   ├── launch_count          ← op_summary 同 Op Name 行数
│   ├── task {…}              ← ★骨架侧：通用msprof op_summary
│   │   ├── task_type         ← `Task Type` (AI_CORE/AIV/MIX)
│   │   ├── task_duration_us  ← `Task Duration(us)`
│   │   ├── block_dim         ← `Block Dim` (核数)
│   │   ├── input_shapes/dtypes ← `Input Shape(s)` / `Input Data Type(s)`
│   │   ├── output_shapes/dtypes
│   │   ├── aicore_time_us / aiv_time_us ← `aicore_time(us)` / `aiv_time(us)`
│   │   ├── total_cycles      ← `Total Cycles`
│   │   ├── pipes_us {}       ← `aic_mac_time(us)`/`aic_mte1/2/3`/`aic_fixpipe`/`aiv_vec/scalar/mte2/mte3`
│   │   │                       ⚠8.5.1 通用msprof 无这些列(需--aic-metrics)→可能空
│   │   ├── est_bytes_in/out  ← shape×dtype 计算（搬运块字节数）
│   │   └── transfers[]       ← bytes÷pipe时间 计算（per-op 每通路估带宽）
│   │
│   ├── deep {…}              ← ★深层侧：msprof op 的 OPPROF 8 CSV (board_<i>.json)
│   │   ├── freq_mhz          ← OpBasicInfo `Current Freq(MHz)`
│   │   ├── bandwidth_gb_s {} ← Memory/MemoryL0/MemoryUB.csv（17 通路）
│   │   │     main_mem_read/write ← Memory `aic_main_mem_read/write_bw`
│   │   │     l1_read/write       ← Memory `aic_l1_read/write_bw`
│   │   │     gm_to_ub / ub_to_gm ← Memory `aiv_gm_to_ub_bw`/`aiv_ub_to_gm_bw`（NA=合法缺）
│   │   │     ub_vector/scalar    ← MemoryUB `aiv_ub_*_bw_vector/scalar`（NA=合法缺）
│   │   │     l0a/l0b/l0c         ← MemoryL0 `aic_l0a/l0b/l0c_*_bw`
│   │   ├── engine_utilization {}← PipeUtilization `aic_cube_ratio`/`aiv_vec_ratio`/`aic_mte1/2/3`/`scalar`/`fixpipe`
│   │   ├── compute {}          ← ArithmeticUtilization `aic_cube_fops`/`aiv_vec_fops`/`*_ratio`/`*_cycles`
│   │   ├── conflict {}         ← ResourceConflictRatio `aiv_vec_bank/bankgroup/total/resc/mte_cflt_ratio`
│   │   ├── l2_hit_rate         ← L2Cache `aic_total_hit_rate(%)`
│   │   └── roofline {}         ← ★计算（用上面 deep 的带宽+算力）
│   │         bottleneck_type / achieved_memory_bw_gb_s / memory_utilization
│   │         / achieved_compute_tflops / compute_utilization / arithmetic_intensity
│   │         （对 1638.4GB/s 和 294.9/73.7TFLOPS fp16/fp32 判 memory/compute/latency/balanced bound）
│   │
│   └── filled_by             ← "msprof op"（有 deep）/ "msprof only"（没跑到 op）
│
├── framework_kernels[]       ← op_summary 里 aclnn*（torch 数据准备），非优化目标，仅观察
├── api_overhead[]            ← api_statistic：Level/API Name/Time(us)/Count/Avg/Min/Max/Variance
├── multi_kernel[]            ← op_statistic：OP Type/Count/Total Time/Core Type/Ratio(%)
└── notes[]
```

---

## 2. 「之前要的字段」分别在哪

| 之前要的 | 产物位置 | 来源 |
|---|---|---|
| kernel 数/核数/耗时 | `summary.num_kernels / total_ns / num_cores` | 通用msprof op_summary |
| 每 kernel 形状/类型/引擎 | `kernels[i].task.*` | 通用msprof op_summary |
| per-op 每 pipe 耗时 | `kernels[i].task.pipes_us` | ⚠op_summary(需--aic-metrics) 或 msprof op PipeUtilization |
| per-op 估搬运带宽 | `kernels[i].task.transfers` | shape×dtype÷pipe时间 计算 |
| 真实每通路带宽 | `kernels[i].deep.bandwidth_gb_s` | msprof op Memory/MemoryL0/MemoryUB |
| 引擎利用率 | `kernels[i].deep.engine_utilization` | msprof op PipeUtilization |
| 算力 fops | `kernels[i].deep.compute` | msprof op ArithmeticUtilization |
| UB 冲突 | `kernels[i].deep.conflict` | msprof op ResourceConflictRatio |
| L2 命中 | `kernels[i].deep.l2_hit_rate` + `summary.l2_hit_rate` | msprof op L2Cache / 通用msprof l2_cache |
| **roofline 一针见血** | `kernels[i].deep.roofline` | deep 带宽+算力 计算 |
| launch 开销 | `api_overhead[]` + `summary.api_overhead_total_us` | 通用msprof api_statistic |
| 多算子分解 | `multi_kernel[]` | 通用msprof op_statistic |

---

## 3. 核心一句话

`kernels[i]` 把「骨架（task，通用msprof）」和「深层（deep，msprof op）」按 **kernel 名** 拼在一起；
`deep.roofline` 是最终要看的判断（memory/compute/latency/balanced bound）。

查看：
```bash
python3 input/matmul/real_report.py input/matmul/e2e_run/06_diagnosis/diagnosis.json
# 或 cat 原始 JSON
```

---

## 4. 已知的合法缺失（不是 bug）

| 字段 | 为何缺 |
|---|---|
| `deep.bandwidth_gb_s.l2_read/write` | Memory.csv 无 L2 带宽列（A2 系） |
| `deep.bandwidth_gb_s.gm_to_ub / ub_to_gm` | `aiv_gm_to_ub_bw`/`aiv_ub_to_gm_bw` = NA（cube matmul 不用 AIV GM↔UB） |
| `deep.bandwidth_gb_s.ub_vector/scalar` | MemoryUB NA（无 vector 运算） |
| `deep.bandwidth_gb_s.ub_mte_read/write` | `ub_read/write_bw_mte` 仅推理产品，910B3 无 |
| `task.pipes_us.*` | 8.5.1 通用msprof 无 per-pipe 列（需 --aic-metrics） |

> 以上在 `check_fields.py` 均判「合法缺」；若 **cube 侧**（`aic_main_mem_*_bw`/`aic_cube_fops`/`aic_mte1/2/3_ratio`）也 NA，才是采集没开，需排查。
