# 字段提取核对清单 — 没提取到的字段，本该从哪来

> 每个字段一行，直接写：**没提取到 → 本该从 [哪个工具] 的 [哪个文件] 的 [哪个具体列名] 提取**。
> 你去真机打开那个文件，找那一列，三选一：
> 1. **列在且有值** → parser 或采集配置问题 → 把真实列名贴给我，我修键
> 2. **列在但值为空** → 该引擎/通路无数据（合法），或需要 `--aic-metrics` 才采集
> 3. **列不在** → 版本无此列（合法缺）
>
> 标记：✅ 基础必有（默认采集就有）｜⚠️ 需要 `--aic-metrics` 才生成｜🔶 版本不同列名不同

---

## A. `msprof op` 产出的文件（在 `OPPROF_{时间戳}/` 目录，8 个 CSV）

### A1. OpBasicInfo.csv（kernel 级基础信息）

- [ ] `execution_summary.total_ns` ← 没提取到 → 本应从 **msprof op** 的 **OpBasicInfo.csv** 的 **`Task Duration(us)`** 提取 ✅
- [ ] `execution_summary.num_cores` ← 没提取到 → 本应从 **msprof op** 的 **OpBasicInfo.csv** 的 **`Block Dim`** 提取 ✅
- [ ] `execution_summary.kernel_name` ← 没提取到 → 本应从 **msprof op** 的 **OpBasicInfo.csv** 的 **`Op Name`** 提取 ✅
- [ ] `execution_summary.freq_mhz` ← 没提取到 → 本应从 **msprof op** 的 **OpBasicInfo.csv** 的 **`Current Freq(MHz)`** 提取 ✅

### A2. PipeUtilization.csv（引擎利用率 + 每 pipe 耗时）

- [ ] `engine_utilization.cube` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_cube_ratio`** 提取（🔶 或 `aic_mac_ratio`）✅
- [ ] `engine_utilization.vec` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aiv_vec_ratio`** 提取 ✅
- [ ] `engine_utilization.mte1` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_mte1_ratio`** 提取 ✅
- [ ] `engine_utilization.mte2` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_mte2_ratio`**（或 `aiv_mte2_ratio`）提取 ✅
- [ ] `engine_utilization.mte3` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_mte3_ratio`**（或 `aiv_mte3_ratio`）提取 ✅
- [ ] `engine_utilization.scalar` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_scalar_ratio`**（或 `aiv_scalar_ratio`）提取 ✅
- [ ] `engine_utilization.fixpipe` ← 没提取到 → 本应从 **msprof op** 的 **PipeUtilization.csv** 的 **`aic_fixpipe_ratio`** 提取（🔶 或 `aic_fixp_ratio`）✅

### A3. ArithmeticUtilization.csv（算力）

- [ ] `compute.cube_fops` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aic_cube_fops`** 提取 ✅
- [ ] `compute.vector_fops` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aiv_vec_fops`** 提取 ✅
- [ ] `compute.cube_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aic_cube_ratio`** 提取 ✅
- [ ] `compute.cube_fp16_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aic_cube_fp16_ratio`** 提取 ✅
- [ ] `compute.cube_int8_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aic_cube_int8_ratio`** 提取 ✅
- [ ] `compute.aic_total_cycles` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aic_total_cycles`** 提取 ✅
- [ ] `compute.aiv_total_cycles` ← 没提取到 → 本应从 **msprof op** 的 **ArithmeticUtilization.csv** 的 **`aiv_total_cycles`** 提取 ✅

### A4. Memory.csv（GM/L1/UB↔GM 带宽）

- [ ] `main_mem_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aic_main_mem_read_bw`** 提取 ✅
- [ ] `main_mem_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aic_main_mem_write_bw`** 提取 ✅
- [ ] `l1_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aic_l1_read_bw`** 提取 ✅
- [ ] `l1_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aic_l1_write_bw`** 提取 ✅
- [ ] `gm_to_ub_gb_s` (GM→UB, MTE2) ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aiv_gm_to_ub_bw`** 提取 ✅
- [ ] `ub_to_gm_gb_s` (UB→GM, MTE3) ← 没提取到 → 本应从 **msprof op** 的 **Memory.csv** 的 **`aiv_ub_to_gm_bw`** 提取 ✅
- [ ] `l2_read_gb_s` / `l2_write_gb_s` ← 没提取到 → **⚠️ Memory.csv 没有 L2 带宽列**（A2 系），别去找，属合法缺

### A5. MemoryL0.csv（L0A/L0B/L0C 带宽）

- [ ] `l0a_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`aic_l0a_read_bw`** 提取 ✅
- [ ] `l0a_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`aic_l0a_write_bw`** 提取 ✅
- [ ] `l0b_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`aic_l0b_read_bw`** 提取 ✅
- [ ] `l0b_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`aic_l0b_write_bw`** 提取 ✅
- [ ] `l0c_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`l0c_read_bw_cube`** 提取 ✅
- [ ] `l0c_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryL0.csv** 的 **`l0c_write_bw_cube`** 提取 ✅

### A6. MemoryUB.csv（UB 读写带宽）🔶 列名版本差异大，重点核对

- [ ] `ub_vector_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryUB.csv** 的 **`aiv_ub_read_bw_vector`** 提取（🔶 或 `ub_read_bw_vector`）✅
- [ ] `ub_vector_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryUB.csv** 的 **`aiv_ub_write_bw_vector`** 提取 ✅
- [ ] `ub_scalar_read_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryUB.csv** 的 **`aiv_ub_read_bw_scalar`** 提取（🔶 或 `aic_ub_read_bw_scalar`）✅
- [ ] `ub_scalar_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryUB.csv** 的 **`aiv_ub_write_bw_scalar`** 提取 ✅
- [ ] `ub_mte_read_gb_s` / `ub_mte_write_gb_s` ← 没提取到 → 本应从 **msprof op** 的 **MemoryUB.csv** 的 **`ub_read_bw_mte`** / **`ub_write_bw_mte`** 提取（⚠️ 仅推理产品，910B3 合法缺）
- [ ] 🔶 cannbot-skills 真实解析 MemoryUB 用的列名是 **`aiv_fixp2ub_write_bw(GB/s)`** / **`aic_fixp2ub_write_bw(GB/s)`** —— 你的 MemoryUB.csv 是不是这套？贴列头确认

### A7. L2Cache.csv（L2 命中率）

- [ ] `l2_hit_rate` ← 没提取到 → 本应从 **msprof op** 的 **L2Cache.csv** 的 **`aic_total_hit_rate(%)`** 提取（🔶 或 `aiv_total_hit_rate(%)`）✅

### A8. ResourceConflictRatio.csv（资源冲突）

- [ ] `conflict.bank_cflt_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ResourceConflictRatio.csv** 的 **`aiv_vec_bank_cflt_ratio`** 提取 ✅
- [ ] `conflict.bankgroup_cflt_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ResourceConflictRatio.csv** 的 **`aiv_vec_bankgroup_cflt_ratio`** 提取 ✅
- [ ] `conflict.total_cflt_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ResourceConflictRatio.csv** 的 **`aiv_vec_total_cflt_ratio`** 提取 ✅
- [ ] `conflict.resc_cflt_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ResourceConflictRatio.csv** 的 **`aiv_vec_resc_cflt_ratio`** 提取 ✅
- [ ] `conflict.mte_cflt_ratio` ← 没提取到 → 本应从 **msprof op** 的 **ResourceConflictRatio.csv** 的 **`aiv_vec_mte_cflt_ratio`** 提取 ✅

---

## B. 通用 `msprof` 产出的文件（在 `mindstudio_profiler_output/` 目录）

### B1. op_summary_*.csv（每 kernel 一行）

- [ ] `kernel.op_name` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Op Name`** 提取 ✅
- [ ] `kernel.task_type` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Task Type`** 提取 ✅
- [ ] `kernel.task_duration_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Task Duration(us)`** 提取 ✅
- [ ] `kernel.block_dim` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Block Dim`** 提取 ✅
- [ ] `kernel.input_shapes` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Input Shape(s)`** 提取 ✅
- [ ] `kernel.input_dtypes` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Input Data Type(s)`** 提取 ✅
- [ ] `kernel.aicore_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`aicore_time(us)`** 提取（🔶 或 `aic_time(us)`）✅
- [ ] `kernel.aiv_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`aiv_time(us)`** 提取 ✅
- [ ] `kernel.total_cycles` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`Total Cycles`** 提取 ✅
- [ ] `task.pipes_us.aic_mac_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`aic_mac_time(us)`** 提取（🔶 或 `aic_cube_time(us)`）**⚠️ 8.5.1 通用 msprof `--ai-core=on` 无这些列（需 `--aic-metrics`）→ 合法缺，数据从 msprof op 的 PipeUtilization.csv 拿**
- [ ] `task.pipes_us.aic_fixpipe_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`aic_fixpipe_time(us)`** 提取（🔶 或 `aic_fixp_time(us)`）⚠️ 同上
- [ ] `task.pipes_us.aiv_vec_time_us` / `aiv_mte2_time_us` / `aiv_mte3_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_summary_*.csv** 的 **`aiv_vec_time(us)`** / **`aiv_mte2_time(us)`** / **`aiv_mte3_time(us)`** 提取 ⚠️ 同上

### B2. op_statistic_*.csv（类型统计）

- [ ] `multi_kernel[].op_type` ← 没提取到 → 本应从 **通用 msprof** 的 **op_statistic_*.csv** 的 **`OP Type`** 提取 ✅
- [ ] `multi_kernel[].count` ← 没提取到 → 本应从 **通用 msprof** 的 **op_statistic_*.csv** 的 **`Count`** 提取 ✅
- [ ] `multi_kernel[].total_time_us` ← 没提取到 → 本应从 **通用 msprof** 的 **op_statistic_*.csv** 的 **`Total Time(us)`** 提取 ✅
- [ ] `multi_kernel[].ratio` ← 没提取到 → 本应从 **通用 msprof** 的 **op_statistic_*.csv** 的 **`Ratio(%)`** 提取 ✅

### B3. api_statistic_*.csv（API 开销）

- [ ] `api_overhead[].api_name` ← 没提取到 → 本应从 **通用 msprof** 的 **api_statistic_*.csv** 的 **`API Name`** 提取 ✅
- [ ] `api_overhead[].total_us` ← 没提取到 → 本应从 **通用 msprof** 的 **api_statistic_*.csv** 的 **`Time(us)`** 提取 ✅
- [ ] `api_overhead[].count` ← 没提取到 → 本应从 **通用 msprof** 的 **api_statistic_*.csv** 的 **`Count`** 提取 ✅

### B4. l2_cache_*.csv（任务级 L2）

- [ ] `l2_hit_rate` ← 没提取到 → 本应从 **通用 msprof** 的 **l2_cache_*.csv** 的 **`Hit Rate`** 提取 ✅

---

## C. 真机上核对命令（逐文件打列头）

```bash
# msprof op 的 8 个 CSV 列头
for f in OpBasicInfo PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio; do
  echo "── $f ──"
  find input/matmul/e2e_run/04_board -name "$f.csv" | head -1 | xargs head -1
done

# 通用 msprof 的 4 个文件列头
for f in op_summary op_statistic api_statistic l2_cache; do
  echo "── $f ──"
  find input/matmul/e2e_run/05_task -name "${f}_*.csv" | head -1 | xargs head -1
done

# 缺字段逐一核对（check_fields 会打印每个缺字段的期望来源）
python3 analyzers/check_fields.py input/matmul/e2e_run/06_diagnosis/board_1.json \
  input/matmul/e2e_run/06_diagnosis/task.json input/matmul/e2e_run/06_diagnosis/diagnosis.json
```

把上面的列头输出贴回来，我按真实列名订正 parser 键和 `msprof_fields_reference.md`。
