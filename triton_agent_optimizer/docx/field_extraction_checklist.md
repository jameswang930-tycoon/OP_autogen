# 字段提取期望清单 — 没提取到的字段本该从哪来

> 目的：`check_fields.py` 报某个字段「合法缺」或值为空时，按本清单去真机核对——**该字段本应来自哪个工具、哪个文件、哪个列名**，到底有没有。
> 用法：`python3 analyzers/check_fields.py board_1.json task.json diagnosis.json` → 输出里每个缺字段会带「期望[文件 列名]」→ 去真机对应文件看那一列。

## 核对三步法

对每个缺字段，去期望文件找期望列，三选一：

| 核对结果 | 结论 | 处理 |
|---|---|---|
| 文件/列存在 **且有值** | parser 或 run 配置问题 | 贴 raw 列名给我，修 parser 键 |
| 文件/列存在 **但值为空** | 该引擎/通路无数据（合法）或采集配置没开 | 看标记 ⚠️ 依赖 `--aic-metrics` 的列；向量算子无 cube 列是正常的 |
| 文件/列 **不存在** | 版本无此列（合法缺） | 无操作，或用 `msprof op` 补采 |

## 标记含义

- ✅ 基础必有（默认采集就有）
- ⚠️ 依赖 `--aic-metrics=<group>` 才生成（8.5.1 通用 msprof 该参数报错 → 用 msprof op 拿）
- 🔶 版本差异（`fixpipe`/`fixp`、`cube`/`mac` 两种名都认）

---

## 一、Board 侧 — `msprof op` → `OPPROF_*/`（8 CSV）

### 1. OpBasicInfo.csv — kernel 级基础 ✅

| normalized | 期望列名 |
|---|---|
| `execution_summary.total_ns` | `Task Duration(us)` |
| `execution_summary.num_cores` | `Block Dim` |
| `execution_summary.kernel_name` | `Op Name` |
| `execution_summary.freq_mhz` | `Current Freq(MHz)` |

### 2. PipeUtilization.csv — 引擎利用率 + 每 pipe 耗时 ✅

| normalized | 期望列名 |
|---|---|
| `engine_utilization.cube` | `aic_cube_ratio`（🔶 或 `aic_mac_ratio`） |
| `engine_utilization.vec` | `aiv_vec_ratio` |
| `engine_utilization.mte1` | `aic_mte1_ratio` |
| `engine_utilization.mte2` | `aic_mte2_ratio` / `aiv_mte2_ratio` |
| `engine_utilization.mte3` | `aic_mte3_ratio` / `aiv_mte3_ratio` |
| `engine_utilization.scalar` | `aic_scalar_ratio` / `aiv_scalar_ratio` |
| `engine_utilization.fixpipe` | `aic_fixpipe_ratio`（🔶 `aic_fixp_ratio`） |
| 每 pipe 耗时 | `aic_cube_time(us)` / `aic_mte1_time(us)` / `aic_mte2_time(us)` / `aiv_vec_time(us)` … |

### 3. ArithmeticUtilization.csv — 算力 ✅

| normalized | 期望列名 |
|---|---|
| `compute.cube_fops` | `aic_cube_fops` |
| `compute.vector_fops` | `aiv_vec_fops` |
| `compute.cube_ratio` | `aic_cube_ratio` |
| `compute.cube_fp16_ratio` | `aic_cube_fp16_ratio` |
| `compute.cube_int8_ratio` | `aic_cube_int8_ratio` |
| `compute.aic_total_cycles` / `aiv_total_cycles` | `aic_total_cycles` / `aiv_total_cycles` |

### 4. Memory.csv — GM/L1/UB↔GM 带宽 ✅

| normalized | 期望列名 |
|---|---|
| `main_mem_read_gb_s` | `aic_main_mem_read_bw` |
| `main_mem_write_gb_s` | `aic_main_mem_write_bw` |
| `l1_read_gb_s` | `aic_l1_read_bw` |
| `l1_write_gb_s` | `aic_l1_write_bw` |
| `gm_to_ub_gb_s` (GM→UB MTE2) | `aiv_gm_to_ub_bw` |
| `ub_to_gm_gb_s` (UB→GM MTE3) | `aiv_ub_to_gm_bw` |
| `l2_read/write_gb_s` | **⚠️ Memory.csv 无 L2 带宽列（A2 系）** → 合法缺，别找 |

### 5. MemoryL0.csv — L0A/L0B/L0C 带宽 ✅

| normalized | 期望列名 |
|---|---|
| `l0a_read/write_gb_s` | `aic_l0a_read_bw` / `aic_l0a_write_bw` |
| `l0b_read/write_gb_s` | `aic_l0b_read_bw` / `aic_l0b_write_bw` |
| `l0c_read/write_gb_s` | `l0c_read_bw_cube` / `l0c_write_bw_cube` |

### 6. MemoryUB.csv — UB 读写带宽 ✅（列名版本差异大 🔶）

| normalized | 期望列名（候选） |
|---|---|
| `ub_vector_read_gb_s` | `aiv_ub_read_bw_vector`（🔶 也可能 `ub_read_bw_vector`） |
| `ub_vector_write_gb_s` | `aiv_ub_write_bw_vector` |
| `ub_scalar_read_gb_s` | `aiv_ub_read_bw_scalar` / `aic_ub_read_bw_scalar` |
| `ub_scalar_write_gb_s` | `aiv_ub_write_bw_scalar` / `ub_write_bw_scalar` |
| `ub_mte_read/write_gb_s` | `ub_read_bw_mte` / `ub_write_bw_mte` — ⚠️ 仅推理产品，910B3 合法缺 |
| fixpipe→UB 写 | 🔶 `aiv_fixp2ub_write_bw(GB/s)` / `aic_fixp2ub_write_bw(GB/s)`（cannbot-skills 真实名） |

> **重点核对**：cannbot-skills（CANN 官方 agent 技能）真实解析 MemoryUB 用的是 `aiv_fixp2ub_write_bw` 这类名字，和我文档里的 `aiv_ub_read_bw_vector` 不同。你 8.5.1 到底是哪套，贴 MemoryUB.csv 列头给我确认。

### 7. L2Cache.csv — L2 命中率 ✅

| normalized | 期望列名 |
|---|---|
| `l2_hit_rate` | `aic_total_hit_rate(%)`（🔶 `aiv_total_hit_rate(%)`，值>1 归一化为 0~1） |

### 8. ResourceConflictRatio.csv — 冲突 ✅

| normalized | 期望列名 |
|---|---|
| `conflict.bank_cflt_ratio` | `aiv_vec_bank_cflt_ratio` |
| `conflict.bankgroup_cflt_ratio` | `aiv_vec_bankgroup_cflt_ratio` |
| `conflict.total_cflt_ratio` | `aiv_vec_total_cflt_ratio` |
| `conflict.resc_cflt_ratio` | `aiv_vec_resc_cflt_ratio` |
| `conflict.mte_cflt_ratio` | `aiv_vec_mte_cflt_ratio` |

---

## 二、Task 侧 — 通用 `msprof` → `mindstudio_profiler_output/`

### 9. op_summary_*.csv — 每 kernel 一行 ✅基础 / ⚠️per-pipe 需 --aic-metrics

| normalized | 期望列名 |
|---|---|
| `kernel.op_name` | `Op Name` |
| `kernel.task_type` | `Task Type` |
| `kernel.task_duration_us` | `Task Duration(us)` |
| `kernel.block_dim` | `Block Dim` |
| `kernel.input_shapes` | `Input Shape(s)` |
| `kernel.input_dtypes` | `Input Data Type(s)` |
| `kernel.aicore_time_us` | `aicore_time(us)`（🔶 `aic_time(us)`） |
| `kernel.aiv_time_us` | `aiv_time(us)` |
| `kernel.total_cycles` | `Total Cycles` |
| `task.pipes_us.*`（每 pipe 耗时） | `aic_mac_time(us)`🔶`aic_cube_time(us)` / `aic_mte1_time(us)` / `aic_mte2_time(us)` / `aic_fixpipe_time(us)`🔶`aic_fixp_time(us)` / `aiv_vec_time(us)` / `aiv_mte2_time(us)` / `aiv_mte3_time(us)` — **⚠️ 8.5.1 通用 msprof `--ai-core=on` 无这些列（需 --aic-metrics）→ 应判合法缺，数据从 msprof op 的 PipeUtilization.csv 拿** |

### 10. op_statistic_*.csv — 类型统计 ✅

| normalized | 期望列名 |
|---|---|
| `multi_kernel[].op_type` | `OP Type` |
| `multi_kernel[].count` | `Count` |
| `multi_kernel[].total_time_us` | `Total Time(us)` |
| `multi_kernel[].ratio` | `Ratio(%)` |

### 11. api_statistic_*.csv — API 开销 ✅

| normalized | 期望列名 |
|---|---|
| `api_overhead[].api_name` | `API Name` |
| `api_overhead[].total_us` | `Time(us)` |
| `api_overhead[].count` | `Count` |

### 12. l2_cache_*.csv — 任务级 L2 ✅

| normalized | 期望列名 |
|---|---|
| `l2_hit_rate` | `Hit Rate` |

### 13. task_time_*.csv — 调度信息（字段见文档 5.7.3.12–13 节，未用）

---

## 三、diagnosis 侧 — 计算字段（不来自任何 CSV 原始列）

| 字段 | 来源计算 |
|---|---|
| `summary.num_kernels` | op_summary 去重 Op Name 数 |
| `summary.filled_kernels` | 有 board 匹配的 kernel 数 |
| `kernels[i].deep.roofline.*` | `main_mem_read/write_bw` ÷ 1.8TB/s；`cube_fops+vec_fops` ÷ 294.9TFLOPS |

---

## 四、真机上核对命令（每类各一行）

```bash
# board 侧 (msprof op) — 8 个 CSV 的列头
for f in OpBasicInfo PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio; do
  echo "── $f ──"; head -1 input/matmul/e2e_run/04_board/op_1/*/$f.csv 2>/dev/null || head -1 input/matmul/e2e_run/04_board/*/$f.csv 2>/dev/null
done

# task 侧 (通用 msprof) — 各文件列头
for f in op_summary op_statistic api_statistic l2_cache; do
  echo "── $f ──"; find input/matmul/e2e_run/05_task -name "${f}_*.csv" | head -1 | xargs head -1
done

# 缺字段逐一核对 (每个带期望来源)
python3 analyzers/check_fields.py input/matmul/e2e_run/06_diagnosis/board_1.json \
  input/matmul/e2e_run/06_diagnosis/task.json input/matmul/e2e_run/06_diagnosis/diagnosis.json
```

把上面列头 + check_fields 输出贴回来，我按真实列名订正 `docx/msprof_fields_reference.md` 和 parser 键。
