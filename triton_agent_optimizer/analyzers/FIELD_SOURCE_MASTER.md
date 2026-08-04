# 字段来源总表（定稿）— 4 来源 × 输出文件 × 精确字段 → 6 优化策略

> 2026-08-03 网上核实（CANN 8.5 / msopprof / AscendNPU-IR 官方文档）。回答：每个工具输出什么文件、每个文件有哪些精确字段，映射到 6 个优化策略该看什么。

---

## 一、4 个来源 → 输出文件 → 精确字段

### 来源 A：msprof op simulator（指令级仿真）
```
命令: msprof op simulator --kernel-name=X --soc-version=Ascend910B3 --output=./sim_prof
产物: sim_prof/OPPROF_xxx/simulator/
  ├── core0~23.cubecore0/*_instr_exe.csv   ← 每核指令级
  └── trace.json                           ← Chrome trace (全核)
```
**instr_exe.csv 字段**（7 列）：`instr, addr, pipe, call_count, cycles, running_time(us), detail`
- `pipe` 取值：VECTOR/SCALAR/CUBE/MTE1/MTE2/MTE3/FIXP/FLOWCTRL/CACHEMISS/ALL
- `detail`：含 `Dtype:`、`Id:`、`XD:X3=`(搬运字节) 等 → 数据块大小
- ⚠ running_time(us) 常为 0，用 `cycles ÷ 1.9GHz` 换算

**trace.json**：Chrome trace 格式（`ph=X` 事件 + `ts`/`dur`/`pid`/`tid`/`name`/`cat`），可算 total_ns/并行/关键路径

**给策略**：每 op 真实指令耗时(per-call)、cycles、pipe、搬运块大小、串/并、指令级气泡 → **Tier1,4,5**

### 来源 B：msprof op（真机单算子，★主源）
```
命令: msprof op --kernel-name=X --warm-up=10 --output=./board_prof   # 不指定 --aic-metrics = 默认全量 8 CSV
产物: board_prof/OPPROF_xxx/  8 个 CSV:
```
| 文件 | 关键字段 | 用途 |
|---|---|---|
| **OpBasicInfo.csv** | Op Name, Op Type, **Task Duration(us)**, **Block Dim**, Device Id, Pid, Current/Rated Freq | 端到端耗时/核数 |
| **PipeUtilization.csv** | block_id, **aic_time(us)**, **aiv_time(us)**, **aic_total_cycles**, **aiv_total_cycles**, **aic_cube_time/ratio**, **aiv_vec_time/ratio**, **aic_mte1/2/3_time/ratio**, aic_scalar/fixpipe_time, icache_miss_rate | 各 pipe 占比 |
| **ArithmeticUtilization.csv** | **aic_cube_ratio**, **aic_cube_fops**(算力), aic_cube_total_instr_number, **aiv_vec_ratio**, **aiv_vec_fops**, aiv_vec_fp32/fp16/int32_ratio | cube/vec 算力 |
| **Memory.csv** | **main_mem_read/write_bw**, ub_read/write_bw, l1_read/write_bw, l2_read/write_bw | ★每通路真实带宽 |
| **MemoryL0.csv** | l0a/l0b/l0c_read/write_bw | L0 带宽 |
| **MemoryUB.csv** | mte/vector/scalar 到 UB 的读写带宽 | UB 带宽 |
| **L2Cache.csv** | **Hit Rate**, Victim Rate | ★L2 命中 |
| **ResourceConflictRatio.csv** | vec_bankgroup_cflt_ratio, vec_bank_cflt_ratio, vec_resc_cflt_ratio | UB bank 冲突 |

**给策略**：真实带宽/L2/cube/pipe → **Tier3,4,5,6**（真机校准）

### 来源 C：通用 msprof（真机任务级）
```
命令: msprof --output=./task_prof --application="python3 test_matmul.py" --ai-core=on
     (8.5.1 不认 --aic-metrics; 用 --ai-core=on 拿基础 op_summary)
产物: task_prof/PROF_xxx/mindstudio_profiler_output/
  ├── op_summary_*.csv   ← 每 kernel 一行
  ├── op_statistic_*.csv
  ├── task_time_*.csv
  ├── api_statistic_*.csv
  ├── l2_cache_*.csv
  └── msprof_*.json (timeline)
```
**op_summary 字段**：Device_id, Task ID, Stream ID, **Op Name**, **OP Type**, **Task Type**, **Task Start Time(us)**, **Task Duration(us)**, **Task Wait Time(us)**, **Block Dim**, HF32 Eligible, **Input/Output Shapes/Data Types**, aicore_time(us)*, aiv_time(us)*, total_cycles*
> * `aicore_time/aiv_time/total_cycles` 需 `--task-time=l1`（8.5.1 可能不支持，见 diagnose）

**给策略**：每 kernel 真实耗时/核数/输入输出 shape → **Tier1,6**（多 kernel 全抓）

### 来源 D：HIVM（bishengir 打印）
```
命令: 流程 D → hivm_try.txt
```
**字段**：`ops[]`: op_type, engine, dst, src, src2, memory_region, size_kb(从类型算), dtype, attrs(block_sizes), dependencies(RAW/WAR/WAW); `buffers`: 每 buffer 大小/region/生产者/消费者; `execution_summary`

**给策略**：op 图/依赖/数据流/尺寸 → **Tier1,2,3,4**（结构核心源）

---

## 二、6 策略 ×（瓶颈 → 字段 → 来源）

| Tier | 瓶颈信号 | 精确字段 | 来源 |
|---|---|---|---|
| **1 算法** | 串/并、归约形态、op 数、grid | execution_mode, num_ops, ops[].op_type, num_cores | C op_summary + D hivm + A trace |
| **2 融合** | WAR 依赖、GM↔UB 往返、逐元素链 | dependencies[], ops[].transfer_path, ops[].size_kb | D hivm |
| **3 分块** | 每通路带宽利用率、regime、k0 | transfer_paths[].real_bw_gb_s, bw_utilization, regime, ops[].size_kb | B Memory.csv + D hivm |
| **4 访存** | 小传输、L2 低、搬运大小 | ops[].size_kb/data_size_bytes, l2_hit_rate | A instr_exe + B L2Cache |
| **5 计算** | cube/vec 失衡、气泡、pipe 忙 | cube_fops, vector_fops, engine_utilization, ops[].cycles | B Arithmetic/Pipe + A |
| **6 架构** | pipe 利用率、L2、Block Dim | engine_utilization, l2_hit_rate, num_cores | B PipeUtilization + C op_summary |

---

## 三、diagnosis.json 最终输出（定稿 schema）

```json
{
  "summary": { "total_ns","num_cores","kernel_name","execution_mode",
               "l2_hit_rate","engine_utilization","total_gm_read_bytes","total_gm_write_bytes" },
  "ops": [ { "op_id","op_type","engine","transfer_path","path_desc",
             "dst","src","src2","dst_region","src_region",
             "size_kb","dtype","attrs",
             "duration_ns","cycles","pipe","call_count","data_size_bytes",
             "real_duration_ns","real_bw_gb_s","l2_hit",
             "effective_bw_gb_s","peak_bw_gb_s","bw_utilization","regime",
             "dependencies":[{"from_op","type","buffer"}],"sim_instr" } ],
  "transfer_paths": [ { "path","desc","num_ops","total_size_kb","total_duration_ns",
                        "real_bw_gb_s","effective_bw_gb_s","peak_bw_gb_s","bw_utilization","regime" } ],
  "dependencies": [ {"from_op","to_op","type","buffer"} ],
  "bottlenecks": {
    "tier1_algorithm": {"execution_mode","num_ops","top_ops","hint"},
    "tier2_fusion":    {"war_deps","gm_roundtrips","fusion_candidates","hint"},
    "tier3_tiling":    {"path_regimes","bottleneck_path","hint"},
    "tier4_memory":    {"l2_hit_rate","small_transfer_ops","hint"},
    "tier5_compute":   {"cube_fops","vector_fops","engine_utilization","hint"},
    "tier6_arch":      {"block_num","pipe_utilization","hint"}
  }
}
```

字段来源：summary ← B(task/cores/L2) + A(exec_mode)；ops 结构 ← D，时序 ← A，真机指标 ← B；通路 ← B Memory；依赖 ← D。

**合计 ~50 字段概念**（summary 8 + 每 op 22 + 每通路 10 + 每边 4 + 每 Tier ~5×6）。

---

## 四、关键提醒（8.5.1 实测边界）

- **msprof op 默认全量 8 CSV**（含 Memory/L2Cache/ArithmeticUtilization）= 真实带宽/L2/cube 的唯一真源 ✅
- **通用 msprof 的 --aic-metrics 8.5.1 不认** → op_summary 只有基础字段（耗时/核数/shape）✅
- **op_summary 的 aicore_time/aiv_time/total_cycles** 需 `--task-time=l1`（8.5.1 可能不支持，但 msprof op 的 OpBasicInfo/PipeUtilization 有等价数据）
- **simulator running_time 常为 0** → 用 cycles÷1.9GHz（已修）
