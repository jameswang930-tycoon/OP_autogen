# 内网 Agent 交接教学文档 — Triton 910B3 优化数据流水线

> 目的：让内网 agent 能独立完成「4 源采集 → 解析 → 整合 → diagnosis.json」全流程，并继续完善。
> 本文件是**唯一权威交接文档**。照着做，遇到不懂的先读 `knowledge/hivm.md`（完整 HIVM 方言文档）再上网搜。

---

## 1. 项目目标

把 Triton kernel（如 `input/matmul/triton_kernel.py`）在 **Ascend 910B3** 上优化。
不盲试，而是用 **4 个真实数据源**精确诊断瓶颈，驱动 6 层优化（算法→融合→分块→访存→计算→架构）。

```
Triton kernel → 4源采集 → 解析 → 整合 diagnosis.json → 每个 Tier 的瓶颈信号 + 优化提示 → LLM 按提示改代码
```

---

## 2. 4 个数据源（在哪、怎么产生）

### 源 A：真实 HIVM IR
- **文件**：`input/matmul/e2e_run/02_hivm/hivm_try.txt`
- **怎么产生**：`bash analyzers/run_server_flow.sh` 阶段 2（bishengir 手动打印）
- **内容**：`hivm.hir.load/store/mmadL1/vadd/set_flag/...` 语义 op + address_space + 依赖
- **参考**：`knowledge/hivm.md`（完整方言文档）

### 源 B：simulator 指令级（msprof op simulator）
- **文件**：`input/matmul/e2e_run/03_sim/sim_prof/OPPROF_*/simulator/`
  - `core*/..._instr_exe.csv`（每核，**按指令名聚合**：instr/addr/pipe/call_count/cycles/running_time(us)/detail）
  - `trace.json`（Chrome trace：每条指令 ts/dur = start/end + 通道）
- **产生**：阶段 3

### 源 C：msprof op 真机单算子（★主源，8 个 CSV）
- **文件**：`input/matmul/e2e_run/04_board/board_prof/OPPROF_*/`
  - `OpBasicInfo.csv`(耗时/核数) `PipeUtilization.csv`(各pipe占比)
  - `ArithmeticUtilization.csv`(cube/vec fops) `Memory.csv`(★每层级真实带宽)
  - `MemoryL0.csv`(L0A/B/C带宽) `MemoryUB.csv`(UB带宽) `L2Cache.csv`(★L2命中) `ResourceConflictRatio.csv`
- **产生**：阶段 4（**不指定 --aic-metrics = 默认全量**；指定会限制/报错）

### 源 D：通用 msprof 任务级
- **文件**：`input/matmul/e2e_run/05_task/task_prof/PROF_*/mindstudio_profiler_output/`
  - `op_summary_*.csv`（每 kernel：Task Duration/Block Dim/Input-Output Shapes）
  - 目录名拼写可能不同（`mind_studio_profile_output`）→ **find 宽找**
- **产生**：阶段 5（**--ai-core=on**，8.5.1 不认 --aic-metrics）

---

## 3. 完整流水线（6 阶段，一键脚本）

```bash
bash analyzers/run_server_flow.sh [M] [N] [K]   # 默认 64³
```
| 阶段 | 产出目录 | 检查 |
|---|---|---|
| 1 编译 | `01_compile/` ttir+ttadapter | 两个 .mlir 存在 |
| 2 HIVM | `02_hivm/hivm_try.txt` | `grep -c 'hivm.hir'` > 0 |
| 3 sim | `03_sim/sim_prof/` | instr_exe.csv + trace.json |
| 4 board | `04_board/board_prof/` | OpBasicInfo + **Memory.csv**（真实带宽） |
| 5 task | `05_task/task_prof/` | op_summary_*.csv |
| 6 整合 | `06_diagnosis/` 4个源JSON + diagnosis.json | diagnosis.json 生成 |

脚本末尾自动打印「字段预期校验」——告诉你每个 JSON 哪些该有值、哪些合法缺失。

---

## 4. 每个源怎么提取/解析（脚本已写好）

### 解析脚本（产出源 JSON）
```bash
python analyzers/pipeline_parse_hivm.py  <hivm_try.txt>  <out/hivm.json>   # 结构字段
python analyzers/pipeline_parse_sim.py   <sim_prof目录>   <out/sim.json>    # 指令时序+start/end
python analyzers/pipeline_parse_board.py <board_prof目录> <out/board.json>  # 真机带宽/L2/cube
python analyzers/pipeline_parse_task.py  <task_prof目录>  <out/task.json>   # 每kernel耗时
```
- 4 个 JSON 是**统一格式**（pipeline_schema.py 定义），缺的字段=None
- sim 的 running_time(us) 常为 0 → **用 cycles÷1.9GHz 兜底**（已处理）
- board 的 Memory.csv 是真实带宽（main_mem/ub/l1/l2 read-write bw）；L2Cache.csv 是命中率

### 翻译脚本（人话，排错用）
```bash
python analyzers/translate_hivm.py    # hivm_try.txt → 引擎/从哪到哪/操作/数据块/时长
python analyzers/translate_trace.py   # trace.json+instr_exe → 时间线/管道中文/搬运字节
```
- 顶部有 INPUT/OUTPUT 变量可改
- **不删任何内容**，翻译不了标 [待补]

### 整合脚本（核心）
```bash
python analyzers/integrate.py <hivm.json> <sim.json> <task.json> <board.json> <out/diagnosis.json>
```
- 按优化策略组织成 diagnosis.json

---

## 5. 最终输出：diagnosis.json（完整 schema + 每字段怎么填）

```json
{
  "summary": {
    "total_ns":          真机 Task Duration×1000 (task.json 或 board OpBasicInfo),
    "num_cores":         Block Dim (board/task),
    "kernel_name":       Op Name,
    "execution_mode":    trace.json 并行检测 (sim),
    "l2_hit_rate":       board L2Cache.csv Hit Rate,
    "engine_utilization": board PipeUtilization ratios (cube/vec/mte2/3/scalar),
    "total_gm_read_bytes": 总 GM 读字节 (算带宽用, 待补),
    "total_gm_write_bytes": 总 GM 写字节 (待补)
  },
  "ops": [ /* 每个 HIVM 语义 op, 从 hivm.json 遍历 */
    {
      "op_id": hivm 顺序,
      "op_type": hivm (gm_to_ub/mmadL1/set_flag...),
      "engine": OP_TO_ENGINE 映射 (mmadL1→CubeUnit),
      "transfer_path": op_type→通路 (GM→UB/UB→GM/L1→L0/CubeUnit/Sync),
      "path_desc": 通路描述 (从哪搬到哪),
      "dst": hivm, "src": hivm, "src2": hivm, "dst_region": hivm 区域,
      "size_kb": hivm (从 op 类型算, 动态维=0),
      "dtype": hivm, "attrs": hivm (block_sizes 等),
      "duration_ns": sim per-call (按 pipe/指令名对齐),
      "cycles": sim, "pipe": sim pipe, "call_count": sim,
      "data_size_bytes": sim detail 搬运字节,
      "real_duration_ns": task Task Duration (每 kernel),
      "real_bw_gb_s": board Memory.csv 该通路带宽,
      "l2_hit": board L2Cache,
      "effective_bw_gb_s": size÷duration 算,
      "peak_bw_gb_s": board Memory 实测峰值 (校准用),
      "bw_utilization": effective÷peak,
      "regime": floor/ramp/saturated (bw_util 分档),
      "dependencies": hivm RAW/WAR/WAW,
      "sim_instr": 对齐到的 sim 指令名
    }
  ],
  "transfer_paths": [ /* 每通路聚合 */
    { "path", "desc", "num_ops", "total_size_kb", "total_duration_ns",
      "real_bw_gb_s": board Memory, "effective_bw_gb_s", "peak_bw_gb_s",
      "bw_utilization", "regime" }
  ],
  "dependencies": [ {"from_op","to_op","type","buffer"} ],
  "bottlenecks": {
    "tier1_algorithm": {"execution_mode","num_ops","top_ops","hint"},
    "tier2_fusion":    {"war_deps","fusion_candidates","hint"},
    "tier3_tiling":    {"path_regimes","bottleneck_path","hint"},
    "tier4_memory":    {"l2_hit_rate","small_transfer_ops","hint"},
    "tier5_compute":   {"cube_fops","vector_fops","engine_utilization","hint"},
    "tier6_arch":      {"block_num","pipe_utilization","hint"}
  }
}
```

**字段填充规则**：
- 结构字段（op_type/dst/src/size/region/attrs/deps）← hivm.json（真实）
- 指令时序（duration/cycles/pipe/call_count/data_size）← sim.json（instr_exe 按指令名聚合）
- 真机指标（real_duration/real_bw/l2/engine_util）← board.json（Memory/L2Cache/PipeUtilization）+ task.json
- **合法缺失（不是 bug）**：bw_utilization/regime（需 peak 校准）；无 Memory 的通路 real_bw=None；sync op 的 size_kb=None（同步不搬数据）

---

## 6. 网上怎么搜（关键词 + 参考链接）

| 要查 | 关键词 | 参考 |
|---|---|---|
| HIVM 方言全文档 | `AscendNPU-IR HIVMDialect` | 已在你本地 `knowledge/hivm.md` |
| simulator 格式 | `msopprof simulator instr_exe trace.json` | hiascend 官方 |
| msprof op 字段 | `msprof op OpBasicInfo PipeUtilization Memory L2Cache` | hiascend CANN 文档 |
| op_summary | `op_summary 算子详细信息 CANN` | hiascend |
| 910B 内存层级 | `Ascend 910B Atlas A2 UB L1 L0A L0B L0C L2 容量` | 已核实: UB192/L1 512/L0A·B 64/L0C 128 |
| 性能分析案例 | `triton-ascend profiling layer_norm 流水利用率` | 官方示例（aiv_vec_ratio 等） |

**搜索要点**：华为文档在 `www.hiascend.com/document`，代码在 `gitcode.com/Ascend/*`。WebFetch 可能被挡 → 用 WebSearch。

---

## 7. 关键经验与避坑（务必记住）

1. **npuir.mlir 不生成** → 手动 bishengir D 打印（pass 名 `hivm-inject-sync` 或 `hivm-graph-sync-solver` 都试）
2. **sim 的 running_time(us) 常为 0** → 用 `cycles ÷ 1.9GHz`（已处理）
3. **msprof op 不指定 --aic-metrics** = 默认全量 8 CSV（指定会限制/报错）
4. **通用 msprof 的 --aic-metrics 在 8.5.1 不认** → 用 `--ai-core=on`
5. **op_summary 目录拼写不一** → find 宽找
6. **每次运行自动清理 e2e_run**
7. **容量修正**：UB=192/L1=512/L0A·B=64/L0C=128/L2≈4-8MB（config.py + simulator.py 已改）

## 8. 待办（最重要的下一步）

**按操作聚合（不是按顺序）**——这是当前最大缺口：
1. HIVM op_type → sim 指令名 的映射官方没有，**必须从真实 sim.json 的 op_name 学**
2. 跑 `bash analyzers/run_server_flow.sh` → 看 `06_diagnosis/sim.json` 里真实的指令名
3. 建映射表（gm_to_ub→MTE2载入类, mmadL1→CUBE-mmad类, set_flag→SET_FLAG...）
4. 改 `integrate.py` 按**指令名（操作身份）**对齐，不按 pipe+顺序
5. 用 `trace_events`（trace.json start/end）算调度/关键路径

---

## 9. 已完成的文件清单

| 文件 | 作用 |
|---|---|
| `run_server_flow.sh` | 6 阶段一键采集+解析+整合+字段校验 |
| `pipeline_parse_{hivm,sim,board,task}.py` | 4 源 → 统一格式 JSON |
| `integrate.py` | 4 源整合 → diagnosis.json（按优化策略） |
| `translate_hivm.py` / `translate_trace.py` | 完整翻译（人话，排错用） |
| `pipeline_schema.py` | 统一 op 字段定义 |
| `input/matmul/real_report.py` | 读 diagnosis.json 打印诊断报告 |
| `FIELD_SOURCE_MASTER.md` / `OPTIMIZATION_DATA_MAP.md` | 字段来源/策略映射参考 |
| `knowledge/hivm.md` | 完整 HIVM 方言文档（权威） |
