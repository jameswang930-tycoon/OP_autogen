# 输出目录架构设计

> 完整目录架构说明。确认后实现到 `outputs/` 目录中。

---

## 目录全景

```
outputs/
│
└── <triton_kernel_name>/                       # ★ 每个 Triton kernel 一个大文件夹
    │
    ├── round0/                                  # ★ 基准分析 (优化前原始数据)
    │   │
    │   ├── kernel.py                            # 1. 原始 Triton kernel 代码
    │   │
    │   ├── bench.py                             # 2. msprof 基准测试脚本
    │   │   #    - 运行 msprof op ./binary
    │   │   #    - 测量绝对延迟 (ms)、端到端加速比、吞吐量 (GB/s or TFLOPS)
    │   │   #    - 计算各引擎时间占比
    │   │
    │   ├── benchmark_result.json                # 3. 基准测试结果
    │   │   #    { "absolute_latency_ms": ..., "speedup_vs_baseline": 1.0,
    │   │   #      "throughput_gb_s": ..., "time_ratios": {...},
    │   │   #      "total_ops": ..., "execution_mode": "..." }
    │   │
    │   ├── msprof/                              # 4. msprof op simulator 分析产物
    │   │   ├── OPPROF_{timestamp}_XXX/          #    msprof 中间产物目录 (原始输出)
    │   │   │   └── simulator/
    │   │   │       ├── trace.json
    │   │   │       └── core0.veccore0/
    │   │   │           ├── trace.json
    │   │   │           ├── core0.veccore0_code_exe.csv
    │   │   │           └── core0.veccore0_instr_exe.csv
    │   │   └── pipeline_report.json             #    ★ msprof 最终解析产物 (JSON)
    │   │       #    msprof字段已填(16✅), HIVMIR字段标记"待补充"(13❌)
    │   │       #    格式: 29字段完整流水线报告 (PARTIAL)
    │   │
    │   ├── hivmir/                              # 5. HIVMIR 编译器中间产物分析
    │   │   ├── compiler_output/                 #    HIVMIR 中间产物目录 (编译器原始输出)
    │   │   │   └── hivmir_output.mlir           #    HIVMIR IR 文本
    │   │   └── hivmir_report.json               #    ★ HIVMIR 最终解析产物 (JSON)
    │   │       #    HIVMIR字段已填(9✅), msprof字段标记"待补充"(16❌)
    │   │       #    格式: 与 pipeline_report.json 完全对齐的 29字段结构
    │   │
    │   └── merged/                              # 6. 合并产物 (dsl_merger.py)
    │       ├── merged_report.json               #    ★ 合并后完整报告 (29字段全填)
    │       ├── final_report_llm.txt             #    LLM 消费: 对齐 simulator --llm 格式
    │       │   #    7 section: EXEC SUMMARY / TIME BREAKDOWN / PER-OP STATS /
    │       │   #    ENGINE UTIL / BW UTIL / PARALLELISM / CRITICAL PATH
    │       └── final_report_human.txt           #    人读: ASCII Gantt图 + 柱状图 + 表格
    │           #    内容: Pipeline Execution Graph / 操作表格 /
    │           #    时间占比柱状图 / 引擎利用率 / 带宽利用率 / 关键路径
    │
    ├── 01_block_size_launch/                    # Tier 1: Block Size & Launch Config 优化
    │   ├── round1/
    │   │   ├── kernel.py                        # 优化后 kernel
    │   │   ├── bench.py
    │   │   ├── benchmark_result.json
    │   │   ├── msprof/
    │   │   │   ├── OPPROF_{timestamp}_XXX/
    │   │   │   └── pipeline_report.json
    │   │   ├── hivmir/
    │   │   │   ├── compiler_output/
    │   │   │   └── hivmir_report.json
    │   │   ├── merged/
    │   │   │   ├── merged_report.json
    │   │   │   ├── final_report_llm.txt
    │   │   │   └── final_report_human.txt
    │   │   └── optimization_record.json          # ★ 本轮独有: 优化记录
    │   │       # {
    │   │       #   "round": 1, "strategy_tier": 1,
    │   │       #   "strategy": "increase_tile_size",
    │   │       #   "optimization_target": "增大 BLOCK_SIZE 256→8192",
    │   │       #   "bottleneck_before": {"op_id": 2, "type": "memory_bandwidth",
    │   │       #       "time_ratio": 0.47, "bw_util": 0.21, "regime": "ramp"},
    │   │       #   "bottleneck_after": {"op_id": 1, "type": "compute_vec",
    │   │       #       "time_ratio": 0.35, "bw_util": 0.88, "regime": "saturated"},
    │   │       #   "target_speedup": 1.10,
    │   │       #   "actual_speedup": 1.12,
    │   │       #   "cumulative_speedup": 1.12,
    │   │       #   "decision": "KEEP",
    │   │       #   "decision_reason": "GM→UB 带宽利用率 21→78%, 总时间 -12%",
    │   │       #   "code_diff": "BLOCK_SIZE 256 → 8192"
    │   │       # }
    │   ├── round2/
    │   │   └── ...
    │   └── roundN/
    │
    ├── 02_memory_access/                        # Tier 2: Memory Access & Coalescing
    │   ├── round1/ ... roundN/
    │
    ├── 03_operator_fusion/                      # Tier 3: Operator Fusion
    │   ├── round1/ ... roundN/
    │
    ├── 04_compute_optimization/                 # Tier 4: Compute Optimization
    │   ├── round1/ ... roundN/
    │
    ├── 05_architecture_specific/                # Tier 5: 910B3 Architecture
    │   ├── round1/ ... roundN/
    │
    ├── 06_algorithmic_restructure/              # Tier 6: Algorithmic Restructure
    │   ├── round1/ ... roundN/
    │
    ├── optimization_trajectory.json              # ★ 跨轮汇总: 每轮瓶颈+策略+加速比
    │   # [ {"round":0, "phase":"baseline", "speedup":1.0, "bottleneck":"ub_to_gm"},
    │   #   {"round":1, "phase":"Tier1", "strategy":"increase_tile", "speedup":1.12,
    │   #    "bottleneck":"vadd"}, ... ]
    │
    ├── trajectory_chart.png                      # ★ 优化轨迹图: 双面板(加速比+延迟)
    │
    ├── optimization_summary.md                   # ★ 优化总结报告
    │   # - 关键瓶颈变化历史
    │   # - 成功的优化策略清单
    │   # - 失败的优化策略清单 (避免重复)
    │   # - 最终加速比 + 各阶段贡献
    │   # - 优化建议 (供后续类似算子参考)
    │
    └── optimized_kernel.py                       # ★ 最终优化版 kernel (通过全部验证)
```

---

## 各文件详细说明

### round0 — 基准分析 (必做, 优化前)

| 文件 | 内容 | 格式 | 谁产出 |
|---|---|---|---|
| `kernel.py` | 原始 Triton kernel 代码 | Python | 用户提供 |
| `bench.py` | 基准测试脚本 (msprof 采集) | Python | execution/hardware_runner.py |
| `benchmark_result.json` | 绝对延迟/加速比/吞吐量/时间占比 | JSON | hardware_runner.py |
| `msprof/pipeline_report.json` | msprof 解析最终产物 (29字段, 16✅+13❌) | JSON | analyzers/msprof_analyzer.py |
| `msprof/OPPROF_*/` | msprof 原始中间产物 | raw | msprof op simulator |
| `hivmir/hivmir_report.json` | HIVMIR 解析最终产物 (29字段, 9✅+16❌) | JSON | analyzers/hivmir_analyzer.py |
| `hivmir/compiler_output/` | HIVMIR 编译器原始输出 | .mlir | Ascend 编译器 |
| `merged/merged_report.json` | 合并后完整报告 (29字段全填) | JSON | analyzers/dsl_merger.py |
| `merged/final_report_llm.txt` | LLM 消费: 7-section 结构化文本 | TXT | dsl_merger.py |
| `merged/final_report_human.txt` | 人读: ASCII Gantt + 柱状图 + 表格 | TXT | dsl_merger.py |

### roundN — 每轮优化 (在对应 Tier 文件夹下)

除 round0 的全部内容外, 额外多:

| 文件 | 内容 | 格式 | 谁产出 |
|---|---|---|---|
| `optimization_record.json` | 本轮优化详情: 策略/瓶颈前后/加速比/决策 | JSON | agents/orchestrator.py |

### kernel 顶层文件

| 文件 | 内容 | 谁产出 |
|---|---|---|
| `optimization_trajectory.json` | 跨轮汇总 (每轮一条记录) | feedback/optimization_journal.py |
| `trajectory_chart.png` | 优化轨迹图 | feedback/trajectory_chart.py |
| `optimization_summary.md` | 优化总结报告 | feedback/case_template.py |
| `optimized_kernel.py` | 最终优化版代码 | agents/orchestrator.py |

---

## optimization_record.json 详细 Schema

```json
{
  "round": 3,
  "phase": "Tier2_memory_access",
  "timestamp": "2026-07-23T16:00:00",

  "plan": {
    "strategy": "merge_small_transfers",
    "strategy_tier": 2,
    "description": "合并 4×1KB gm_to_ub 为 1×4KB 传输",
    "plan_file": "round3_plan.md"
  },

  "bottleneck_before": {
    "op_id": 0,
    "op_type": "gm_to_ub",
    "engine": "GM→UB",
    "time_ratio": 0.35,
    "bw_utilization": 0.21,
    "regime": "ramp",
    "size_kb": 1.0,
    "target_k0": 6.65
  },

  "bottleneck_after": {
    "op_id": 2,
    "op_type": "ub_to_gm",
    "engine": "UB→GM",
    "time_ratio": 0.42,
    "bw_utilization": 0.91,
    "regime": "saturated"
  },

  "code_change": {
    "diff_file": "round3_diff.patch",
    "lines_changed": 2,
    "summary": "合并 4个1KB tile → 1个4KB tile"
  },

  "verification": {
    "stage1_emulator": {"passed": true, "max_abs_error": 4.77e-07},
    "stage2_simulator": {"estimated_speedup": 1.08},
    "stage3_hardware": {"passed": true, "actual_speedup": 1.07}
  },

  "performance": {
    "target_speedup": 1.05,
    "actual_speedup": 1.07,
    "cumulative_speedup": 1.21,
    "latency_ms_before": 0.0183,
    "latency_ms_after": 0.0171,
    "throughput_gb_s_before": 850.0,
    "throughput_gb_s_after": 910.0
  },

  "decision": "KEEP",
  "decision_reason": "GM→UB 合并后带宽利用率 21→91%, 瓶颈转移到 ub_to_gm"
}
```

## 关于融合分析

Tier 3 (算子融合) 的 round 目录包含 **所有被融合算子的串联流水线**。

例如融合 `add + relu`:
- `merged/final_report_llm.txt` 包含 6 个 op (gm_to_ub→vadd→vrelu→...), 全部显示
- 需要单算子视图时, data_extractor 按 op_id 过滤即可

---

## 实现状态

| 组件 | 状态 |
|---|---|
| msprof_analyzer (产出 pipeline_report.json) | ✅ 已实现 |
| hivmir_analyzer (产出 hivmir_report.json) | ✅ 已实现 |
| dsl_merger (合并 → merged_report.json) | ❌ 待实现 |
| hardware_runner (bench.py + benchmark_result.json) | ❌ 待实现 |
| optimization_record.json (每轮记录) | ❌ 待实现 |
| optimization_trajectory.json (跨轮汇总) | ❌ 待实现 |
| trajectory_chart.png (轨迹图) | ❌ 待实现 |
| optimization_summary.md (总结报告) | ❌ 待实现 |
