# 输出目录架构设计

> 与 `README.md` §6 和 `ARCHITECTURE_DESIGN.md` §6 保持一致。

---

## 完整目录全景

```
outputs/<kernel_name>/
│
├── round0/                                    # ★ 基准分析 (仅分析层, 无优化轮)
│   │
│   │  # ═══ 分析产物 (由分析层脚本生成) ═══
│   │
│   ├── kernel.py                              # ← 用户提供 (原始 Triton kernel)
│   ├── benchmark_result.json                  # ← execution/hardware_runner
│   │
│   ├── msprof/                                # msprof 分析
│   │   ├── OPPROF_{timestamp}_XXX/            #   ← msprof op simulator (原始)
│   │   │   └── simulator/trace.json
│   │   └── pipeline_report.json               #   ← analyzers/msprof_analyzer
│   │                                              (16字段填✅, 13字段待补充❌)
│   │
│   ├── hivmir/                                # HIVMIR 分析
│   │   ├── compiler_output/hivmir_output.mlir  #   ← execution/compiler (原始)
│   │   └── hivmir_report.json                 #   ← analyzers/hivmir_analyzer
│   │                                              (9字段填✅, 16字段待补充❌)
│   │
│   └── merged/                                # 合并产物
│       ├── merged_report.json                 #   ← analyzers/dsl_merger
│       │                                          ★ 29字段全部填充
│       ├── final_report_llm.txt               #   ← dsl_merger (7-section 文本)
│       └── final_report_human.txt             #   ← dsl_merger (ASCII Gantt 图)
│
│   # round0 没有以下优化文件: plan.md / diff.patch / optimization_record / verification
│
├── 01_algorithmic_structure/                  # Tier 1: Algorithmic Structure
│   ├── round1/                                #   第1轮优化
│   ├── round2/                                #   第2轮
│   └── ...                                    #   最多 N 轮
│
├── 02_operator_fusion/                        # Tier 2: Operator Fusion
│   ├── round1/ ... roundN/
│
├── 03_tiling_block_config/                    # Tier 3: Tiling & Block Config
│   ├── round1/ ... roundN/
│
├── 04_memory_access/                          # Tier 4: Memory Access
│   ├── round1/ ... roundN/
│
├── 05_compute_occupancy/                      # Tier 5: Compute & Occupancy
│   ├── round1/ ... roundN/
│
├── 06_910b3_architecture/                     # Tier 6: 910B3 Architecture
│   ├── round1/ ... roundN/
│
│   # ════════════════════════════════════════════════════════
│   # 以 03_tiling_block_config/round5/ 为例:
│   # ════════════════════════════════════════════════════════
│   #
│   03_tiling_block_config/round5/
│   │
│   │  # ── 分析产物 (每轮自动重跑) ──
│   │
│   ├── kernel.py                              # ← agents/coder (上一轮改的)
│   │                                            (round1 从 round0 拷贝)
│   │
│   ├── benchmark_result.json                  # ← execution/hardware_runner
│   │                                            (910B3 benchmark 结果)
│   │
│   ├── msprof/                                # msprof 分析
│   │   ├── OPPROF_xxx/simulator/trace.json     #   ← msprof op simulator
│   │   └── pipeline_report.json               #   ← analyzers/msprof_analyzer
│   │
│   ├── hivmir/                                # HIVMIR 分析
│   │   ├── compiler_output/hivmir_output.mlir  #   ← execution/compiler
│   │   └── hivmir_report.json                 #   ← analyzers/hivmir_analyzer
│   │
│   ├── merged/                                # 合并产物
│   │   ├── merged_report.json                 #   ← analyzers/dsl_merger (29字段)
│   │   ├── final_report_llm.txt               #   ← dsl_merger (LLM 文本)
│   │   └── final_report_human.txt             #   ← dsl_merger (Gantt 图)
│   │
│   │  # ── 优化产物 (roundN 独有, round0 没有) ──
│   │
│   ├── plan.md                                # ← agents/planner (LLM Plan)
│   │                                            (本轮优化策略+具体改动+预期效果)
│   │
│   ├── plan.json                              # ← agents/planner (LLM Plan)
│   │                                            (plan.md 的机器可读 JSON 版本)
│   │
│   ├── diff.patch                             # ← agents/coder (LLM Code)
│   │                                            (本轮代码变更 unified diff)
│   │
│   ├── optimization_record.json               # ← feedback/record_manager
│   │                                            {round, tier, strategy,
│   │                                             actual_speedup, decision,
│   │                                             bottleneck_before/after, ...}
│   │
│   ├── verification.json                      # ← agents/verifier
│   │                                            {stage1_passed, stage1_error,
│   │                                             stage2_actual_speedup, ...}
│   │
│   ├── AGENT_TASK_PLAN.md                     # ← agents/orchestrator
│   │                                            (自包含 Planner 任务文件,
│   │                                             无 API key 环境时生成)
│   │
│   └── AGENT_TASK_CODE.md                     # ← agents/orchestrator
│                                                (自包含 Coder 任务文件,
│                                                 无 API key 环境时生成)
│
├── optimization_trajectory.json               # ★ 全局中枢 (← feedback/record_manager)
│   {                                          #    每轮更新 state + history
│     "state": {"tier":3, "round":12, ...},
│     "baseline": {...},                       #    详见下方 schema
│     "history": [{...每轮一条...}, ...]
│   }
│
└── final_output/                              # ★ 最终产物 (达标/停止后生成)
    ├── optimized_kernel.py                    #   ← Orchestrator._finalize()
    ├── trajectory_chart.png                   #   ← feedback/trajectory_chart
    ├── optimization_summary.md                 #   ← record_manager (案例模板)
    ├── final_merged_report.json               #   ← 最后一轮 merged 拷贝
    ├── final_report_llm.txt                   #   ← dsl_merger (最终 LLM 文本)
    └── final_report_human.txt                 #   ← dsl_merger (最终 Gantt 图)
```

---

## 每轮文件来源速查

| 文件 | 谁写入 | 何时写入 | round0有? |
|---|---|---|---|
| `kernel.py` | Coder (roundN) / 用户 (round0) | Coder 完成后 | ✅ |
| `benchmark_result.json` | hardware_runner | Stage 2 验证后 | ✅ |
| `msprof/pipeline_report.json` | msprof_analyzer | Analyzers 第1步 | ✅ |
| `hivmir/hivmir_report.json` | hivmir_analyzer | Analyzers 第2步 | ✅ |
| `merged/merged_report.json` | dsl_merger | Analyzers 第3步 | ✅ |
| `merged/final_report_llm.txt` | dsl_merger | Analyzers 第3步 | ✅ |
| `merged/final_report_human.txt` | dsl_merger | Analyzers 第3步 | ✅ |
| `plan.md` | Planner (LLM) | Planner 完成后 | — |
| `plan.json` | Planner (LLM) | Planner 完成后 | — |
| `diff.patch` | Coder (LLM) | Coder 完成后 | — |
| `optimization_record.json` | RecordManager | 决策后 | — |
| `verification.json` | Verifier | 验证完成后 | — |
| `AGENT_TASK_PLAN.md` | Orchestrator | 无 API key 时 | — |
| `AGENT_TASK_CODE.md` | Orchestrator | 无 API key 时 | — |

---

## optimization_trajectory.json Schema

```json
{
  "kernel": {"name": "...", "dtype": "fp16", "initial_kernel_path": "..."},
  "state": {
    "tier": 3, "round": 12, "best_speedup": 1.52,
    "consecutive_reverts": 0, "consecutive_no_improvement": 0,
    "started_at": "2026-07-24T10:00:00", "last_updated": "2026-07-24T12:30:00"
  },
  "baseline": {
    "total_ns": 3655.6, "num_ops": 3, "execution_mode": "sequential",
    "bottleneck_op_id": 2, "bottleneck_type": "memory_bandwidth",
    "engine_utilization": {"GM->UB": 0.44, "UB->GM": 0.47}
  },
  "tier_progress": {
    "tier_1": {"rounds_spent": 3, "best_in_tier": 1.00},
    "tier_2": {"rounds_spent": 4, "best_in_tier": 1.09}
  },
  "history": [
    {
      "round": 1, "tier": 1, "tier_name": "Algorithmic Structure",
      "strategy": "evaluate_algorithm_choice",
      "actual_speedup": 1.0, "cumulative_speedup": 1.0,
      "decision": "KEEP", "decision_reason": "...",
      "bottleneck_before": {}, "code_lines_changed": 0,
      "emulator_passed": true, "hardware_tested": false,
      "timestamp": "2026-07-24T10:05:00"
    }
  ]
}
```

---

## 6 层优化顺序

| Tier | 目录 | 名称 | 晋升条件 |
|---|---|---|---|
| 1 | `01_algorithmic_structure` | Algorithmic Structure | 算法已最优 |
| 2 | `02_operator_fusion` | Operator Fusion | 无可融合 op |
| 3 | `03_tiling_block_config` | Tiling & Block Config | 连续 3 轮无改进 |
| 4 | `04_memory_access` | Memory Access | 连续 3 轮无改进 |
| 5 | `05_compute_occupancy` | Compute & Occupancy | 连续 3 轮无改进 |
| 6 | `06_910b3_architecture` | 910B3 Architecture | 连续 3 轮无改进 → 停止 |
