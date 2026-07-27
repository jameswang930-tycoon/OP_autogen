# 输出目录架构设计

> 完整目录架构说明。与 `README.md` §6 保持一致。

---

## 目录全景

```
outputs/
│
└── <triton_kernel_name>/                       # ★ 每个 kernel 独立目录
    │
    ├── round0/                                  # ★ 基准分析 (优化前, 无优化文件)
    │   │
    │   ├── kernel.py                            # 原始 Triton kernel
    │   ├── benchmark_result.json                # 基准: 延迟/加速比/吞吐/时间占比
    │   │
    │   ├── msprof/                              # msprof 分析产物
    │   │   ├── OPPROF_{timestamp}_XXX/          #   msprof 原始中间产物
    │   │   │   └── simulator/
    │   │   │       ├── trace.json
    │   │   │       └── core0.veccore0/...
    │   │   └── pipeline_report.json             #   msprof 解析产物 (16✅+13❌)
    │   │
    │   ├── hivmir/                              # HIVMIR 分析产物
    │   │   ├── compiler_output/                 #   HIVMIR 编译器原文
    │   │   │   └── hivmir_output.mlir
    │   │   └── hivmir_report.json               #   HIVMIR 解析产物 (9✅+16❌)
    │   │
    │   └── merged/                              # 合并产物
    │       ├── merged_report.json               #   ★ 29字段全填 (dsl_merger)
    │       ├── final_report_llm.txt             #   LLM 读: 7-section 文本
    │       └── final_report_human.txt           #   人读: ASCII Gantt 图
    │
    ├── 01_algorithmic_structure/                # Tier 1: Algorithmic Structure (最先)
    ├── 02_operator_fusion/                      # Tier 2: Operator Fusion
    ├── 03_tiling_block_config/                  # Tier 3: Tiling & Block Config
    ├── 04_memory_access/                        # Tier 4: Memory Access & Coalescing
    ├── 05_compute_occupancy/                    # Tier 5: Compute & Occupancy
    ├── 06_910b3_architecture/                   # Tier 6: 910B3 Architecture (最后)
    │   │
    │   └── roundN/                              #   每轮优化目录
    │       │
    │       │  ┌─ 分析产物 (每轮都有) ──────────────┐
    │       │  │ msprof/ + hivmir/ + merged/       │
    │       │  │ benchmark_result.json             │
    │       │  │ kernel.py (本轮当前代码)           │
    │       │  └───────────────────────────────────┘
    │       │
    │       │  ┌─ 优化产物 (roundN 独有) ───────────┐
    │       │  │ plan.md           Planner 产出     │
    │       │  │ plan.json         Planner 产出(JSON)│
    │       │  │ diff.patch        Coder 产出       │
    │       │  │ optimization_record.json 决策记录   │
    │       │  │ verification.json Verifier 产出    │
    │       │  │ AGENT_TASK_PLAN.md  任务文件        │
    │       │  │ AGENT_TASK_CODE.md  任务文件        │
    │       │  └───────────────────────────────────┘
    │
    ├── optimization_trajectory.json              # ★ 全局中枢状态
    │
    └── final_output/                             # ★ 最终产物
        ├── optimized_kernel.py                   #   最终优化版 kernel
        ├── trajectory_chart.png                  #   6阶段加速比曲线图
        ├── optimization_summary.md                #   总结报告
        ├── final_merged_report.json              #   最终合并报告
        ├── final_report_llm.txt                  #   最终 LLM 文本
        └── final_report_human.txt                #   最终 Gantt 图
```

---

## round0 vs roundN 对比

| 文件 | round0 (基准) | roundN (优化) | 写入者 |
|---|---|---|---|
| `kernel.py` | 原始 kernel | Coder 修改后 | Orchestrator / Coder |
| `benchmark_result.json` | ✅ | ✅ | hardware_runner |
| `msprof/` | ✅ | ✅ | msprof_analyzer |
| `hivmir/` | ✅ | ✅ | hivmir_analyzer / compiler |
| `merged/` | ✅ | ✅ | dsl_merger |
| `plan.md` | — | ✅ | Planner |
| `plan.json` | — | ✅ | Planner |
| `diff.patch` | — | ✅ | Coder |
| `optimization_record.json` | — | ✅ | RecordManager |
| `verification.json` | — | ✅ | Verifier |
| `AGENT_TASK_PLAN.md` | — | ✅ | Orchestrator |
| `AGENT_TASK_CODE.md` | — | ✅ | Orchestrator |

---

## optimization_trajectory.json Schema

```json
{
  "kernel": {
    "name": "rms_norm_residual",
    "dtype": "fp16",
    "initial_kernel_path": "outputs/.../round0/kernel.py"
  },
  "state": {
    "tier": 3,
    "round": 12,
    "best_speedup": 1.52,
    "consecutive_reverts": 0,
    "consecutive_no_improvement": 0,
    "started_at": "2026-07-24T10:00:00",
    "last_updated": "2026-07-24T12:30:00"
  },
  "baseline": {
    "total_ns": 3655.6,
    "num_ops": 3,
    "execution_mode": "sequential",
    "bottleneck_op_id": 2,
    "bottleneck_op_type": "ub_to_gm",
    "bottleneck_engine": "UB->GM",
    "bottleneck_type": "memory_bandwidth",
    "bottleneck_time_ratio": 0.4677,
    "engine_utilization": {"GM->UB": 0.44, "UB->GM": 0.47, "VecUnit": 0.09}
  },
  "tier_progress": {
    "tier_1": {"rounds_spent": 3, "best_in_tier": 1.00},
    "tier_2": {"rounds_spent": 4, "best_in_tier": 1.09},
    "tier_3": {"rounds_spent": 7, "best_in_tier": 1.19}
  },
  "history": [
    {
      "round": 1,
      "tier": 1,
      "tier_name": "Algorithmic Structure",
      "strategy": "evaluate_algorithm_choice",
      "target_speedup": 1.0,
      "actual_speedup": 1.0,
      "cumulative_speedup": 1.0,
      "decision": "KEEP",
      "decision_reason": "algorithm already optimal",
      "bottleneck_before": {"op_id": 2, "type": "memory_bandwidth"},
      "bottleneck_after": {},
      "code_lines_changed": 0,
      "emulator_passed": true,
      "hardware_tested": false,
      "coder_retries": 0,
      "timestamp": "2026-07-24T10:05:00"
    }
  ]
}
```

## optimization_record.json Schema (每轮独有)

```json
{
  "round": 3,
  "tier": 3,
  "tier_name": "Tiling & Block Config",
  "strategy": "increase_tile_size",
  "target_speedup": 1.15,
  "actual_speedup": 1.12,
  "cumulative_speedup": 1.21,
  "decision": "KEEP",
  "decision_reason": "GM->UB bw_util 21->78%, bottleneck shifted to VecUnit",
  "bottleneck_before": {"op_id": 0, "type": "memory_latency", "time_ratio": 0.35},
  "bottleneck_after":  {"op_id": 1, "type": "compute_vec", "time_ratio": 0.30},
  "code_lines_changed": 2,
  "emulator_passed": true,
  "hardware_tested": false,
  "coder_retries": 0,
  "verification": {
    "stage1_passed": true,
    "stage1_error": "",
    "stage2_estimated_speedup": null,
    "stage3_actual_speedup": null
  }
}
```

## 六层优化策略顺序

**核心原则: 从结构影响最大到最小。改了上层，下层要重做。**

| Tier | 文件夹 | 名称 | 做什么 | 晋升条件 |
|---|---|---|---|---|
| 1 | `01_algorithmic_structure` | Algorithmic Structure | Online Softmax / Split-K / Persistent Kernel | 算法已最优 |
| 2 | `02_operator_fusion` | Operator Fusion | 逐元素融合 / WAR打破 / 激活融合 | 无可融合op |
| 3 | `03_tiling_block_config` | Tiling & Block Config | BLOCK_SIZE / num_warps / num_stages | 连续3轮无改进 |
| 4 | `04_memory_access` | Memory Access | 小传输合并 / double buffer / coalescing | 连续3轮无改进 |
| 5 | `05_compute_occupancy` | Compute & Occupancy | 计算-传输重叠 / 向量化 / 精度取舍 | 连续3轮无改进 |
| 6 | `06_910b3_architecture` | 910B3 Architecture | Grid / Pipeline / L2驻留 / 混合精度 | 连续3轮无改进→停止 |

**降级规则**: 融合新算子→回退 Tier 3; 改了算法→回退 Tier 2
