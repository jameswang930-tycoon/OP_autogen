# Triton Agent Optimizer

> 基于 DSL 流水线分析 (HIVMIR + msprof op simulator) 的 Triton Kernel 自动优化框架。
> 目标硬件: **华为 Ascend 910B3 NPU**。

---

## 1. 核心理念

**不是盲试。** AutoKernel 只能知道 "kernel X 占 60% 时间" → 盲目试 300~400 轮。我们的方案通过 DSL 流水线精确知道：哪个 op 是瓶颈、为什么慢、带宽利用率多少、该改什么参数。**精准度比盲试高一个数量级, 预估 20~50 轮收敛。**

---

## 2. 整体流程

```
Round 0 (Baseline):
  Analyzers (msprof→hivmir→merger→diagnoser→extractor) → trajectory.json

Round 1..N (Optimize):
  ┌─────────────────────────────────────────────────────────┐
  │ ① Analyzers (分析层, 5脚本)                             │
  │    msprof → hivmir → merger → diagnoser → extractor    │
  │    每轮重跑, 生成最新 merged_report + diagnosis         │
  ├─────────────────────────────────────────────────────────┤
  │ ② Planner (LLM Agent)                                   │
  │    输入: diagnosis + extracted + playbook + history     │
  │    输出: round_N_plan.md (优化策略 + 具体改动 + 预期效果)│
  ├─────────────────────────────────────────────────────────┤
  │ ③ Coder (LLM Agent)                                     │
  │    输入: plan.md + kernel.py                            │
  │    输出: optimized kernel.py + diff.patch               │
  │    约束: 只能改 kernel.py                               │
  ├─────────────────────────────────────────────────────────┤
  │ ④ Verifier (脚本, 两阶段)                               │
  │    Stage 1: CPU Emulator (正确性, 秒级)                 │
  │    Stage 2: 910B3 Hardware (性能, 分钟级, 本地跳过)     │
  │    FAIL → 错误回传 Coder 重试 (最多3次)                  │
  ├─────────────────────────────────────────────────────────┤
  │ ⑤ RecordManager (反馈层 — 决策引擎)                     │
  │    KEEP/REVERT → 写 optimization_record.json            │
  │    → 更新 optimization_trajectory.json                  │
  │    → 管理 Tier 晋升/降级                                 │
  │    → 检查 7 条停止条件                                   │
  │    → 达标时生成案例模板 + 轨迹图                         │
  └─────────────────────────────────────────────────────────┘
     │
     └──→ Round N+1 (自动切换 Tier 目录) → ... → Finalize
```

---

## 3. 6 层优化策略

**原则: 从结构影响最大到最小。改了上层, 下层要重做。**

| Tier | 名称 | 做什么 | Playbook |
|---|---|---|---|
| **1** | Algorithmic Structure | Online Softmax / Split-K / Persistent Kernel | `playbook_tier1_algorithm.md` |
| **2** | Operator Fusion | 逐元素融合 / WAR打破 / 激活融合 | `playbook_tier2_fusion.md` |
| **3** | Tiling & Block Config | BLOCK_SIZE / num_warps / num_stages | `playbook_tier3_tiling.md` |
| **4** | Memory Access | 小传输合并 / double buffer / coalescing | `playbook_tier4_memory.md` |
| **5** | Compute & Occupancy | 计算-传输重叠 / 向量化 / 精度取舍 | `playbook_tier5_compute.md` |
| **6** | 910B3 Architecture | Grid / Pipeline / L2驻留 / 混合精度 | `playbook_tier6_architecture.md` |

**晋升**: 连续 3 轮无改进 → Tier+1
**降级**: 融合新算子 → 回退 Tier 3; 改了算法 → 回退 Tier 2

---

## 4. 文件架构

```
triton_agent_optimizer/
│
├── README.md                           # 本文档
├── config.py                           # ★ 全局配置 (路径/硬件参数/阈值)
├── main.py                             # (待实现) 主入口
│
├── prepare/                            # 环境准备
│   ├── setup_env.sh                    #   Linux/910B3 环境设置
│   ├── setup_env.bat                   #   Windows 环境设置
│   └── env_check.py                    #   35项环境验证
│
├── analyzers/                          # ★ 分析层 (5文件, 全部完成)
│   ├── msprof_analyzer.py              #   ① msprof trace.json → timing/engine/channel
│   ├── hivmir_analyzer.py              #   ② HIVMIR .mlir → buffer/size/deps
│   ├── dsl_merger.py                   #   ③ 合并 → 29字段完整报告 + LLM文本 + Gantt图
│   ├── bottleneck_diagnoser.py         #   ④ Tier-aware 瓶颈诊断 (~2KB)
│   └── data_extractor.py               #   ⑤ 按需提取关键列 (Tier×列过滤, ~2KB)
│
├── agents/                             # ★ 智能体层 (4文件, 全部完成)
│   ├── orchestrator.py                 #   调度器 (Python 状态机, 薄循环)
│   ├── planner.py                      #   规划器 (LLM Prompt 编排器)
│   ├── coder.py                        #   编码器 (LLM Prompt 编排器, 只改kernel.py)
│   └── verifier.py                     #   验证器 (脚本: CPU仿真 + 910B3实测)
│
├── execution/                          # ★ 执行层 (2文件 + 2 stub)
│   ├── emulator_runner.py              #   Stage 1: CPU Emulator 正确性验证
│   ├── compiler.py                     #   Ascend 编译器接口 (需910B3)
│   ├── hardware_runner.py              #   Stage 2: 910B3 benchmark (需910B3)
│   └── simulator_runner.py             #   (备用) cost simulator 包装器
│
├── feedback/                           # ★ 反馈与记录层 (2文件, 全部完成)
│   ├── record_manager.py               #   决策引擎: KEEP/REVERT + Tier + 停止 + 轨迹
│   └── trajectory_chart.py             #   优化轨迹图: 6阶段加速比曲线
│
├── memory/                             # ★ 记忆层 (3文件, 全部完成)
│   ├── sliding_window.py               #   热/温/冷三层滑动窗口
│   ├── context_manager.py              #   Token估算 + 裁剪 + prompt构建
│   ├── experience_retriever.py         #   分Tier检索 + 3级匹配 + 经验记录
│   └── experiences/                    #   6个Tier经验库JSON
│
├── playbooks → docx/                   # 优化指导手册 (7文件, 全部完成)
│   ├── OPTIMIZATION_METHODOLOGY.md      #   总纲: 方法论 + 参考文献
│   ├── playbook_tier1_algorithm.md      #   Tier 1: 算子→算法对照表
│   ├── playbook_tier2_fusion.md         #   Tier 2: 融合决策树 + WAR打破
│   ├── playbook_tier3_tiling.md         #   Tier 3: K0参考 + BLOCK_SIZE启发式
│   ├── playbook_tier4_memory.md         #   Tier 4: 合并传输 + double buffer
│   ├── playbook_tier5_compute.md        #   Tier 5: 计算-传输重叠 + 向量化
│   └── playbook_tier6_architecture.md   #   Tier 6: Grid/Pipeline/L2
│
├── outputs/                            # 优化产物 (自动生成, 不入git)
│   └── <kernel_name>/
│       ├── round0/                     #   基准分析 (msprof/hivmir/merged)
│       ├── 01_algorithmic_structure/    #   Tier 1 优化轮次
│       ├── 02_operator_fusion/          #   Tier 2
│       ├── 03_tiling_block_config/      #   Tier 3
│       ├── 04_memory_access/            #   Tier 4
│       ├── 05_compute_occupancy/        #   Tier 5
│       ├── 06_910b3_architecture/       #   Tier 6
│       ├── optimization_trajectory.json  #   ★ 全局状态文件
│       └── final_output/                #   最终产物
│           ├── optimized_kernel.py
│           ├── trajectory_chart.png
│           └── optimization_summary.md
│
├── example_output/                     # 参考示例 (simulator输出)
├── scripts/                            # 工具脚本
│   └── init_output_structure.py        #   初始化 outputs/ 目录
└── docx/ → playbooks                   # 优化指导手册
```

---

## 5. 数据流与对齐

```
                    msprof trace.json          HIVMIR .mlir
                          │                        │
                    msprof_analyzer          hivmir_analyzer
                          │                        │
                    pipeline_report.json     hivmir_report.json
                    (16 fields ✅)           (9 fields ✅)
                    (13 fields ❌)           (16 fields ❌)
                          │                        │
                          └──────────┬─────────────┘
                                     ▼
                               dsl_merger
                          (op_id 对齐, 互相填补)
                                     │
                                     ▼
                            merged_report.json
                            (29 fields 全部填充)
                                     │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                   bottleneck_diagnoser    data_extractor
                   (瓶颈分类+评估)         (Tier×列过滤, ~2KB)
                          │                     │
                          └──────────┬──────────┘
                                     ▼
                              Planner (LLM)
                        (诊断 + Playbook + 历史 → 计划)
```

---

## 6. 输出目录

每轮输出到 `outputs/<kernel>/<tier_dir>/roundN/`:

```
roundN/
├── kernel.py                  # Coder 修改后的代码
├── plan.md                    # Planner 优化计划
├── diff.patch                 # 代码变更
├── optimization_record.json   # 本轮决策+效果 (RecordManager)
├── verification.json          # 验证结果 (Verifier)
├── msprof/pipeline_report.json   # 分析产物
├── hivmir/hivmir_report.json     # 分析产物
└── merged/merged_report.json     # 合并产物
```

全局状态文件: `optimization_trajectory.json`

```json
{
  "state": {"tier": 3, "round": 12, "best_speedup": 1.52},
  "baseline": {"total_ns": 3655.6, "bottleneck_type": "memory_bandwidth"},
  "history": [
    {"round": 1, "tier": 1, "strategy": "...", "speedup": 1.0, "decision": "KEEP"},
    ...
  ]
}
```

---

## 7. 快速开始

```bash
# 1. 环境准备
source prepare/setup_env.sh          # Linux/910B3
prepare\setup_env.bat                # Windows

# 2. 初始化输出目录
python scripts/init_output_structure.py

# 3. 运行优化
python agents/orchestrator.py

# 4. 查看结果
ls outputs/<kernel>/final_output/
#   optimized_kernel.py
#   trajectory_chart.png
#   optimization_summary.md
```

---

## 8. 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 瓶颈诊断 | 脚本 (规则引擎) | 阈值分类是确定性的, OPAL: 压缩到 ~2KB |
| 策略规划 | LLM Agent | 需要 Playbook + 历史 + 代码上下文推理 |
| 代码修改 | LLM Agent | 需要理解语义, 不能只做文本替换 |
| 验证 | 脚本 (CPU仿真 + Hardware) | 确定性, 不需要推理 |
| 调度器 | Python 状态机 | 管循环+状态, 不做决策 |
| 决策/记录 | RecordManager | 所有决策集中, 调度器只管循环 |
| 优化顺序 | Algorithm→Fusion→Tiling→Memory→Compute→Arch | 结构影响从大到小 |
| 输出格式 | JSON (LLM) + TXT (7-section) + TXT (Gantt) | 同时满足 AI 和人 |

---

## 9. 参考

| 项目 | 借鉴点 |
|---|---|
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent) (Meta) | 多智能体协作 (Planner+Coder+Verifier) |
| [AutoKernel](https://github.com/rightnow-ai/autokernel) | 6层优化手册、闭环优化、轨迹图 |
| [GEAK](https://github.com/AMD-AIG-AIMA/GEAK-Agent) (AMD) | Reflexion 迭代修复 |
| [OPAL](https://arxiv.org/abs/2510.00932) (2025) | 脚本压缩诊断→LLM推理 |
| [TritonForge](https://arxiv.org/abs/2512.09196) (2025) | Profiling-guided 自动优化 |

### 配套文档

| 文档 | 内容 |
|---|---|
| `docx/OPTIMIZATION_METHODOLOGY.md` | 优化方法论 + 参考文献 |
| `example_output/FIELD_SOURCE_MATRIX.md` | 29字段 × 来源矩阵 (msprof vs HIVMIR) |
| `../claude_resume_summary/resume_2026-07-23.md` | 对话上下文恢复文档 |
