# Triton Agent Optimizer

> 基于 DSL 流水线分析 (HIVMIR + msprof op simulator) 的 Triton Kernel 自动优化框架。
> 目标硬件: **华为 Ascend 910B3 NPU**。

---

## 1. 核心理念

**不是盲试。** AutoKernel 只能知道 "kernel X 占 60% 时间" → 盲目尝试 300~400 轮。我们的方案通过 **DSL 流水线分析** 精确知道：
- 哪个 op 是瓶颈 (如 op2: ub_to_gm)
- 为什么慢 (带宽利用率 46%, regime=ramp, k0=10.72KB, 当前 tile=1KB)
- 怎么改 (合并小传输, tile > 10.72KB 进入饱和区)

**精准度比盲试高一个数量级, 预估 20~50 轮即可收敛。**

### 完整闭环

```
Triton Kernel (.py)
  → HIVMIR 编译器中间产物 (变量名/依赖/数据大小)
  → msprof op simulator (时序/带宽/引擎利用率/瓶颈识别)
  → DSL 流水线合并 (精确到每个 op: 时间占比/瓶颈类型/依赖关系)
  → Agent 智能体 (Planner→Coder→Verifier, 按 Playbook 精准优化)
  → CPU Emulator 验证 (秒级正确性检查, 不上板)
  → 910B3 真机实测 (最终性能验证)
  → 记录本轮结果 → 下一轮 → 直到收敛
```

---

## 2. 目录结构

```
triton_agent_optimizer/
│
├── README.md                           # 本文档
├── ARCHITECTURE_DESIGN.md              # 完整架构设计
├── IMPLEMENTATION_PLAN.md              # 逐文件实现计划 (36 文件)
├── OUTPUT_STRUCTURE.md                 # 输出目录架构
├── config.py                           # ★ 全局配置中心
│
├── analyzers/                          # ★ 分析层 (已完成 4/5)
│   ├── msprof_analyzer.py              #   ① msprof op simulator 解析 → timing
│   ├── hivmir_analyzer.py              #   ② HIVMIR 编译器产物解析 → buffer/size/deps
│   ├── dsl_merger.py                   #   ③ 合并 → 29字段完整报告 + LLM文本 + Gantt图
│   ├── bottleneck_diagnoser.py         #   ④ 瓶颈诊断 → ~2KB 结构化数据 (Tier-aware)
│   └── data_extractor.py               #   ⑤ 按需数据提取 (待实现)
│
├── agents/                             # 智能体层 (待实现)
│   ├── orchestrator.py                 #   调度器: 管理 6-tier 优化循环
│   ├── planner.py                      #   规划器: 读诊断+Playbook+历史 → 生成优化计划
│   ├── coder.py                        #   编码器: 按计划做最小化代码改动
│   └── verifier.py                     #   验证器: 三阶段验证 (CPU→Sim→HW)
│
├── optimizers/                         # 优化器 (待实现)
│   ├── tile_optimizer.py, memory_optimizer.py,
│   ├── fusion_optimizer.py, compute_optimizer.py
│
├── execution/                          # 执行层 (待实现)
│   ├── emulator_runner.py              #   CPU Emulator (秒级正确性)
│   ├── simulator_runner.py             #   Cost Simulator (秒级预估)
│   ├── compiler.py                     #   Ascend 编译器接口
│   └── hardware_runner.py              #   910B3 真机运行 (分钟级实测)
│
├── feedback/                           # 反馈层 (待实现)
│   ├── round_logger.py                 #   每轮记录 (JSONL)
│   ├── optimization_journal.py         #   跨轮日志管理
│   ├── stop_condition.py               #   7 条停止条件
│   ├── trajectory_chart.py             #   优化轨迹图
│   └── case_template.py               #   优秀案例模板生成
│
├── memory/                             # 上下文记忆层 (待实现)
│   ├── context_manager.py              #   上下文构建 (滑窗+摘要+token管理)
│   ├── experience_retriever.py         #   经验检索 (对接 memory/ 模块)
│   └── sliding_window.py               #   三层滑动窗口
│
├── playbooks/                          # 优化指导手册 (待编写, ~20章)
│   ├── optimization_playbook.md        #   总纲
│   ├── playbook_algorithmic.md         #   Tier 1: 算法选择
│   ├── playbook_fusion.md              #   Tier 2: 算子融合
│   ├── playbook_tiling.md              #   Tier 3: 分块配置
│   ├── playbook_memory.md              #   Tier 4: 内存访问
│   ├── playbook_compute.md             #   Tier 5: 计算调优
│   └── playbook_910b3_arch.md          #   Tier 6: 硬件专属
│
├── docx/                               # 文档
│   └── OPTIMIZATION_METHODOLOGY.md     #   ★ 优化方法论文档 (含参考文献)
│
├── example_output/                     # 参考示例
│   ├── README.md                       #   索引
│   ├── FIELD_SOURCE_MATRIX.md          #   29字段 × 来源矩阵 (msprof vs HIVMIR)
│   ├── mock_pipeline_report_vector_add.json
│   ├── mock_hivmir_report_vector_add.json
│   └── 01~07_*.txt                     #   simulator 输出示例
│
├── scripts/
│   └── init_output_structure.py        #   初始化 outputs/ 目录结构
│
└── output/                             # (旧, 已废弃 → 用 outputs/)
```

---

## 3. 6 层优化策略 (正确顺序)

**核心原则: 从结构影响最大到最小。改了上层结构, 下层要重做 → 结构性的先做。**

| Tier | 名称 | 做什么 | 晋升条件 |
|---|---|---|---|
| **1** | Algorithmic Structure | 选择最优算法 (Online Softmax/Flash Attention/Split-K) | 算法已最优 |
| **2** | Operator Fusion | 融合逐元素/激活/残差操作, 打破 WAR | 无可融合 op |
| **3** | Tiling & Block Config | BLOCK_SIZE, num_warps, num_stages, grid | 连续 3 轮无改进 |
| **4** | Memory Access | 合并小传输, coalescing, double buffering | 连续 3 轮无改进 |
| **5** | Compute & Occupancy | 计算-传输重叠, 向量化, 精度取舍 | 连续 3 轮无改进 |
| **6** | 910B3 Architecture | grid 选择, pipeline 切换, L2 驻留 | 连续 3 轮无改进 → 停止 |

**降级规则**: 融合了新算子 → 回退到 Tier 3; 改了算法 → 回退到 Tier 2。

详见: `docx/OPTIMIZATION_METHODOLOGY.md`

---

## 4. 数据流 (一轮优化)

```
Step 0: 准备
  Orchestrator 决定当前 Tier
     │
Step 1: 分析
  ├─ ① msprof_analyzer  → outputs/<kernel>/roundN/msprof/pipeline_report.json
  ├─ ② hivmir_analyzer  → outputs/<kernel>/roundN/hivmir/hivmir_report.json
  ├─ ③ dsl_merger       → outputs/<kernel>/roundN/merged/merged_report.json
  │                       + final_report_llm.txt (LLM 读, 7-section)
  │                       + final_report_human.txt (人读, Gantt图)
  └─ ④ bottleneck_diagnoser → BottleneckDiagnosis (~2KB JSON, Tier-aware)
     │
Step 2: 规划 (Planner LLM)
  输入: BottleneckDiagnosis + merged 完整数据 + Playbook Tier N 章节 + 历史 + kernel 代码
  输出: round_N_plan.md (具体优化操作/预期效果/验证方法)
     │
Step 3: 编码 (Coder LLM)
  输入: plan.md + kernel.py
  输出: optimized_kernel.py + diff.patch
     │
Step 4: 验证 (Verifier)
  ① CPU Emulator (秒级, 多 shape/dtype)
  ② Cost Simulator (秒级, 预估加速比)
  ③ 910B3 Hardware (分钟级, 实测加速比)
     │
Step 5: 决策
  Keep (>1% 提升) / Revert
     │
Step 6: 记录
  optimization_record.json → optimization_journal.jsonl
     │
Step 7: 检查停止条件 → 继续下一轮 或 停止
```

---

## 5. 输出目录结构

```
outputs/<kernel_name>/
│
├── round0/                              # 基准分析 (优化前)
│   ├── kernel.py                        # 原始 kernel
│   ├── benchmark_result.json            # 延迟/加速比/吞吐/时间占比
│   ├── msprof/
│   │   ├── OPPROF_*/simulator/trace.json # msprof 中间产物
│   │   └── pipeline_report.json         #   解析产物 (16✅+13❌)
│   ├── hivmir/
│   │   ├── compiler_output/hivmir_output.mlir
│   │   └── hivmir_report.json           #   解析产物 (9✅+16❌)
│   └── merged/
│       ├── merged_report.json            #   ★ 合并产物 (29字段全填)
│       ├── final_report_llm.txt          #   LLM 读: 7-section 文本
│       └── final_report_human.txt        #   人读: ASCII Gantt 图
│
├── 01_algorithmic_structure/            # Tier 1
│   └── round1/...roundN/               # (每轮含 round0 全部 + optimization_record.json)
├── 02_operator_fusion/                  # Tier 2
├── 03_tiling_block_config/             # Tier 3
├── 04_memory_access/                   # Tier 4
├── 05_compute_occupancy/               # Tier 5
├── 06_910b3_architecture/              # Tier 6
│
├── optimization_trajectory.json         # 跨轮汇总
├── optimization_summary.md              # 总结报告
└── final_output/                        # 最终产物
    ├── optimized_kernel.py
    ├── trajectory_chart.png
    └── optimization_summary.md
```

---

## 6. 核心设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| **瓶颈诊断** | 脚本 (规则引擎) | 阈值分类是确定性的, 不依赖 LLM; OPAL 论文: 990MB→6KB 压缩 |
| **策略规划** | LLM | 需要结合 Playbook + 历史 + 代码上下文做推理 |
| **验证顺序** | CPU → Simulator → Hardware | 逐级过滤, 节省硬件时间 |
| **优化顺序** | Algorithm → Fusion → Tiling → Memory → Compute → Arch | 结构影响从大到小, 上层改了底层白做 |
| **上下文管理** | 热(5轮完整) / 温(15轮摘要) / 冷(数据点) | 1M 上下文窗口, 控制 token 用量 |
| **输出格式** | JSON (LLM 直接读取) + TXT (7-section 文本) + TXT (Gantt 图) | 同时满足 AI 和人 |
| **字段对齐** | 29 字段全结构, msprof 和 HIVMIR 输出格式完全一致 | dsl_merger 通过 op_id 直接对齐 |

---

## 7. 关键文件说明

### 7.1 分析层 (已实现, 可运行)

| 文件 | 做什么 | 怎么运行 |
|---|---|---|
| `analyzers/msprof_analyzer.py` | 运行 msprof op simulator → 解析 trace.json → timing 数据 | `python analyzers/msprof_analyzer.py` (自测) |
| `analyzers/hivmir_analyzer.py` | 解析 HIVMIR 编译器文本 → buffer/size/deps | `python analyzers/hivmir_analyzer.py` (自测) |
| `analyzers/dsl_merger.py` | 合并 msprof + HIVMIR → 29字段完整报告 | `python analyzers/dsl_merger.py outputs/.../round0` |
| `analyzers/bottleneck_diagnoser.py` | 从 merged_report.json 诊断瓶颈 (Tier-aware) | `python analyzers/bottleneck_diagnoser.py outputs/.../merged/merged_report.json 3` |

### 7.2 29 字段完整清单

| # | 字段 | 来源 | 说明 |
|---|---|---|---|
| 1 | op_id | msprof + HIVMIR | 操作序号 |
| 2 | op_type | msprof + HIVMIR | gm_to_ub/vadd/ub_to_gm/... |
| 3 | engine | msprof | GM→UB/VecUnit/UB→GM/... |
| 4 | instruction | HIVMIR | gm_to_ub(ub_1, gm_1) |
| 5-7 | dst/src/src2 | HIVMIR | buffer 名 |
| 8 | size_kb | HIVMIR | 精确数据大小 |
| 9 | memory_region | HIVMIR | GM/UB/L1/L0 |
| 10 | variable_name | HIVMIR | 变量名 |
| 11-13 | duration_ns/start_ns/end_ns | msprof | 时序 |
| 14 | time_ratio | msprof (计算) | 时间占比 |
| 15-17 | effective_bw/peak_bw/bw_util | merger (计算) | 带宽 |
| 18 | regime | merger (计算) | floor/ramp/saturated/flat |
| 19 | wait_before_start_ns | msprof | 等待时间 |
| 20 | blocked_by | HIVMIR + msprof | 阻塞关系 |
| 21 | pipeline_channel | msprof | MTE2/VECTOR/MTE3/... |
| 22-23 | core_id/trace_event_name | msprof | 核/事件名 |
| 24-29 | dependencies/scalar/address_offset/line_number | HIVMIR | 依赖/标量/偏移/行号 |

详见: `example_output/FIELD_SOURCE_MATRIX.md`

---

## 8. 快速开始

```bash
# 1. 检查环境
python config.py

# 2. 初始化输出目录 (示例)
python scripts/init_output_structure.py

# 3. 合并 round0 数据
python analyzers/dsl_merger.py outputs/vector_add_fp16_N65536/round0

# 4. 诊断瓶颈 (Tier 2: Fusion)
python analyzers/bottleneck_diagnoser.py outputs/vector_add_fp16_N65536/round0/merged/merged_report.json 2
```

---

## 9. 参考文档

| 文档 | 内容 |
|---|---|
| `ARCHITECTURE_DESIGN.md` | 完整架构设计: 架构图/数据流/智能体模式/上下文管理/停止条件 |
| `IMPLEMENTATION_PLAN.md` | 36 个文件的实现顺序/依赖关系/接口定义 |
| `OUTPUT_STRUCTURE.md` | outputs/ 目录架构 + optimization_record.json schema |
| `docx/OPTIMIZATION_METHODOLOGY.md` | 6 层优化方法论 + 参考文献 |
| `example_output/FIELD_SOURCE_MATRIX.md` | 29 字段 × 来源矩阵 (msprof vs HIVMIR) |
| `../claude_resume_summary/resume_2026-07-23.md` | 对话上下文恢复文档 |
| `../costModel/cost_emulator/simulator.py` | 7-engine pipeline simulator |
| `../emulators/common/__init__.py` | CPU Triton Emulator |

## 10. 相关项目

| 项目 | 借鉴点 |
|---|---|
| [KernelAgent](https://github.com/meta-pytorch/KernelAgent) (Meta) | 多智能体协作 (Planner+Coder+Verifier) |
| [AutoKernel](https://github.com/rightnow-ai/autokernel) | 6层优化手册、迭代优化闭环、轨迹图 |
| [GEAK](https://github.com/AMD-AIG-AIMA/GEAK-Agent) (AMD) | Reflexion 迭代修复、pass@k 验证 |
| [OPAL](https://arxiv.org/abs/2510.00932) (2025) | 脚本压缩诊断数据 → LLM 推理 |
| [TritonForge](https://arxiv.org/abs/...) (2025) | Profiling-guided 自动化优化循环 |
