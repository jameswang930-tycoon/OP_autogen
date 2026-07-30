# Triton Agent Optimizer — 完整架构设计 (v3.0)

> **核心差异化优势**: 不靠盲试（AutoKernel 300~400轮），而是通过 **Triton→TTIR→HIVM + msprof op simulator** 精确诊断瓶颈——知道哪个 op、哪个引擎、带宽利用率多少、为什么慢、该改什么参数。精准度比盲试高一个数量级。
>
> **环境**: WSL2 Ubuntu 24.04 + CANN 9.0 + triton 2.3.1 (无需 NPU 硬件，纯 CPU 闭环)
> **更新**: 2026-07-28 — Triton .py → HIVM MLIR 全自动化链路打通

---

## 0. 完整数据流 (已验证闭环)

```
Triton Kernel (.py)
  │  triton 2.3.1 (ast_to_ttir, 纯CPU)
  ▼
TTIR MLIR (triton intermediate representation)
  │  ttir_to_hivm.py (自研转换器)
  ▼
HIVM MLIR (Ascend NPU 指令级 IR)
  │                        │
  │  hivmir_analyzer.py    │  bishengir-compile → .o
  │  解析 11 语义字段       │  → msprof op simulator
  ▼                        ▼
HIVM Report              OPPROF_xxx/
(ops, buffers, deps)       ├── trace.json
                           └── instr_exe.csv
                              │
                              │  msprof_analyzer.py
                              ▼
                           msprof Report
                           (14 timing/pipe 字段)
                              │
  └────────── dsl_merger.py ──┘
              ▼
         29 字段全填充 merged_report.json
              │
              ▼
         Planner (LLM) → Coder (LLM) → Verifier → RecordManager
```

### 环境需求

| 步骤 | 环境 | 工具 |
|---|---|---|
| Triton .py → TTIR | WSL2 + CUDA driver | triton 2.3.1 + LD_PRELOAD stub |
| TTIR → HIVM | 任何 Python | ttir_to_hivm.py (自研) |
| HIVM MLIR compile | WSL2 + CANN 9.0 | bishengir-compile |
| msprof trace | WSL2 + CANN 9.0 | msprof op simulator |
| Analyer + Optimizer | 任何 Python | hivmir/msprof/dsl_merger analyzer |

---

## 0. Agent 实现模式

**Agent 的 Python 文件 = Prompt 编排器 + 执行框架。决策智能在 LLM 侧。**

```
  Python 骨架 (代码做的事)           LLM 大脑 (AI 做的事)
  ─────────────────────────          ─────────────────────
  1. 构建上下文 (诊断+Playbook+历史) → 1. 理解瓶颈原因
  2. 估算token用量，裁剪超限        → 2. 参考 Playbook 选择策略
  3. 调用 LLM API                   → 3. 生成具体优化计划 (JSON)
  4. 解析 JSON 输出                  → 4. 生成代码 diff
  5. 写入文件 / 执行动作             → 5. 判断优化是否有效
```

| 组件 | 实现方式 | 说明 |
|---|---|---|
| **Orchestrator** | Python 状态机 | 薄循环，不做决策 |
| **Planner** | LLM Agent | 读诊断+Playbook+历史 → 生成计划 |
| **Coder** | LLM Agent | 读计划+代码 → 最小化改动 |
| **Verifier** | Python 脚本 | CPU仿真 + 910B3实测，零推理需求 |
| **Analyzers** | Python 脚本 | 5个确定性函数链 |
| **RecordManager** | Python 规则引擎 | KEEP/REVERT + Tier + 停止条件 |

---

## 1. 完整闭环架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                       │
│  Triton kernel (.py)  ·  Shape/Dtype  ·  PyTorch reference  ·  Target HW    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS LAYER (5 Scripts)                          │
│                                                                              │
│    HIVMIR Analyzer              msprof Analyzer                              │
│    ┌──────────────────┐        ┌──────────────────┐                          │
│    │ .mlir 文本       │        │ trace.json       │                          │
│    │ buffer名/size    │        │ timing/engine    │                          │
│    │ RAW/WAR/WAW      │        │ pipeline channel │                          │
│    │ 操作序列         │        │ core breakdown   │                          │
│    └────────┬─────────┘        └────────┬─────────┘                          │
│             │          DSL Merger       │                                    │
│             └──────────┬────────────────┘                                    │
│                        ▼                                                     │
│             ┌─────────────────────┐                                          │
│             │ 29-field Report     │  op_id 对齐, 互相填补                    │
│             │ merged_report.json  │                                          │
│             └──────────┬──────────┘                                          │
│                        ▼                                                     │
│          ┌─────────────────────────┐                                         │
│          │ BottleneckDiagnoser     │  瓶颈分类 + 优化空间评估                 │
│          │ Tier-aware 规则引擎     │  memory_latency/bandwidth/compute/...    │
│          └──────────┬──────────────┘                                         │
│                     ▼                                                        │
│          ┌─────────────────────────┐                                         │
│          │ DataExtractor           │  Tier x 列过滤, 聚合分析                 │
│          │ ~2KB 精简文本           │  输出 → 注入 Planner prompt              │
│          └─────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          AGENT LAYER                                           │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │  Planner (LLM)   │  │   Coder (LLM)    │  │ Verifier (Script)│            │
│  │  ─────────────── │  │  ─────────────── │  │  ─────────────── │            │
│  │  读 Playbook  N  │  │  读 Plan + Code  │  │  Stage1:CPU仿真  │            │
│  │  读 Diagnosis    │  │  最小化代码改动  │  │  Stage2:910B3实测│            │
│  │  检索记忆经验    │  │  只改 kernel.py  │  │  FAIL→Coder重试  │            │
│  │  生成本轮计划    │  │  语法检查+diff   │  │  最多 3 次       │            │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘            │
│           │                     │                      │                      │
│           └─────────────────────┼──────────────────────┘                      │
│                                 ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              Orchestrator (Python 状态机, 非 LLM)                │        │
│  │  Analyzers → Plan → Code → Verify(retry) → Decide → Record      │        │
│  │  6-Tier Manager · 7 Stop Conditions · optimization_trajectory    │        │
│  └──────────────────────────────────────────────────────────────────┘        │
│                                 │                                             │
│  ┌──────────────────────────────────────────────────────────────────┐        │
│  │              RecordManager (反馈层 — 决策引擎)                   │        │
│  │  KEEP/REVERT · Tier晋升/降级 · 7条停止条件                       │        │
│  │  写 optimization_record.json · 更新 trajectory.json              │        │
│  │  达标 → 案例模板 · 经验入库 · Gantt图                            │        │
│  └──────────────────────────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                      │
│                                                                               │
│  ┌────────────────────────────┐    ┌────────────────────────────┐            │
│  │  Stage 1: CPU Emulator     │    │  Stage 2: 910B3 Hardware   │            │
│  │  ─────────────────────     │    │  ─────────────────────     │            │
│  │  emulators/common 模拟执行 │    │  Ascend Compiler 编译      │            │
│  │  Triton→Emulator import转换│    │  msprof 性能数据采集       │            │
│  │  多 shape/dtype 测试       │    │  benchmark (warmup+repeat) │            │
│  │  verify() 数值对比         │    │  真实延迟 / 吞吐 / 加速比  │            │
│  │  秒级反馈                  │    │  分钟级反馈 (本地跳过)     │            │
│  └────────────────────────────┘    └────────────────────────────┘            │
│                                                                               │
│  注: Cost Simulator 在分析层 (瓶颈诊断), 不在验证环节                          │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                        │
│                                                                               │
│  outputs/<kernel>/                                                             │
│  ├── round0/                        # Baseline                                 │
│  ├── 01_algorithmic_structure/      # Tier 1: 3 rounds                        │
│  ├── 02_operator_fusion/            # Tier 2: 4 rounds                        │
│  ├── 03_tiling_block_config/        # Tier 3: 7 rounds  ← 主要加速阶段         │
│  ├── 04_memory_access/              # Tier 4: 5 rounds                        │
│  ├── 05_compute_occupancy/          # Tier 5: 4 rounds                        │
│  ├── 06_910b3_architecture/         # Tier 6: 2 rounds                        │
│  ├── optimization_trajectory.json   # 全局状态 (中枢文件)                      │
│  └── final_output/                  # 最终产物                                 │
│      ├── optimized_kernel.py        # 优化后 kernel                            │
│      ├── trajectory_chart.png       # 6阶段加速比曲线图                         │
│      └── optimization_summary.md    # 总结报告                                 │
└──────────────────────────────────────────────────────────────────────────────┘

                             循环: Output → Analysis → Agent → Execution → Output
```

---

## 1B. 完整数据流图 (端到端)

```
                              ┌──────────────┐
                              │ Triton Kernel │  (.py, 用户输入)
                              └──────┬───────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │ msprof op      │    │ Ascend Compiler│    │ roundN/kernel.py│
     │ simulator      │    │ (910B3 only)   │    │ (Coder 产出)    │
     │ ────────────── │    │ ────────────── │    │ ──────────────  │
     │ trace.json     │    │ .mlir HIVMIR   │    │ Triton kernel   │
     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
             │                     │                     │
             ▼                     ▼                     ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │ msprof_analyzer.py │  │ hivmir_analyzer.py │  │ emulator_runner.py │
  │ ─────────────────  │  │ ─────────────────  │  │ ─────────────────  │
  │ 解析 Chrome Trace  │  │ 解析 MLIR 文本     │  │ import转换→CPU仿真 │
  │ timing·engine·ch   │  │ buffer·size·deps   │  │ verify()数值对比   │
  └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
           │                       │                        │
           ▼                       ▼                        ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │pipeline_report.json│  │ hivmir_report.json │  │ verification.json  │
  │  16 fields [OK]     │  │   9 fields [OK]    │  │  PASS/FAIL+error   │
  │  13 fields TBD      │  │  16 fields TBD     │  │  + speedup         │
  └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
           │                       │                        │
           └───────────┬───────────┘                        │
                       ▼                                    │
        ┌──────────────────────────┐                        │
        │   dsl_merger.py          │                        │
        │   ──────────────         │                        │
        │   op_id 对齐             │                        │
        │   互相填补 TBD 字段      │                        │
        │   SATURATION_PARAMS 计算 │                        │
        │   bw_util·regime·peak    │                        │
        └───────────┬──────────────┘                        │
                    ▼                                       │
        ┌──────────────────────────┐                        │
        │   merged_report.json     │                        │
        │   ★ 29 fields ALL FILLED │                        │
        └───────────┬──────────────┘                        │
                    │                                       │
         ┌──────────┴──────────┐                            │
         ▼                     ▼                            │
  ┌──────────────────┐  ┌──────────────────┐               │
  │bottleneck_       │  │data_extractor.py │               │
  │diagnoser.py      │  │────────────────  │               │
  │────────────────  │  │Tier×列过滤      │               │
  │Tier-aware 分类   │  │聚合分析          │               │
  │op_id·type·引擎   │  │~2KB 精简文本     │               │
  │HIGH/MED/LOW/UNCR │  └────────┬─────────┘               │
  │BottleneckDignosis│           │                         │
  └────────┬─────────┘           │                         │
           │                     │                         │
           └──────────┬──────────┘                         │
                      │                                    │
                      ▼                                    │
  ┌────────────────────────────────────┐                   │
  │         Planner (LLM)              │                   │
  │         ─────────────              │                   │
  │  输入: diagnosis                  │                   │
  │       + extracted_text            │                   │
  │       + playbook_tier_N.md        │                   │
  │       + history[-5:]              │                   │
  │       + similar_cases (memory)    │                   │
  │       + kernel_code               │                   │
  │                                   │                   │
  │  输出: round_N_plan.md            │                   │
  │       {strategy, specific_change, │                   │
  │        target_speedup,            │                   │
  │        expected_impact,           │                   │
  │        verification_method}       │                   │
  └──────────────┬────────────────────┘                   │
                 │                                        │
                 ▼                                        │
  ┌────────────────────────────────────┐                   │
  │         Coder (LLM)                │                   │
  │         ─────────                  │                   │
  │  输入: plan_text + kernel_code     │                   │
  │       + previous_error (if retry)  │                   │
  │                                   │                   │
  │  输出: round_N/kernel.py (优化后)  │                   │
  │        round_N/diff.patch          │                   │
  │                                   │                   │
  │  约束: 只改 kernel.py              │                   │
  │        Python 语法检查             │                   │
  └──────────────┬────────────────────┘                   │
                 │                                        │
                 ▼                                        │
  ┌──────────────────────────────────────────┐             │
  │         Verifier (Script)                │◄────────────┘
  │         ─────────────────                │  (verification.json)
  │                                          │
  │  Stage 1: CPU Emulator                   │
  │    输入: round_N/kernel.py               │
  │    执行: emulators/common 模拟            │
  │         多shape/dtype测试                │
  │         与 NumPy reference 数值对比       │
  │    输出: PASS/FAIL + error_details       │
  │                                          │
  │    FAIL → 回传错误到 Coder (最多3次) ──→ Coder
  │    PASS → Stage 2                        │
  │                                          │
  │  Stage 2: 910B3 Hardware (本地跳过)      │
  │    编译 → benchmark → msprof             │
  │    输出: actual_speedup                  │
  └──────────────┬───────────────────────────┘
                 │
                 ▼  VerifyResult {stage1_passed, speedup, error_details}
  ┌──────────────────────────────────────────┐
  │      RecordManager (Feedback Engine)     │
  │      ───────────────────────────         │
  │                                          │
  │  ① Decide:  speedup>1.01? KEEP:REVERT   │
  │  ② Save:    round_N/optimization_record  │
  │  ③ Update:  optimization_trajectory.json │
  │  ④ Manage:  Tier 晋升/降级               │
  │  ⑤ Check:   7 条停止条件                 │
  │  ⑥ Record:  speedup>1.05→记忆库SUCCESS   │
  │             speedup<0.98→记忆库FAIL       │
  └──────────────┬───────────────────────────┘
                 │
                 ▼  (should_stop? CONTINUE / STOP)
       ┌─────────┴─────────┐
       ▼                   ▼
  [CONTINUE]           [STOP]
   Round N+1            Finalize
       │                   │
       │                   ▼
       │     ┌─────────────────────────────┐
       │     │        Output Layer         │
       │     │  ───────────────────────    │
       │     │  optimized_kernel.py        │
       │     │  trajectory_chart.png       │
       │     │  optimization_summary.md    │
       │     │  final_merged_report.json   │
       │     │  Gantt chart (按需)         │
       │     └─────────────────────────────┘
       │
       └──→ Analyzers (重跑, 拿最新kernel的DSL数据) → Planner → ...
```

### 关键文件读/写映射

| 文件 | 写入者 | 读取者 |
|---|---|---|
| `roundN/kernel.py` | Coder | Emulator, Hardware, Analyzers(下轮) |
| `roundN/plan.md` | Planner | Coder, 人工debug |
| `roundN/diff.patch` | Coder | 人工review |
| `roundN/optimization_record.json` | RecordManager | 人工, 后续分析 |
| `roundN/verification.json` | Verifier | RecordManager |
| `msprof/pipeline_report.json` | msprof_analyzer | dsl_merger |
| `hivmir/hivmir_report.json` | hivmir_analyzer | dsl_merger |
| `merged/merged_report.json` | dsl_merger | diagnoser, extractor, 人工 |
| `merged/final_report_llm.txt` | dsl_merger | Planner (LLM prompt) |
| `merged/final_report_human.txt` | dsl_merger | 人工 |
| `optimization_trajectory.json` | RecordManager | Orchestrator, Planner, trajectory_chart |
| `final_output/trajectory_chart.png` | trajectory_chart | 人工 |
| `memory/experiences/tier*.json` | RecordManager | Planner (经验检索) |

---

## 2. 每轮执行流程

```
Round N:
  Orchestrator._run_one_round()
  │
  ├─ ① Analyzers.run()         [脚本, 5步链, 每轮重跑]
  │   msprof→hivmir→merger→diagnoser→extractor
  │   → merged_report.json + BottleneckDiagnosis + extracted_text
  │
  ├─ ② Planner.generate()      [LLM]
  │   输入: diagnosis + extracted + playbook_tier_N + history[-5:] + kernel_code
  │   输出: round_N_plan.md
  │
  ├─ ③ Coder.apply()           [LLM]
  │   输入: plan_text + kernel_code (+ previous_error if retry)
  │   输出: round_N/kernel.py + round_N/diff.patch
  │
  ├─ ④ Verifier.verify()       [脚本, 两阶段]
  │   Stage1: CPU Emulator → PASS/FAIL
  │     FAIL → error回传Coder, 重试最多3次
  │   Stage2: 910B3 → actual_speedup (本地跳过)
  │   输出: round_N/verification.json
  │
  └─ ⑤ RecordManager.evaluate() [规则引擎]
      decide KEEP/REVERT
      save round_N/optimization_record.json
      update optimization_trajectory.json
      check stop → CONTINUE / STOP

Round 0 (Baseline):
  只跑 Analyzers, 写入 trajectory.json baseline
```

---

## 3. 六层优化策略

**原则: 从结构影响最大到最小。改了上层, 下层要重做。**

| Tier | 名称 | Playbook | 做什么 | 晋升条件 |
|---|---|---|---|---|
| **1** | Algorithmic Structure | `playbook_tier1_algorithm.md` | Online Softmax / Split-K / Persistent Kernel | 算法已最优 |
| **2** | Operator Fusion | `playbook_tier2_fusion.md` | 逐元素融合 / WAR打破 / 激活融合 | 无可融合op |
| **3** | Tiling & Block Config | `playbook_tier3_tiling.md` | BLOCK_SIZE / num_warps / num_stages | 连续3轮无改进 |
| **4** | Memory Access | `playbook_tier4_memory.md` | 小传输合并 / double buffer / coalescing | 连续3轮无改进 |
| **5** | Compute & Occupancy | `playbook_tier5_compute.md` | 计算-传输重叠 / 向量化 / 精度取舍 | 连续3轮无改进 |
| **6** | 910B3 Architecture | `playbook_tier6_architecture.md` | Grid / Pipeline / L2驻留 / 混合精度 | 连续3轮无改进→停止 |

**降级**: 融合新算子→回到Tier3; 改了算法→回到Tier2

---

## 4. 文件架构总览

```
triton_agent_optimizer/
│
├── README.md                    # 本文档
├── ARCHITECTURE_DESIGN.md       # 架构设计 (本文档)
├── config.py                    # 全局配置 (路径/硬件参数/阈值)
│
├── prepare/                     # 环境准备
│   ├── setup_env.sh             #   Linux/910B3 一键设置
│   ├── setup_env.bat            #   Windows 一键设置
│   └── env_check.py             #   35项环境验证检查
│
├── analyzers/                   # 分析层 (5文件, 全部完成 ✅)
│   ├── msprof_analyzer.py       #   ① trace.json → timing/engine
│   ├── hivmir_analyzer.py       #   ② .mlir → buffer/size/deps
│   ├── dsl_merger.py            #   ③ 合并→29字段报告+LLM文本+Gantt图
│   ├── bottleneck_diagnoser.py  #   ④ Tier-aware瓶颈诊断
│   └── data_extractor.py        #   ⑤ Tier×列过滤精简提取
│
├── agents/                      # 智能体层 (4文件, 全部完成 ✅)
│   ├── orchestrator.py          #   调度器: Python状态机, 薄循环
│   ├── planner.py               #   规划器: LLM Prompt编排
│   ├── coder.py                 #   编码器: LLM改代码, 只改kernel.py
│   └── verifier.py              #   验证器: CPU仿真+910B3实测
│
├── execution/                   # 执行层 (4文件)
│   ├── emulator_runner.py       #   ✅ Stage1: CPU正确性验证
│   ├── simulator_runner.py      #   (备用) cost simulator
│   ├── compiler.py              #   🔶 Ascend编译器 (需910B3)
│   └── hardware_runner.py       #   🔶 910B3 benchmark (需910B3)
│
├── feedback/                    # 反馈层 (2文件, 全部完成 ✅)
│   ├── record_manager.py        #   决策引擎: KEEP/REVERT+Tier+Stop
│   └── trajectory_chart.py      #   6阶段加速比曲线图
│
├── memory/                      # 记忆层 (3文件, 全部完成 ✅)
│   ├── sliding_window.py        #   热/温/冷三层窗口
│   ├── context_manager.py       #   Token估算+裁剪+prompt构建
│   ├── experience_retriever.py  #   分Tier检索+3级匹配+记录
│   └── experiences/             #   6个Tier经验库JSON
│
├── docx/                        # Playbook 知识库 (7文件, 全部完成 ✅)
│   ├── OPTIMIZATION_METHODOLOGY.md   # 总纲+参考文献
│   ├── playbook_tier1_algorithm.md   # Tier 1: 算子→算法表
│   ├── playbook_tier2_fusion.md      # Tier 2: 融合决策树
│   ├── playbook_tier3_tiling.md      # Tier 3: K0+BLOCK_SIZE表
│   ├── playbook_tier4_memory.md      # Tier 4: 传输合并+double buffer
│   ├── playbook_tier5_compute.md     # Tier 5: 计算优化
│   └── playbook_tier6_architecture.md # Tier 6: 910B3专属
│
├── example_output/              # 参考示例
├── scripts/                     # 工具
│   └── init_output_structure.py
│
├── outputs/                     # 优化产物 (自动生成, 不入git)
│   └── <kernel_name>/
│       ├── round0/              #   Baseline (msprof/hivmir/merged)
│       ├── 01~06_tier_folders/  #   各Tier优化轮次
│       ├── optimization_trajectory.json  # ★ 中枢状态
│       └── final_output/        #   最终产物
│
└── cases/                       # 案例库
    └── template.md
```

---

## 5. 数据流: msprof + HIVMIR → 完整报告

```
  msprof trace.json              HIVMIR .mlir
        │                              │
  msprof_analyzer               hivmir_analyzer
        │                              │
  pipeline_report.json          hivmir_report.json
  (16 fields [OK])               (9 fields [OK])
  (13 fields TBD)                (16 fields TBD)
        │                              │
        └──────────┬───────────────────┘
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
  bottleneck_diagnoser   data_extractor
  (瓶颈分类+评估)         (Tier×列过滤)
        │                     │
        └──────────┬──────────┘
                   ▼
           Planner (LLM)
      (诊断+Playbook+历史→计划)
```

---

## 6. 输出目录

```
outputs/<kernel_name>/
│
├── round0/                                    # ★ 基准分析 (仅Analyzer, 无优化)
│   │
│   │  ┌─ 分析产物 ───────────────────────────┐
│   │  │ kernel.py            原始 Triton kernel        │
│   │  │ benchmark_result.json 延迟/加速比/吞吐/占比    │
│   │  │                                          │
│   │  │ msprof/                                  │
│   │  │   OPPROF_*/simulator/  msprof 中间产物    │
│   │  │   pipeline_report.json msprof 解析(16✅+13❌)│
│   │  │                                          │
│   │  │ hivmir/                                  │
│   │  │   compiler_output/      HIVMIR 原文       │
│   │  │   hivmir_report.json    HIVMIR 解析(9✅+16❌)│
│   │  │                                          │
│   │  │ merged/                                  │
│   │  │   merged_report.json    ★ 合并 29字段全填 │
│   │  │   final_report_llm.txt  LLM 7-section文本 │
│   │  │   final_report_human.txt 人读 Gantt图    │
│   │  └──────────────────────────────────────────┘
│
├── 01_algorithmic_structure/                  # Tier 1
│   ├── round1/                                # 第1轮优化
│   │   │
│   │   │  ┌─ 分析产物 (同 round0) ───────────┐
│   │   │  │ msprof/ + hivmir/ + merged/      │
│   │   │  └──────────────────────────────────┘
│   │   │
│   │   │  ┌─ 优化产物 (roundN 独有) ─────────┐
│   │   │  │ kernel.py             Coder 改后代码        │
│   │   │  │ plan.md               Planner 优化计划      │
│   │   │  │ diff.patch            代码变更diff          │
│   │   │  │ optimization_record.json RecordManager 决策  │
│   │   │  │ verification.json      Verifier 验证结果    │
│   │   │  └──────────────────────────────────────────┘
│   │   │
│   │   ├── round2/ ... roundN/
│   │
├── 02_operator_fusion/round1..N/              # Tier 2
├── 03_tiling_block_config/round1..N/          # Tier 3
├── 04_memory_access/round1..N/                # Tier 4
├── 05_compute_occupancy/round1..N/            # Tier 5
├── 06_910b3_architecture/round1..N/           # Tier 6
│
├── optimization_trajectory.json               # ★ 全局中枢状态 (RecordManager)
│
└── final_output/                              # ★ 最终产物 (达标或停止后生成)
    ├── optimized_kernel.py                    # 最终优化版 kernel
    ├── trajectory_chart.png                   # 6阶段加速比曲线图
    ├── optimization_summary.md                 # 总结报告
    ├── final_merged_report.json               # 最后一轮的合并报告
    ├── final_report_llm.txt                   # LLM可读的最终报告
    └── final_report_human.txt                 # 人读的最终Gantt图
```

### round0 vs roundN 对比

| 文件 | round0 (基准) | roundN (优化轮) | 说明 |
|---|---|---|---|
| `kernel.py` | 原始 kernel | Coder 修改后 kernel | round0 是用户原始代码 |
| `plan.md` | ❌ | ✅ Planner 产出 | round0 没有优化计划 |
| `diff.patch` | ❌ | ✅ Coder 产出 | round0 没有代码变更 |
| `optimization_record.json` | ❌ | ✅ RecordManager 产出 | round0 不记录优化效果 |
| `verification.json` | ❌ | ✅ Verifier 产出 | round0 不验证优化 |
| `msprof/` | ✅ | ✅ | 分析产物, 每轮都有 |
| `hivmir/` | ✅ | ✅ | 分析产物, 每轮都有 |
| `merged/` | ✅ | ✅ | 分析产物, 每轮都有 |
| `benchmark_result.json` | ✅ | ✅ | 基准性能数据 |

### optimization_trajectory.json

```json
{
  "state":  {"tier": 3, "round": 12, "best_speedup": 1.52,
             "consecutive_reverts": 0, "consecutive_no_improvement": 0},
  "baseline": {"total_ns": 3655.6, "num_ops": 3,
               "bottleneck_type": "memory_bandwidth"},
  "history": [
    {"round": 1, "tier": 1, "strategy": "...",
     "actual_speedup": 1.0, "cumulative_speedup": 1.0,
     "decision": "KEEP", "decision_reason": "..."}
  ]
}
```

---

## 7. 设计决策速查

| 决策 | 选择 | 理由 |
|---|---|---|
| 瓶颈诊断 | 脚本规则引擎 | 确定性, 压缩到 ~2KB |
| 策略规划 | LLM (Planner) | 需要上下文推理 |
| 代码修改 | LLM (Coder) | 语义理解, 最小化改动 |
| 验证 | 脚本 (CPU+Hardware) | 确定性, 不推理 |
| 调度 | Python 状态机 | 管循环, 不做决策 |
| 决策/记录 | RecordManager | 集中管理 Tier+Stop+轨迹 |
| 优化顺序 | Algorithm→Fusion→Tile→Memory→Compute→Arch | 结构影响从大到小 |
| 记忆 | 分Tier JSON+3级匹配 | 简单, LLM可读, 无需向量库 |
