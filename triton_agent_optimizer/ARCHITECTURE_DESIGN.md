# Triton Agent Optimizer — 完整架构设计

> **核心差异化优势**: 不靠盲试（AutoKernel 300~400轮），而是通过 **DSL 流水线分析（HIVMIR + cost simulator）** 精确诊断瓶颈——知道哪个 op、哪个引擎、带宽利用率多少、为什么慢、该改什么参数。精准度比盲试高一个数量级。

---

## 0. 前置概念：Agent 是什么？

**Agent 的 Python 文件不是纯代码逻辑，而是 "Prompt 编排器 + 执行框架"。**

```
┌─ Python 骨架（代码做的事）───────┐    ┌─ LLM 大脑（AI 做的事）───────┐
│                                   │    │                               │
│  1. 构建上下文                    │    │  1. 理解瓶颈原因               │
│     - 从 DSL 报告提取关键数据     │    │  2. 参考 Playbook + 历史案例  │
│     - 注入 Playbook 相关章节      │    │  3. 生成本轮优化计划            │
│     - 检索相似历史案例            │    │  4. 生成代码 diff              │
│     - 估算 token 用量，裁剪       │    │  5. 判断优化是否有效            │
│                                   │    │                               │
│  2. 调用 LLM                      │ ←→ │  (推理+生成发生在 LLM 侧)     │
│                                   │    │                               │
│  3. 解析输出                      │    │                               │
│     - JSON → 结构化计划           │    │                               │
│     - diff → 文件写入             │    │                               │
│                                   │    │                               │
│  4. 执行动作                      │    │                               │
│     - 写入计划文件                │    │                               │
│     - 应用代码 diff               │    │                               │
│     - 运行验证流程                │    │                               │
│     - 记录本轮结果                │    │                               │
└───────────────────────────────────┘    └───────────────────────────────┘
```

**所以 `agents/planner.py` 是一个 Python 类，它不自己做优化决策，而是：**
1. 准备正确的上下文（DSL 关键数据 + Playbook 章节 + 历史）
2. 调用 LLM 生成计划
3. 解析 LLM 输出
4. 写入文件 / 执行动作

**决策智能在 LLM 侧，框架代码负责管理信息流。**

---

## 参考项目

| 项目 | 核心借鉴点 |
|---|---|
| **KernelAgent** (Meta) | 多智能体协作：Planner + Coder + Verifier，闭环优化 |
| **AutoKernel** (RightNow AI) | 6层优化手册、Amdahl调度、5级正确性验证、909行 playbook |
| **GEAK** (AMD) | Reflexion 迭代修复、pass@k 验证、debugging trap 防护 |
| **Kernel Agent** (Yaalalabs) | Hook拦截 + 挥发/持久双层缓存，上下文管理 |

---

## 1. 完整闭环架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                      输入层 (Input Layer)                           │
├─────────────────────────────────────────────────────────────────────┤
│  • Triton kernel (.py)          • Shape 信息                        │
│  • PyTorch 参考实现              • 目标硬件配置 (910B3)              │
│  • 优化目标（加速比 / 带宽利用率 / 延迟上限）                       │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                分析层 (Analysis Layer)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐        ┌──────────────────────┐          │
│  │  HIVMIR 分析          │        │  msprof 分析          │          │
│  │  ────────────         │        │  ────────────         │          │
│  │  • 变量名/数据大小    │        │  • 时序信息 (ns)      │          │
│  │  • 依赖关系 RAW/WAR   │        │  • 带宽利用率 (%)     │          │
│  │  • 操作类型分类       │        │  • 瓶颈识别           │          │
│  │  • 引擎归属           │        │  • 引擎利用率         │          │
│  └──────────────────────┘        └──────────────────────┘          │
│           ↓                                ↓                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              DSL 流水线合并 (DataMerger)                 │       │
│  │  • 完整操作流水（Op序号、类型、引擎、SIZE、Times...）    │       │
│  │  • 依赖图（RAW/WAR/WAW）                                 │       │
│  │  • 时间占比分析 + 关键路径提取                           │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              瓶颈诊断 (BottleneckDiagnoser)              │       │
│  │  • 识别瓶颈操作（时间占比最大）                          │       │
│  │  • 分类瓶颈类型（memory_bandwidth / memory_latency /     │       │
│  │    compute_vec / compute_cube / dependency / engine_contention）│
│  │  • 评估可优化空间（是否已达到理论峰值）                  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│              智能体层 (Agent Layer) — 多智能体协作                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│  │ Planner Agent  │  │  Coder Agent   │  │ Verifier Agent │       │
│  │   规划智能体    │  │   编码智能体    │  │   验证智能体    │       │
│  │  ────────────  │  │  ────────────  │  │  ────────────  │       │
│  │ • 读瓶颈报告   │  │ • 读优化计划   │  │ • 三阶段验证   │       │
│  │ • 选优化策略   │  │ • 最小化代码改 │  │   ① CPU Emulator│       │
│  │ • 写本轮计划   │  │ • 单文件变更   │  │   ② Simulator  │       │
│  │ • 参考历史案例 │  │ • 保持可回退   │  │   ③ 910B3 实测  │       │
│  └────────────────┘  └────────────────┘  └────────────────┘       │
│         │                    │                    │                  │
│         └────────────────────┼────────────────────┘                  │
│                              ↓                                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              Orchestrator (调度器)                       │       │
│  │  • 轮次管理（每轮：Plan→Code→Verify→Decide→Record）     │       │
│  │  • 上下文管理（滑动窗口 + 摘要压缩 + 经验检索）         │       │
│  │  • 停止条件判断                                          │       │
│  │  • 全局状态维护                                          │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│              执行层 (Execution Layer)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐   │
│  │  CPU Emulator    │  │  Cost Simulator  │  │  910B3 Hardware│   │
│  │  ────────────    │  │  ────────────    │  │  ────────────  │   │
│  │  • 秒级反馈      │  │  • 秒级反馈      │  │  • 分钟级反馈  │   │
│  │  • 正确性验证    │  │  • 性能预估      │  │  • 真实性能     │   │
│  │  • Shape sweep   │  │  • 瓶颈分析      │  │  • msprof 数据  │   │
│  │  • 边界条件       │  │  • 引擎利用率    │  │  • HIVMIR 提取  │   │
│  └──────────────────┘  └──────────────────┘  └────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │          决策 (Keep / Revert)                            │       │
│  │  • 正确性通过 + 性能提升 > 1% → Keep                     │       │
│  │  • 正确性失败 or 性能下降    → Revert                    │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│            反馈与记录层 (Feedback & Record Layer)                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │          轮次日志 (Round Journal)                        │       │
│  │  • 每轮记录：轮次号 / 策略 / 代码diff / 验证结果         │       │
│  │  • 性能追踪：本轮加速比 / 累计加速比 / 瓶颈变化          │       │
│  │  • 决策记录：Keep/Revert + 原因                           │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │          策略调整 (Strategy Adjuster)                    │       │
│  │  • 瓶颈转移 → 切换优化方向                               │       │
│  │  • 连续失败 → 提升策略层级                               │       │
│  │  • 达到平台期 → 标记该方向已耗尽                         │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │          案例生成 (Case Generator)                       │       │
│  │  • 加速比 ≥ 目标 → 生成优秀案例模板                      │       │
│  │  • 包含：优化前后对比 / 关键变更 / 每轮决策链            │       │
│  │  • 入库经验库供后续参考                                  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                     │
│  循环：分析层 → 智能体层 → 执行层 → 反馈层 → 分析层                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│                 输出层 (Output Layer)                               │
├─────────────────────────────────────────────────────────────────────┤
│  • 优化后的 Triton kernel                                          │
│  • 优化轮次记录（Round Journal JSONL）                              │
│  • 性能报告（加速比、瓶颈变化曲线）                                │
│  • 优秀案例文档（template，可入库）                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1.5 数据流与输出格式对齐

### 1.5.1 核心数据流

```
Kernel 代码 (.py)
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step A: 生成 DSL 程序                                       │
│  kernel → DSL program (alloc + gm_to_ub + vadd + ub_to_gm...) │
│  对接: analyzers/msprof_analyzer.py → generate_dsl_from_kernel() │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step B: 运行 Cost Simulator                                 │
│  python simulator.py --llm --critical-path "<DSL program>"   │
│  对接: costModel/cost_emulator/simulator.py (不改动)         │
│                                                              │
│  输出 (simulator --llm 格式，7 个 section):                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ === EXECUTION SUMMARY ===                             │    │
│  │   total_ns, num_ops, execution_mode                   │    │
│  │ === TIME BREAKDOWN ===                                │    │
│  │   每 op: duration_ns, time_ratio (从大到小)            │    │
│  │ === PER-OP STATISTICS ===                             │    │
│  │   每 op: engine, size, cycles_ns, bandwidth,          │    │
│  │          wait_ns, blocked_by, fix suggestion (WAR)     │    │
│  │ === ENGINE UTILIZATION ===                            │    │
│  │   每引擎: busy/total ns, utilization%                  │    │
│  │ === BANDWIDTH UTILIZATION ===                         │    │
│  │   每 op: effective/peak GB/s, utilization%, regime     │    │
│  │ === PARALLELISM ===                                   │    │
│  │   并行对 or root_cause_of_sequential_execution         │    │
│  │ === CRITICAL PATH ===                                 │    │
│  │   path, edges, per-op detail, fraction_of_makespan     │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step C: 提取 HIVMIR（从编译器中间产物）                      │
│  对接: fusion_pipeline/extract_hivmir_from_compiler.py       │
│  对接: analyzers/hivmir_analyzer.py → HIVMIRParser           │
│                                                              │
│  输出 (HIVMIR 富化数据):                                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  每 op: variable_name, precise_size_kb,              │    │
│  │         memory_region (GM/UB/L1/L0),                  │    │
│  │         dependencies [(op_id, RAW/WAR/WAW), ...]      │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step D: 数据合并 (DSL Merger)                                │
│  对接: analyzers/dsl_merger.py → DataMerger                   │
│                                                              │
│  操作: 将 Step B 的 simulator 输出（性能数据）与              │
│        Step C 的 HIVMIR 输出（语义信息）合并                   │
│                                                              │
│  合并方式: 通过 op_id 对齐 —— simulator 的 op0 对应           │
│            HIVMIR 的 op0（两者按程序顺序同序）                 │
│                                                              │
│  输出: 完整 DSL 流水线报告 (CombinedOp 列表)                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Op | 操作类型 | 引擎    | SIZE  | 变量名 | Times    │    │
│  │     | BW util | Regime | waitFor | 依赖类型 | 时间占比│    │
│  │  ─────────────────────────────────────────────────── │    │
│  │   0 | gm_to_ub | GM→UB  | 128KB | ub_1   | 1621.6ns │    │
│  │     | 100%    | saturated | -    | -       | 44.36%  │    │
│  │   1 | vadd     | VecUnit| 128KB | ub_2   |  324.4ns │    │
│  │     | 100%    | saturated | op0  | RAW     |  8.88%  │    │
│  │   2 | ub_to_gm | UB→GM  | 128KB | gm_2   | 1709.6ns │    │
│  │     | 100%    | saturated | op1  | RAW     | 46.77%  │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  + ENGINE UTILIZATION + CRITICAL PATH + PARALLELISM          │
│  (完整保留 simulator 的 7 个 section 结构)                    │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step E: 瓶颈诊断 (Bottleneck Diagnoser)                      │
│  对接: analyzers/bottleneck_diagnoser.py                      │
│                                                              │
│  输入: 合并后的 CombinedOp 列表                               │
│  输出:                                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  bottleneck_op: op2 (ub_to_gm)                         │    │
│  │  bottleneck_type: memory_bandwidth                     │    │
│  │  time_ratio: 46.77%                                    │    │
│  │  bw_utilization: 100% (已饱和 → 不能靠合并传输提速)     │    │
│  │  regime: saturated                                     │    │
│  │  on_critical_path: true                                │    │
│  │  optimization_headroom: LOW (已达峰值带宽)             │    │
│  │  suggested_strategies:                                  │    │
│  │    - 减少数据传输量 (减小 tile size → 增大并行度)      │    │
│  │    - 使用更快引擎 (UB→GM 76.67 vs GM→UB 80.83 GB/s)   │    │
│  │    - overlap 传输与计算 (double buffering)             │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│  Step F: Agent 优化循环 (Planner → Coder → Verify → Decide) │
│  (使用 Step E 的精确诊断结果，不再是盲试)                    │
└──────────────────────────────────────────────────────────────┘
```

### 1.5.2 输出格式对齐原则

**最终优化报告 = simulator `--llm --critical-path` 格式 + HIVMIR 富化字段 + 优化元数据。**

每一轮优化前后的性能对比，严格使用 simulator 的 7 个 section 结构：

| Simulator Section | 用途 | 优化前后对比 |
|---|---|---|
| EXECUTION SUMMARY | total_ns 变化 | 优化前 3655ns → 优化后 2800ns (23%↓) |
| TIME BREAKDOWN | 哪个 op 占比变了 | op2 从 47%→25%, op0 从 44%→55% (瓶颈转移) |
| PER-OP STATISTICS | 具体参数变化 | blocked_by 消除, wait_ns 减少 |
| ENGINE UTILIZATION | 引擎负载均衡 | GM→UB 利用率从 44%→55%, UB→GM 从 47%→25% |
| BANDWIDTH UTILIZATION | 带宽利用率 | op2 从 ramp→saturated, bw_util 46%→92% |
| PARALLELISM | 并行度变化 | parallel_pairs 从 0→2 (打破 WAR 依赖) |
| CRITICAL PATH | 关键路径变化 | 路径缩短, fraction_of_makespan 从 100%→85% |

> **为什么必须对齐 simulator 格式？** 因为 `.plan.json` 的 `raw_llm` 字段就是 simulator 的 `--llm` 输出。整个 `/triton-plan → triton-gen` 流水线已经依赖这个格式。Agent 优化器的输出与这个格式对齐，意味着优化后的 kernel 可以直接回到现有流水线中。

---

## 1.6 按需数据提取 (On-Demand Data Extraction)

### 问题
复杂 kernel 的 DSL 流水线可能有 50~200 个 op，完整输出塞进 LLM prompt 会超过上下文窗口。

### 方案：DataExtractor 按瓶颈类型提取关键数据

```
┌─────────────────────────────────────────────────────────────┐
│  全量 DSL 流水线报告（存文件，不入 prompt）                  │
│  200 op × 15 字段 = 3000+ 数据点                             │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼  DataExtractor (自动脚本)
┌─────────────────────────────────────────────────────────────┐
│  注入 prompt 的精简数据（~20 行）                             │
│                                                              │
│  瓶颈类型 = memory_bandwidth → 提取:                          │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [关键瓶颈] op2 (ub_to_gm) time_ratio=46.77%           │    │
│  │   引擎: UB→GM peak=76.67 GB/s/核                      │    │
│  │   当前: tile=1KB, bw=16.2 GB/s (21.1%峰值), regime=ramp│   │
│  │   半饱和点: k0=10.72KB → tile需>10.72KB才能饱和        │    │
│  │                                                       │    │
│  │ [依赖链] op0(gm_to_ub)→op1(vadd)→op2(ub_to_gm)       │    │
│  │   全 RAW 串行，无 WAR 可打破                           │    │
│  │                                                       │    │
│  │ [注入 Playbook] playbook_memory.md §2 (小传输合并)    │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  瓶颈类型 = dependency → 提取:                                │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [依赖图] op0→op1(RAW), op0→op2(WAR, avoidable)       │    │
│  │   WAR on 'ub_1': op0 reads ub_1, op2 writes ub_1      │    │
│  │   fix: allocate ub_3 instead of reusing ub_1          │    │
│  │                                                       │    │
│  │ [注入 Playbook] playbook_fusion.md §3 (WAR 打破)     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  瓶颈类型 = compute_vec → 提取:                               │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [关键瓶颈] op1 (vadd) time_ratio=35.2%                │    │
│  │   引擎: VecUnit peak=404 GB/s/核                       │    │
│  │   当前: tile=24KB, bw=320 GB/s (79.2%峰值), regime=saturated│
│  │   → 已达峰值，无法通过增大 tile 提升                    │    │
│  │   → 优化方向: 与传输 overlap (double buffering)        │    │
│  │                                                       │    │
│  │ [注入 Playbook] playbook_compute.md §3 (计算传输重叠) │    │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### DataExtractor 实现

```python
# analyzers/data_extractor.py

class DataExtractor:
    """从完整 DSL 报告中按需提取关键数据"""

    EXTRACTORS = {
        'memory_bandwidth': extract_memory_bottleneck_data,
        'memory_latency':    extract_memory_bottleneck_data,
        'compute_vec':       extract_compute_bottleneck_data,
        'compute_cube':      extract_compute_bottleneck_data,
        'dependency':        extract_dependency_bottleneck_data,
        'engine_contention': extract_engine_contention_data,
    }

    def extract(self, combined_ops, bottleneck_type, bottleneck_op_id):
        """根据瓶颈类型提取关键数据点"""
        extractor = self.EXTRACTORS.get(bottleneck_type)
        if not extractor:
            return self._extract_generic(combined_ops, bottleneck_op_id)
        return extractor(combined_ops, bottleneck_op_id)

    # 每个 extractor 返回:
    # {
    #   'critical_data': str,       # 关键数据点 (注入 prompt)
    #   'relevant_playbook': str,   # 应注入的 playbook 文件名
    #   'relevant_section': str,    # 应注入的 playbook 章节
    #   'suggested_strategies': list  # 建议策略列表
    # }
```

### Playbook 细拆分

```
playbooks/
├── optimization_playbook.md        # 总纲 (索引 + 策略层级总览)
├── playbook_tiling.md             # Tier 1: Tiling (3 章)
│   ├── §1 BLOCK_SIZE 选择启发式
│   ├── §2 num_warps / num_stages 调优
│   └── §3 910B3 UB=192KB tile 上限计算
├── playbook_memory.md             # Tier 2: Memory (4 章)
│   ├── §1 910B3 各引擎带宽参数速查
│   ├── §2 小传输合并 (何时+如何合并)
│   ├── §3 coalescing 对齐 & double buffering
│   └── §4 UB 容量管理 (buffer 分配策略)
├── playbook_fusion.md             # Tier 3: Fusion (3 章)
│   ├── §1 融合机会识别 (读 RAW chain)
│   ├── §2 逐元素融合 (vadd+vmul+vrelu 合并)
│   └── §3 WAR 依赖打破 (独立 buffer 策略)
├── playbook_compute.md            # Tier 4: Compute (3 章)
│   ├── §1 VecUnit: 饱和参数 & 调优方法
│   ├── §2 CubeUnit: matrixmul 优化
│   └── §3 计算与传输重叠 (pipeline overlap)
├── playbook_910b3_arch.md         # Tier 5: 910B3 专属 (4 章)
│   ├── §1 核心配置速查 (20 AI Core + 40 Vec Core)
│   ├── §2 内存层级 (GM→L2→UB→L1→L0 大小+带宽)
│   ├── §3 Vector Pipeline vs Matrix Pipeline 选择
│   ├── §4 grid 数选择: transfer=20, compute=40
│   └── §5 L2 驻留策略 (L2=192MB shared)
└── playbook_algorithmic.md        # Tier 6: 算法重构 (按算子类型)
    ├── §1 Online Softmax (数值稳定 + 减少归约)
    ├── §2 Persistent Kernel (减少 launch overhead)
    └── §3 Split-K (大 K 维度分解)
```

---

## 1.7 910B3 专属优化手册核心内容 (示例)

`playbook_memory.md §1: 910B3 各引擎带宽参数速查` 必须精确到这种程度：

```markdown
# playbook_memory.md §1: 引擎带宽参数速查

## 饱和曲线模型
bw = vpeak × size_kb / (size_kb + k0)   (clamped to peak_clamp)
duration_ns = size_kb × 1024 / bw

## 参数表 (单核, GB/s)

| 引擎   | vpeak  | k0 (KB) | peak_clamp | 1KB 实际 | 10KB 实际 | 饱和点 |
|--------|--------|---------|------------|----------|-----------|--------|
| GM→UB  | 121.08 | 6.65    | 80.83      | 15.8     | 58.8      | >12KB  |
| UB→GM  | 190.19 | 10.72   | 76.67      | 16.2     | 46.7      | >30KB  |
| VecUnit| 461.0  | 4.50    | 404.0      | 83.8     | 317.9     | >24KB  |
| GM→L1  | 37.5   | 6.65    | 37.5       | —       | —         | placeholder |
| L1→L0  | 100.0  | 6.65    | 100.0      | —       | —         | placeholder |
| CubeUnit| 150.0 | 0       | 150.0      | 150.0    | 150.0     | flat    |
| L0→GM  | 37.5   | 6.65    | 37.5       | —       | —         | placeholder |

## 关键结论
1. **GM→UB 做小传输很亏**: 1KB tile 只有 15.8 GB/s (19.5%峰值), 合并到 >12KB 可达到 78+ GB/s
2. **UB→GM 饱和更慢**: k0=10.72KB, 需要更大的 tile 才能饱和 (>30KB)
3. **VecUnit 最容易饱和**: k0=4.5KB, tile>24KB 即达到 404 GB/s
4. **CubeUnit/M1→L1/L1→L0/L0→GM 全是 placeholder** — 不要基于它们做精确决策

## 聚合带宽 (多核并行)
- GM→UB 聚合: 20核 × 80.83 ≈ 1616 GB/s (接近 HBM 理论 1.54 TB/s)
- VecUnit 聚合: 40核 × 404 ≈ 16160 GB/s (≈ 8 TFLOPS fp16)
```

---

## 1.8 优化轨迹可视化 (Performance Trajectory Chart)

从 `optimization_journal.jsonl` 自动生成，参照 AutoKernel 的双面板图：

```
┌──────────────────────────────────────────────────────────────┐
│           Optimization Trajectory: vector_add_fp16_N65536     │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Top Panel: 累计加速比 (cumulative speedup)                   │
│  ─────────────────────────────────────                        │
│  1.5x │                                        ●─●─●─●      │
│       │                              ●─●─●──┘                │
│  1.3x │                    ●─●─●──┘                           │
│       │          ●─●─●──┘                                     │
│  1.1x │  ●─●──┘                                               │
│       │ ●  (●=Kept  ○=Reverted  ─=running best)              │
│  1.0x ├────────────────────────────────────────────           │
│       0    5   10   15   20   25   30   35   40               │
│                          Round                                │
│                                                               │
│  Bottom Panel: 延迟 (latency μs)                              │
│  ─────────────────────────────────────                        │
│   110 │ ○                        ○                            │
│       │  ●─●  ●─●  ○  ●─●─●     ···                          │
│   100 │       ○       ○      ●─●─●                          │
│       │                                                       │
│    90 │                                        ●─●─●─●      │
│       │ baseline (104.3 μs)                                  │
│    80 ├────────────────────────────────────────────           │
│       0    5   10   15   20   25   30   35   40               │
│                          Round                                │
│                                                               │
│  Legend: ● Kept (18)  ○ Reverted (5)  ─ Running Best         │
│  Final: 1.42× speedup, 73.4 μs (from 104.3 μs)              │
└──────────────────────────────────────────────────────────────┘
```

生成脚本 `feedback/trajectory_chart.py`:
- 从 `optimization_journal.jsonl` 读取每轮的 `cumulative_speedup`, `decision`, `hardware_speedup_actual`
- 用 `np.minimum.accumulate()` 计算 running-best 延迟曲线
- 用 `np.maximum.accumulate()` 计算 running-best 加速比曲线
- 绿色=Keep, 红色=Revert, 蓝色虚线=running best, 灰色虚线=baseline

---

## 2. 完整项目文件结构

```
triton_agent_optimizer/
│
├── main.py                              # 主入口
├── config.py                            # 全局配置（硬件参数、阈值、路径）
├── README.md                            # 使用文档
├── ARCHITECTURE_DESIGN.md               # 本文档
│
├── agents/                              # 智能体层
│   ├── __init__.py
│   ├── orchestrator.py                  # 调度器：协调整个优化循环
│   │   # class Orchestrator:
│   │   #   - run() → 完整优化循环
│   │   #   - run_single_round() → 单轮：Plan→Code→Verify→Decide→Record
│   │   #   - _manage_context() → 上下文管理（滑动窗口+摘要）
│   │   #   - _check_stop() → 停止条件判断
│   │   #   - _generate_final_report() → 最终报告
│   │
│   ├── planner.py                       # 规划智能体
│   │   # class PlannerAgent:
│   │   #   - analyze_bottleneck(dsl_pipeline) → 瓶颈诊断
│   │   #   - generate_round_plan(bottleneck, history, cases) → 本轮优化计划文档
│   │   #   - select_strategy(bottleneck, iteration, playbook) → 选择策略
│   │   #   - prioritize_optimizations(opportunities) → 排序优化方向
│   │
│   ├── coder.py                         # 编码智能体
│   │   # class CoderAgent:
│   │   #   - apply_optimization(kernel_code, strategy, plan) → 修改后代码
│   │   #   - generate_diff(original, optimized) → 本轮 diff
│   │   #   - validate_syntax(kernel_code) → 语法检查
│   │   #   原则：单文件变更、最小化改动、保持可回退
│   │
│   └── verifier.py                      # 验证智能体
│       # class VerifierAgent:
│       #   - verify_cpu(kernel_code, test_cases) → (pass, error_details)
│       #   - verify_simulator(kernel_code, dsl) → (pass, perf_estimate, bottleneck)
│       #   - verify_hardware(kernel_code) → (pass, real_perf, msprof_data)
│       #   - full_verification(kernel_code) → 三阶段完整验证结果
│
├── analyzers/                           # 分析层（对接 fusion_pipeline + simulator）
│   ├── __init__.py
│   ├── hivmir_analyzer.py              # HIVMIR 分析器
│   │   # class HIVMIRAnalyzer:
│   │   #   - extract_from_compiler(kernel) → 从编译器提取 HIVMIR
│   │   #   - parse(hivmir_text) → 解析为结构化数据
│   │   #   - extract_dependencies() → 提取 RAW/WAR/WAW 依赖
│   │   #   - extract_variable_info() → 变量名、数据大小
│   │   #   - extract_op_sequence() → 操作序列与引擎归属
│   │
│   ├── msprof_analyzer.py              # msprof 分析器
│   │   # class MsprofAnalyzer:
│   │   #   - run_simulator(dsl_program) → 运行 cost model simulator
│   │   #   - parse_simulator_output() → 解析 --llm 输出的 7 个 section
│   │   #   - generate_dsl_from_kernel() → kernel → DSL 程序
│   │
│   ├── dsl_merger.py                   # DSL 数据合并
│   │   # class DSLMerger:
│   │   #   - merge(sim_ops, hivmir_ops) → CombinedOp 列表
│   │   #   - generate_pipeline_report() → 完整流水线报告
│   │   #      (对齐 simulator --llm 7-section 格式 + HIVMIR 字段)
│   │   #   - compute_engine_utilization()
│   │   #   - compute_critical_path()
│   │
│   ├── data_extractor.py               # 按需数据提取器 ★ NEW
│   │   # class DataExtractor:
│   │   #   - extract(combined_ops, bottleneck_type, op_id) → 精简关键数据
│   │   #   - 每种瓶颈类型对应一个 extractor 子函数
│   │   #   - 返回: critical_data + relevant_playbook + suggested_strategies
│   │   #   原则: 200 op × 15 字段 → ~10 行关键数据注入 prompt
│   │
│   └── bottleneck_diagnoser.py         # 瓶颈诊断器
│       # class BottleneckDiagnoser:
│       #   - diagnose(combined_ops) → 瓶颈诊断报告
│       #   - classify_bottleneck(op) → 分类瓶颈类型
│       #   - assess_optimization_headroom(op) → 评估可优化空间
│       #   - compare_with_theoretical_peak(op, engine) → 与理论峰值对比
│
├── optimizers/                          # 优化策略实现
│   ├── __init__.py
│   ├── base_optimizer.py               # 优化器基类
│   │   # class BaseOptimizer:
│   │   #   - apply(kernel_code, params) → 修改后代码
│   │   #   - validate(kernel_code) → 是否可编译
│   │   #   - revert(kernel_code) → 回退
│   │
│   ├── tile_optimizer.py               # Tiling 优化
│   │   # - adjust_block_size() → 调整 BLOCK_SIZE
│   │   # - tune_num_warps() → 调整 num_warps
│   │   # - tune_num_stages() → 调整 num_stages（pipeline stages）
│   │   # - auto_tile_search() → 自动搜索最优 tile
│   │
│   ├── memory_optimizer.py             # 内存访问优化
│   │   # - coalesce_access() → 合并内存访问
│   │   # - merge_transfers() → 合并相邻传输（小→大）
│   │   # - prefetch_memory() → 预取优化
│   │   # - reduce_register_pressure() → 减少寄存器压力
│   │   # - use_faster_memory_level() → 使用更快的存储层级
│   │   # - optimize_data_layout() → 数据布局优化
│   │
│   ├── fusion_optimizer.py             # 算子融合
│   │   # - fuse_elementwise() → 融合逐元素操作
│   │   # - fuse_reduction() → 融合归约操作
│   │   # - fuse_activation() → 融合激活函数
│   │   # - eliminate_intermediate_buffer() → 消除中间 buffer (WAR→独立)
│   │   # - break_dependency_chain() → 打破依赖链
│   │
│   └── compute_optimizer.py            # 计算优化
│       # - optimize_vectorization() → 向量化优化
│       # - optimize_matmul_tiling() → 矩阵乘法 tiling
│       # - use_tensor_cores() → 使用 tensor/cube core
│       # - pipeline_overlap() → 计算与传输重叠
│       # - persistent_kernel() → 持久化 kernel
│
├── execution/                           # 执行层
│   ├── __init__.py
│   ├── emulator_runner.py              # CPU Emulator 运行器
│   │   # class EmulatorRunner:
│   │   #   - run(kernel_code, test_inputs) → (pass, output, trace)
│   │   #   - run_shape_sweep(kernel_code, shapes) → 多 shape 测试
│   │   #   - run_edge_cases(kernel_code) → 边界条件测试
│   │   #   - compare_with_reference(output, reference) → 数值对比
│   │   #   - generate_error_report() → 错误报告（line no, expected vs actual）
│   │   #   对接：emulators/common/__init__.py (tl, launch_kernel, verify)
│   │
│   ├── simulator_runner.py             # Cost Model Simulator 运行器
│   │   # class SimulatorRunner:
│   │   #   - run(dsl_program) → 完整模拟结果
│   │   #   - get_bottleneck() → 瓶颈操作
│   │   #   - get_critical_path() → 关键路径
│   │   #   - get_engine_utilization() → 引擎利用率
│   │   #   - get_perf_estimate() → 性能预估
│   │   #   对接：costModel/cost_emulator/simulator.py
│   │
│   ├── hardware_runner.py              # 910B3 真机运行器
│   │   # class HardwareRunner:
│   │   #   - compile_and_run(kernel_code) → 编译 + 运行
│   │   #   - collect_msprof() → 收集 msprof 性能数据
│   │   #   - extract_hivmir() → 提取 HIVMIR
│   │   #   - benchmark(iterations) → 性能基准测试
│   │   #   注意：此步骤在 emulator 验证通过后才执行
│   │
│   └── compiler.py                     # 编译器接口
│       # class CompilerInterface:
│       #   - compile_triton(kernel_code) → 编译 Triton → NPU 二进制
│       #   - extract_hivmir(kernel_code) → 提取 HIVMIR 中间表示
│       #   - check_compile_errors() → 编译错误检查
│
├── feedback/                            # 反馈与记录层
│   ├── __init__.py
│   ├── round_logger.py                 # 每轮日志记录器
│   │   # class RoundLogger:
│   │   #   - log_round(round_data) → 写入本轮完整数据
│   │   #   - log_code_diff(diff) → 记录代码变更
│   │   #   - log_verification(result) → 记录三阶段验证结果
│   │   #   - log_decision(decision, reason) → 记录 Keep/Revert 决策
│   │   #   输出格式: JSONL，每行一轮
│   │
│   ├── optimization_journal.py         # 优化日志（结构化 JSONL）
│   │   # class OptimizationJournal:
│   │   #   - append(entry) → 追加一条记录
│   │   #   - query(round_range) → 查询指定轮次
│   │   #   - get_summary() → 生成摘要
│   │   #   - get_performance_curve() → 性能变化曲线数据
│   │
│   ├── trajectory_chart.py             # 优化轨迹可视化 ★ NEW
│   │   # class TrajectoryChart:
│   │   #   - generate(journal) → 从 JSONL 生成双面板图
│   │   #   - 上图: 累计加速比 (running-best 曲线 + 每轮散点)
│   │   #   - 下图: 延迟 μs (running-best + baseline 虚线)
│   │   #   - 颜色: 绿=Keep, 红=Revert
│   │   #   - 输出: trajectory_<op_name>.png
│   │
│   ├── case_template.py               # 优秀案例模板生成器
│   │   # class CaseGenerator:
│   │   #   - should_generate(final_result) → 判断是否值得生成案例
│   │   #   - generate(kernel, journal, final_result) → 生成案例文档
│   │   #   - fill_template() → 填充模板
│   │   #   - publish_to_case_library() → 发布到案例库
│   │
│   └── stop_condition.py              # 停止条件检查器
│       # class StopChecker:
│       #   - check(journal, current_state) → 是否应停止
│       #   条件:
│       #     1. 连续 N 轮 Revert（默认5轮）→ 无改进空间
│       #     2. 达到理论峰值 90% → 接近硬件极限
│       #     3. 总轮次/时间预算耗尽 → 资源约束
│       #     4. 达到目标加速比 → 目标达成
│       #     5. 所有策略层级已尝试且无效果 → 策略耗尽
│       #     6. 瓶颈不可进一步优化（compute/bandwidth-bound at peak）
│
├── memory/                              # 上下文记忆系统
│   ├── __init__.py
│   ├── context_manager.py              # 上下文管理器
│   │   # class ContextManager:
│   │   #   - build_prompt_context(round_num, journal, cases) → 构建当前轮次上下文
│   │   #   - compact_old_rounds(old_rounds) → 旧轮次摘要压缩
│   │   #   - estimate_token_usage(context) → 估算 token 用量
│   │   #   - trim_context_if_needed(context, max_tokens) → 裁剪上下文
│   │   #   策略：
│   │   #     - 最近 5 轮：保留完整上下文（代码+diff+结果）
│   │   #     - 5~15 轮前：保留摘要（策略+加速比+决策）
│   │   #     - 15 轮之前：仅保留关键数据点（加速比、瓶颈类型）
│   │
│   ├── experience_retriever.py         # 经验检索
│   │   # class ExperienceRetriever:
│   │   #   - retrieve_similar_cases(kernel_fingerprint) → 相似案例
│   │   #   - retrieve_effective_strategies(bottleneck_type) → 有效策略
│   │   #   - retrieve_failed_approaches(bottleneck_type) → 已失败方法
│   │   #   对接：项目级 memory/ 模块 (fingerprint + retrieve + store)
│   │
│   └── sliding_window.py              # 滑动窗口
│       # class SlidingWindow:
│       #   - add_round(round_data) → 添加轮次
│       #   - get_recent_rounds(n) → 获取最近 N 轮完整数据
│       #   - get_window_summary() → 窗口摘要
│       #   - window_size: int = 5
│
├── playbooks/                           # 优化指导手册（LLM 上下文注入，细拆分 ~20 章节）
│   ├── optimization_playbook.md         # 总纲：策略层级总览 + 章节索引
│   │   # 6 层策略 (Tier 1~6) 概览 + 晋升规则 + 每层对应的手册章节
│   │
│   ├── playbook_tiling.md              # Tier 1: Tiling 优化 (3 章)
│   │   # §1 BLOCK_SIZE 选择启发式
│   │   # §2 num_warps / num_stages 调优
│   │   # §3 910B3 UB=192KB tile 上限计算
│   │
│   ├── playbook_memory.md              # Tier 2: Memory 优化 (4 章)
│   │   # §1 910B3 各引擎带宽参数速查 (vpeak/k0/clamp 表)
│   │   # §2 小传输合并 (何时触发 + 如何合并 + UB 容量约束)
│   │   # §3 coalescing & double buffering 模板
│   │   # §4 UB 容量管理 (buffer 分配策略 + overflow 预防)
│   │
│   ├── playbook_fusion.md              # Tier 3: Fusion 优化 (3 章)
│   │   # §1 融合机会识别 (读 RAW chain)
│   │   # §2 逐元素融合 (vadd+vmul+vrelu 等)
│   │   # §3 WAR 依赖打破 (独立 buffer → 解锁并行)
│   │
│   ├── playbook_compute.md             # Tier 4: Compute 优化 (3 章)
│   │   # §1 VecUnit: 饱和参数 + 调优 (fp16 vadd, vpeak=461/k0=4.5)
│   │   # §2 CubeUnit: matrixmul (placeholder, 待实测)
│   │   # §3 计算与传输重叠 (double buffer + pipeline overlap)
│   │
│   ├── playbook_910b3_arch.md          # Tier 5: 910B3 专属优化 (5 章)
│   │   # §1 核心配置: 20 AI Core (transfer) + 40 Vec Core (compute) @ 1.8 GHz
│   │   # §2 内存层级: GM 64GB → L2 192MB → UB 192KB/核 → L1 2MB/核 → L0 1MB/核
│   │   # §3 Vector Pipeline (GM→UB→VecUnit→UB→GM) vs Matrix Pipeline (GM→L1→L0→CubeUnit→L0→GM)
│   │   # §4 grid 数选择: transfer 用 20, compute 用 40
│   │   # §5 L2 驻留策略 (192MB shared, hit vs miss 行为差异)
│   │
│   └── playbook_algorithmic.md         # Tier 6: 算法重构 (按算子类型)
│       # §1 Online Softmax (数值稳定 + 减少归约)
│       # §2 Persistent Kernel (减少 launch overhead)
│       # §3 Split-K (大 K 维度分解)
│       # §4 FP32 accumulation for fp16 compute (精度保证)
│
├── cases/                               # 优秀案例库
│   ├── template.md                      # 案例模板
│   │   # 包含：算子信息 / 初始性能 / 优化过程总览 / 每轮详细记录 /
│   │   #       最终性能 / 关键变更清单 / 经验总结 / 适用场景
│   └── README.md                        # 案例库使用说明
│
└── tests/                               # 测试
    ├── __init__.py
    ├── test_kernels/                    # 测试用 kernel
    └── test_orchestrator.py             # Orchestrator 单元测试
```

---

## 3. 单轮优化详细流程

```
Round N 开始
│
├── Step 0: 上下文准备 (ContextManager)
│   ├── 构建滑窗上下文：最近5轮完整 + 前15轮摘要 + 更早轮次关键数据点
│   ├── 检索相似案例：从 experience_retriever 获取相关优化经验
│   ├── 注入 playbook 相关章节
│   └── 估算 token 用量，超限则裁剪
│
├── Step 1: 瓶颈重分析 (PlannerAgent)
│   ├── 如果本轮有新的 msprof/HIVMIR 数据 → 重新分析
│   ├── 否则复用上轮分析（若 Keep）或回退到上上轮（若 Revert）
│   └── 输出：当前瓶颈 op + 类型 + 时间占比
│
├── Step 2: 生成本轮优化计划 (PlannerAgent)
│   ├── 根据瓶颈类型 + 历史尝试 + 策略层级 选择优化方向
│   ├── 生成 《Round N 优化计划》文档：
│   │   ├── 当前瓶颈描述
│   │   ├── 本轮优化目标（预期加速比/效果）
│   │   ├── 具体优化手段（1个，少量改动）
│   │   ├── 预期变更范围（哪些参数/哪些行）
│   │   └── 验证方法（需跑什么测试）
│   └── 输出：round_N_plan.md
│
├── Step 3: 代码修改 (CoderAgent)
│   ├── 读取当前 kernel 代码
│   ├── 按计划做最小化代码改动（单文件）
│   ├── 生成 diff
│   └── 输出：optimized_kernel.py + round_N_diff.patch
│
├── Step 4: CPU Emulator 验证 (VerifierAgent → EmulatorRunner)
│   ├── 4a. 基础正确性：单 shape + 标准输入
│   ├── 4b. Shape sweep：多 shape 测试（小/中/大/非整除）
│   ├── 4c. 边界条件：dtype sweep、极端值、空张量
│   ├── 4d. 数值精度：与 PyTorch reference 对比（max_abs / max_rel）
│   │
│   ├── PASS → 进入 Step 5
│   └── FAIL → 记录错误详情 → 进入 Step 3 重试（最多 3 次）
│       └── 3 次全失败 → 本轮标记为 FAIL，Revert，记录原因
│
├── Step 5: Cost Simulator 预估 (VerifierAgent → SimulatorRunner)
│   ├── 生成 DSL 程序
│   ├── 运行 simulator（--llm --critical-path）
│   ├── 分析瓶颈变化
│   └── 输出：预估性能 + 瓶颈对比（优化前 vs 优化后）
│
├── Step 6: 910B3 真机验证 (VerifierAgent → HardwareRunner)
│   ├── 编译 kernel（Ascend 编译器）
│   ├── 运行基准测试（warmup + repeat）
│   ├── 收集 msprof 数据
│   ├── 提取 HIVMIR（供下轮分析使用）
│   └── 输出：真实加速比 + msprof 报告 + HIVMIR 文件
│
├── Step 7: 决策 (Orchestrator)
│   ├── 正确性: PASS + 加速比 > 1.01 → KEEP
│   ├── 正确性: PASS + 加速比 ≤ 1.01 → REVERT（无明显改进）
│   ├── 正确性: FAIL → REVERT（代码有 bug）
│   └── 更新 current_best_kernel
│
├── Step 8: 记录本轮 (RoundLogger)
│   ├── 写入 optimization_journal.jsonl：
│   │   {
│   │     "round": N,
│   │     "timestamp": "...",
│   │     "plan": "round_N_plan.md",
│   │     "strategy": "increase_tile_size",
│   │     "strategy_tier": 1,
│   │     "bottleneck_before": {"op": "op2", "type": "memory_bandwidth", "ratio": 0.47},
│   │     "bottleneck_after":  {"op": "op0", "type": "compute_vec", "ratio": 0.35},
│   │     "code_diff": "round_N_diff.patch",
│   │     "emulator_result": "PASS",
│   │     "simulator_speedup_est": 1.15,
│   │     "hardware_speedup_actual": 1.12,
│   │     "cumulative_speedup": 1.45,
│   │     "decision": "KEEP",
│   │     "decision_reason": "带宽利用率 46%→78%，GM→UB 传输合并有效"
│   │   }
│   └── 更新性能曲线数据
│
├── Step 9: 检查停止条件 (StopChecker)
│   ├── 连续 Revert ≥ 5? → STOP (diminishing returns)
│   ├── 达到理论峰值 90%? → STOP (hardware limit)
│   ├── 累计轮次 ≥ max_rounds? → STOP (budget exhausted)
│   ├── 加速比 ≥ target_speedup? → STOP (goal achieved)
│   ├── 所有策略层级已耗尽? → STOP (strategy exhausted)
│   └── 否则 → CONTINUE (进入 Round N+1)
│
└── Round N 结束
```

---

## 4. 验证三阶段详解

```
┌──────────────────────────────────────────────────────────────────┐
│                    Verification Pipeline                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Stage 1: CPU Emulator (秒级，每轮必跑)                           │
│  ─────────────────────────────────────                            │
│  输入: optimized_kernel.py                                        │
│  执行:                                                            │
│    a. 基础正确性: 单 shape 输入，输出 vs PyTorch reference        │
│    b. Shape Sweep:                                                │
│       - 小 shape (边界条件: 1, 3, 7)                              │
│       - 中 shape (典型值: 256, 512)                               │
│       - 大 shape (压力测试: 4096, 8192)                           │
│       - 非整除 shape (测试 mask 逻辑: 1025, 2049)                 │
│    c. Dtype Sweep: fp16, fp32, bf16 (如适用)                      │
│    d. 边界条件: 全零输入、极值 (±max)、空张量                     │
│    e. 数值精度: max_abs_error, max_rel_error, ULP 分布            │
│  输出: PASS/FAIL + 详细错误报告（行号、期望值、实际值）           │
│  对接: emulators/common/ (tl, launch_kernel, verify, TraceLogger) │
│  策略: 借鉴 "The Correctness Illusion" 论文的 seeded fuzzing      │
│        oracle — 多 shape + 多 dtype + 严格 tolerance              │
│                                                                   │
│  Stage 2: Cost Simulator (秒级，Stage 1 PASS 后跑)                 │
│  ─────────────────────────────────────                            │
│  输入: kernel → DSL program                                       │
│  执行:                                                            │
│    python simulator.py --llm --critical-path "<DSL program>"      │
│  输出:                                                            │
│    - total_ns (预估总时间)                                        │
│    - 瓶颈 op + time_ratio                                         │
│    - 关键路径 (critical path)                                     │
│    - 引擎利用率                                                   │
│    - 带宽利用率 (per op)                                          │
│  对接: costModel/cost_emulator/simulator.py                       │
│                                                                   │
│  Stage 3: 910B3 Hardware (分钟级，Stage 1+2 PASS 后跑)              │
│  ─────────────────────────────────────                            │
│  输入: optimized_kernel.py                                        │
│  执行:                                                            │
│    a. 编译 (Ascend 编译器)                                        │
│    b. 基准测试 (warmup=30, repeat=200)                            │
│    c. 收集 msprof 数据                                            │
│    d. 提取 HIVMIR                                                 │
│  输出:                                                            │
│    - 真实延迟 (ms) / 吞吐 (GB/s, TFLOPS)                          │
│    - msprof 报告 (时序、带宽、引擎利用率)                         │
│    - HIVMIR 文件 (供下轮分析)                                     │
│                                                                   │
│  特殊处理:                                                        │
│    - 编译失败 → 记录错误，Revert                                  │
│    - UB overflow → 提示 tile 太大，Revert + 调整策略             │
│    - 运行时错误 → 记录完整错误栈，Revert                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. 上下文管理策略

### 问题

一次优化可能 50~200 轮，每轮包含完整代码 + diff + 验证报告 + 决策记录。不管理的话上下文很快爆炸。

### 三层上下文策略

```
Layer 1: 热上下文 (Hot Context) — 最近 5 轮
  ├── 完整保留：kernel 代码、diff、验证结果、决策原因
  ├── 注入到每轮 LLM prompt 中
  └── 目的：保持优化方向连贯，避免反复尝试同一策略

Layer 2: 温上下文 (Warm Context) — 6~15 轮前
  ├── 摘要保留：策略名称、加速比、瓶颈类型、决策（Keep/Revert）
  ├── 注入到 planner prompt 中（作为历史参考）
  └── 目的：知道试过什么，防止重复

Layer 3: 冷上下文 (Cold Context) — 16 轮以上
  ├── 仅保留关键数据点：(round, speedup, bottleneck_type)
  ├── 不注入 prompt，但可通过 journal query 查询
  └── 目的：性能曲线、长期趋势分析

特殊注入 (每轮都注入):
  ├── Playbook 相关章节 (当前策略层级对应的手册)
  ├── 相似历史案例 (从 experience_retriever 检索)
  └── 当前 kernel 的指纹 + 最相关的 2~3 条经验
```

### 与现有 memory/ 模块的关系

```
triton_agent_optimizer/memory/context_manager.py
  ├── 调用 memory/retrieve.py → 检索相似算子经验
  ├── 调用 memory/schema.py → Fingerprint 数据结构
  └── 新增：滑窗 + 摘要压缩逻辑（memory/ 不做这些）

不修改 memory/ 包 — 只读调用其检索 API。
```

---

## 6. 优化策略层级 (6-Tier Playbook)

借鉴 AutoKernel 的 909 行 playbook，结合 910B3 硬件特点：

| Tier | 策略类别 | 典型手段 | 适用瓶颈 | 预期收益 |
|---|---|---|---|---|
| **1** | Block Size & Launch | 调整 BLOCK_SIZE, num_warps, num_stages | 带宽利用率低 (floor/ramp) | 大 (1.5~10×) |
| **2** | Memory Access | 合并小传输、coalescing、prefetch | memory_bandwidth, memory_latency | 中 (1.2~2×) |
| **3** | Operator Fusion | 融合逐元素操作、消除中间 buffer | dependency (RAW chain) | 中 (1.3~2×) |
| **4** | Compute Optimization | 向量化、tiling 调优、指令级优化 | compute_vec, compute_cube | 小~中 (1.1~1.5×) |
| **5** | 910B3 Architecture | Cube core 利用、UB 容量管理、L2 驻留 | engine_contention | 视情况 |
| **6** | Algorithmic Restructure | 换算法（如 online softmax）、persistent kernel | 以上都无效时 | 大但风险高 |

### 策略晋升规则

```
Tier N 中连续 3 轮无改进 → 晋升到 Tier N+1
Tier N 中有改进 → 继续在当前 Tier 深挖
Tier N 中瓶颈转移（如 memory→compute）→ 切换策略类别
所有 Tier 耗尽 → 触发停止条件
```

---

## 7. 停止条件

| # | 条件 | 默认阈值 | 说明 |
|---|---|---|---|
| 1 | 连续 Revert | 5 轮 | 连续失败 = 无改进空间 |
| 2 | 理论峰值 | 90% of peak | 已接近硬件物理极限 |
| 3 | 轮次预算 | 200 轮 | 单 kernel 最大优化轮次 |
| 4 | 时间预算 | 6 小时 | 单 kernel 最大优化时间 |
| 5 | 目标加速比 | 用户指定 | 达成目标即停止 |
| 6 | 策略耗尽 | 所有 Tier 尝试完毕 | 无更多优化手段可用 |
| 7 | 平台期检测 | 最近 10 轮加速比波动 < 2% | 性能已收敛 |

---

## 8. 优秀案例模板

当一个 kernel 优化完成后（累计加速比 ≥ 目标），自动生成案例文档：

```markdown
# 优化案例: {op_name}

## 基本信息
- 算子名称: `vector_add`
- 初始性能: 1200 GB/s (HBM 带宽利用率 74%)
- 最终性能: 1520 GB/s (HBM 带宽利用率 94%)
- 累计加速比: 1.27×
- 总优化轮次: 23 轮 (18 Keep, 5 Revert)
- 总耗时: 2.3 小时

## 优化过程总览

| 轮次 | 策略 | 变更 | 加速比 | 决策 |
|------|------|------|--------|------|
| 1-5 | Tile 增大 (Tier 1) | BLOCK_SIZE 256→2048 | 1.15× | Keep |
| 6-8 | 传输合并 (Tier 2) | 4×1KB → 1×4KB tile | 1.08× | Keep |
| 9-11 | 计算优化 (Tier 4) | 向量化对齐 | 1.02× | Keep |
| 12-15 | 融合尝试 (Tier 3) | 合并 vadd+vmul | — | Revert (UB overflow) |
| 16-23 | 架构优化 (Tier 5) | Double buffer + L2 驻留 | 1.03× | Keep |

## 关键变更

### 变更 1: BLOCK_SIZE 调整 (Round 1)
```diff
- BLOCK_SIZE = 256
+ BLOCK_SIZE = 2048
```
效果: 带宽利用率 46% → 78%，瓶颈从 GM→UB 转移到 VecUnit

### 变更 2: 传输合并 (Round 6)
```diff
- for m in range(0, 10, 1) { gm_to_ub(ub_1, gm_1 + m*1KB) }
+ gm_to_ub(ub_1, gm_1)  # 一次 10KB 传输
```
效果: 消除 9 次小传输开销，总时间减少 8%

## 经验总结

1. **先调 tile size 再调别的** — 这是收益最大的一步
2. **910B3 UB=192KB 注意** — 融合时容易 overflow
3. **小传输是杀手** — 1KB 传输只有 ~16 GB/s，合并到 10KB 上升到 ~60 GB/s
4. **WAR 可以通过独立 buffer 消除** — 不用改算法

## 适用场景
- 所有 memory-bound 的逐元素算子
- 小 tile size 场景（初始 BLOCK_SIZE < 1024）
```

---

## 9. 轮次日志 (Round Journal) JSONL Schema

每轮一行，JSONL 格式：

```json
{
  "round": 12,
  "timestamp": "2026-07-23T14:30:00",
  "kernel_fingerprint": "vadd_fp16_N65536_B2048",

  "plan": {
    "strategy": "increase_tile_size",
    "strategy_tier": 1,
    "target_speedup_this_round": 1.10,
    "plan_file": "rounds/round_012_plan.md"
  },

  "bottleneck_before": {
    "op_id": 2,
    "op_type": "gm_to_ub",
    "engine": "GM→UB",
    "time_ratio": 0.47,
    "bw_utilization": 0.46,
    "regime": "ramp"
  },

  "bottleneck_after": {
    "op_id": 1,
    "op_type": "vadd",
    "engine": "VecUnit",
    "time_ratio": 0.35,
    "bw_utilization": 0.88,
    "regime": "saturated"
  },

  "code_change": {
    "diff_file": "rounds/round_012_diff.patch",
    "lines_changed": 3,
    "files_changed": 1
  },

  "verification": {
    "stage1_emulator": {
      "passed": true,
      "shapes_tested": [256, 512, 1024, 1025, 65536],
      "dtypes_tested": ["fp16", "fp32"],
      "max_abs_error": 4.77e-07,
      "max_rel_error": 1.14e-04
    },
    "stage2_simulator": {
      "total_ns_before": 3655.57,
      "total_ns_after": 3180.12,
      "estimated_speedup": 1.15,
      "critical_path_changed": true
    },
    "stage3_hardware": {
      "passed": true,
      "actual_speedup": 1.12,
      "latency_us_before": 18.3,
      "latency_us_after": 16.3,
      "msprof_file": "rounds/round_012_msprof.json",
      "hivmir_file": "rounds/round_012_hivmir.mlir"
    }
  },

  "decision": "KEEP",
  "decision_reason": "GM→UB 带宽利用率从 46% 提升到 78%，总时间减少 12%",
  "cumulative_speedup": 1.45,
  "cumulative_rounds_kept": 10,
  "cumulative_rounds_reverted": 2
}
```

---

## 10. 与现有项目组件的对接关系

```
triton_agent_optimizer/          ← 本项目（新建）
│
├── analyzers/                   对接 →
│   ├── hivmir_analyzer.py       fusion_pipeline/extract_hivmir_from_compiler.py
│   ├── msprof_analyzer.py       fusion_pipeline/complete_data_merge.py
│   └── dsl_merger.py            costModel/cost_emulator/simulator.py
│
├── execution/                   对接 →
│   ├── emulator_runner.py       emulators/common/__init__.py
│   │                             emulators/test/<op>/__init__.py
│   └── simulator_runner.py      costModel/cost_emulator/simulator.py
│
├── memory/                      对接 →
│   └── experience_retriever.py  memory/retrieve.py, memory/schema.py
│
└── playbooks/                   独立（LLM 上下文注入用）
```

---

## 11. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 每轮改动粒度 | **单文件，最小化改动（1个参数/1个模式）** | 容易归因、容易回退、防止改乱 |
| 验证顺序 | **CPU → Simulator → Hardware** | 逐步过滤，节省硬件时间 |
| 上下文管理 | **滑窗 + 摘要 + 经验检索** | 保持连贯性同时控制 token 成本 |
| 经验记忆 | **复用现有 memory/ 模块** | 已有 fingerprint + retrieve + writeback |
| 策略表达 | **Markdown playbook 注入 LLM prompt** | 灵活、可迭代、LLM 可直接理解 |
| 优化方向规划 | **每轮先生成 plan 文档，再执行** | 防止盲目改动，提供可追溯的决策链 |
| 硬件参数 | **从 simulator.py SATURATION_PARAMS 读取** | 唯一实测来源，避免硬编码 |
| 停止条件 | **多条件 OR，满足任一即停** | 防止无限循环 |

---

## 12. 实现路线图

### Phase 1: 核心闭环 (当前)
- [ ] 完善 `orchestrator.py` 单轮流程
- [ ] 实现 `round_logger.py` + `optimization_journal.py`
- [ ] 实现 `stop_condition.py`
- [ ] 对接 `emulator_runner.py` → `emulators/common/`
- [ ] 对接 `simulator_runner.py` → `costModel/cost_emulator/simulator.py`

### Phase 2: 上下文与记忆
- [ ] 实现 `context_manager.py` (滑窗+摘要)
- [ ] 实现 `experience_retriever.py` (对接 memory/)
- [ ] 实现 `case_template.py` (案例生成)

### Phase 3: 策略实现
- [ ] 实现 `tile_optimizer.py` (Tier 1)
- [ ] 实现 `memory_optimizer.py` (Tier 2)
- [ ] 实现 `fusion_optimizer.py` (Tier 3)
- [ ] 实现 `compute_optimizer.py` (Tier 4)

### Phase 4: Playbook 编写
- [ ] `optimization_playbook.md` (总纲)
- [ ] `playbook_tiling.md`
- [ ] `playbook_memory.md`
- [ ] `playbook_fusion.md`
- [ ] `playbook_compute.md`

### Phase 5: 真机集成
- [ ] 实现 `hardware_runner.py`
- [ ] 实现 `compiler.py` (HIVMIR 提取)
- [ ] 910B3 端到端测试
