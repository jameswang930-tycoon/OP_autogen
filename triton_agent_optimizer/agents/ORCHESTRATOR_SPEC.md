# Orchestrator 实现规范 — AI Agent 执行手册

> 本文档是**强制规范**。实现 Orchestrator 时必须严格遵守。
> 配套阅读: `ARCHITECTURE_DESIGN.md` (架构), `ORCHESTRATOR_LOGIC.md` (逻辑摘要)。

---

## 0. 铁律 (不可违反)

### 0.1 六层必须依次执行，不能跳，不能乱序

```
Tier 1 → Tier 2 → Tier 3 → Tier 4 → Tier 5 → Tier 6
Algorithm → Fusion → Tiling → Memory → Compute → Architecture
```

**只有满足晋升条件才能进入下一 Tier。绝对不能跳过。**

晋升条件: 同一 Tier 内**连续 3 轮无改进** (speedup 变化 < 1% 或连续 REVERT)。

### 0.2 每轮 Planner 只能读自己 Tier 的文档

| 当前 Tier | 只能读这个文件 | 绝对不能读 |
|---|---|---|
| Tier 1 | `docx/playbook_tier1_algorithm.md` | playbook_tier2/3/4/5/6 |
| Tier 2 | `docx/playbook_tier2_fusion.md` | playbook_tier1/3/4/5/6 |
| Tier 3 | `docx/playbook_tier3_tiling.md` | playbook_tier1/2/4/5/6 |
| Tier 4 | `docx/playbook_tier4_memory.md` | playbook_tier1/2/3/5/6 |
| Tier 5 | `docx/playbook_tier5_compute.md` | playbook_tier1/2/3/4/6 |
| Tier 6 | `docx/playbook_tier6_architecture.md` | playbook_tier1/2/3/4/5 |

**原因**: Tier 1 (Algorithm) 不需要知道 Tier 3 (Tiling) 的 K0 参数表——看了也没用，还可能误导。每层只关注自己该做的事情。

### 0.3 每轮 Planner 只能检索自己 Tier 的经验库

| 当前 Tier | 经验库文件 |
|---|---|
| Tier 1 | `memory/experiences/tier1_algorithm.json` |
| Tier 2 | `memory/experiences/tier2_fusion.json` |
| Tier 3 | `memory/experiences/tier3_tiling.json` |
| Tier 4 | `memory/experiences/tier4_memory.json` |
| Tier 5 | `memory/experiences/tier5_compute.json` |
| Tier 6 | `memory/experiences/tier6_architecture.json` |

**原因**: Tier 1 看到 "merge_small_transfers 在 Tier 4 成功了 1.3x" 毫无意义——Tier 还没到那里。分层隔离防止噪声。

### 0.4 禁止使用模拟数据

所有 msprof trace.json 和 HIVMIR .mlir 必须来自真实编译器/硬件采集。
**不能用 `cost_emulator/simulator.py` 的 DSL 输出来冒充真实数据。**
simulator 仅在本地无 CANN 环境时作为临时 fallback，不可用于最终决策。

---

## 1. 完整执行流程

### 1.1 Round 0: 基准分析

```
Orchestrator._run_round0():
  ① 拷贝 kernel.py → outputs/<kernel>/round0/
  ② 运行完整分析链:
     a. compiler.py: 编译 kernel → .o binary + HIVMIR .mlir
     b. msprof op simulator ./binary → trace.json
     c. msprof_analyzer.parse(trace.json) → pipeline_report.json
     d. hivmir_analyzer.parse(.mlir) → hivmir_report.json
     e. dsl_merger: pipeline + hivmir → merged_report.json
     f. bottleneck_diagnoser.diagnose(merged, tier=1) → baseline diagnosis
  ③ RecordManager.init_baseline() → optimization_trajectory.json
```

**验证**: `round0/merged/merged_report.json` 存在，29 字段全填。`optimization_trajectory.json` 有 baseline 条目。

### 1.2 Round N: 优化轮

```
Orchestrator._run_one_round():
  tier = trajectory.state.tier   ← 从状态文件读取当前 Tier
  rn = trajectory.state.round + 1
  rd = outputs/<kernel>/TIER_DIRS[tier]/roundN/   ← 自动选目录

  ① ANALYZE: 重新分析当前 kernel
     _run_analyzers(rd): 同 round0 的 6 步分析链
     输出: merged_report + BottleneckDiagnosis + extracted_text

  ② PLAN: 生成优化计划
     必须使用当前 Tier 的 playbook 和经验库 (铁律 0.2 + 0.3)
     _call_planner(diag, extracted, tier, history):
       有 API key → PlannerAgent.generate() (LLM)
       无 API key → 写 AGENT_TASK_PLAN.md (自包含任务文件)
     输出: rd/plan.md + rd/plan.json

  ③ CODE: 修改代码
     _call_coder(plan, kernel_code):
       有 API key → CoderAgent.apply() (LLM)
       无 API key → 写 AGENT_TASK_CODE.md
     输出: rd/kernel.py + rd/diff.patch

  ④ VERIFY: 验证正确性+性能
     _verify_with_retry(optimized_code):
       Stage 1: CPU Emulator → PASS/FAIL
         FAIL → 错误回传 Coder → 重新修改 (最多 3 次)
         PASS → Stage 2
       Stage 2: 910B3 Hardware (本地跳过)
     输出: rd/verification.json

  ⑤ DECIDE: KEEP or REVERT
     speedup > 1.01? → KEEP (更新 current_kernel)
     speedup ≤ 1.01? → REVERT (current_kernel 不变)

  ⑥ RECORD
     RecordManager.evaluate():
       - 写 rd/optimization_record.json
       - 更新 optimization_trajectory.json (state + history)
       - 检查停止条件 → CONTINUE / STOP
       - 记录经验: speedup>1.05→SUCCESS, speedup<0.98→FAIL
```

### 1.3 Tier 晋升

```python
StopChecker.check():
    # 规则: 连续 5 轮 REVERT → 晋升
    if history[-5:] all REVERT:
        if tier >= 6: return STOP
        else: tier += 1

    # 其他停止条件:
    # - 平台期 (最近10轮 speedup 波动 < 2%)
    # - 轮次预算耗尽 (max_rounds)
    # - 目标达成 (best_speedup >= target)
    # - Tier6 + 3连败
    # - 连续10轮无改进
```

**晋升时重置计数器**:
```python
state.consecutive_reverts = 0
state.consecutive_no_improvement = 0
```

---

## 2. 每个 Tier 的职责边界

### Tier 1: Algorithmic Structure

**目标**: 确认当前算法是否最优。不调参数。

**能做的事**:
- 对照 playbook_tier1 §算子→算法对照表，判断是否需要换算法
- 检查 execution_mode (sequential→persistent kernel?)
- 检查 op_count (过多→算法重构?)

**不能做的事**:
- ❌ 不能修改 BLOCK_SIZE (那是 Tier 3)
- ❌ 不能融合算子 (那是 Tier 2)
- ❌ 不能改 num_warps (那是 Tier 3)

**晋升条件**: 确认当前算法已最优 (每轮 plan 写 "algorithm already optimal")

### Tier 2: Operator Fusion

**目标**: 融合可合并的 op，消除中间 GM 读写。不调参数。

**能做的事**:
- 读 dependencies_summary.raw → 找可融合的 RAW 链
- 读 dependencies_summary.war → 打破 WAR 假依赖
- 读 buffers → 检查 producer/consumer 关系
- 检查融合后 UB 容量 (sum ≤ 192KB)

**不能做的事**:
- ❌ 不能调 BLOCK_SIZE (那是 Tier 3)
- ❌ 不能改算法 (那应该回退 Tier 1)

**晋升条件**: 确认所有可融合 op 已融合，或 UB 容量不允许更多融合

### Tier 3: Tiling & Block Config

**目标**: 调整 BLOCK_SIZE / num_warps / num_stages。只在融合后的稳定结构上做。

**能做的事**:
- 读 critical_path 上 op 的 bw_utilization + regime
- 找 regime=floor/ramp 的 op → 增大 tile
- 对照 playbook_tier3 §K0参数表 → 计算目标 tile size
- 调整 num_warps (1~8) 和 num_stages (0~4)

**不能做的事**:
- ❌ 不能融合算子 (那是 Tier 2)
- ❌ 不能改算法 (那应该回退 Tier 1)

**晋升条件**: 连续 3 轮无改进，或所有传输 op 已饱和

### Tier 4: Memory Access

**目标**: 传输已饱和 → 优化数据存取模式。不调 tile size。

**能做的事**:
- 合并小传输 (多个 <k0 的传输 → 一次大传输)
- Double buffering (传输和计算重叠)
- Coalescing (内存对齐)

**不能做的事**:
- ❌ 不能调 BLOCK_SIZE (那是 Tier 3，而且饱和了调也没用)

**晋升条件**: 连续 3 轮无改进

### Tier 5: Compute & Occupancy

**目标**: 计算效率优化。非计算密集型 op 可能很快跳过。

**能做的事**:
- 计算-传输重叠 (double buffer 变体)
- 向量化对齐 (SIMD 256-bit)
- 精度取舍 (fp16 compute + fp32 accumulate)

**不能做的事**:
- ❌ 不能调传输参数 (那是 Tier 3/4)

**晋升条件**: 连续 3 轮无改进

### Tier 6: 910B3 Architecture

**目标**: 硬件专属微调。全部通用手段用尽后才到这里。

**能做的事**:
- 调整 grid (transfer=20, compute=40)
- Pipeline 切换 (Vector ↔ Matrix)
- L2 驻留策略

**警告**: Engine 3-6 (GM→L1/L1→L0/CubeUnit/L0→GM) 参数是 PLACEHOLDER。
基于 PLACEHOLDER 的优化建议必须标注 UNCERTAIN。

**晋升条件**: 连续 3 轮无改进 → **停止优化**

---

## 3. 验收标准

### 3.1 代码验收

```bash
# 必须跑通
python main.py input/rms_norm_residual/triton_kernel.py --max-rounds 5

# 输出目录必须存在
ls outputs/rms_norm_residual/
# round0/ 01_algorithmic_structure/ ... optimization_trajectory.json final_output/
```

### 3.2 功能验收

| # | 检查项 | 验收方法 |
|---|---|---|
| 1 | 6 个 Tier 目录全部创建 | `ls outputs/<kernel>/` 有 01~06 开头的目录 |
| 2 | round0 分析产物完整 | `round0/merged/merged_report.json` 29 字段全填 |
| 3 | trajectory.json 每轮更新 | `cat optimization_trajectory.json` 的 history 每轮一条 |
| 4 | Tier 晋升自动触发 | 连续 5 REVERT → state.tier 自动 +1 |
| 5 | Playbook 隔离 | Tier 3 的 Planner 不能读 playbook_tier4 |
| 6 | 经验库隔离 | Tier 3 只检索 tier3_tiling.json |
| 7 | 停止条件触发 | 达到 max_rounds → 自动停止 |
| 8 | 最终产物生成 | `final_output/` 有 optimized_kernel.py + trajectory_chart.png |

### 3.3 数据验收

| # | 检查项 | 验收方法 |
|---|---|---|
| 1 | 不使用模拟数据 | msprof trace.json 来自真实编译器，非 simulator.py |
| 2 | SATURATION_PARAMS 已实测 | dsl_merger.py 中 Engine 3-6 不是 PLACEHOLDER |
| 3 | HIVMIR 真实 | .mlir 来自 Ascend 编译器，非手工编写 |

### 3.4 单轮完整性验收

取任意一轮 (如 `03_tiling_block_config/round5/`)，检查以下文件存在:

- [x] `kernel.py`
- [x] `plan.md` + `plan.json`
- [x] `diff.patch`
- [x] `optimization_record.json`
- [x] `verification.json`
- [x] `msprof/pipeline_report.json`
- [x] `hivmir/hivmir_report.json`
- [x] `merged/merged_report.json`
- [x] `merged/final_report_llm.txt`
- [x] `merged/final_report_human.txt`

---

## 4. 可以改的 vs 绝对不能改的

| 可以改 | 绝对不能改 |
|---|---|
| `analyzers/` 中的正则提取规则 (适配真实数据格式) | 29 字段 JSON schema |
| `config.py` 中的路径/阈值 | 6 Tier 的顺序 |
| `compiler.py` 中的编译器路径/参数 | Playbook 隔离规则 (每 Tier 只看自己的) |
| LLM model 名称 | 经验库隔离规则 |
| `execution/` 中的 benchmark 实现 | Plan→Code→Verify→Record 流程 |
| Playbook 文档内容 (补充细节) | 晋升规则 (连续 3 轮无改进→Tier+1) |
| SATURATION_PARAMS 参数值 (实测后填入) | KEEP/REVERT 阈值 (1.01) |

---

## 5. 参考文件速查

| 看什么 | 去哪里 |
|---|---|
| 完整架构图 | `ARCHITECTURE_DESIGN.md` §1 |
| 每轮执行流程 | `ARCHITECTURE_DESIGN.md` §2 |
| 输出目录结构 | `OUTPUT_STRUCTURE.md` |
| 文件架构 | `README.md` §4 |
| 调度逻辑摘要 | `agents/ORCHESTRATOR_LOGIC.md` |
| 6 Tier 方法论 | `docx/OPTIMIZATION_METHODOLOGY.md` |
| 每层 Playbook | `docx/playbook_tier1~6_*.md` |
| 各层部署指南 | `analyzers|agents|execution|feedback|memory/DEPLOYMENT_GUIDE_910B3.md` |
| Agent 使用方式 | `prepare/AGENT_USAGE_GUIDE.md` |
