# Triton Agent Optimizer v4 — 架构设计（按当前实现）

> **核心思路**: 不靠盲试（AutoKernel 300~400 轮），而是用 **真机 msprof 双源采集（通用 msprof + msprof op）** 精确诊断每个 kernel 的耗时/带宽/算力/瓶颈 → 6 层优化策略逐层推进 → 每轮**端到端 msprof 实测**决定保留/回退/晋升。
>
> **环境**: Ascend 910B3 (NPU) + CANN 9.0 + triton-ascend + nga (本地 LLM codeagent)
> **更新**: 2026-08-06 — 按 v4 实际实现重写（单文件驱动 + 端到端验证 + keep/revert/promote）

---

## 0. 一句话总结

```
输入单个 kernel_op.py（config+kernel+test 一体）
  → 采集+解析出 diagnosis.json（骨架+deep+roofline）
  → 按当前 tier 筛字段喂 Planner(LLM)
  → Planner 出 changes[]（old_code→new_code 精确替换）+ promote 决策
  → Coder 确定性应用 changes[] → 本轮 kernel_op.py
  → 端到端 msprof 实测 → 加速比（vs 初始基线）
  → 对比上一轮决定 KEEP / REVERT / 晋升 / 回退 / 停止
```

---

## 1. 整体数据流（v4 闭环）

```
input/<op>/kernel_op.py            (源单文件: ① config + ② kernel + ③ test main)
   │  main.py --fresh 可选清 outputs/<op>
   ▼
Scheduler (agents/scheduler.py) — 状态机主循环 (tier 1~6 × round N)
   │
   ├─① 采集+解析  _run_optimize()
   │    bash analyzers/run_optimize.sh <input_dir> <round_dir> M N K   (TIER env)
   │      ├─ warmup 裸跑 (JIT 预热)
   │      ├─ 通用 msprof → pipeline_parse_task.py → task.json      (骨架: 每kernel耗时/launch/api)
   │      ├─ 逐 kernel msprof op → pipeline_parse_board.py → board_<i>.json (deep: 带宽/L2/cube/conflict)
   │      ├─ integrate.py → diagnosis.json   (骨架+deep 合并, roofline 用 hardware_peak.json 校准)
   │      ├─ 07_tier<N>_fields/*.txt|.json  (extract_tier_fields 筛出全局+当前tier字段)
   │      ├─ check_fields.py → field_check.log   (字段校验明细)
   │      └─ (仅 TIER==2) MULTI 路径: bishengir-compile → HIVM → filter_hivm_for_fusion.py
   │            → 08_fusion/hivm_fusion_view.txt
   │
   ├─(round1 基准) baseline_ns/num_kernels/baseline_mnk/initial_tflops/pytorch_tflops
   │    + verify_end_to_end 复测源 kernel (warmup + 1×msprof KERNEL_LOOP) → 同口径 baseline
   │
   ├─② 诊断筛字段  _diagnose() → 07_tier<N>_fields (Planner 只读这个)
   ├─(Tier2 多一步) _run_fusion() → run_hivm_fusion.py → 08_fusion/fusion_analysis.json
   │
   ├─③ Planner  _plan()  (agents/planner.py generate_v4)
   │    读: skills/triton-op-planner/SKILL.md + docx/playbook_tier<N>.md + 当前 kernel_op.py
   │        + 07 字段 + config 常量 + 历史(前层进度) [+ 融合分析]
   │    出: roundN/plan.md  (JSON: strategy, changes[], promote, promote_to)
   │
   ├─④ Coder  _code()  (agents/coder.py apply)
   │    确定性应用 changes[] (old_code→new_code 逐字符替换全部出现处) + 语法检查
   │    出错/LLM 超时 → 带错误回传 LLM 修复 (≤3 次)
   │    出: roundN/kernel_op.py + diff.patch
   │
   ├─⑤ 验证  _verify()  (agents/verifier.py verify_end_to_end)
   │    warmup(3) + 一次 msprof 内循环 KERNEL_LOOP(30) 次
   │    → 非 aclnn 目标 kernel 耗时之和 ÷ 实测遍数 = 单次端到端 ns
   │    → speedup = baseline_ns / ns
   │    (源码无 KERNEL_LOOP 循环 → 自动改除实测遍数, 防虚高)
   │
   ├─⑥ 决策  加速比 = 初始基线/本轮 (累计输出)
   │    speedup ≥ prev_speedup(上一轮已接受) → KEEP, 进 kernel 链
   │    speedup <  prev_speedup → REVERT, 沿用上一轮 kernel
   │    记录 history[] {round,tier,strategy,change,speedup,prev_speedup,ns,decision,result,error,tflops}
   │
   └─⑦ 晋升/停止
        planner.promote+promote_to → 晋升/回退目标层 (支持回退前层)
        或 本 tier 连续 3 轮无改进 → 晋升下一层
        speedup ≥ target → 停止; Tier6 无改进 → 停止
   │
   ▼
outputs/<op>/optimization_trajectory.json   (全局状态 + history, 每轮落盘)
   │
   ▼
feedback/trajectory_chart.py → final_output/trajectory_chart.png
   (加速比曲线 + 各 tier 色带 + KEEP/REVERT 点 + PyTorch 虚线[自动读 bench json])
```

---

## 1A. 完整闭环架构（分层图，v4 更新）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                 │
│   input/<op>/kernel_op.py   (config+kernel+test 一体, 源文件)            │
│   main.py --fresh / --resume / --target X / --max-rounds N / --stub      │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                   ANALYSIS LAYER (run_optimize.sh 采集链)                │
│  通用 msprof ──────┐       逐 kernel msprof op ──────┐                   │
│  op_summary.csv    │       OPPROF 8 CSV               │                   │
│       ▼            │            ▼                     │                   │
│  pipeline_parse    │       pipeline_parse             │                   │
│  _task.py          │       _board.py                  │                   │
│  → task.json 骨架  │       → board_<i>.json deep      │                   │
│  (kernel/launch/   │       (带宽/L2/cube/conflict)    │                   │
│   api_overhead)    │                                  │                   │
│       └────────────┴────────────┬─────────────────────┘                   │
│                                 ▼                                        │
│   integrate.py → diagnosis.json   (骨架+deep 合并)                        │
│   (roofline 峰值用 bench_910b3/hardware_peak.json 校准)                   │
│                                 ▼                                        │
│   extract_tier_fields → 07_tier<N>_fields/*.txt|.json                    │
│   (全局摘要[前层信号] + 当前 tier 专属字段)                               │
│   check_fields.py → 05_task/field_check.log                              │
│   (TIER==2) bishengir-compile → filter_hivm → 08_fusion/                 │
│   run_hivm_fusion.py → 08_fusion/fusion_analysis.json                    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │ Planner (LLM)  │  │  Coder (确定性)│  │  Verifier      │             │
│  │ ────────────── │  │  ───────────── │  │  ─────────────  │             │
│  │ 读 SKILL +     │  │ 应用 changes[]  │  │ warmup(3) +    │             │
│  │ playbook_tierN │  │ old→new 逐字符 │  │ 1×msprof       │             │
│  │ + 07字段       │  │ 替换全部出现处 │  │ KERNEL_LOOP(30) │             │
│  │ + 当前kernel   │  │ +LLM修错(≤3)   │  │ → 非aclnn和÷   │             │
│  │ + 历史(前层)   │  │ 语法检查       │  │  实测遍数       │             │
│  │ → plan.md      │  │ → roundN/      │  │ → ns/speedup   │             │
│  │  (changes[]+   │  │   kernel_op.py │  │  (循环丢自校正) │             │
│  │   promote_to)  │  │   +diff.patch  │  │                │             │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘             │
│          └───────────────────┼───────────────────┘                       │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │           Scheduler (Python 状态机 — 决策核心)                │       │
│  │  KEEP: speedup ≥ prev_speedup (进 kernel 链)                  │       │
│  │  REVERT: speedup < prev_speedup (沿用上一轮)                   │       │
│  │  promote_to 晋升/回退 (支持回退前层) · 本层无改进3轮兜底晋升    │       │
│  │  target 达标停 · Tier6 无改进停                                │       │
│  │  → optimization_trajectory.json (每轮落盘, --resume 可续)      │       │
│  └──────────────────────────────────────────────────────────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     EXECUTION LAYER (910B3 真机)                         │
│   triton-ascend 编译 · msprof 通用采集 · msprof op 逐 kernel ·           │
│   msprof 端到端验证 (warmup + KERNEL_LOOP 内循环)                        │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT LAYER                                │
│  outputs/<op>/<tier_name>/roundN/                                        │
│    {kernel_op.py, diff.patch, plan.md, 07_tier*_fields/, msprof_0/}      │
│  outputs/<op>/optimization_trajectory.json  (★ 中枢状态)                 │
│  outputs/<op>/final_output/trajectory_chart.png  (加速比曲线+PyTorch线)   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 1B. 完整数据流图（端到端，v4）

```
input/<op>/kernel_op.py
   │  (round1 采集源; 之后采集上一轮输出目录 — kernel 链连续)
   ▼
Scheduler._run_optimize → bash analyzers/run_optimize.sh <input> <round> M N K
   ├─① warmup 裸跑 (JIT 预热)
   ├─② 通用 msprof → task.json            (骨架: 每kernel耗时/launch/api_overhead/l2)
   ├─③ 逐 kernel msprof op → board_<i>.json  (deep: 带宽/引擎/conflict/fops)
   ├─④ integrate.py → diagnosis.json      (骨架+deep 合并 + roofline)
   ├─⑤ extract_tier_fields → 07_tier<N>_fields
   └─⑥ (TIER==2) HIVM → 08_fusion/fusion_analysis.json
   │
   ▼
round1 基准: baseline_ns / num_kernels / baseline_mnk / initial_tflops / pytorch_tflops
   + verify_end_to_end 复测源 kernel (warmup+msprof, 与后续轮同口径)
   │
   ▼
_diagnose → 07 字段 (全局摘要 + 本层字段)
   │
   ▼
Planner.generate_v4 → plan.md (JSON: strategy + changes[] + promote + promote_to)
   │
   ▼
Coder.apply → roundN/kernel_op.py + diff.patch   (old_code→new_code 确定性替换)
   │
   ▼
verify_end_to_end → warmup + 1×msprof KERNEL_LOOP → 非aclnn求和÷实测遍数 → ns → speedup
   │
   ▼
KEEP / REVERT (对比 prev_speedup) + history 记录 (含每轮 tflops) + promote 决策
   │
   ▼
optimization_trajectory.json
   │
   ▼
feedback/trajectory_chart.py → final_output/trajectory_chart.png
   (加速比曲线 · 各 tier 色带 · KEEP/REVERT 点 · PyTorch 虚线[自动读 bench json])
```

---

## 2. 逐轮执行流程（Scheduler 状态机）

```
main.py input/<op> [--fresh] [--resume] [--max-rounds N] [--target X] [--stub]
  │
  ├─ 单文件 kernel_op.py 若缺 → merge_single_file.py 生成
  ├─ --fresh → 清空 outputs/<op>
  ├─ _Tee → stdout/stderr 双写终端 + outputs/<op>/optimization.log (UTF-8)
  └─ Scheduler(op_dir, max_rounds, target, stub, resume)

while rn <= max_rounds:                     # 每轮一个 round_dir
    round_dir = outputs/<op>/<tier_name>/roundN
    diagnosis = _run_optimize(round_dir, tier)   # 采集失败→重试1次→跳过→连3次停(H2)
    if baseline_ns is None:                      # 首轮设基准
        baseline_ns / num_kernels / baseline_mnk / initial_tflops / pytorch_tflops
        + verify_end_to_end 复测源 kernel (VERIFY_BASELINE=1) → 与后续轮同口径
    extracted = _diagnose(diagnosis, tier, round_dir)   # → 07_tier<N>_fields
    if tier == 2: fusion_analysis = _run_fusion(round_dir)   # → 08_fusion/
    plan = _plan(diagnosis, extracted, tier, rn, round_dir, fusion_analysis)
    if plan.promote:                                 # 晋升轮: 原样拷贝 kernel, 不调 LLM
        copy current_kernel → roundN/kernel_op.py
    else:
        for attempt in range(3):
            new_code = _code(plan, rn, round_dir, prev_err)   # → roundN/kernel_op.py
            v = _verify(round_dir, baseline_ns)               # 端到端 msprof
            if v.ok: break
    speedup = baseline_ns / ns
    if speedup >= prev_speedup: current_kernel = roundN/kernel_op.py; KEEP
    else: REVERT (沿用上一轮)
    history.append(...); _save_traj()
    # 晋升/停止判定 → 下一轮/下一 tier/停止
```

---

## 3. 文件职责总览（谁产出 → 谁消费）

| 文件 | 角色 | 产出 → 消费 |
|---|---|---|
| `main.py` | 入口 | `_Tee` 日志 → optimization.log; 创建 kernel_op.py; 启动 Scheduler |
| `agents/scheduler.py` | **调度状态机** (核心) | 驱动每轮; 读 diagnosis.json → 写 07_tier 字段 / plan.md 触发 / roundN 目录 / optimization_trajectory.json |
| `agents/planner.py` | Planner (LLM 决策) | 读 SKILL+playbook+07字段+kernel+历史 → 写 plan.md (changes[]+promote) |
| `agents/coder.py` | Coder (确定性改码+LLM修复) | 读 plan.md changes[] + 当前 kernel → 写 roundN/kernel_op.py + diff.patch |
| `agents/verifier.py` | 验证 (端到端 msprof) | `verify_end_to_end` 读 roundN/kernel_op.py → 返回 ns/speedup |
| `agents/llm_client.py` | LLM 统一入口 | nga run CLI / API / stub; `LLM_CLI_TIMEOUT` 默认 3600s |
| `analyzers/run_optimize.sh` | 采集驱动 | 调下面 4 个脚本 → diagnosis.json + 07 字段 (+08_fusion) |
| `analyzers/pipeline_parse_task.py` | 通用 msprof 解析 | task_prof → task.json (骨架: kernel_slots/launch/api_overhead/l2) |
| `analyzers/pipeline_parse_board.py` | msprof op 解析 | OPPROF 8 CSV → board_<i>.json (deep: 带宽/引擎/conflict/fops) |
| `analyzers/integrate.py` | 骨架+deep 合并 | task.json + board_*.json → diagnosis.json (roofline 用 hardware_peak.json 校准) |
| `analyzers/check_fields.py` | 字段校验 | 明细 → 05_task/field_check.log (终端只留摘要) |
| `analyzers/run_hivm_fusion.py` | Tier2 融合分析 | 编译 kernel → HIVM MLIR → nga run fusion skill → 08_fusion/fusion_analysis.json |
| `analyzers/filter_hivm_for_fusion.py` | HIVM 文本过滤 | bishengir 输出 → 08_fusion/hivm_fusion_view.txt |
| `agents/scheduler.py:extract_tier_fields` | 字段筛选 | diagnosis → 07_tier<N>_fields/*.txt|.json |
| `bench_910b3/run_bench.py` | 硬件峰值校准 | 多变体 sweep 取最大 → hardware_peak.json (integrate 读它做 roofline 峰值) |
| `bench_910b3/bench_pytorch*.py` | PyTorch 基准线 | torch 同场景 → pytorch_{mlp,attention}_tflops.json (轨迹图 vs-PyTorch) |
| `feedback/trajectory_chart.py` | 轨迹图 | optimization_trajectory.json → final_output/trajectory_chart.png (PyTorch 线自动读 bench json) |
| `skills/triton-op-planner\|coder\|fusion/SKILL.md` | nga run 技能 | 指导 LLM 输出格式/铁律 (changes[], promote_to, 本层专属) |
| `docx/playbook_tier1~6.md` | 每层策略知识 | 给 Planner/Coder 参考的「问题→方案→正确代码」 |

---

## 4. 各组件详解

### 4.1 Scheduler — 状态机核心
- **状态**: `traj["state"]` = {tier, round, best_speedup, baseline_ns, num_kernels, baseline_mnk, initial_tflops, current_speedup, current_kernel}
- **tier 目录名**: 1→`01_algorithmic_structure`, 2→`02_operator_fusion`, ..., 6→`06_910b3_architecture`
- **每轮产物** (round_dir): `kernel_op.py` + `diff.patch` + `plan.md` + `07_tier<N>_fields/` + `msprof_0/` (验证数据)
- **轨迹**: `outputs/<op>/optimization_trajectory.json`（每轮落盘, 可 `--resume` 续跑）
- **采集链演进**: round1 采集源目录 → 之后采集上一轮输出目录 (current_kernel.parent), kernel 链连续

### 4.2 采集链 (run_optimize.sh)
- **双数据源**: 通用 msprof（骨架，task.json） + 逐 kernel msprof op（deep，board_<i>.json）
- **尺寸传递**: scheduler 从 kernel config 提取 M/N/K 传给脚本 (MATMUL_M/N/K env)，保证 baseline/verify 同尺寸
- **07 字段**: `extract_tier_fields` 输出「全局摘要(前层信号) + 当前 tier 专属字段」，Planner 每轮先看前层
- **Tier2 MULTI 路径**: 多 kernel 才编译 HIVM（门控 `TIER==2`），产出融合视图给 fusion skill

### 4.3 Planner (LLM)
- 输入: SKILL.md + playbook_tier<N> + 当前 kernel_op.py 路径 + 07 字段 + config 常量 + 历史(前层进度: 每层最佳加速比×轮数×促成改动) [+ 融合分析]
- 输出: **plan.md JSON**
  ```json
  {
    "strategy": "增大BLOCK_K减MTE1次数",
    "target_speedup": 1.1,
    "changes": [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                 "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                 "reason": "...", "section": "① config", "tier": 3}],
    "expected_impact": "...",
    "promote": false, "promote_to": 0, "promote_reason": ""
  }
  ```
- 铁律: `old_code` 逐字符匹配; `tier` = 当前层; 先做前层优先检查; changes 只属本层; `promote_to` 支持回退

### 4.4 Coder (确定性改码 + LLM 修复)
- **Step 0**: 无 previous_error 时，确定性 `code.replace(old_code, new_code)`（替换全部出现处）→ 不靠 LLM 最稳
- **Step 1**: 有报错时，走 LLM 带错误修复（查 memory.codeerror 已知方案）
- 校验: Python 语法 + 函数完整性（防截断）+ no-op 检测 → diff.patch

### 4.5 Verifier (端到端 msprof)
- `verify_end_to_end`: warmup(VERIFY_WARMUP=3) + 一次 msprof 内 kernel 循环 KERNEL_LOOP(VERIFY_LOOP=30) 次
- `_read_target_duration`: 读 op_summary **非 aclnn** 目标 kernel `Task Duration(us)` 之和
- `per_pass = total_us / 实测遍数`；实测遍数 = 行数/每遍 kernel 数（源码无 KERNEL_LOOP 循环时用，防 ÷30 虚高）
- 返回 {ok, ns, speedup, loop, rows, duration_us}

### 4.6 LLMClient
- cli (`LLM_CLI_COMMAND='nga run'`) / api (DEEPSEEK/ANTHROPIC) / stub
- CLI 超时 `LLM_CLI_TIMEOUT` 默认 3600s；子进程输出 `errors="backslashreplace"`（非 UTF-8 字节保留为 \xNN，不丢）

---

## 5. 六层优化策略

| Tier | 名称 | Playbook | 本层做什么 | 晋升/停止 |
|---|---|---|---|---|
| **1** | 算法结构 | `playbook_tier1_algorithm.md` | 算法选择/fp16累加/flash rescale/split-k | planner 判算法已优 → 晋升 |
| **2** | 算子融合 | `playbook_tier2_fusion.md` | 逐元素并入 matmul epilogue / 残差 / 冗余 load | 融合分析后无候选 → 晋升 |
| **3** | 分块配置 | `playbook_tier3_tiling.md` | BLOCK_M/N/K（L0/UB 约束） | 连续 3 轮无改进 → 晋升 |
| **4** | 访存 | `playbook_tier4_memory.md` | 连续化/128-bit对齐/L2复用/流水线 | 连续 3 轮无改进 → 晋升 |
| **5** | 计算占用 | `playbook_tier5_compute.md` | 向量化/rsqrt/FMA/ILP/bank冲突 | 连续 3 轮无改进 → 晋升 |
| **6** | 架构专属 | `playbook_tier6_architecture.md` | 引擎失衡/wait_ratio/mte冲突/代码风格 | Tier6 无改进 → **停止** |

**晋升/回退由 Planner 的 `promote_to` 决策**: 可晋升后层，也可**回退前层**（如分块调 8 轮发现算法问题 → 回 Tier1）。无改进兜底 = 本 tier 连续 3 轮 speedup ≤ prev_speedup。

---

## 6. 测量口径（加速比可信的前提）

- **加速比 = baseline_ns / ns**，两端都是 ns，无量纲
- **端到端 = 目标 kernel（非 aclnn）耗时之和**，同尺寸、同 loop、同口径
- **baseline 复测**: round1 用 verify 机制（warmup+msprof）重测源 kernel，与后续轮完全同口径（VERIFY_BASELINE=1）
- **尺寸一致**: scheduler 传真实 M/N/K 给 run_optimize（防 512 默认覆盖 2048）
- **保留判定**: 本轮 speedup ≥ 上一轮已接受 → KEEP 进链；否则 REVERT 沿用上一轮
- **每轮 tflops**: 用本轮诊断 cube_fops ÷ 本轮 ns（kernel 结构变化后 FLOPs 变 → 轨迹图不失真）

---

## 7. 输出目录结构

```
outputs/<op>/
├── optimization.log                  # 全流程运行日志 (Tee 双写)
├── baseline_verify/                  # round1 基准复测 (msprof_0/)
├── 01_algorithmic_structure/round1..N/
│   ├── kernel_op.py                  # Coder 产出的本轮优化代码
│   ├── diff.patch                    # 与上一轮的差异
│   ├── plan.md                       # Planner 计划 (JSON: changes[]+promote)
│   ├── 07_tier1_fields/              # 本轮筛好的诊断字段 (tier1_fields.txt|.json)
│   ├── msprof_0/                     # 验证用的 msprof 产物
│   ├── 04_board/ 05_task/ 06_diagnosis/   # run_optimize 采集中间产物
│   └── (Tier2) 08_fusion/            # HIVM 融合分析
├── 02_operator_fusion/round1..N/ ...   # 其余 tier 同构
├── 03_tiling_block_config/ ...
├── 04_memory_access/ ...
├── 05_compute_occupancy/ ...
├── 06_910b3_architecture/ ...
├── optimization_trajectory.json      # ★ 全局状态+history (中枢)
└── final_output/trajectory_chart.png # 轨迹图 (跑 chart 生成)
```

`optimization_trajectory.json`:
```json
{
  "v": 4,
  "state": {"tier": 3, "round": 8, "best_speedup": 17.793, "current_speedup": 17.793,
            "baseline_ns": 5900000.0, "num_kernels": 3, "baseline_mnk": [2048,2048,2048],
            "initial_tflops": 5.8, "pytorch_tflops": 42.0, "pytorch_baseline": "pytorch_mlp_tflops.json",
            "current_kernel": "outputs/<op>/03_tiling_block_config/round8/kernel_op.py"},
  "history": [
    {"round": 1, "tier": 1, "strategy": "...", "change": "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64",
     "speedup": 1.5, "prev_speedup": 1.0, "ns": 3933333.0, "decision": "KEEP", "result": "OK",
     "error": "", "tflops": 8.7}
  ]
}
```

---

## 8. 环境与运行

```bash
conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd triton_agent_optimizer

# 完整优化循环 (无目标跑满; 目标达标即停)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15
LLM_CLI_COMMAND='nga run' python3 main.py input/attention_mlp --fresh --max-rounds 15 --target 2.0

# 单文件能跑 + 数值校验
python3 input/matmul/kernel_op.py && MATMUL_VERIFY=1 python3 input/matmul/kernel_op.py

# 只采集+解析
bash analyzers/run_optimize.sh input/matmul input/matmul/e2e_run

# 硬件峰值校准 + PyTorch 基准
cd bench_910b3 && python3 run_bench.py
python3 bench_pytorch_mlp.py           # matmul(MLP) 对照
python3 bench_pytorch_attention.py     # attention_mlp 对照

# 轨迹图 (自动读 PyTorch 基准)
cd .. && python3 feedback/trajectory_chart.py outputs/matmul

# 运行日志
cat outputs/<op>/optimization.log
```

---

## 9. 与 v3 的差异（为什么重写）

| 项 | v3 (旧) | v4 (当前) |
|---|---|---|
| 输入 | 三文件 (test_matmul.py+triton_kernel.py+config.json) | **单文件 kernel_op.py** (config+kernel+test) |
| 调度 | Orchestrator + RecordManager | **Scheduler 状态机** 一体 |
| 诊断 | CPU 仿真 + HIVMIR 文本解析 (dsl_merger 29字段) | **真机 msprof 双源** (通用+op) → diagnosis.json |
| 验证 | CPU Emulator Stage1 + 真机 Stage2 | **端到端 msprof** (warmup + 1×msprof KERNEL_LOOP) |
| 改码 | LLM 全改 | **确定性 changes[] 替换** + LLM 仅修错 |
| 晋升 | 规则 | planner `promote_to` (可回退前层) + 无改进兜底 |
| HIVM | 每轮全流程 | 仅 **Tier2 融合** |
| 记忆 | 经验库检索 | 基本弃用 (仅 coder 错修参考) |
```
