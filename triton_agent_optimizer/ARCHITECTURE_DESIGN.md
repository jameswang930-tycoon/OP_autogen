# Triton Agent Optimizer v4 — 架构设计（按当前实现）

> **核心思路**: 不靠盲试（AutoKernel 300~400 轮），而是用 **真机 msprof 双源采集（通用 msprof + msprof op）** 精确诊断每个 kernel 的耗时/带宽/算力/瓶颈 → 6 层优化策略逐层推进 → 每轮**端到端 msprof 实测**（纯 kernel+端到端双口径, 主指标=端到端）决定保留/回退/晋升，并以**工业级最优**（torch.compile/TorchAir、CANN 融合、CANN-FA 各 mode 取 min）为对比天花板。
>
> **环境**: Ascend 910B3 (NPU) + CANN 9.0 + triton-ascend + nga (本地 LLM codeagent)
> **更新**: 2026-08-10 — v4.2: 端到端口径统一（纯kernel+E2E 双指标, 主=E2E）+ 工业级基准（各 mode 取 min）+ PyTorch 统一 msprof + 基准产物收纳 bench_910b3/outputs/

---

## 0. 一句话总结

```
输入单个 kernel_op.py（config+kernel+test 一体）
  → 采集+解析出 diagnosis.json（骨架+deep+roofline）
  → 按当前 tier 筛字段喂 Planner(LLM)
  → Planner 出 changes[]（old_code→new_code 精确替换）+ promote 决策
  → Coder 确定性应用 changes[] → 本轮 kernel_op.py
  → 端到端 msprof 实测（纯kernel+E2E 双口径, 主加速比=端到端 vs 初始基线; 另 vs 工业级最优）
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
   ├─(round1 基准) baseline_ns(纯)/baseline_e2e_ns(端到端)/num_kernels/num_launches/baseline_mnk/initial_tflops/pytorch基准/工业级基准(各mode取min)
   │    + verify_end_to_end 复测源 kernel (正确性校验 + warmup + 1×msprof KERNEL_LOOP) → 同口径 baseline
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
   │    正确性校验(MATMUL_VERIFY) + warmup(3) + 一次 msprof 内循环 KERNEL_LOOP(30) 次
   │    读全部 op_summary*.csv 合并 → 同一次同算两种口径 ÷ 实测遍数:
   │      纯kernel ns = Σ非aclnn行; 端到端 e2e_ns = Σ全部行(含框架kernel)
   │    → 主加速比 = baseline_e2e_ns / e2e_ns (纯kernel 作参考)
   │    (源码无 KERNEL_LOOP 循环 → 自动改除实测遍数, 防虚高)
   │
   ├─⑥ 决策  主加速比 = 初始端到端基线/本轮端到端 (累计输出; 纯kernel 作参考)
   │    speedup ≥ prev_speedup×噪声地板(1.01) → KEEP, 进 kernel 链
   │    否则 → REVERT, 沿用上一轮 kernel
   │    记录 history[] {round,tier,strategy,change,speedup(e2e),kernel_speedup(纯),prev_speedup,ns,e2e_ns,decision,result,error,tflops}
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
   (加速比曲线 + 各 tier 色带 + KEEP/REVERT 点 + PyTorch 虚线 + 工业级红线[自动读 bench_910b3/outputs/])
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
│  │ 读 SKILL +     │  │ 应用 changes[]  │  │ 正确性校验+    │             │
│  │ playbook_tierN │  │ old→new 逐字符 │  │ warmup(3) +    │             │
│  │ + 07字段       │  │ 替换全部出现处 │  │ 1×msprof       │             │
│  │ + 当前kernel   │  │ +LLM修错(≤3)   │  │ KERNEL_LOOP(30) │             │
│  │ + 历史(前层)   │  │ 语法检查       │  │ → 纯kernel ns +│             │
│  │ → plan.md      │  │ → roundN/      │  │  端到端 e2e_ns │             │
│  │  (changes[]+   │  │   kernel_op.py │  │  (循环丢自校正) │             │
│  │   promote_to)  │  │   +diff.patch  │  │                │             │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘             │
│          └───────────────────┼───────────────────┘                       │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │           Scheduler (Python 状态机 — 决策核心)                │       │
│  │  KEEP: 端到端加速比 ≥ prev×噪声地板1.01 (进 kernel 链)          │       │
│  │  REVERT: 否则沿用上一轮 (纯kernel 作参考)                       │       │
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
│  outputs/<op>/final_output/  ★最终产物: kernel_op.py + final_summary.json│
│    + trajectory_chart.png  (加速比曲线+PyTorch虚线+工业级红线)             │
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
round1 基准: baseline_ns(纯) / baseline_e2e_ns(端到端) / num_kernels / num_launches / baseline_mnk / initial_tflops / pytorch基准(msprof) / 工业级基准(各mode取min)
   + verify_end_to_end 复测源 kernel (正确性校验+warmup+msprof, 与后续轮同口径)
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
verify_end_to_end → 正确性校验 + warmup + 1×msprof KERNEL_LOOP → 读全部op_summary合并 → 纯kernel ns + 端到端 e2e_ns → 主加速比=端到端 (纯kernel参考)
   │
   ▼
KEEP / REVERT (对比 prev_speedup) + history 记录 (含每轮 tflops) + promote 决策
   │
   ▼
optimization_trajectory.json
   │
   ▼
feedback/trajectory_chart.py → final_output/trajectory_chart.png
   (加速比曲线 · 各 tier 色带 · KEEP/REVERT 点 · PyTorch 虚线 + 工业级红线[自动读 bench_910b3/outputs/])
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
        baseline_ns / baseline_e2e_ns / num_kernels / num_launches / baseline_mnk / initial_tflops
        + pytorch基准(msprof) + 工业级基准(各mode取min)
        + verify_end_to_end 复测源 kernel (VERIFY_BASELINE=1) → 与后续轮同口径
    extracted = _diagnose(diagnosis, tier, round_dir)   # → 07_tier<N>_fields
    if tier == 2: fusion_analysis = _run_fusion(round_dir)   # → 08_fusion/
    if rn == 1 or tier == 3: tier3_sweep = _tier3_sweep_data(tier, rn, round_dir)
        # 程序化枚举 L0 合法 BLOCK 候选 → 真机实测 → 最优写回 kernel_op.py
    plan = _plan(diagnosis, extracted, tier, rn, round_dir, fusion_analysis, tier3_sweep)
    if plan.promote:                                 # 晋升轮: 原样拷贝 kernel, 不调 LLM
        copy current_kernel → roundN/kernel_op.py
    else:
        for attempt in range(3):
            new_code = _code(plan, rn, round_dir, prev_err)   # → roundN/kernel_op.py
            v = _verify(round_dir, baseline_ns)               # 正确性校验+端到端 msprof (双口径)
            if v.ok: break
    speedup = baseline_e2e_ns / e2e_ns        # 主加速比=端到端 (缺则纯 kernel 兜底)
    if speedup >= prev_speedup × 噪声地板(1.01): current_kernel = roundN/kernel_op.py; KEEP
    else: REVERT (沿用上一轮)
    history.append(...); _save_traj()
    # 晋升/停止判定 → 下一轮/下一 tier/停止
```

---

## 3. 完整文件架构图（所有文件夹/文件 → 主要作用）

> 标注: ★ = 核心链路关键文件; 括号内为该文件/文件夹的主要职责。

```
triton_agent_optimizer/
├── main.py                        # ★入口: 解析 input/<op> 参数 → 启动 Scheduler
│                                  #   (--fresh/--resume/--max-rounds/--target/--stub); Tee 日志双写 optimization.log
├── config.py                      # 全局配置中心: 检测本地/服务器环境, 集中所有路径/阈值 (from config import Config)
├── _sim_flow_test.py              # ★端到端链路模拟回归: 打桩真机(msprof/HIVM/LLM), 真实 scheduler 逐轮执行 → 验证全流程
├── _sim_edge_test.py              # ★边界场景回归 #2: 采集失败/回退/晋升/目标/基准 等失败分支 (全真实执行 scheduler/coder/verifier)
├── README.md                      # 项目总览 + 开发记录 (会话开始先扫这里回忆进度)
├── ARCHITECTURE_DESIGN.md         # ★本文档: 架构设计 + 逐组件说明 (按当前实现维护)
├── IMPLEMENTATION_PLAN.md         # 逐文件实现计划 (实现已完成, 作架构参考保留)
├── .env / .env.template           # 环境密钥/配置 (LLM key 等, 不入库)
├── .gitignore                     # git 忽略规则

├── agents/                        # ── Agent 层: 调度状态机 + 规划/改码/验证 + LLM 入口 ──
│   ├── scheduler.py               # ★调度状态机 (核心): 驱动每轮 采集→诊断→Planner→Coder→Verify→决策→晋升;
│   │                              #   round1 设基准(纯kernel+端到端+工业级); 主加速比=端到端; extract_tier_fields 筛字段;
│   │                              #   sweep 触发; 优秀案例记录; 最终产物写 final_output/
│   ├── planner.py                 # Planner (LLM): 读 SKILL+playbook+07字段+planner_context+轨迹+手递 → plan.md (changes[]+promote)
│   ├── coder.py                   # Coder (确定性改码+LLM修复): 应用 changes[] 逐字符替换 → roundN/kernel_op.py + diff.patch
│   │                              #   语法/函数完整性/no-op 校验; Unicode 脏字符清洗
│   ├── verifier.py                # Verifier: 先 MATMUL_VERIFY 正确性校验 → warmup + 1×msprof KERNEL_LOOP
│   │                              #   → 读全部 op_summary 合并 → 返回 ns(纯kernel)+e2e_ns(端到端)+speedup
│   ├── llm_client.py              # LLM 统一入口: nga run CLI / API / stub; LLM_CLI_TIMEOUT=3600s
│   └── __init__.py

├── analyzers/                     # ── 采集+解析层: run_optimize.sh 驱动 msprof 双源 → diagnosis.json + 07 字段 ──
│   ├── run_optimize.sh            # ★采集驱动: warmup → 通用msprof→task.json + msprof op→board_<i>.json
│   │                              #   → integrate → 07字段 → (Tier2)HIVM; 参数 M/N/K + TIER env
│   ├── pipeline_parse_task.py     # 通用 msprof 解析 → task.json (骨架: kernel_slots/launch/api_overhead/l2)
│   ├── pipeline_parse_board.py    # msprof op 解析 → board_<i>.json (deep: 带宽/引擎/conflict/fops)
│   ├── integrate.py               # 骨架+deep 合并 → diagnosis.json (roofline 用 hardware_peak.json 校准)
│   ├── pipeline_schema.py         # 诊断数据 schema 定义
│   ├── check_fields.py            # 字段校验明细 → 05_task/field_check.log (终端只留摘要)
│   ├── run_hivm_fusion.py         # Tier2 融合: 编译 kernel → HIVM MLIR → nga run fusion skill → 08_fusion/fusion_analysis.json
│   ├── filter_hivm_for_fusion.py  # HIVM 文本过滤 → 08_fusion/hivm_fusion_view.txt
│   ├── hivmir_analyzer.py         # HIVMIR 解析 (供 filter_hivm_for_fusion 用)
│   ├── merge_single_file.py       # 旧三文件 → 单文件 kernel_op.py 合并工具 (main.py 兜底)
│   ├── sweep_blocks.py            # ★Tier3 分块扫描 v2 (从根目录移入): 程序化枚举 L0 合法 BLOCK
│   │                              #   + 单进程 torch.npu.Event 实测; ×0.9 边距/增量保存/崩溃续跑/设备污染恢复
│   └── __init__.py

├── bench_910b3/                   # ── 基准测量层: 工业级基准 + PyTorch 基准 + 硬件峰值 (产物统一 outputs/) ──
│   ├── bench_common.py            # ★msprof 测量工具: BENCH_OUT=outputs/; clean_bench_out(); measure_pytorch_msprof
│   │                              #   (端到端+纯kernel+kernels_per_iter 周期检测); measure_msprof / measure_msprof_op
│   ├── bench_industrial.py        # ★工业级基准: 每算子 eager(aclnn厂商)/compile(TorchAir)/cann-fused(aclnn融合)/fa(CANN-FA)
│   │                              #   → outputs/industrial_<op>_<mode>_tflops.json (含 actual_mode / kernels_per_iter)
│   ├── bench_all.py               # ★全算子最优: 跑全部模式 → 每算子取 min 端到端(仅真执行) → outputs/industrial_summary.json
│   │                              #   --clean 一键清空产物; 明细表含 执行状态/融合判定
│   ├── pt_msprof.py               # ★PyTorch 统一 msprof 包装: 一次 msprof 包 bench_pytorch_*.py
│   │                              #   → outputs/pytorch_*_tflops.json (端到端+纯kernel, 与 verify 同口径)
│   ├── bench_pytorch.py           # torch.matmul (两层 MLP) 基准
│   ├── bench_pytorch_mlp.py       # torch MLP 基准 (matmul 对照)
│   ├── bench_pytorch_attention.py # 自注意力+MLP 基准 (attention_mlp 对照)
│   ├── bench_pytorch_flash_attention.py   # CANN FA 基准 (flash_attention 对照)
│   ├── bench_pytorch_conv2d.py / bench_pytorch_conv_bias_relu.py   # 卷积基准
│   ├── bench_pytorch_matmul_relu.py / bench_pytorch_matmul_transpose.py   # matmul 变体基准
│   ├── bench_pytorch_rms_norm.py / bench_pytorch_layernorm.py / bench_pytorch_sigmoid.py   # 归一化/逐元素基准
│   ├── bench_config.py            # 变体注册表 + 静态 bytes/flops 计算 + PT_BENCH_MAP (算子→pytorch json 映射)
│   ├── bench_kernels.py           # 硬件基准 triton kernels (read/write/copy/l2/cube/vec 多变体)
│   ├── run_bench.py               # ★硬件峰值校准: 多策略实测取最大 → hardware_peak.json (integrate 读它做 roofline 峰值)
│   ├── bench_theory.py            # 910B3 理论峰值计算 + 理论/实测对照 (纯本地, 无 NPU 依赖)
│   ├── README.md                  # bench 使用说明
│   └── outputs/                   # ★基准产物收纳 (运行时生成): industrial_*.json / pytorch_*.json / .actual_*.txt
│                                  #   / industrial_summary.json / msprof_pt / _pt_msprof_tmp

├── input/                         # ── 算子源文件层: 每算子一目录, kernel_op.py 单文件 (config+kernel+test 一体) ──
│   ├── matmul/            kernel_op.py   # 两层 MLP (GELU(X@W1+b1)@W2)
│   ├── attention_mlp/     kernel_op.py   # 自注意力 + MLP (9-kernel)
│   ├── matmul_relu/       kernel_op.py   # matmul + ReLU
│   ├── matmul_transpose/  kernel_op.py   # matmul (B 转置)
│   ├── flash_attention/   kernel_op.py   # FlashAttention (K 预转置)
│   ├── conv2d/            kernel_op.py   # 卷积
│   ├── conv_bias_relu/    kernel_op.py   # 卷积 + bias + ReLU
│   ├── rms_norm/          kernel_op.py   # 行级 RMSNorm
│   ├── rms_norm_residual/ kernel_op.py   # RMSNorm + 残差
│   ├── layernorm/         kernel_op.py   # LayerNorm
│   ├── sigmoid/           kernel_op.py   # 逐元素 sigmoid
│   ├── softmax/           kernel_op.py   # 行级 softmax
│   ├── vector_add/        kernel_op.py   # 逐元素加法
│   └── fused_add_mul/     kernel_op.py   # (x+z)*w

├── skills/                        # ── 本地 nga 技能 (Planner/Coder/Fusion 的 prompt 铁律) ──
│   ├── triton-op-planner/SKILL.md # 指导 LLM 输出 plan.md 格式/铁律 (changes[], promote_to)
│   ├── triton-op-coder/SKILL.md   # 指导 LLM 改码格式 (确定性替换铁律)
│   └── triton-op-fusion/SKILL.md  # 指导 LLM 融合分析 (Tier2)

├── docx/                          # ── 知识库: 6 层策略 playbook + 编码规范 + 字段参考 ──
│   ├── playbook_tier1_algorithm.md ~ playbook_tier6_architecture.md   # 每层「问题→方案→正确代码」
│   ├── CODING_GUIDE.md            # 编码规范 (前三 tier 优化指导)
│   ├── OPTIMIZATION_METHODOLOGY.md  # 优化方法论
│   ├── msprof_fields_reference.md # msprof 字段参考
│   ├── field_extraction_checklist.md / aggregation_rules.md / final_product_spec.md   # 字段提取/聚合/产物规范

├── feedback/                      # ── 结果反馈层 ──
│   ├── trajectory_chart.py        # ★轨迹图: optimization_trajectory.json → final_output/trajectory_chart.png
│   │                              #   (加速比曲线+各tier色带+KEEP/REVERT点+PyTorch虚线+工业级红线, 自动读 bench_910b3/outputs/)
│   └── __init__.py

├── memory/                        # ── 记忆层 ──
│   ├── excellent_cases.py         # 优秀案例自动记录/检索 (EXCELLENT_THRESHOLD=1.3×, planner 优化前参考)
│   ├── codeerror/                 # 已知代码错误修复方案 (softmax.json / triton_agent_optimizer.json)
│   └── __init__.py

├── knowledge/                     # ── 领域知识 ──
│   └── hivm.md                    # HIVM 知识笔记

├── paper_reference/               # ── 参考资料 (CANN-Bot 论文技能包, 只读不改) ──
│   └── cannbot-skills-paper/
│       ├── ops/
│       │   ├── npu-arch/                  # NPU 架构技能 (SKILL.md + references/npu-arch-guide*.md)
│       │   ├── triton-latency-optimizer/  # Triton 延迟优化技能 (SKILL.md + references/: docs_triton_IR ~350 篇 HIVM/triton 架构参考)
│       │   ├── triton-op-coding/          # 算子编码技能 (references/triton-ascend-*.md)
│       │   ├── triton-op-designer/        # 算子设计技能 (references/cases/*, sketch-design.md)
│       │   ├── triton-op-verifier/        # 验证技能 (scripts/verify.py 等)
│       │   └── triton-task-extractor/     # 任务提取技能
│       └── plugins-official/
│           └── triton-op-generator/       # 官方算子生成插件 (template/*, utils/*)

├── outputs/                       # ── 运行产物 (运行时生成, 不入库) ──
│                                  #   outputs/<op>/ = 优化结果 (optimization_trajectory.json + tier/roundN/ + final_output/)

├── autogen/                       # ── 编译配置 (gitignore 忽略) ──
│   └── custom_compile_options.ini # 自定义编译选项

├── prepare/                       # ── 环境准备 ──
│   └── setup_and_verify.sh        # 环境安装/校验脚本
```

---

## 4. 各组件详解

### 4.1 Scheduler — 状态机核心
- **状态**: `traj["state"]` = {tier, round, best_speedup, baseline_ns(纯), baseline_e2e_ns(端到端), num_kernels, num_launches, baseline_mnk, initial_tflops, pytorch基准, industrial_time_us(工业级), current_speedup, current_kernel}
- **tier 目录名**: 1→`01_algorithmic_structure`, 2→`02_operator_fusion`, ..., 6→`06_910b3_architecture`
- **每轮产物** (round_dir): `kernel_op.py` + `diff.patch` + `plan.md` + `07_tier<N>_fields/` + `msprof_0/` (验证数据)
- **轨迹**: `outputs/<op>/optimization_trajectory.json`（每轮落盘, 可 `--resume` 续跑）
- **采集链演进**: round1 采集源目录 → 之后采集上一轮输出目录 (current_kernel.parent), kernel 链连续

### 4.2 采集链 (run_optimize.sh)
- **双数据源**: 通用 msprof（骨架，task.json） + 逐 kernel msprof op（deep，board_<i>.json）
- **尺寸传递**: scheduler 从 kernel config 提取 M/N/K 传给脚本 (MATMUL_M/N/K env)，保证 baseline/verify 同尺寸
- **07 字段**: `extract_tier_fields` 输出「全局摘要(前层信号) + Per-Kernel 概览 + 当前 tier 专属字段」，Planner 每轮先看前层
- **Tier2 MULTI 路径**: 多 kernel 才编译 HIVM（门控 `TIER==2`），产出融合视图给 fusion skill
- **Tier3 分块 sweep**: round1/tier3 自动调 sweep_blocks 程序化枚举 BLOCK → 真机实测 → 最优写回 (09_tier3_sweep/)

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

### 4.5 Verifier (正确性校验 + 端到端 msprof 双口径)
- 先跑 MATMUL_VERIFY=1 正确性校验 (必须输出 `result check: PASS`; 否则本轮 FAIL 回传 coder 修)
- `verify_end_to_end`: warmup(VERIFY_WARMUP=3) + 一次 msprof 内 kernel 循环 KERNEL_LOOP(VERIFY_LOOP=30) 次
- `_read_durations`: 读**全部 op_summary*.csv 合并**, 同一次同算两种口径之和 → 纯kernel(Σ非aclnn) + 端到端(Σ全部含框架)
- `per_pass = total_us / 实测遍数`；实测遍数 = 行数/每遍 kernel 数（源码无 KERNEL_LOOP 循环时用, 传 num_launches 兜底, 防 ÷30 虚高）
- 主加速比 = baseline_e2e_ns / e2e_ns (纯 kernel 参考); 返回 {ok, ns, e2e_ns, speedup, loop, rows, duration_us}

### 4.6 LLMClient
- cli (`LLM_CLI_COMMAND='nga run'`) / api (DEEPSEEK/ANTHROPIC) / stub
- CLI 超时 `LLM_CLI_TIMEOUT` 默认 3600s；子进程输出 `errors="backslashreplace"`（非 UTF-8 字节保留为 \xNN，不丢）

---

## 5. 六层优化策略

| Tier | 名称 | Playbook | 本层做什么 | 晋升/停止 |
|---|---|---|---|---|
| **1** | 算法结构 | `playbook_tier1_algorithm.md` | 算法选择/fp16累加/flash rescale/split-k | planner 判算法已优 → 晋升 |
| **2** | 算子融合 | `playbook_tier2_fusion.md` | 逐元素并入 matmul epilogue / 残差 / 冗余 load | 融合分析后无候选 → 晋升 |
| **3** | 分块配置 | `playbook_tier3_tiling.md` | BLOCK_M/N/K（L0/UB 约束; round1/tier3 自动 sweep 实测最优块） | 连续 3 轮无改进 → 晋升 |
| **4** | 访存 | `playbook_tier4_memory.md` | 连续化/128-bit对齐/L2复用/流水线 | 连续 3 轮无改进 → 晋升 |
| **5** | 计算占用 | `playbook_tier5_compute.md` | 向量化/rsqrt/FMA/ILP/bank冲突 | 连续 3 轮无改进 → 晋升 |
| **6** | 架构专属 | `playbook_tier6_architecture.md` | 引擎失衡/wait_ratio/mte冲突/代码风格 | Tier6 无改进 → **停止** |

**晋升/回退由 Planner 的 `promote_to` 决策**: 可晋升后层，也可**回退前层**（如分块调 8 轮发现算法问题 → 回 Tier1）。无改进兜底 = 本 tier 连续 3 轮 speedup ≤ prev_speedup。

---

## 6. 测量口径（加速比可信的前提）

- **主加速比 = baseline_e2e_ns / e2e_ns**（端到端口径: Σ全部 kernel 行含框架），两端都是 ns，无量纲
- **纯 kernel = Σ非 aclnn 目标 kernel 耗时之和**（作参考）；一次 msprof 同算两种口径，同尺寸、同 loop
- **端到端含框架**: torch/工业级侧计算 kernel 全走 aclnn 下发 → 两端"端到端"口径统一可比（纯 kernel 口径 torch 不可分，不做跨侧对比）
- **工业级对比**: round1 读 `bench_910b3/outputs/industrial_<op>_<mode>_tflops.json`，各 mode（eager/compile/cann-fused/fa）取 time_us 最小者 = industrial_time_us；我们最终 kernel vs 它
- **baseline 复测**: round1 用 verify 机制（warmup+msprof）重测源 kernel，与后续轮完全同口径（VERIFY_BASELINE=1）
- **尺寸一致**: scheduler 传真实 M/N/K 给 run_optimize（防 512 默认覆盖 2048）
- **保留判定**: 本轮端到端加速比 ≥ 上一轮已接受×噪声地板(KEEP_FLOOR=1.01) → KEEP 进链；否则 REVERT 沿用上一轮
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
├── 09_tier3_sweep/                   # 分块 sweep 产物 (sweep_result.json / sweep_runner.py)
├── optimization_trajectory.json      # ★ 全局状态+history (中枢)
└── final_output/                     # ★最终产物 (优化结束自动生成)
    ├── kernel_op.py                  #   最优 kernel (可直接取用)
    ├── baseline_kernel.py            #   baseline 副本
    ├── final_summary.json            #   摘要 (含双口径 + industrial_time_us)
    └── trajectory_chart.png          #   轨迹图 (PyTorch 虚线 + 工业级红线)
```

> **bench 基准产物** 统一放 `bench_910b3/outputs/`（industrial_*.json / pytorch_*.json / .actual_*.txt / industrial_summary.json / msprof 临时目录）；
> `python3 bench_910b3/bench_all.py --clean` 一键清空。

`optimization_trajectory.json`:
```json
{
  "v": 4,
  "state": {"tier": 3, "round": 8, "best_speedup": 17.793, "current_speedup": 17.793,
            "baseline_ns": 5900000.0, "baseline_e2e_ns": 6320000.0, "num_kernels": 3,
            "num_launches": 3, "baseline_mnk": [2048,2048,2048],
            "initial_tflops": 5.8, "pytorch_time_us": 45.0,
            "industrial_time_us": 43.0, "industrial_baseline": "industrial_matmul_compile_tflops.json",
            "current_kernel": "outputs/<op>/03_tiling_block_config/round8/kernel_op.py"},
  "history": [
    {"round": 1, "tier": 1, "strategy": "...", "change": "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64",
     "speedup": 1.5, "kernel_speedup": 1.5, "prev_speedup": 1.0, "ns": 3933333.0,
     "e2e_ns": 4210000.0, "decision": "KEEP", "result": "OK",
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

# 硬件峰值校准 (roofline 峰值)
cd bench_910b3 && python3 run_bench.py

# 工业级基准 (各算子各 mode 真机测 → 取每算子 min 作为对比天花板)
python3 bench_industrial.py matmul --mode compile     # TorchAir 图融合
python3 bench_industrial.py flash_attention --mode fa # CANN FlashAttention
python3 bench_all.py                    # 全部算子全部模式 → 自动取最优 + 汇总表
python3 bench_all.py --clean            # 清理 bench_910b3/outputs/ 全部产物

# PyTorch 基准 (统一 msprof 包裹: 端到端+纯kernel 同口径)
python3 pt_msprof.py bench_pytorch_mlp.py        # matmul(MLP) 对照
python3 pt_msprof.py bench_pytorch_attention.py  # attention_mlp 对照

# 轨迹图 (自动读 outputs/ 里的 PyTorch + 工业级基准)
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
| 验证 | CPU Emulator Stage1 + 真机 Stage2 | **正确性校验 + 端到端 msprof** (warmup + 1×msprof KERNEL_LOOP; 纯kernel+E2E 双口径, 主=E2E) |
| 对比基准 | 无 (只 vs naive torch) | **工业级天花板** (eager/compile/cann-fused/fa 各 mode 取 min) + PyTorch 统一 msprof |
| 改码 | LLM 全改 | **确定性 changes[] 替换** + LLM 仅修错 |
| 晋升 | 规则 | planner `promote_to` (可回退前层) + 无改进兜底 |
| HIVM | 每轮全流程 | 仅 **Tier2 融合** |
| 记忆 | 经验库检索 | 基本弃用 (仅 coder 错修参考) |
```

---

## 10. 六层优化策略：读取字段详解

> 每轮 `run_optimize.sh` 产出 `diagnosis.json` 后，scheduler 按当前 tier 用 `TIER_FIELDS` + `TIER_PER_KERNEL` 筛字段 → `07_tier<N>_fields/` 喂 Planner。
> 字段来源: `summary.*`(通用 msprof 骨架) / `kernels[i].task.*`(骨架 per-kernel) / `kernels[i].deep.*`(msprof op 深层画像)。
> **全局摘要(前层信号)** 任何 tier 都喂: `num_kernels`(多→融合空间) / `num_kernels_total`(含框架) / `api_overhead_total_us`(launch 开销) —— 让 planner 先做"前层优先检查"，不能只闷头调本层参数。

### 10.1 Tier1 算法结构

**优化什么**: 算法选择 / 精度(fp16/int8) / 算术强度 → 决定 kernel 走什么算法规格。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `summary.num_kernels` | 目标 kernel 数 | 多 kernel = 复合算子/融合空间; 判断该走单算子还是算法拆分 |
| `summary.total_ns` | 端到端耗时 | 总预算, 各 kernel 占比的分母 |
| `kernels[].deep.compute.cube_fops` | cube 浮点运算数 | 真实计算量; 算 TFLOPS/算术强度 (FLOPs 来源) |
| `kernels[].deep.compute.vector_fops` | 向量运算数 | 非 matmul (softmax/rms_norm/逐元素) 的真实计算量 |
| `kernels[].deep.compute.cube_ratio` | cube 指令占比 | 低 = 算法没走 cube (如 conv 该 im2col 上 cube); 高 = 计算密集 |
| `kernels[].deep.compute.cube_fp16_ratio` | cube fp16 占比 | 精度选择: 该不该降 fp16 提算力 |
| `kernels[].deep.compute.cube_int8_ratio` | cube int8 占比 | 能否走 int8 量化 |
| `kernels[].deep.engine_utilization.vec` | 向量指令占比 | 向量型算子瓶颈 (归约/逐元素 kernel 命门) |
| `kernels[].deep.roofline.compute_utilization` | 算力利用率 | 是否计算瓶颈、离峰值多远 (naive<30% → 有空间) |
| `kernels[].deep.roofline.arithmetic_intensity` | 算术强度(计算/访存) | 计算型 vs 访存型 → 决定走算法优化还是访存优化 |
| `kernels[].deep.roofline.bottleneck_type` | 瓶颈类型 | 一针见血: compute/memory/latency/balanced → 本层是否瓶颈 |
| `kernels[].task.pipes_us.aiv_vec_time_us` | 向量耗时 (per-kernel) | 向量 kernel (rms_norm/softmax/bias_gelu) 的实际耗时占比 |

### 10.2 Tier2 算子融合

**优化什么**: 多 kernel → 单 kernel (逐元素并入 matmul epilogue / 残差 / 冗余 load)。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `summary.num_kernels` | 目标 kernel 数 | 有几个 kernel 可融合; 3 个分离 kernel 是融合目标 |
| `summary.num_kernels_total` | 总 kernel 数(含框架) | aclnn 框架准备 kernel 是否多 → 能否避免 |
| `summary.api_overhead_total_us` | launch 开销 | 大 = 每次 launch 固定开销高 → 融合收益大 |
| `kernels[].task.task_type` | 每 kernel 引擎 | 逐元素 Vec kernel 可并入 matmul epilogue; Cube/Vec 混合找融合点 |
| `kernels[].launch_count` | 每 kernel launch 次数 | 复用 kernel (QKV 复用 matmul) 的融合/拆分候选 |
| `api_overhead` | API 开销明细 | 具体哪个 launch 贵 (换 tensor 不换 launch) |
| `multi_kernel` | 算子类型分解 | 整条计算链 → 找相邻可融合 op (bias→gelu→残差) |
| `framework_kernels` | 框架 kernel(非目标) | 哪些是 aclnn 数据准备, 是否本可避免 |
| `kernels[].deep.roofline.compute_utilization` | 算力利用率 (per-kernel) | 低利用率的独立小 kernel = 最该并入的候选 |
| `kernels[].launch_count` | launch 次数 (per-kernel) | 每 kernel 被 launch 几次, 占比口径 |

### 10.3 Tier3 分块配置

**优化什么**: BLOCK_M/N/K (L0A/L0B/L0C/UB 约束, 2 幂) — round1/tier3 自动 sweep 实测最优块。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `kernels[].task.block_dim` | 核数 | 当前分块用多少核; 太少 = 并行度不足 |
| `kernels[].deep.engine_utilization.mte1` | MTE1(L1→L0A/B)占比 | 内层搬运是否瓶颈 (BK 太大 → L0A/B 搬运占比高) |
| `kernels[].deep.engine_utilization.mte2` | MTE2(GM→L1)占比 | GM 搬运是否瓶颈 (BLOCK 小 → GM 搬运频繁) |
| `kernels[].deep.engine_utilization.cube` | cube 占比 | 计算占比; 太低 = 被搬运拖住 |
| `kernels[].deep.bandwidth_gb_s.l0a_read_gb_s` | L0A 读带宽 | BLOCK_M×BK 与 L0A(64KB) 的贴合度 |
| `kernels[].deep.bandwidth_gb_s.l0a_write_gb_s` | L0A 写带宽 | 同上 |
| `kernels[].deep.bandwidth_gb_s.l0b_read_gb_s` | L0B 读带宽 | BLOCK_N×BK 与 L0B(64KB) 的贴合度 |
| `kernels[].deep.bandwidth_gb_s.l0b_write_gb_s` | L0B 写带宽 | 同上 |
| `kernels[].task.pipes_us.aic_mte1_time_us` | MTE1(L1→L0A/B)耗时 (per-kernel) | 搬运实际耗时 → BLOCK 该调大/调小 |
| `kernels[].deep.conflict.bank_cflt_ratio` | bank 冲突 (per-kernel) | 大块时 UB bank 冲突 |

### 10.4 Tier4 访存

**优化什么**: GM 带宽 / L2 复用 / 连续化 / 128-bit 对齐 / 流水线。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `kernels[].deep.bandwidth_gb_s.main_mem_read_gb_s` | GM 读带宽 | 是否接近 HBM 峰值 (访存瓶颈) |
| `kernels[].deep.bandwidth_gb_s.main_mem_write_gb_s` | GM 写带宽 | 是否访存写瓶颈 |
| `kernels[].deep.bandwidth_gb_s.gm_to_ub_gb_s` | GM→UB 带宽 | 搬运通路效率 |
| `kernels[].deep.bandwidth_gb_s.ub_to_gm_gb_s` | UB→GM 带宽 | 搬运通路效率 |
| `kernels[].deep.l2_hit_rate` | L2 命中率 | 低 = L2 复用差 → 增大块/调访问序复用 |
| `kernels[].task.pipes_us.aic_mte2_time_us` | MTE2(GM读)耗时 | GM 读时间占比 |
| `kernels[].task.pipes_us.aic_mte3_time_us` | MTE3(GM写)耗时 | GM 写时间占比 |
| `kernels[].deep.roofline.memory_utilization` | 访存利用率 | 是否访存瓶颈 + 提升空间 |
| `kernels[].deep.roofline.arithmetic_intensity` | 算术强度 | 确认访存型 → 本层优化优先 |
| `kernels[].task.est_bytes_in/out` | 绝对搬运量 (per-kernel) | 冗余搬运判断: 中间 tensor 写 GM 又读回 → L2 复用/降搬运 |

### 10.5 Tier5 计算占用

**优化什么**: 向量化 / rsqrt / FMA / ILP / bank 冲突。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `kernels[].task.pipes_us.aic_cube_time_us` | cube 耗时 | 计算核心时间; cube 满 = 算力瓶颈 |
| `kernels[].task.pipes_us.aic_scalar_time_us` | 标量耗时 | 标量瓶颈 (地址计算/循环/索引开销) |
| `kernels[].deep.engine_utilization.scalar` | scalar 占比 | 标量是否卡住流水 |
| `kernels[].deep.engine_utilization.fixpipe` | fixpipe 占比 | 定标指令是否卡 |
| `kernels[].deep.compute.cube_ratio` | cube 指令占比 | 是否计算为主 |
| `kernels[].deep.conflict.bank_cflt_ratio` | bank 冲突 | 向量 bank 冲突 (128-bit 对齐可解) |
| `kernels[].deep.conflict.bankgroup_cflt_ratio` | bankgroup 冲突 | bankgroup 冲突 |
| `kernels[].deep.conflict.total_cflt_ratio` | vec 总冲突 | 向量总冲突占比 |
| `kernels[].deep.conflict.wait_ratio` | vec 被阻塞占比 (per-kernel) | 等待/流水气泡 → ILP 提升 |

### 10.6 Tier6 架构专属

**优化什么**: 引擎失衡 / wait_ratio / mte 冲突 / 代码风格。
**读的字段**:

| 字段路径 | 含义 | 为什么读 / 看什么 |
|---|---|---|
| `kernels[].deep.engine_utilization` | 各引擎利用率 | cube/vec/mte2/mte3 是否均衡; 引擎失衡 = 硬件没吃满 |
| `kernels[].deep.conflict.mte_cflt_ratio` | mte 冲突 | 搬运冲突 |
| `kernels[].deep.conflict.wait_ratio` | vec 被阻塞占比 | 向量等搬运/等计算 → 流水线气泡 |
| `kernels[].task.task_type` | 每 kernel 引擎 | 引擎类型分布 |
| `kernels[].task.block_dim` | 核数 | 并行度 |
| `kernels[].deep.roofline.bottleneck_type` | 瓶颈类型 | 最终确认瓶颈是否属本层 |
| `kernels[].deep.engine_utilization.cube/vec/mte2/mte3` | 各引擎占比 (per-kernel) | 逐 kernel 引擎分布 → 调访问/代码结构喂平衡 |

---

## 11. diagnosis.json 完整结构与字段来源

### 11.1 数据流：msprof 产物 → diagnosis.json

```
通用 msprof (任务级)                     msprof op (逐 kernel, 每 kernel 一次)
  05_task/task_prof/                      04_board/op_<i>/OPPROF_*/
  ├─ op_summary*.csv  (每 kernel 一行)     ├─ OpBasicInfo.csv
  ├─ op_statistic*.csv (算子统计)          ├─ PipeUtilization.csv
  ├─ api_statistic*.csv (API 开销)         ├─ ArithmeticUtilization.csv
  └─ l2_cache*.csv     (L2 命中)           ├─ Memory.csv
      │                                    ├─ MemoryL0.csv
      ▼                                    ├─ MemoryUB.csv
  pipeline_parse_task.py                   ├─ L2Cache.csv
      ▼                                    └─ ResourceConflictRatio.csv
  05_task/task.json  (骨架)                    │
      │                                        ▼
      │                            pipeline_parse_board.py
      │                                        ▼
      │                            04_board/board_<i>.json (deep)
      └───────────────┬────────────────────────┘
                      ▼
              integrate.py (按 kernel 名合并 + roofline 计算)
                      ▼
              06_diagnosis/diagnosis.json
```

### 11.2 diagnosis.json 顶层结构

```json
{
  "meta":        {"source": "msprof (generic) + msprof op per-kernel", "num_kernels": N, "filled_kernels": M},
  "summary":     {"num_kernels", "num_kernels_total", "total_ns", "num_cores",
                  "api_overhead_total_us", "l2_hit_rate", "filled_kernels"},
  "kernels": [   {  // 每优化目标 kernel 一个 (= kernel_slot + deep)
      "kernel_name", "framework", "launch_count",
      "task":  { "task_type", "task_duration_us", "block_dim", "input/output_shapes|dtypes",
                 "aicore_time_us", "aiv_time_us", "total_cycles",
                 "pipes_us": {"aic_cube/mac/scalar/mte1/mte2/mte3/fixpipe_time_us",
                              "aiv_vec/scalar/mte2/mte3_time_us"},
                 "est_bytes_in", "est_bytes_out", "transfers": [...] },
      "deep":  { "freq_mhz", "bandwidth_gb_s": {...}, "engine_utilization": {...},
                 "compute": {...}, "conflict": {...}, "l2_hit_rate",
                 "roofline": {"achieved_memory_bw_gb_s", "peak_memory_bw_gb_s", "memory_utilization",
                              "achieved_compute_tflops", "peak_compute_tflops", "compute_utilization",
                              "arithmetic_intensity", "bottleneck_type"} },
      "filled_by": "msprof op"
  }],
  "framework_kernels": [...],   // aclnn* 框架 kernel (非优化目标, 仅观察)
  "api_overhead":    [...],     // 来自 api_statistic.csv
  "multi_kernel":    [...],     // 来自 op_statistic.csv
  "notes": [...]
}
```

### 11.3 字段来源明细（每个字段 ← 哪个文件哪个列 / 怎么算）

#### summary.*（integrate.py 从 task.json 汇总）

| 字段 | 来源文件 → 列 | 怎么解析 |
|---|---|---|
| `num_kernels` | op_summary*.csv `Op Name` 列 | 去重非 aclnn 的 op 名数 = 优化目标 kernel 数 |
| `num_kernels_total` | op_summary*.csv `Op Name` 列 | 去重含 aclnn 的全部 op 名数 |
| `total_ns` | op_summary*.csv `Task Duration(us)` | Σ 非 aclnn 行的耗时 ×1000（与 verify 端到端口径一致） |
| `num_cores` | op_summary*.csv `Block Dim` 列 | 最大值（★实际是 launch grid，910B3 固定 20 核） |
| `api_overhead_total_us` | api_statistic*.csv `Time` 列 | Σ 各行 total_us |
| `l2_hit_rate` | l2_cache*.csv `Hit Rate` 列 | 百分数 >1 归一化到 0~1 |
| `filled_kernels` | — | integrate 统计 deep 被 msprof op 填满的 kernel 数 |

#### kernels[i].task.*（骨架，来自 05_task/task_prof/op_summary*.csv，pipeline_parse_task.py）

| 字段 | 来源列 | 说明 |
|---|---|---|
| `kernel_name` / `framework` | `Op Name` | 是否为 aclnn 前缀（框架 kernel 不进 kernels[]，进 framework_kernels[]） |
| `task_type` | `Task Type` | 引擎类型（Cube/Vec…） |
| `task_duration_us` | `Task Duration(us)` | 单次 launch 耗时 |
| `block_dim` | `Block Dim` | launch grid |
| `input/output_shapes|dtypes` | `Input/Output Shape`、`Input/Output Data Type` | 形状/类型 |
| `aicore_time_us` / `aiv_time_us` / `total_cycles` | `aicore time`/`aiv time`/`total cycles` | cube/向量核时间、周期 |
| `pipes_us.*` | `aic_*_time(us)` / `aiv_*_time(us)` 列 | per-pipe 耗时；兼容列名（cube↔mac、fixpipe↔fixp；cube 缺 mte3 用 aiv_mte3） |
| `est_bytes_in/out` | `Input/Output Shape`×dtype 字节 | 计算估值（每元素搬一次） |
| `transfers[]` | shape+dtype → 字节, pipe 耗时 → 带宽 | 计算：GM读(MTE2)/L1→L0A/B(MTE1)/写回(MTE3)/Cube MAC 每通路 |
| `launch_count` | `Op Name` 同名行数 | 每 kernel 被 launch 几次（占比口径） |

#### kernels[i].deep.*（深层，来自 04_board/op_<i>/OPPROF_*/，pipeline_parse_board.py）

| 字段 | 来源 CSV → 列 | 说明 |
|---|---|---|
| `freq_mhz` | OpBasicInfo `Current Freq` | 运行频率 |
| `bandwidth_gb_s.main_mem_read/write_gb_s` | Memory `main_mem_read/write_bw`（aic 前缀兼容） | GM 读/写带宽 |
| `bandwidth_gb_s.l1_read/write_gb_s` | Memory `l1_read/write_bw` | L1 带宽 |
| `bandwidth_gb_s.l2_read/write_gb_s` | Memory `l2_read/write_bw` | L2 带宽 |
| `bandwidth_gb_s.gm_to_ub/ub_to_gm_gb_s` | Memory `aiv_gm_to_ub_bw` / `aiv_ub_to_gm_bw` | UB↔GM 真实搬运带宽 |
| `bandwidth_gb_s.ub_vector/scalar/mte_read/write_gb_s` | MemoryUB `aiv_ub_*_bw_vector/scalar`、`ub_*_bw_mte` | UB 读写带宽 |
| `bandwidth_gb_s.l0a/l0b/l0c_read/write_gb_s` | MemoryL0 `aic_l0a/l0b/l0c_read/write_bw` | L0 各缓冲带宽 |
| `engine_utilization.cube/vec/mte1/mte2/mte3/scalar/fixpipe` | PipeUtilization `*_ratio`（或 time/总时长） | 各引擎占比 |
| `compute.cube_fops` / `vector_fops` | ArithmeticUtilization `cube_fops` / `aiv_vec_fops` | 真实浮点运算数（FLOPs 来源） |
| `compute.cube_ratio` / `cube_fp16_ratio` / `cube_int8_ratio` | ArithmeticUtilization `cube_ratio` / `cube_fp16_ratio` / `cube_int8_ratio` | 精度分布 |
| `compute.vec_ratio` / `vec_fp32_ratio` / `*_instr_number` | ArithmeticUtilization | 向量占比/指令数 |
| `conflict.*` | ResourceConflictRatio **全列**（bank_cflt_ratio / bankgroup_cflt_ratio / mte_cflt_ratio / wait_ratio / total_cflt_ratio） | 冲突/等待占比 |
| `l2_hit_rate` | L2Cache `Hit Rate`（归一化 0~1） | L2 命中率 |

★带宽统一换算：Ascend Memory*.csv 列是 MB/s，`>=1000` 就 ÷1000 转 GB/s（小带宽列不动，避免误除）。

#### kernels[i].deep.roofline.*（★ integrate.py 计算，不是 CSV 直读）

| 字段 | 计算式 | 用途 |
|---|---|---|
| `achieved_memory_bw_gb_s` | `main_mem_read + main_mem_write` | 实际访存带宽（读+写之和） |
| `peak_memory_bw_gb_s` | hardware_peak.json `gm_bw_gb_s`，无则理论 1638.4 GB/s | 峰值参照 |
| `memory_utilization` | achieved/peak | 访存利用率 |
| `achieved_compute_tflops` | `(cube_fops + vector_fops)/1e12` | 实际算力 |
| `peak_compute_tflops` | hardware_peak.json / 理论 294.9 (fp16) | fp16 峰值 |
| `peak_compute_fp32_tflops` | 73.7 (fp16/4) | fp32 峰值 |
| `compute_utilization` | achieved / max(fp16峰, fp32峰) | 算力利用率（用 max 防 fp32 kernel 被 fp16 峰值低估） |
| `arithmetic_intensity` | `achieved_compute×1e12 / achieved_mem / 1e9` | 算术强度（计算/访存） |
| `bottleneck_type` | `_classify(mem_util, comp_util)` | mem≥0.8&comp<0.5→memory；comp≥0.8&mem<0.5→compute；都<0.3→latency；否则 balanced |

#### framework_kernels[] / api_overhead[] / multi_kernel[]

| 字段 | 来源 CSV → 列 | 说明 |
|---|---|---|
| `framework_kernels[]` | op_summary*.csv（aclnn* 行） | torch_npu 框架 kernel（数据准备/参考），非优化目标，仅观察 |
| `api_overhead[]` | api_statistic*.csv `Level`/`API Name`/`Time`/`Count` | launch/API 开销明细（判断融合收益） |
| `multi_kernel[]` | op_statistic*.csv `OP Type`/`Count`/`Total Time`/`Avg`/`Ratio` | 每类算子次数/耗时（判断是否值得融合） |

> **raw**：task.json 和每个 board_<i>.json 里都保留 `raw` = 对应 CSV 的**全部列+全部行**（不遗漏），diagnosis.json 里只留 normalized 关键字段（Planner 只看筛后的 07 字段）。
