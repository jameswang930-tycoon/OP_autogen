# Triton Agent Optimizer v4.5 — 架构设计（按当前实现）

> **核心思路**: 不靠盲试（AutoKernel 300~400 轮），而是用 **真机 msprof 双源采集（通用 msprof + msprof op）** 精确诊断每个 kernel 的耗时/带宽/算力/瓶颈 → 6 层优化策略逐层推进 → 每轮**两段验证（段1 正确性+Event 快测秒级 → 不快于 best 直接 REVERT; 段2 过门才 msprof 全量确认）**，**严格最优 KEEP（★v4.6: 纯 kernel 绝对延迟 < 历史最小 best_kernel_ns 才进链, msprof 设备侧）**决定保留/回退（Event 快测门只做粗筛省 msprof）, 并以**工业级最优**（各 mode msprof 纯 kernel 口径取 min）为对比天花板，最终算 **vs 工业级比值**（纯 kernel 同尺）看优化效果。
>
> **环境**: Ascend 910B3 (NPU) + CANN 8.5.1 + triton-ascend 3.2.0 + torch_npu 2.9.0 + nga (本地 LLM codeagent)
>
> **版本历史**:
> - **v4.5 (2026-08-13)**: 失败案例库 (memory/failed_cases.py: 按 tier 分文件/指纹去重/两级检索/方案收敛守卫/stuck 黑名单, scheduler 轮内重试累积上下文, coder 检索注入+solved 回填) + 两段验证 (段1 正确性+Event 快测秒级, 不快于 best 直接 REVERT; 段2 过门才 msprof 全量) + Amdahl 显式编排 (planner 每轮看 per-kernel 占比排序) + 跨轮诊断快照 (hist 记 top2 kernel 关键指标 + 全量 JSONL) + v3 就地展开式讲演页
> - **v4.6 (2026-08-18)**: ★口径统一 — KEEP/REVERT 决策主依据从 Event 端到端改为**纯 kernel 绝对延迟** (msprof Task Duration 求和÷遍数 = verify 的 ns) + 欠采硬门槛 (行数<loop 不采纳); speedup/best_speedup/vs_industrial 全部统一纯 kernel 口径 (history speedup 与 best_speedup 同源, 根治「显示 2.x vs best 1.x」矛盾); Event 降为参考 (快测门粗筛 + 报告, best_e2e_event_ns 独立维护不参与决策); 工业级对比只认 method=msprof json (`bench_industrial.py --msprof`); verifier Event 门控改 loop_ok (msprof 漏记不株连); 修 speedup 轮首未初始化崩溃; 回归 32/32
> - **v4.4 (2026-08-12)**: 修复轮: sweep 回滚内容快照/promote 门前置+有效轮计数/rebaseline 同步 best Event/diff.patch 仅成功时写/coder Unicode 清洗 4 缺陷/bench 测量方法学对齐 do_bench (多窗口 median + 输入轮换破 L2); 回归测试 `_sim_fix_regression.py`
> - **v4.3 (2026-08-11)**: bench 全切 Event 设备侧计时 + 每轮补 e2e_event_ns + 严格最优 KEEP (Event 绝对延迟) + best_kernel 绑定 + 失败回滚 + sweep 每 tier3 round 都跑 + 设备污染检测/重置 + vs_industrial 比值

---

## 1. 整体数据流（v4 闭环）

```
input/<op>/kernel_op.py            (源单文件: ① config + ② kernel + ③ test main)
   │  main.py --fresh 可选清 outputs/<op>
   ▼
Scheduler (agents/scheduler.py) — 状态机主循环 (tier 1~6 × round N)
   │
   ├─① 采集+解析  _run_optimize()
   │    bash analyzers/run_optimize.sh <input_dir> <round_dir> M N K   (TIER env; M/N/K 从当前 kernel config 提取, 防 512 默认覆盖)
   │      ├─ warmup 裸跑 (JIT 预热)
   │      ├─ 通用 msprof → pipeline_parse_task.py → task.json      (骨架: 每kernel耗时/launch/api)
   │      ├─ 逐 kernel msprof op → pipeline_parse_board.py → board_<i>.json (deep: 带宽/L2/cube/conflict)
   │      │    (msprof op 崩 AICore → 检测设备级错误 break 剩余 kernel + sync/empty_cache 重置, 防级联)
   │      ├─ integrate.py → diagnosis.json   (骨架+deep 合并, roofline 用 hardware_peak.json 校准)
   │      ├─ 07_tier<N>_fields/*.txt|.json  (extract_tier_fields 筛出全局+当前tier字段)
   │      ├─ check_fields.py → field_check.log   (字段校验明细)
   │      └─ (仅 TIER==2 或 ENABLE_HIVM=1) MULTI 路径: bishengir-compile → HIVM → filter_hivm_for_fusion.py
   │            → 08_fusion/hivm_fusion_view.txt
   │
   ├─(round1 基准) baseline_ns(纯)/baseline_e2e_ns(端到端)/baseline_e2e_event_ns(Event)/num_kernels/num_launches/baseline_mnk/initial_tflops
   │    + pytorch基准(Event, 缺则自动跑 bench_pytorch_*.py) + 工业级基准(各mode取min, 仅真执行)
   │    + verify_end_to_end 复测源 kernel (正确性校验 + warmup + 1×msprof KERNEL_LOOP + Event) → 同口径 baseline
   │
   ├─(round1 + 每个 tier3 round) 分块 sweep  _tier3_sweep_data() → 09_tier3_sweep/
   │    在 best_kernel.py (历史最优) 上程序化枚举全部 L0 合法 BLOCK → 单进程 torch.npu.Event 实测
   │    → 最优写回 round_dir/kernel_op.py → current_kernel 指向它 (结果持久化 st["last_sweep_result"] 每轮传 planner)
   │
   ├─② 诊断筛字段  _diagnose() → 07_tier<N>_fields (Planner 只读这个)
   ├─(Tier2 多一步) _run_fusion() → run_hivm_fusion.py → 08_fusion/fusion_analysis.json
   │
   ├─③ Planner  _plan()  (agents/planner.py generate_v4)
   │    读: skills/triton-op-planner/SKILL.md + docx/playbook_tier<N>.md + 当前 kernel_op.py
   │        + 07 字段 + Amdahl 优先级(per-kernel 占比排序) + planner_context.json(每kernel全量task/deep+占比+Top耗时)
   │        + trajectory(各层进度, 含diag 跨轮诊断快照) + 历史梗概(前层进度) + 手递handoff
   │        + 优秀案例(memory/tier{N}_cases.json) [+ 融合分析 + sweep结论]
   │    出: roundN/plan.md  (JSON: strategy, changes[], promote, promote_to, promote_evidence, handoff)
   │
   ├─④ Coder  _code()  (agents/coder.py apply)
   │    确定性应用 changes[] (old_code→new_code 逐字符替换全部出现处) + 语法检查
   │    出错/LLM 超时 → 失败案例库: 重试上下文 = 前几次(方案+报错)全序列 + 失败库检索注入
   │      (solved方案/stuck黑名单/已试方案 + 禁止原样重试) → 回传 LLM 修复 (≤3 次)
   │    修复成功 → mark_solved 回填失败库; hist 记 error_class 四分类
   │    出: roundN/kernel_op.py + diff.patch
   │
   ├─⑤ 验证 — 两段验证 (TWO_PHASE_VERIFY=1 默认; 0 关闭退全量; stub 自动禁用)
   │    段1 verify_fast_gate (agents/verifier.py, 秒级, 无 msprof):
   │      正确性校验(_correctness_check) + Event 快测(_event_e2e_ns)
   │      → Event ≥ best → 直接 REVERT (省 warmup×3+msprof 几分钟/轮; speedup=None 不拿 Event 派生冒充)
   │    段2 verify_end_to_end (过门才跑, 全量确认+诊断字段):
   │      正确性校验(MATMUL_VERIFY) + warmup(3) + 一次 msprof 内循环 KERNEL_LOOP(30) 次
   │      读全部 op_summary*.csv 合并 → 同一次同算两种口径 ÷ 实测遍数:
   │        纯kernel ns = Σ非aclnn行; 端到端 e2e_ns = Σ全部行(含框架kernel)
   │      + Event 设备侧计时 (e2e_event_ns, 工业级权威绝对值, 无 profiler 扰动):
   │        改写 kernel_op 注入 warmup + KERNEL_EVENT_REPS(5) 个独立窗口 → median → EVENT_E2E_US
   │      (源码无 KERNEL_LOOP 循环 → 自动改除实测遍数, 防虚高; msprof 轮 Event 缺失用段1快测值兜底)
   │
   ├─⑥ 决策  严格最优 KEEP (★v4.6): 本轮纯 kernel 耗时 ns (msprof) < 历史最小 best_kernel_ns 才进链
   │    (欠采硬门槛: 行数<loop 不采纳防假快; 纯 kernel 缺失不保留; Event 快测门已粗筛)
   │    采纳时同步 best_kernel/best_round + 复制 outputs/<op>/best_kernel.py
   │    未采纳/失败 → current_kernel 回滚轮首快照 (内容快照恢复, 防 sweep 轮链污染; 失败代码另存 failed_kernel.py)
   │    记录 history[] {..., ns, e2e_ns(msprof), e2e_event_ns(Event), sweep_ran, sweep_adopted,
   │                    error_class(四分类), diag(跨轮诊断快照紧凑串)}
   │    失败案例: 失败轮自动入库 (指纹去重/attempted_solutions/stuck); 成功轮 solved 回填
   │    全量诊断快照: 每轮写 diag_snapshots.jsonl (per-kernel 关键指标, 审计/讲演, 不入 context)
   │    (每 REBASELINE_EVERY=10 轮: 环境漂移重基准 — 重测原始 baseline + 当前 kernel, 同步 best_e2e_event_ns)
   │    (优秀案例: 本轮相对上一最优 >1.3× → 自动记 memory/tier{N}_cases.json)
   │    (每轮产出策略摘要: strategy_summary.py → final_output/{all,successful}_strategies.md)
   │
   └─⑦ 晋升/停止
        planner.promote+promote_to → 晋升/回退目标层 (支持回退前层)
         或 本 tier 连续 3 轮无改进 → 晋升下一层
         达标不硬停 (D3): speedup ≥ target 后继续探后续层找更大空间 (target 置 -1)
         停止条件: max_rounds 跑满 (有效轮) / Tier6 连续无改进 / 连续3次采集失败
   │
   ▼
outputs/<op>/optimization_trajectory.json   (全局状态 + history, 每轮落盘)
   │
    ▼
feedback/trajectory_chart.py → final_output/trajectory_chart.png
   (加速比曲线 + 各 tier 色带 + KEEP/REVERT 点 + PyTorch 虚线 + 工业级红线[自动读 bench_910b3/outputs/])
   (每轮另有 strategy_summary.py → final_output/{all,successful}_strategies.md)
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
│  extract_tier_fields → 07_tier<N>_fields/*.txt|.json                    │
│  (全局摘要[前层信号] + 当前 tier 专属字段)                               │
│  check_fields.py → 05_task/field_check.log                              │
│  (TIER==2 或 ENABLE_HIVM=1) bishengir-compile → filter_hivm → 08_fusion/│
│  run_hivm_fusion.py → 08_fusion/fusion_analysis.json                    │
│  (round1 + 每 tier3 轮) sweep_blocks.py → 09_tier3_sweep/               │
│  (在 best_kernel.py 上枚举 L0 合法 BLOCK → torch.npu.Event 实测最优)    │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                              AGENT LAYER                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │   LLMClient (agents/llm_client.py) — Planner/Coder 共用的 LLM 入口    │ │
│  │   echo "<prompt>" | nga run (CLI 优先) / API / stub 三模式            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐             │
│  │ Planner (LLM)  │  │  Coder (确定性)│  │  Verifier      │             │
│  │ ────────────── │  │  ───────────── │  │  ─────────────  │             │
│  │ 读 SKILL +     │  │ 应用 changes[]  │  │ 正确性校验+    │             │
│  │ playbook_tierN │  │ old→new 逐字符 │  │ msprof 双口径 +│             │
│  │ + 07字段 +     │  │ 替换全部出现处 │  │ Event 设备侧  │             │
│  │ planner_ctx +  │  │ +LLM修错(≤3)   │  │ (e2e_event_ns) │             │
│  │ 轨迹 + 手递 +  │  │ Unicode 清洗   │  │ → ns/e2e_ns +  │             │
│  │ 优秀案例       │  │ 语法检查       │  │   e2e_event_ns │             │
│  │ → plan.md      │  │ → roundN/      │  │  (循环丢自校正) │             │
│  │  (changes[]+   │  │   kernel_op.py │  │                │             │
│  │   promote_to)  │  │   +diff.patch  │  │                │             │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘             │
│          └───────────────────┼───────────────────┘                       │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │           Scheduler (Python 状态机 — 决策核心)                │       │
│  │  严格最优 KEEP: 纯 kernel ns < 历史最小才进链 (绑定 best)  │       │
│  │  未采纳/失败 → current_kernel 内容快照回滚 (保 input 链)       │       │
│  │  晋升门前置(无依据转正常轮)·有效轮计数(max_rounds 硬上限)      │       │
│  │  设备污染检测: verify 崩 AICore → 下轮采集前 _reset_device     │       │
│  │  promote_to 晋升/回退 · 本层无改进3轮兜底 · target/Tier6 停    │       │
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
│  outputs/<op>/optimization_trajectory.json  ( 中枢状态)                 │
│  outputs/<op>/final_output/  最终产物: kernel_op.py + final_summary.json│
│    + trajectory_chart.png  (加速比曲线+PyTorch虚线+工业级红线)             │
└──────────────────────────────────────────────────────────────────────────┘
```

## 1B. 完整数据流图（每阶段：读什么 → 执行什么 → 写什么/传什么）

> 路径约定：`round_dir` = `outputs/<op>/<tier_name>/roundN/`（每轮一个，自包含）。
> 每个阶段框按「**读**（输入文件: 位置/内容）→ **执行** → **写**（产出文件 + 传给下一步的信息）」标注。

```
═══════════════════════════════════════════════════════════════════════
 阶段 0 │ 输入
═══════════════════════════════════════════════════════════════════════
输入文件:
  input/<op>/kernel_op.py              源算子单文件
    ├ ① config: M/N/K/DTYPE/BLOCK_* (env 可覆盖)
    ├ ② @triton.jit kernel(s)  (被优化的对象)
    └ ③ __main__ 测试驱动: KERNEL_LOOP 循环 + MATMUL_VERIFY 正确性校验块
  main.py 启动校验: validate_kernel_op 检查 ①KERNEL_LOOP ②MATMUL_VERIFY ③__main__
    └ ④同名 kernel 多次调用(A1 聚合风险警告) — 只警告不阻塞
  bench_910b3/outputs/*.json           (仅 round1 读) 基准线
    ├ pytorch_<op>_tflops.json         PyTorch 基准 (Event 设备侧计时, 缺则自动跑 bench_pytorch_*.py)
    └ industrial_<op>_<mode>_tflops.json  工业级各 mode (eager/compile/cann-fused/fa, Event)
  outputs/<op>/optimization_trajectory.json  (--resume 续跑读; 否则 round1 新建)
                    │
                    ▼  (scheduler 每轮建 round_dir, 把源 kernel 自包含拷进 round_dir/input/)

═══════════════════════════════════════════════════════════════════════
 阶段 1 │ 采集+解析  (bash analyzers/run_optimize.sh)  —— 五条支线
═══════════════════════════════════════════════════════════════════════
前置 (进入本阶段前, scheduler 每轮检查):
  - 上轮 verify 崩 AICore (设备污染) → 先 _reset_device (子进程 sync+empty_cache)
  - 每 REBASELINE_EVERY=10 轮 → 环境漂移重基准 (见阶段 6)

 ┌─ 支线 A · warmup 预热 ──────────────────────────────────────────────┐
 │  读: round_dir/input/kernel_op.py                                   │
 │  执行: 裸跑一次 (JIT 编译 + 设备初始化, 防 msprof 首次漏采)          │
 │  写: round_dir/05_task/warmup.txt  (崩溃 trace 在这查)              │
 └────────────────────────────────────────────────────────────────────┘

 ┌─ 支线 B · 通用 msprof (任务级骨架) ─────────────────────────────────┐
 │  读: round_dir/input/kernel_op.py                                   │
 │  执行: msprof --application=python3 kernel_op.py --ai-core=on       │
 │  产出 msprof 原始 CSV (round_dir/05_task/task_prof/):               │
 │    op_summary*.csv     每 kernel 一行: Op Name/Type, Task Type,     │
 │                        Task Duration(us), Block Dim, Input/Output   │
 │                        Shape&Dtype, aic_*/aiv_*_time(us), cycles   │
 │    op_statistic*.csv   每类算子 次数/总耗时/占比                     │
 │    api_statistic*.csv  launch/API 开销明细                          │
 │    l2_cache*.csv       L2 命中率                                    │
 │  解析: pipeline_parse_task.py  (按 distinct Op Name 去重成 slot)    │
 │  写: round_dir/06_diagnosis/task.json                              │
 │    ├ execution_summary: total_ns / num_kernels / num_cores         │
 │    ├ normalized.kernel_slots[]  每 kernel: task{...}, deep=null     │
 │    ├ normalized.framework_kernels[]  (aclnn* 框架, 非优化目标)       │
 │    └ normalized.multi_kernel[] / api_overhead[] / l2_hit_rate      │
 └────────────────────────────────────────────────────────────────────┘
                    │  (task.json 给出 distinct 非 aclnn kernel 名单)
                    ▼
 ┌─ 支线 C · 逐 kernel msprof op (深层画像, 循环每个 kernel) ──────────┐
 │  读: task.json 的 kernel 名单 + round_dir/input/kernel_op.py        │
 │  执行 (循环每个 KNAME): msprof op --kernel-name=KNAME               │
 │  产物 msprof op 原始 (round_dir/04_board/op_<i>/OPPROF_*/, 8 CSV):  │
 │    OpBasicInfo.csv          Task Duration/Block Dim/Freq            │
 │    PipeUtilization.csv      各 pipe time & ratio                    │
 │    ArithmeticUtilization.csv cube/vec fops, 精度占比                 │
 │    Memory.csv               main_mem/l1/l2/gm_to_ub 带宽            │
 │    MemoryL0.csv             l0a/l0b/l0c 带宽                        │
 │    MemoryUB.csv             ub vector/scalar 带宽                   │
 │    L2Cache.csv              hit rate                               │
 │    ResourceConflictRatio.csv bank/mte/wait 冲突                     │
 │  解析: pipeline_parse_board.py  (MB/s→GB/s, 归一化)                 │
 │  设备污染防护: msprof op 失败(缺 Memory.csv) 时 grep 设备级错误    │
 │    (aclrt/575/npu function) → break 剩余 kernel + sync/empty_cache   │
 │    重置设备 (防级联: 污染设备下轮通用 msprof 也采不到)              │
 │  写: round_dir/06_diagnosis/board_<i>.json  (每 kernel 一个 deep)   │
 │    ├ engine_utilization{} / bandwidth_gb_s{} / compute{}           │
 │    ├ conflict{} / l2_hit_rate                                      │
 │    └ raw (8 CSV 全字段)                                            │
 └────────────────────────────────────────────────────────────────────┘
                    │
                    ▼  (board_<i>.json 按 kernel 名回填进 kernel_slot)
 ┌─ 合并 · integrate.py ──────────────────────────────────────────────┐
 │  读: task.json + board_*.json + bench_910b3/hardware_peak.json     │
 │  执行: 按 kernel 名填 slot.deep; 用峰值算 roofline (mem/comp util)  │
 │  写: round_dir/06_diagnosis/diagnosis.json  (本轮诊断中枢)        │
 │    ├ summary: num_kernels/total_ns/api_overhead_total_us/l2/...    │
 │    └ kernels[i]{ task(骨架), deep{bandwidth,engine,compute,        │
 │              conflict,l2, roofline{util,bottleneck_type}}, filled_by│
 └────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─ 字段筛选 + 校验 ──────────────────────────────────────────────────┐
 │  读: diagnosis.json + 当前 TIER                                     │
 │  执行: extract_tier_fields(diagnosis, TIER)  + check_fields.py     │
 │  写: round_dir/07_tier<N>_fields/                                  │
 │    ├ tier<N>_fields.txt    全局摘要 + per-kernel + 本层字段         │
 │    ├ tier<N>_fields.json   结构化                                  │
 │    └ planner_context.json 完整数据上下文 (planner 直接 cat 读)      │
 │  写: round_dir/05_task/field_check.log  (列名不匹配明细)            │
 └────────────────────────────────────────────────────────────────────┘

 ┌─ 支线 D · (TIER==2 多kernel 或 ENABLE_HIVM=1) HIVM 融合分析 ────────┐
 │  读: round_dir/input/kernel_op.py                                  │
 │  执行: (rm ~/.triton 后重编译取 ttadapter) bishengir-compile → HIVM │
 │  写: round_dir/08_fusion/hivm_try.txt → hivm_fusion_view.txt       │
 │  执行: run_hivm_fusion.py (nga run fusion skill 分析 RAW/WAR/WAW)   │
 │  写: round_dir/08_fusion/fusion_analysis.json                      │
 │    └ raw_deps[]/war_deps[]/waw_deps[]/fusion_candidates[]          │
 └────────────────────────────────────────────────────────────────────┘

 ┌─ 支线 E · (仅 round1 或 tier==3) Tier3 分块 sweep ─────────────────┐
 │  读: best_kernel.py (历史最优; 缺则当前 kernel_op.py)              │
 │      (读 BLOCK 当前值 + 拷一份到 round_dir/kernel_op.py, 不碰源)   │
 │  执行: sweep_blocks.sweep() 程序化枚举全部 L0 合法 BLOCK + 单进程    │
 │        torch.npu.Event 实测 (崩溃续跑/增量保存/设备污染恢复)        │
 │  写: round_dir/09_tier3_sweep/                                     │
 │    ├ sweep_result.json  configs[]{block,ns,speedup} + best + written│
 │    └ sweep_runner.py    (生成的 runner 脚本)                       │
 │  写: round_dir/kernel_op.py  (sweep 把最优 BLOCK 写回, 覆盖; 该轮 verify 失败时
 │                               scheduler 用内容快照恢复轮首, 失败代码另存 failed_kernel.py)│
 │  传: best/written 给 planner_context (告诉它 BLOCK 已穷举)         │
 └────────────────────────────────────────────────────────────────────┘
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 2 │ round1 设基准 (仅首轮执行; 后续轮跳过)
═══════════════════════════════════════════════════════════════════════
读: diagnosis.summary (total_ns/num_kernels)
    bench_910b3/outputs/{pytorch,industrial}_*.json  (各 mode 取 min = 工业级天花板)
    源 kernel_op.py (verify 复测用)
执行: verify_end_to_end 复测源 kernel (正确性校验+warmup+msprof+Event → baseline_e2e_ns + baseline_e2e_event_ns)
写: outputs/<op>/optimization_trajectory.json 的 state{}
    ├ baseline_ns(纯) / baseline_e2e_ns(端到端) / baseline_e2e_event_ns(Event 设备侧) / num_kernels / num_launches
    ├ baseline_mnk / initial_tflops
    ├ pytorch_time_us(Event, 缺则自动跑 bench_pytorch_*.py) / industrial_time_us(各 mode 取 min, 仅真执行) / industrial_baseline
    └ current_kernel (→ 源 kernel_op.py)
写: outputs/<op>/baseline_verify/msprof_0/  (复测产物)
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 3 │ Planner  (agents/planner.py → nga run LLM)
═══════════════════════════════════════════════════════════════════════
读 (planner 全部输入文件):
  - skills/triton-op-planner/SKILL.md                      输出格式/铁律
  - docx/playbook_tier<N>.md                               本层「问题→方案→正确代码」
  - round_dir/07_tier<N>_fields/tier<N>_fields.txt         筛好的诊断字段
  - round_dir/07_tier<N>_fields/planner_context.json       完整数据上下文
  - 当前 kernel_op.py (current_kernel, 整份代码)            改的对象
  - outputs/<op>/optimization_trajectory.json              state + 全量 history(前层试过啥)
  - round_dir/08_fusion/fusion_analysis.json               (仅 tier2)
  - round_dir/09_tier3_sweep/sweep_result.json             (sweep 结论: BLOCK 已穷举)
  - state.handoff                                          (跳转来的瓶颈分析+方向)
  - memory/tier{N}_cases.json                              (本层优秀案例: 历史大加速比轮次, 参考学习)
执行: nga run 调 planner skill → LLM 分析瓶颈 → 出策略
写: round_dir/plan.md  (JSON)
    ├ strategy + expected_impact
    ├ changes[]{old_code, new_code, reason, section, tier}  精确替换片段
    ├ promote / promote_to / promote_evidence / handoff      晋升决策
传: plan 对象 → 交给 Coder
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 4 │ Coder  (agents/coder.py apply)
═══════════════════════════════════════════════════════════════════════
读:
  - round_dir/plan.md 的 changes[]            要应用的 old_code→new_code
  - 当前 kernel_op.py (current_kernel)        原始代码
  - skills/triton-op-coder/SKILL.md           改码铁律 (ASCII/不改函数名/...)
  - previous_error (累积重试上下文)           上轮报错(有则走 LLM 修复)
  - memory/failed_cases.py (tier{N}_failed_cases.json)   失败库检索注入 (两级检索)
执行:
  - 无错: 确定性 code.replace(old_code, new_code) (替换全部出现处, 最稳)
  - 有错: 重试上下文 = 前几次尝试(方案+报错)全序列 + 本次报错 + 失败库注入
          (solved 方案/stuck 黑名单/已试方案, 只读参考严禁抄写; 禁止原样重试已失败方案)
          → nga run 调 coder skill 带错误修复; 修复成功 → mark_solved 回填失败库
  - Python 语法(报错带行内容) + 函数完整性 + no-op 校验
  - Unicode 清洗: 千分位归一(1，024→1024) / 全角数字转ASCII / markdown剥壳 / 智能引号(定界符换·内容删) / 兜底二次验证
写: round_dir/kernel_op.py   (本轮新代码, 不碰源 input/)
    round_dir/diff.patch    (与上一轮差异)
传: optimized_code 写入 round_dir/kernel_op.py → 交给 Verifier
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 5 │ Verifier  (agents/verifier.py)  —— 两段验证
═══════════════════════════════════════════════════════════════════════
══ 段1 verify_fast_gate (秒级, 无 msprof):
读: round_dir/kernel_op.py
执行: _correctness_check (MATMUL_VERIFY=1, 必须 "result check: PASS") + _event_e2e_ns (Event 快测)
传: {"ok": bool, "e2e_event_ns": float|None}
    → scheduler 判门: Event ≥ best → 直接 REVERT (省 warmup×3+msprof 几分钟/轮)
    → 过门 / Event 不可用 → 段2 全量

══ 段2 verify_end_to_end (全量, 过门才跑):
读: round_dir/kernel_op.py (验证 coder 的新输出)
执行 (三步):
  ① warmup × VERIFY_WARMUP(3)  裸跑预热 JIT/cache
  ② 正确性校验  _correctness_check (同上; 不过 → 本轮 FAIL, 错误回传 Coder 同轮重试 ≤3 次)
  ③ msprof 测时  一次 msprof, app 内 KERNEL_LOOP(30) 遍
  ④ Event 设备侧 (多窗口 median): 注入 KERNEL_EVENT_REPS(5) 个独立 Event 窗口
     (每窗口包 LOOP 次, 设备流水连续最后 sync) → 取 median → e2e_event_ns
读: round_dir/msprof_0/op_summary*.csv  (全部合并; msprof 可能拆多文件)
执行: _read_durations 同一次算两种口径 ÷ 实测遍数:
       纯 kernel ns = Σ 非 aclnn 行;   端到端 e2e_ns = Σ 全部行(含框架)
写: round_dir/msprof_0/  (msprof 原始产物)
传 (返回 dict, 不写业务文件): {ok, ns(纯), e2e_ns(端到端), e2e_event_ns(Event 设备侧), speedup, loop, rows}
    (msprof 轮 Event 缺失 → scheduler 用段1快测值兜底)
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 6 │ 决策 + 记录  (scheduler)  —— 回路
═══════════════════════════════════════════════════════════════════════
执行:
  - 主加速比(显示) = baseline_e2e_ns / e2e_ns  (纯 kernel 兜底)
  - 严格最优 (★v4.6): ns(纯 kernel, msprof) < best_kernel_ns (历史最小) → KEEP (current_kernel → 本轮 round_dir)
    (欠采 行数<loop / 纯 kernel 缺失 → 不采纳; best_speedup = baseline_ns/best_kernel_ns 派生, 与 history speedup 同源)
    否则 → REVERT (内容快照回滚轮首 = 沿用历史最优 kernel)
  - 晋升/回退: planner.promote_to (需 promote_evidence, 可回退前层) + 连续 3 轮无改进兜底晋升
  - (晋升时) 写跳转手递: round_dir/10_tier_handoff.json
  - (每 REBASELINE_EVERY=10 轮, 在阶段1前) 环境漂移重基准: 重测原始 baseline_kernel.py + 当前 kernel
    → 校正 baseline_ns/baseline_e2e_ns/baseline_e2e_event_ns + current_speedup, 同步 best_e2e_event_ns
    (用 best_kernel.py 新环境复测; 否则新环境改进轮永远比不过旧 best → 卡死 REVERT)
  - (优秀案例) 本轮相对上一最优 > EXCELLENT_THRESHOLD(1.3×) → 自动记 memory/tier{N}_cases.json
  - (失败案例) 失败轮自动入库 memory/tier{N}_failed_cases.json: 归一化签名指纹去重 + attempts+1
    + attempted_solutions 方案历史 (同方案不重复记); attempts≥3 → stuck (封原方案, 不封新方案);
    成功轮 mark_solved 回填 (方案 + fix_diff); solved 再现 → 自动降级重计
  - (error_class) hist 记失败四分类: env / code_compile / code_numeric / code_runtime
  - (诊断快照) hist 记 top2 kernel 紧凑串 (bn/cu/mu/l2/redun/引擎) → planner 看"改法→指标变化→结果";
    全量 per-kernel 快照写 diag_snapshots.jsonl (审计/讲演, 不入 context)
  - (每轮) strategy_summary.py → final_output/{all,successful}_strategies.md
 写: outputs/<op>/optimization_trajectory.json
    ├ state 更新 (tier/round/current_speedup/current_kernel/best_speedup/best_e2e_event_ns/sweep/...)
    └ history 追加一条 {round,tier,strategy,change,changes_full,
                         speedup(e2e),kernel_speedup(纯),ns,e2e_ns,e2e_event_ns,
                         decision(KEEP/REVERT/FAIL),result(OK/NOOP/FAIL),error,error_class,
                         diag(诊断快照),tflops,sweep_ran,sweep_adopted}
                    │
                    ▼  回【阶段 1】跑下一轮 (tier/round 推进)
                       直到有效轮跑满 max_rounds / Tier6 连续无改进 / 连续3次采集失败 → 出循环
                    │
                    ▼

═══════════════════════════════════════════════════════════════════════
 阶段 7 │ 最终产物 (循环结束后, 一次性生成)
═══════════════════════════════════════════════════════════════════════
读: outputs/<op>/optimization_trajectory.json (state + history)
    bench_910b3/outputs/{pytorch,industrial}_*.json (轨迹图虚线)
写: outputs/<op>/final_output/
    ├ kernel_op.py         当前已采纳的最优 kernel (取 best_kernel, 可直接取用)
    ├ baseline_kernel.py   baseline 副本
    ├ final_summary.json   双口径加速比 + Event 延迟 + vs_industrial_ratio
    ├ trajectory_chart.png 加速比曲线 + tier 色带 + KEEP/REVERT 点
    │                      + PyTorch 灰虚线 + 工业级红虚线
    ├ all_strategies.md    全部轮次策略记录 (每轮由 strategy_summary.py 产出)
    └ successful_strategies.md  仅成功优化策略 (KEEP+严格超越上一轮)
```

**一句话主线**：`kernel_op.py` →(采集)→ `diagnosis.json` →(筛字段)→ `07_fields` →(planner)→ `plan.md` →(coder)→ `roundN/kernel_op.py` →(verify)→ `ns/e2e_ns` →(决策)→ `optimization_trajectory.json` →(结束)→ `final_output/`。

---

## 2. 逐轮执行流程（Scheduler 状态机）

```
main.py input/<op> [--fresh] [--resume] [--max-rounds N] [--target X] [--stub]
  │
  ├─ 单文件 kernel_op.py 若缺 → merge_single_file.py 生成
  ├─ --fresh → 清空 outputs/<op>
  ├─ _Tee → stdout/stderr 双写终端 + outputs/<op>/optimization.log (UTF-8)
  └─ Scheduler(op_dir, max_rounds, target, stub, resume)

while (total_rounds - promote_budget) < max_rounds:      # 有效优化轮计数 (promote 轮免费, max_rounds 硬上限)
    round_dir = outputs/<op>/<tier_name>/roundN
    diagnosis = _run_optimize(round_dir, tier)   # 采集失败→重试1次→跳过→连3次停(H2); 设备污染→下轮采集前重置
    if baseline_ns is None:                      # 首轮设基准
        baseline_ns / baseline_e2e_ns / baseline_e2e_event_ns / num_kernels / num_launches / baseline_mnk / initial_tflops
        + pytorch基准(Event) + 工业级基准(各mode取min, 仅真执行)
        + verify_end_to_end 复测源 kernel (VERIFY_BASELINE=1) → 与后续轮同口径
    extracted = _diagnose(diagnosis, tier, round_dir)   # → 07_tier<N>_fields
    if tier == 2: fusion_analysis = _run_fusion(round_dir)   # → 08_fusion/
    if rn == 1 or tier == 3: tier3_sweep = _tier3_sweep_data(tier, rn, round_dir)
        # 程序化枚举 L0 合法 BLOCK 候选 → 真机 Event 实测 → 最优写回 kernel_op.py
    plan = _plan(diagnosis, extracted, tier, rn, round_dir, fusion_analysis, tier3_sweep)
    # 晋升门前置: promote 无 promote_evidence/reason → 本轮转正常优化轮 (不白耗, 不涨 budget)
    if plan.promote and 有依据:                     # 晋升轮: 原样拷贝 kernel, 不调 LLM
        copy current_kernel → roundN/kernel_op.py
    else:
        for attempt in range(3):
            new_code = _code(plan, rn, round_dir, prev_err)   # → roundN/kernel_op.py (+diff.patch 仅成功时写)
            v = _verify(round_dir, baseline_ns)               # 正确性校验 + msprof 双口径 + Event 注入计时
            if v.ok: break
    # KEEP 决策 = 纯 kernel 绝对延迟 (msprof ns; ★v4.6 主口径, Event 为参考):
    if ns < best_kernel_ns (且非欠采): current_kernel = roundN/kernel_op.py; KEEP
    else: REVERT (内容快照恢复轮首 = 沿用历史最优; 失败代码另存 failed_kernel.py)
    history.append(...); _save_traj()
    # 晋升/停止判定 → 下一轮/下一 tier/停止
```

---

## 3. 完整文件架构图（所有文件夹/文件 → 主要作用）

> 标注:  = 核心链路关键文件; 括号内为该文件/文件夹的主要职责。

```
triton_agent_optimizer/
├── main.py                        # 入口: 解析 input/<op> 参数 → 启动 Scheduler
│                                  #   (--fresh/--resume/--max-rounds/--target/--stub); Tee 日志双写 optimization.log
├── config.py                      # 全局配置中心: 检测本地/服务器环境, 集中所有路径/阈值 (from config import Config)
├── _sim_flow_test.py              # 端到端链路模拟回归: 打桩真机(msprof/HIVM/LLM), 真实 scheduler 逐轮执行 → 验证全流程
├── _sim_edge_test.py              # 边界场景回归 #2: 采集失败/回退/晋升/目标/基准 等失败分支 (全真实执行 scheduler/coder/verifier)
├── _sim_fix_regression.py         # 修复回归 #3 (30 项断言): 全链路模拟 + 4 处修复验证 (sweep回滚链/promote门前置/max_rounds硬上限/
│                                  #   rebaseline best同步) + coder 清洗回归 (千分位/全角/markdown/引号两难) + resume 续跑 + tier3 sweep 场景
├── README.md                      # 项目总览 + 开发记录 (会话开始先扫这里回忆进度)
├── ARCHITECTURE_DESIGN.md         # 本文档: 架构设计 + 逐组件说明 (按当前实现维护)
├── IMPLEMENTATION_PLAN.md         # 逐文件实现计划 (实现已完成, 作架构参考保留)
├── .env / .env.template           # 环境密钥/配置 (LLM key 等, 不入库)
├── .gitignore                     # git 忽略规则

├── agents/                        # ── Agent 层: 调度状态机 + 规划/改码/验证 + LLM 入口 ──
│   ├── scheduler.py               # 调度状态机 (核心): 驱动每轮 采集→诊断→Planner→Coder→Verify→决策→晋升;
│   │                              #   round1 设基准(纯kernel+端到端+Event+工业级); KEEP 决策=纯 kernel ns (欠采/缺失不采纳; Event 快测门粗筛);
│   │                              #   晋升门前置(无依据转正常轮)+有效轮计数(max_rounds 硬上限); 回滚内容快照(failed_kernel 留证);
│   │                              #   rebaseline 同步 best Event; extract_tier_fields 筛字段(+Amdahl 优先级行);
│   │                              #   两段验证接入(段1 Event 快测门); 失败案例库接入(重试上下文+solved 回填+error_class);
│   │                              #   跨轮诊断快照(hist diag + diag_snapshots.jsonl); sweep 触发; 优秀案例; 最终产物 final_output/
│   ├── planner.py                 # Planner (LLM): 读 SKILL+playbook+07字段+planner_context+轨迹+手递 → plan.md (changes[]+promote)
│   ├── coder.py                   # Coder (确定性改码+LLM修复): 应用 changes[] 逐字符替换 → roundN/kernel_op.py + diff.patch
│   │                              #   语法(报错带行内容)/函数完整性/no-op 校验; Unicode 清洗(千分位/全角/markdown/引号两难/兜底二次验证)
│   ├── verifier.py                # Verifier: MATMUL_VERIFY 正确性校验 → msprof 双口径(ns/e2e_ns)
│   │                              #   + Event 设备侧计时 (e2e_event_ns, _inject_event_timing 注入 KERNEL_LOOP)
│   │                              #   返回 {ok, ns, e2e_ns, e2e_event_ns, speedup}
│   ├── llm_client.py              # LLM 统一入口: nga run CLI / API / stub; LLM_CLI_TIMEOUT=3600s
│   └── __init__.py

├── analyzers/                     # ── 采集+解析层: run_optimize.sh 驱动 msprof 双源 → diagnosis.json + 07 字段 ──
│   ├── run_optimize.sh            # 采集驱动: warmup → 通用msprof→task.json + msprof op→board_<i>.json
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
│   ├── sweep_blocks.py            # Tier3 分块扫描 v2 (从根目录移入): 程序化枚举 L0 合法 BLOCK
│   │                              #   + 单进程 torch.npu.Event 实测; ×0.9 边距/增量保存/崩溃续跑/设备污染恢复
│   └── __init__.py

├── bench_910b3/                   # ── 基准测量层: 工业级基准 + PyTorch 基准 + 硬件峰值 (产物统一 outputs/) ──
│   ├── bench_common.py            # msprof 测量工具 (诊断用): BENCH_OUT=outputs/; clean_bench_out();
│   │                              #   measure_msprof / measure_msprof_op (bandwidth/board 解析)
│   ├── bench_industrial.py        # 工业级基准 (Event 设备侧计时): eager/compile/cann-fused/fa → outputs/industrial_<op>_<mode>_tflops.json
│   ├── bench_all.py               # 全算子最优: 跑全部模式 → 每算子 Event 取 min(仅真执行) → outputs/industrial_summary.json
│   │                              #   --clean 一键清空产物; 明细表含 执行状态/融合判定
│   ├── bench_pytorch.py           # torch.matmul (两层 MLP) 基准 (自带 Event 计时)
│   ├── bench_pytorch_mlp.py       # torch MLP 基准 (matmul 对照)
│   ├── bench_pytorch_attention.py # 自注意力+MLP 基准 (attention_mlp 对照)
│   ├── bench_pytorch_flash_attention.py   # CANN FA 基准 (flash_attention 对照)
│   ├── bench_pytorch_conv2d.py / bench_pytorch_conv_bias_relu.py   # 卷积基准
│   │                              # 新算子 (conv1d/batchnorm2d/maxpool2d) 无 pytorch 基准 → 图不画 PT 虚线, 不阻塞
│   ├── bench_pytorch_matmul_relu.py / bench_pytorch_matmul_transpose.py   # matmul 变体基准
│   ├── bench_pytorch_rms_norm.py / bench_pytorch_layernorm.py / bench_pytorch_sigmoid.py   # 归一化/逐元素基准
│   ├── bench_config.py            # 变体注册表 + 静态 bytes/flops 计算 + PT_BENCH_MAP (算子→pytorch json 映射)
│   ├── bench_kernels.py           # 硬件基准 triton kernels (read/write/copy/l2/cube/vec 多变体)
│   ├── run_bench.py               # 硬件峰值校准: 多策略实测取最大 → hardware_peak.json (integrate 读它做 roofline 峰值)
│   ├── bench_theory.py            # 910B3 理论峰值计算 + 理论/实测对照 (纯本地, 无 NPU 依赖)
│   ├── README.md                  # bench 使用说明
│   └── outputs/                   # 基准产物收纳 (运行时生成): industrial_*.json / pytorch_*.json
│                                  #   / industrial_summary.json / msprof 临时目录

├── input/                         # ── 算子源文件层: 每算子一目录, kernel_op.py 单文件 (config+kernel+test 一体) ──
│   ├── matmul/            kernel_op.py   # 两层 MLP (GELU(X@W1+b1)@W2)
│   ├── attention_mlp/     kernel_op.py   # 自注意力 + MLP (9-kernel)
│   ├── matmul_relu/       kernel_op.py   # matmul + ReLU
│   ├── matmul_transpose/  kernel_op.py   # matmul (B 转置)
│   ├── flash_attention/   kernel_op.py   # FlashAttention (K 预转置)
│   ├── conv2d/            kernel_op.py   # 卷积
│   ├── conv_bias_relu/    kernel_op.py   # 卷积 + bias + ReLU
│   ├── conv1d/            kernel_op.py   # 1D 卷积 (外积版, Tier1 im2col 教学)
│   ├── batchnorm2d/       kernel_op.py   # BatchNorm2d 推理 (按通道归约)
│   ├── maxpool2d/         kernel_op.py   # MaxPool2d 窗口最大池化
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
│   ├── trajectory_chart.py        # 轨迹图: optimization_trajectory.json → final_output/trajectory_chart.png
│   │                              #   (加速比曲线+各tier色带+KEEP/REVERT点+PyTorch虚线+工业级红线, Event-vs-Event 同口径)
│   ├── strategy_summary.py        # 策略摘要: 每轮自动产 final_output/{all,successful}_strategies.md
│   │                              #   (successful 仅 KEEP+严格超越上一轮; 排除 promote/REVERT/FAIL/采集失败)
│   ├── pipeline_diagrams.html     # 讲演页 v1: 静态 SVG 双图 (闭环架构 + 完整数据流)
│   ├── pipeline_diagrams_v2.html  # 讲演页 v2: 弹出式下钻
│   ├── pipeline_diagrams_v3.html  # 讲演页 v3: 就地展开式 (点击块内 + 就地展开面板, CSS grid 固定主链一行, 零依赖)
│   └── __init__.py

├── memory/                        # ── 记忆层 ──
│   ├── excellent_cases.py         # 优秀案例自动记录/检索 (EXCELLENT_THRESHOLD=1.3×, planner 优化前参考)
│   ├── failed_cases.py            # 失败案例库 (按 tier 分文件 tier{N}_failed_cases.json):
│   │                              #   归一化签名指纹去重 + 两级检索(指纹精确+关键词相似) + attempted_solutions
│   │                              #   方案收敛守卫 + open→solved|stuck 状态机 + 负正闭环 + LRU 上限
│   │                              #   coder 修复注入 / scheduler 重试上下文累积 / 成功 solved 回填
│   ├── codeerror/                 # 早期 coder 错误修复记录 (按 kernel 分, 已废弃由 failed_cases 取代)
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

├── prepare/                       # ── 环境准备 ──
│   └── setup_and_verify.sh        # 环境安装/校验脚本
```

---

## 4. 各组件详解

### 4.1 Scheduler — 状态机核心
- **状态**: `traj["state"]` = {tier, round, total_rounds, promote_budget, best_speedup, **best_kernel_ns(KEEP 主依据: 历史最小纯 kernel 耗时, msprof)**, best_e2e_event_ns(快测门粗筛+报告参考), baseline_ns(纯), baseline_e2e_ns(端到端 msprof), **baseline_e2e_event_ns(Event)**, num_kernels, num_launches, baseline_mnk, initial_tflops, pytorch基准, industrial_time_us(Event 各 mode min), current_speedup, current_kernel, **best_kernel, best_round, best_e2e_ns**, vs_industrial_ratio, last_sweep_result, handoff, tier_jumps, last_rebase_round}
- **严格最优 KEEP (★v4.6 口径统一)**: 本轮纯 kernel 耗时 `ns < best_kernel_ns`（历史最小, msprof Task Duration 求和÷遍数）才进链 — 与优化对象 (triton.jit kernel) 同尺, host/框架成分不掺入; **欠采硬门槛**（行数 < loop → 求和偏小假快 → 不采纳）+ 纯 kernel 缺失不保留; speedup/best_speedup 全部 `baseline_ns/ns` 派生（与 history 同源, 根治「显示 2.x vs best 1.x」口径矛盾）
- **假小 Event 防护 (2026-08-12, 用户报告 "加速比突然 200x → 真实优化轮永不 KEEP"; 同日简化)**: coder 改坏的代码在 KERNEL_EVENT_TIME 模式下窗口没跑满 (launch 被移走/条件包裹/循环改坏) → Event 假小 → 毒 best。防护原则（用户定版）: **Event 测对就保留, 初始代码差时几百上千倍加速比真实存在, 绝无比值拦截**:
  - Event 真实性保证 (★2026-08-18 改为按 loop_ok 门控): 源码 KERNEL_LOOP 循环丢失 (loop_ok=False) → **不测 Event (返回 None)** → Event 参考值缺失（原因进 hist error）; msprof 漏记 (行数 < loop 但循环完整) → **Event 照测**（Event 注入独立于 op_summary 行数, 不受株连 — 旧行为误把漏记当循环改坏, 真实改进轮 Event=None 误 REVERT）; ★v4.6 起 Event 为**参考口径**（快测门粗筛 + 报告展示）, Event 缺失不再阻断 KEEP（主决策已换纯 kernel, 由欠采门槛防护）
  - best 更新: 只在 kept 时更新 (与 best_kernel/best_round 强绑定, 防 best_speedup 与代码脱钩)
  - rebaseline 同步: 复测 Event 同样由行数保证真实 (循环异常 → None → 不覆盖); `best_speedup` 是派生显示值 = baseline_e2e_event_ns / best_e2e_event_ns
- **input 链不变量**: 未采纳/失败 → `current_kernel` 回滚轮首快照 — sweep 把 current 指向 round_dir 时 coder 会覆写同路径, 必须**内容快照恢复**（失败代码另存 failed_kernel.py 留证）
- **晋升门前置**: planner promote 无 promote_evidence/reason → 本轮**转正常优化轮**（不白耗轮次、不涨 budget）; budget 只在真晋升轮 +1
- **max_rounds 硬上限**: loop 条件 = `(total_rounds - promote_budget) < max_rounds`（有效优化轮计数, promote 轮免费; 旧实现 budget 无限膨胀 → 上限失效）
- **rebaseline 环境漂移**: 每 REBASELINE_EVERY(默认10) 轮重测原始 baseline + 当前 kernel → 校正加速比基数, **并同步 best_e2e_event_ns**（用 best_kernel.py 新环境复测, 否则新环境改进轮永远比不过旧 best → 卡死 REVERT）
- **设备污染恢复**: verify 崩 AICore (HIVM/OOM/aclrt/575) → 标 `_dev_poisoned` → 下轮采集前 `_reset_device` (否则 msprof 在污染设备上"找不到 kernel"→采集失败级联)
- **tier 目录名**: 1→`01_algorithmic_structure`, 2→`02_operator_fusion`, ..., 6→`06_910b3_architecture`
- **每轮产物** (round_dir): `kernel_op.py` + `diff.patch`（只在 coder 成功时写, 失败尝试不覆盖真实 diff）+ `plan.md` + `07_tier<N>_fields/` + `msprof_0/` (验证) + `event_kernel.py` (Event 计时注入) + (失败时) `failed_kernel.py`
- **轨迹**: `outputs/<op>/optimization_trajectory.json`（每轮落盘, 可 `--resume` 续跑）
- **采集链演进**: round1 采集源目录 → 之后采集上一轮输出目录 (current_kernel.parent), kernel 链连续
- **Amdahl 显式编排 (2026-08-13)**: `_plan` 每轮从 diagnosis 算 per-kernel 耗时占比（×launch_count 加权）,
  多 kernel 时注入"优化优先级: A (60%) → B (25%) → C (15%), 本轮优先动占比最大 kernel, 动其他必须给理由"
- **失败案例库接入 (2026-08-13)**: 轮内 coder/verify 失败 → `build_retry_context`（前几次 方案+报错 全序列
  + 本次报错 + 失败库检索注入 + 禁止原样重试）→ 回传 coder; 成功轮 `mark_solved` 回填; hist 记 `error_class` 四分类
  （env/code_compile/code_numeric/code_runtime, 统一入口 `memory.failed_cases.classify_error`）
- **跨轮诊断快照 (2026-08-13)**: hist 记 `diag`（top2 kernel 紧凑串: bn/cu/mu/l2/redun/引擎）
  → planner 通过 history 看"改法→指标变化→结果"因果链; 全量 per-kernel 快照每轮追加写
  `diag_snapshots.jsonl`（审计/讲演用, 不入 context 防膨胀）
- **两段验证接入 (2026-08-13)**: `TWO_PHASE_VERIFY=1`（默认, 0 关闭; stub 自动禁用）—
  段1 `verify_fast_gate` 秒级（正确性+Event 快测）→ Event ≥ best 直接 REVERT（快测轮 speedup 用
  Event 口径派生, 决策走现有"Event ≥ best → REVERT"分支）; 过门才段2 msprof 全量;
  msprof 轮 Event 缺失用段1值兜底

### 4.2 采集链 (run_optimize.sh)
- **双数据源**: 通用 msprof（骨架，task.json） + 逐 kernel msprof op（deep，board_<i>.json）
- **尺寸传递**: scheduler 从 kernel config 提取 M/N/K 传给脚本 (MATMUL_M/N/K env)，保证 baseline/verify 同尺寸
- **07 字段**: `extract_tier_fields` 输出「全局摘要(前层信号) + Per-Kernel 概览 + 当前 tier 专属字段」，Planner 每轮先看前层
- **Tier2 MULTI 路径**: 多 kernel 才编译 HIVM（门控 `TIER==2`），产出融合视图给 fusion skill
- **Tier3 分块 sweep**: **round1 一次 + 每个 tier3 round 都跑** (不再 hash 跳过) — sweep 在 `best_kernel.py` (历史最优) 上程序化枚举 L0 合法 BLOCK → Event 实测 → 最优写回 (09_tier3_sweep/); history 每轮记 `sweep_ran`/`sweep_adopted`

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

### 4.4 Coder (确定性改码 + LLM 修复 + 失败案例库)
- **Step 0**: 无 previous_error 时，确定性 `code.replace(old_code, new_code)`（替换全部出现处）→ 不靠 LLM 最稳
- **Step 1**: 有报错时，走 LLM 带错误修复（失败案例库 `memory/failed_cases.py` 两级检索注入:
  指纹精确命中必中 + 关键词交集近似; 注入 solved 方案 / stuck 黑名单 / 已试方案, 只读参考严禁抄写;
  修复成功且代码有变 → `mark_solved` 回填失败库）
- 校验: Python 语法 + 函数完整性（防截断）+ no-op 检测 → diff.patch（只在成功时写）
- **Unicode 清洗**（用户高频报障 "syntaxerror/非法字符但看文件没有" 的根因治理）:
  - **千分位逗号**: LLM 输出 `1，024`(全角逗号) 被表替换转成 `,` 后变 `1,024` → Python 报 leading zeros/invalid decimal literal → 千分位归一 `\b\d{1,3}(,\d{3})+\b`→去逗号（不误伤 `2048,1024`、`range(0,100,16)` 等合法写法）
  - **全角数字/字母** `６４` → 转 ASCII（删除会留空产生 `= , 64` 新语法错）
  - **markdown 包裹**: 剥离先于"去行首垃圾"（否则开头 ``` 被吃 → 结尾 ``` 残留 SyntaxError）
  - **智能引号两难**: 定界符场景要换 ASCII（`print(“x”)`），字符串内容场景要删（`print("他说“好”")`）→ 试删/试换取第一个编译通过者
  - 兜底逐行清洗后二次 compile 验证; `_validate_python` 报错**带出错行真实内容**(repr) — 消除"看代码没有非法字符"的困惑

### 4.5 Verifier (两段验证: 正确性 + Event 快测门 + msprof 双口径 + Event 设备侧计时)
- **段1 `verify_fast_gate`**（2026-08-13, 秒级, 无 msprof）: `_correctness_check`（MATMUL_VERIFY 必须输出 `result check: PASS`）+ `_event_e2e_ns`（Event 快测）→ scheduler 判门: Event ≥ best → 直接 REVERT（省 warmup×3+msprof 几分钟/轮）; 过门/Event 不可用 → 段2
- **段2 `verify_end_to_end`**（全量, 过门才跑）: 先 `_correctness_check`（不过 → 本轮 FAIL 回传 coder 修）+ warmup(VERIFY_WARMUP=3) + 一次 msprof 内 kernel 循环 KERNEL_LOOP(VERIFY_LOOP=30) 次
- `_read_durations`: 读**全部 op_summary*.csv 合并**, 同一次同算两种口径之和 → 纯kernel(Σ非aclnn) + 端到端(Σ全部含框架)
- `_event_e2e_ns` (2026-08-12 改多窗口 median): 改写 kernel_op 注入 warmup + KERNEL_EVENT_REPS(默认5) 个独立 Event 窗口 (每窗口包 LOOP 次) → 取 median → `e2e_event_ns` (设备侧权威绝对值, 工业级, 无 profiler 扰动; 单窗口 ÷N 只有 1 个样本, median 抗抖动)
- 返回 {ok, ns(msprof纯), e2e_ns(msprof端到端), e2e_event_ns(Event, median), speedup, loop, rows, duration_us}
- msprof 用于诊断/纯kernel 拆解; Event 用于工业级绝对 latency; 两者并列保留

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
**严格晋升**: `promote=true` 必须给 `promote_evidence`（数据依据）且过晋升门前置校验，否则本轮转正常优化轮；同一跳转路径（from→to）≥3 次拒绝（防死循环），跳转时写 `10_tier_handoff.json` 手递瓶颈分析给目标层。

---

## 6. 测量口径（加速比可信的前提）

- **两种计时并存**（口径分离）:
  - **msprof**（诊断 + 纯kernel 拆解）: `e2e_ns` = Σ全部 kernel 行含框架, `ns` = Σ非aclnn 纯 kernel; 有 profiler 挂载开销但**跨轮一致**, 相对 speedup 显示用它
  - **Event**（工业级权威绝对值）: `e2e_event_ns` = `torch.npu.Event` 设备侧计时（多窗口 median: warmup + KERNEL_EVENT_REPS 个独立窗口包 LOOP 次 → median）; 无 profiler 扰动, **最终报告绝对 latency 和 KEEP 决策都用它**; bench（industrial/PyTorch）全走 Event → 两端同口径
- **bench 测量纪律 (2026-08-12 修复, 对齐 triton testing.do_bench)**:
  - **多窗口 median**: 先 5 次估时长 → warmup/rep 次数按 ms 预算自适应 (快 kernel 自动加次) → n_rep 个**独立 Event 对** → 取 median (另报 min/mean)。旧"单窗口 ÷N"只有 1 个样本。
  - **输入轮换破 L2**: 连续 forward 同一批张量, 工作集 <192MB(L2) 时后 N 次全 L2 命中 → 测到 L2 带宽 (数字虚高); Ascend 无清 L2 API → n_buf 组输入轮换 (组数×单组工作集 > L2) 等效 do_bench 的 clear_cache。json 记 `n_buf`。
  - **口径声明**（2026-08-13 逐行核对）: 两边 Event 都是**一次完整调用的设备侧耗时**（输入均预创建不在窗口内）——
    工业级 = torch forward（多 kernel 链 + kernel 间 host 下发 gap + forward 内部**中间张量分配**，均在窗口内）;
    我们 verify Event = kernel_op.py 循环体（融合/单遍后 **kernel 数更少** + 连续 launch gap≈0 + 中间结果预分配）。
    计时方法完全一致（Event + 多窗口 median + 破 L2）; **kernel 数/gap/中间分配的差异 = 融合优化的真实收益,
    不是测量差异** — 对比公平。大算子 (ms 级) 差异可忽略; 小算子 (us 级) 我们天然占优（正是融合/单遍优化目标）→
    报告同时给 `time_us_min/mean`, 对比时声明。
- **主加速比 = baseline_ns / ns**（★v4.6 纯 kernel 口径, 与 KEEP 决策同源）; best_speedup = baseline_ns / best_kernel_ns 派生; e2e_ns/e2e_event_ns 为参考口径并列记录
- **严格最优保留判定 (★v4.6) = 纯 kernel 绝对延迟**: 本轮 `ns < best_kernel_ns`（历史最小, msprof）才 KEEP 进链 — 与优化对象同尺 (纯 triton kernel), host 成分不掺; **欠采硬门槛**（行数 < loop → 不采纳）堵假快后门; Event 降为参考（快测门粗筛 + 报告）, best_e2e_event_ns 独立维护不参与决策
- **工业级对比**: round1 读 `bench_910b3/outputs/industrial_<op>_<mode>_tflops.json`（★2026-08-18 起只认 **method=msprof 纯 kernel 口径**, 由 `bench_industrial.py --msprof` 产出; Event 版 json 跳过+提示重测）, 各 mode 取 time_us 最小者（**仅真正执行**, actual_mode==mode） = industrial_time_us
- **vs 工业级比值**（优化效果终极指标, ★v4.6 纯 kernel 同尺）: `vs_industrial_ratio` = 我们最优纯 kernel ns（best_round 轮的 ns）/ 工业级 time_us×1000（<1=快于工业级）; 另有 `vs_industrial_speedup` = 工业级/我们
- **baseline 复测**: round1 用 verify 机制（warmup+msprof+Event）重测源 kernel，与后续轮完全同口径（VERIFY_BASELINE=1）
- **尺寸一致**: scheduler 传真实 M/N/K 给 run_optimize（防 512 默认覆盖 2048）
- **每轮 tflops**: 用本轮诊断 cube_fops ÷ 本轮 ns（kernel 结构变化后 FLOPs 变 → 轨迹图不失真）

---

## 7. 输出目录结构

```
outputs/<op>/
├── optimization.log                  # 全流程运行日志 (Tee 双写)
├── baseline_verify/                  # round1 基准复测 (msprof_0/ + Event)
├── best_kernel.py                    # 历史最高加速比那轮的代码 (best_speedup 绑定; sweep 输入)
├── 01_algorithmic_structure/round1..N/
│   ├── kernel_op.py                  # Coder 产出的本轮优化代码
│   ├── diff.patch                    # 与上一轮的差异
│   ├── plan.md                       # Planner 计划 (JSON: changes[]+promote)
│   ├── 07_tier1_fields/              # 本轮筛好的诊断字段 (tier1_fields.txt|.json)
│   ├── msprof_0/                     # 验证用的 msprof 产物
│   ├── event_kernel.py               # Event 计时注入版 (verify 生成, KERNEL_EVENT_TIME 触发)
│   ├── failed_kernel.py              # (coder/verify 失败时) 崩掉的中间产物, 排查用
│   ├── 04_board/ 05_task/ 06_diagnosis/   # run_optimize 采集中间产物
│   └── (Tier2) 08_fusion/            # HIVM 融合分析
├── 02_operator_fusion/round1..N/ ...   # 其余 tier 同构
├── 03_tiling_block_config/ ...
├── 04_memory_access/ ...
├── 05_compute_occupancy/ ...
├── 06_910b3_architecture/ ...
│   └── (每轮 roundN/ 内含 09_tier3_sweep/ — sweep 产物: sweep_result.json / sweep_runner.py; 仅 sweep 轮有)
├── optimization_trajectory.json      #  全局状态+history (中枢)
└── final_output/                     # 最终产物 (优化结束自动生成)
    ├── kernel_op.py                  #   最优 kernel (取 best_kernel, 可直接用)
    ├── baseline_kernel.py            #   baseline 副本
    ├── final_summary.json            #   摘要 (双口径 + Event + vs_industrial_ratio)
    ├── trajectory_chart.png          #   轨迹图 (PyTorch 虚线 + 工业级红线, Event-vs-Event)
    ├── all_strategies.md             #   全部轮次策略记录 (strategy_summary.py 每轮产)
    └── successful_strategies.md      #   仅成功优化策略 (KEEP+严格超越上一轮)
```

> **bench 基准产物** 统一放 `bench_910b3/outputs/`（industrial_*.json / pytorch_*.json / industrial_summary.json / msprof 临时目录）；
> `python3 bench_910b3/bench_all.py --clean` 一键清空。

`optimization_trajectory.json`（完整展开样例 + 每字段说明；`...` 仅表示 history 再重复若干轮）:
```json
{
  "v": 4,                              // schema 版本 (v4 状态机; 旧 v3 trajectory 会被检测并重置)

  "state": {                           // ── 全局状态机 (scheduler 每轮读写, 每轮落盘, --resume 续跑用它) ──
    "tier": 3,                         // 当前所在优化层 (1~6)
    "round": 8,                        // 轮次编号 (全局连续递增, 跨 tier 不重置; 目录按 <tier>/roundN 分放)
    "total_rounds": 10,                // 总执行轮 (含 promote 轮; max_rounds 配额按「有效轮」= total−promote_budget 计)
    "promote_budget": 1,               // 已用 promote 额度 (真晋升轮才 +1; 被拒晋升不涨)
    "best_speedup": 17.793,            // 历史最高加速比 (Event 派生显示值 = baseline_e2e_event_ns/best_e2e_event_ns)
    "best_kernel_ns": 355000.0,       // ★v4.6 KEEP 主依据: 历史最小纯 kernel 耗时 (msprof; rebaseline 随环境复测同步)
    "best_e2e_event_ns": 361000.0,     // 历史最小 Event 设备侧延迟 (快测门粗筛+报告参考, 不参与决策)
    "current_speedup": 17.793,         // 当前已采纳 kernel 的加速比 (端到端口径, 保留判定的参考)

    "baseline_ns": 5900000.0,          // 纯 kernel 基线 ns (源 kernel 的 Σ非aclnn 耗时; 纯 kernel 加速比的分母, 参考口径)
    "baseline_e2e_ns": 6320000.0,      // 端到端基线 ns (msprof Σ全部含框架; 主加速比的分母)
    "baseline_e2e_event_ns": 6100000.0,// Event 设备侧基线 ns (工业级口径, vs 工业级/轨迹图用)
    "num_kernels": 3,                  // 优化目标 kernel 数 (verifier 行数告警用)
    "num_launches": 3,                 // 每遍实际 launch 总数 (Σ launch_count; 循环丢失时 ÷ 的兜底除数)
    "baseline_mnk": [2048, 2048, 2048],// 基准 M/N/K (跨尺寸失真 guard: 本轮 mnk 不同则告警)
    "initial_tflops": 5.8,             // 源 kernel 初始 TFLOPS (轨迹图左轴; cube_fops÷baseline_ns 算)

    "pytorch_time_us": 45.0,           // PyTorch 基准端到端 us (Event 测, 轨迹图灰虚线)
    "pytorch_kernel_time_us": null,    // PyTorch 纯 kernel (Event 给不出拆解 → null)
    "pytorch_baseline": "pytorch_mlp_tflops.json",  // PyTorch 基准来源 json
    "industrial_time_us": 43.0,        // 工业级基准端到端 us (各 mode Event 取 min 仅真执行; 轨迹图红虚线)
    "industrial_kernel_time_us": null, // 工业级纯 kernel (Event 无 → null)
    "industrial_baseline": "industrial_matmul_compile_tflops.json",  // 最优来源 mode

    "current_kernel": "outputs/<op>/03_tiling_block_config/round8/kernel_op.py",  // 当前已采纳 kernel 路径
    "best_kernel": "outputs/<op>/best_kernel.py",   // 历史最高加速比那轮的代码 (best_speedup 绑定)
    "best_round": 8,                   // 打出 best_speedup 的那一轮
    "best_e2e_ns": 355000.0,           // best 轮的端到端 ns (msprof)

    "handoff": null,                   // 跳转手递 (tier 间传递瓶颈分析+优化方向; 消费后置 null)
    "tier_jumps": [                    // 跳转路径记录 (防死循环: 同 from->to ≥3 次拒绝)
      {"pair": "2->3", "round": 5}
    ],
    "promote_budget": 1,               // 已用 promote 额度 (promote 轮不挤占 max_rounds)
    "last_rebase_round": 0,            // 上次环境漂移重基准的轮号 (每 REBASELINE_EVERY 轮重测)
    "vs_industrial_ratio": 0.82,       // 我们最优 Event / 工业级 Event (<1=快于工业级)
    "tier3_swept": true,               // Tier3 分块是否已扫
    "last_sweep_result": {             // 上次 sweep 结论 (每轮传给 planner; 含真实状态 ran/skipped/reused/failed)
      "available": true,
      "best": {"block": [128, 128, 64], "ns": 4200000, "speedup": 1.12, "is_current": true},
      "vars": ["BLOCK_M", "BLOCK_N", "BLOCK_K"],
      "n_configs": 287,
      "result_path": "outputs/<op>/.../09_tier3_sweep/sweep_result.json",
      "written": true                  // sweep 是否真写回了新块 (false = 当前块保留)
    }
  },

  "history": [                         // ── 每轮一条 (改了啥 + 结果), planner 看前层试过什么, 轨迹图打点 ──
    {
      "round": 1,                      // 本轮轮次号 (全局连续, 跨 tier 不重置)
      "tier": 1,                       // 本轮所在层
      "strategy": "fp16 累加 + im2col 上 cube",   // planner 给的策略名
      "change": "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64",  // 改动梗概 (压成一句, 给 planner 上下文 + 图标签)
      "changes_full": [                // 完整 changes[] (old_code/new_code 全文, 审计/复盘不丢信息)
        {"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32",
         "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
         "reason": "...", "section": "config", "tier": 1}
      ],
      "expected_impact": "1.10x",      // planner 预期加速比 (下轮反馈"预期 vs 实际"学习闭环)

      "speedup": 1.5,                  // 本轮端到端加速比 (= baseline_e2e_ns / e2e_ns; 主指标, 累计输出)
      "kernel_speedup": 1.5,           // 本轮纯 kernel 加速比 (= baseline_ns / ns; 参考口径)
      "prev_speedup": 1.0,             // 上一轮已接受 kernel 的加速比 (保留判定基准)

      "ns": 3933333.0,                 // 本轮纯 kernel 耗时 ns (baseline_ns / kernel_speedup)
      "e2e_ns": 4210000.0,             // 本轮端到端耗时 ns (msprof Σ全部, 主加速比的实测值)
      "e2e_event_ns": 4100000.0,       // 本轮 Event 设备侧端到端 ns (工业级绝对值, 无 profiler 扰动)

      "decision": "KEEP",              // 决策: KEEP(纯kernel ns<历史最小才采纳, ★v4.6)/REVERT(未超越)/FAIL(coder或verify失败)
      "result": "OK",                  // 结果: OK/NOOP(coder没改动)/FAIL; NOOP 检测本轮输出==改前

      "sweep_ran": true,               // 本轮是否执行了分块 sweep
      "sweep_adopted": true,           // 是否采纳了 sweep 测的最优块
      "sweep_status": "ran_this_round",// sweep 状态 (ran/failed/skipped_no_free_params/reused)

      "error": "",                     // 报错文本 (空=正常; coder/verify 失败时记, 下轮 planner 可见)
      "tflops": 8.7                    // 本轮真实 TFLOPS (cube_fops÷ns; kernel 结构变 FLOPs 也变, 图不失真)
    }
    // ... 每轮一条, 同上结构
  ]
}
```

**字段层级速记**：
- `state` = 全局状态机（当前 tier/round、msprof+Event 双基线、PyTorch/工业级基准、best_kernel/best_round、vs_industrial_ratio、sweep 结果）— 每轮读写，`--resume` 续跑靠它
- `state.baseline_e2e_ns` / `current_speedup` = **主指标对**（端到端基线 + 当前加速比）；`baseline_e2e_event_ns` = 工业级 Event 口径基线
- `state.best_kernel` / `best_round` = **最高加速比绑定**（与 best_speedup 同更新，防"最高和最优代码脱钩"）
- `state.industrial_time_us` = 工业级天花板（Event 各 mode min，轨迹图红线）
- `state.vs_industrial_ratio` = 优化效果终极指标（★v4.6: 我们最优纯 kernel ns / 工业级 msprof 纯 kernel，<1=快于工业级）
- `history[i]` = 每轮一行（策略+改动梗概+完整 changes+msprof/Event 双耗时+sweep 状态+决策+报错+TFLOPS）— planner 看前层试过什么、轨迹图打点、strategy_summary 出策略清单

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

# 工业级基准 (各算子各 mode 真机测 → 取每算子 min 作为对比天花板; Event 设备侧计时)
python3 bench_industrial.py matmul --mode compile     # TorchAir 图融合
python3 bench_industrial.py flash_attention --mode fa # CANN FlashAttention
python3 bench_all.py                    # 全部算子全部模式 → 自动取最优 + 汇总表
python3 bench_all.py --clean            # 清理 bench_910b3/outputs/ 全部产物

# PyTorch 基准 (bench_pytorch_*.py 自带 Event 计时, 无需 msprof 包裹)
python3 bench_pytorch_mlp.py            # matmul(MLP) 对照
python3 bench_pytorch_attention.py      # attention_mlp 对照

# 轨迹图 (自动读 outputs/ 里的 PyTorch + 工业级基准)
cd .. && python3 feedback/trajectory_chart.py outputs/matmul

# 运行日志
cat outputs/<op>/optimization.log
```

---

## 9. 六层优化策略：读取字段详解

> 每轮 `run_optimize.sh` 产出 `diagnosis.json` 后，scheduler 按当前 tier 用 `TIER_FIELDS` + `TIER_PER_KERNEL` 筛字段 → `07_tier<N>_fields/` 喂 Planner。
> 字段来源: `summary.*`(通用 msprof 骨架) / `kernels[i].task.*`(骨架 per-kernel) / `kernels[i].deep.*`(msprof op 深层画像)。
> **全局摘要(前层信号)** 任何 tier 都喂: `num_kernels`(多→融合空间) / `num_kernels_total`(含框架) / `api_overhead_total_us`(launch 开销) —— 让 planner 先做"前层优先检查"，不能只闷头调本层参数。

### 9.1 Tier1 算法结构

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



### 9.2 Tier2 算子融合

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



### 9.3 Tier3 分块配置

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



### 9.4 Tier4 访存

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



### 9.5 Tier5 计算占用

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



### 9.6 Tier6 架构专属

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


## 9A. 六层优化策略：判断瓶颈与针对性修改（独立记录）

> 本节是独立的策略判断块：**读什么字段 → 判断出什么瓶颈 → 针对性改什么**。
> 与上文 9.1-9.6 的"读取字段"表配套使用（字段含义/路径看上文，这里给判断与动作）。
> 每个动作标注对应 `docx/playbook_tier<N>_*.md` 的情况编号。

### 9A.1 Tier1 算法结构

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `roofline.compute_utilization` | **低 (<0.3)** 且非 memory | 算法选错 → 该换算法 |
| `compute.cube_fp16_ratio` | 低 且 compute_bound | 精度没吃满 cube fp16 算力 |
| `roofline.bottleneck_type` | `memory_bound` 且算术强度低 | 冗余访存（如 attention 物化 S[seq²]） |
| `summary.num_kernels` / `api_overhead_total_us` | 多同结构 matmul / launch 大 | 重复计算 + launch 开销 |
| `roofline.arithmetic_intensity` | 明显低于平衡点(≈86) | 算法/访存结构问题 |

**针对性修改（判出瓶颈 → 改什么）**:
- compute_bound + cube_fp16_ratio 低 → **fp16 输入 + fp32 累加**（playbook 情况A；换 DTYPE，别只调参数）
- memory_bound + 巨大中间张量(S[seq²]) → **Flash Attention**（情况B：online softmax 单 kernel 省 S；rescale 别漏，mask 先于 max）
- 多个同结构小 matmul 串行（QKV/FFN 两段）→ **合并单 GEMM**（情况C：拼列一次 dot，kernel 内切分）
- 归约类算子多遍扫数据 → **online/单遍**（情况D：一次 load 同时累 sum/sum_sq）
- 大 K matmul（K>4096）→ **split-K**（情况E）；小 grid+高 launch → **persistent kernel**（情况F）
- conv 无 `tl.dot`（向量外积模拟）→ **软件 im2col + tl.dot 走 cube**（情况G，实测慢 PyTorch 12×）
- GQA 组内复制 → **kernel 内按组索引消复制**（情况H）；MoE 全量 expert → **topk 稀疏**（情况I）；Mamba torch cumsum → **chunked scan**（情况J）
- 结构改动后必须 `MATMUL_VERIFY=1` 数值校验 + 回 Tier2/3 重做

### 9A.2 Tier2 算子融合

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `summary.num_kernels` | >1 且是 matmul→逐元素→matmul 链 | 分离小 kernel 有融合空间 |
| `summary.api_overhead_total_us` | 大（launch 占端到端比例高） | launch 开销拖累 |
| `bottleneck_type` (全局) | `memory_bound` 且中间张量大 | 中间 GM 往返浪费 |
| `main_mem_read/write_gb_s` | 高但算力利用率低 | 中间量在 GM 来回 |
| `08_fusion/fusion_analysis.json` | `raw_deps` 里 from=cube、to=vector | RAW 链可融合；`war/waw` 决定换 buffer |

**针对性修改（判出瓶颈 → 改什么）**:
- matmul 后跟独立 bias/激活 → **并进 epilogue**（情况A：省中间 Z 的 GM 往返，GEMM+Bias+ReLU 实测 28~39%）
- 大 kernel 后跟独立 add(残差) → **并进 epilogue**（情况B）
- 同一张量读多次（QKV/up/gate）→ **单次 load 复用/拼列合并**（情况C + 情况I：SwiGLU 双路合并、BN 并入 conv epilogue 情况H）
- attention 内 scale/mask/softmax 独立步骤 → **并入 QK^T/softmax**（情况F）
- GEMM 输出格式≠下游输入（隐式格式转换）→ **UB 内直接消费**（情况G）
- 融合前先算收益：中间 kernel 耗时 > 2×launch(~10-40us) 才融；两个都 compute_bound 的大 kernel 别硬融（成本模型）
- 不可融合边界：跨 GM store/跨迭代/归约/副作用/寄存器溢出 → 不融或拆

### 9A.3 Tier3 分块配置

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `task.block_dim` | **< 40**（核没吃满） | 分块太小并行度不足 |
| `engine_utilization.mte1` | 高（L1→L0 搬运瓶颈） | BLOCK_K 太大 |
| `engine_utilization.mte2` | 高（GM→L1 瓶颈） | BLOCK_M/N 太小搬运频繁 |
| `engine_utilization.cube` | 低（cube 没满） | 分块不够大喂不饱 cube |
| `bandwidth_gb_s.l0a/l0b_read` | 低 | BLOCK 与 L0A/L0B 贴合差 |
| `roofline.bottleneck_type` | `compute_bound` | 不调分块，回 Tier1/5 |
| `roofline.compute_utilization` | 极低(<0.3) | 回 Tier1 查算法，不是分块 |

**针对性修改（判出瓶颈 → 改什么）**:
- block_dim<40 → **减 BLOCK_M/N**（§二-B，核数上去）
- mte1 高 → **增 BLOCK_K**（§二-A）；mte2 高/cube 低/l0a 低 → **增 BLOCK_M/N**（§二-A）
- memory_bound → **增 tile + swizzle**（§三）
- **round1/tier3 自动 sweep 实测最优块**（`sweep_blocks.py`，L0A/B≤64KB、L0C≤128KB、UB 3 缓冲 ≤192KB、2 的幂、grid∈[16,3000]；结构变化后必须重扫）
- 贴边界候选有 ×0.9 安全余量（防 L0/UB 溢出打崩设备）；寄存器溢出悬崖：tile 过大性能崩塌，以 sweep 数据为准

### 9A.4 Tier4 访存

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `main_mem_read/write_gb_s` | 高但算力利用率低 → memory_bound | GM 流量过大 |
| `gm_to_ub` / `ub_to_gm` | 低（搬运效率差） | 跨步/非连续访问 |
| `l2_hit_rate` | 低 | L2 复用差 |
| `task.pipes_us.aic_mte2/mte3_time` | 高（搬运耗时长） | 搬运量大 + 无流水线 |
| 对齐警告 / `main_mem` 流量异常 | 非 16B 对齐 | 拆分事务浪费带宽 |

**针对性修改（判出瓶颈 → 改什么）**:
- 有跨步/非连续访问 → **连续化**（情况A：最快维匹配内存布局，DMA 一次搬满；逐元素扁平化 1D）
- 指针/维度未对齐 → **128-bit 对齐 + padding**（情况B：fp16 N=10 → pad 到 16B 倍数）
- l2_hit_rate 低 → **L2 复用**（情况C：访问顺序/权重预排/swizzle）
- mte 忙但 cube 空等 → **流水线/UB 控制**（情况D：load 独立成步骤让编译器双缓冲）
- 零散小 load → **合并大块搬运**（情况E）；行宽对齐 512B（情况F）；输出转置先在 UB 内转再连续 store（情况G）
- 中间张量 S/P 落 GM → **UB 内直接消费**（情况H，flash 同源）

### 9A.5 Tier5 计算占用

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `aic_scalar_time_us` / `engine_utilization.scalar` | 高（标量拖累） | 指针 div-mod / int64 索引 → 标量降级 |
| `compute.vector_fops` / `vec` 利用率 | 低 | 逐元素标量加载未向量化 |
| 数学运算慢（cube 已满） | 1/sqrt、erf 等手动组合 | 没用原生指令 |
| `conflict.bank_cflt_ratio` | >4~5% | UB bank 冲突 |
| 性能不升反降 | 寄存器溢出 | 过度展开/ILP 崩塌 |
| `aic_scalar_time` 高 + 代码含 `x/s` 除法 | 逐元素除法 | 标量除法拖累 |
| `vec` 高 `cube` 低 + 外积累加 | conv 没上 cube | 3D 广播展开/vector 模拟 |

**针对性修改（判出瓶颈 → 改什么）**:
- scalar_time/ratio 高 → **消除标量降级**（情况A：指针 div/mod 改 2D 索引、int64 改 int32）
- 逐元素标量加载 → **向量化**（情况B：一次向量 load）
- 1/sqrt → `tl.math.rsqrt`；erf → tanh（情况C）；mul+add → 直接 `x*w+b` 让 FMA 融合（情况D）
- 逐元素 `x/s` 除法 → **倒数+乘法**（情况G）
- bank_cflt_ratio>4% → **访问调整/swizzle + 尾轴 32B/512B 对齐**（情况F/I）
- 寄存器溢出/性能反降 → **控制展开/ILP**（情况E）
- vec 高 cube 低 + 外积累加 → **分块 2D 累加或 tl.dot**（情况H）
- cube 已满（compute_bound）且 scalar/conflict 不高 → **停手/晋升**（情况J，别再试指令级改动）

### 9A.6 Tier6 架构专属

**判断瓶颈（读字段 → 判什么）**:

| 字段 | 触发阈值/模式 | 判断出的瓶颈 |
|---|---|---|
| `conflict.vec_wait_ratio` | 高（vec 被阻塞） | 计算等数据 → 流水线问题 |
| `conflict.mte_cflt_ratio` | 高 | MTE 搬运冲突 |
| `conflict.bank_cflt_ratio` | >4% | UB bank 冲突 |
| `engine_utilization.cube` | 低但任务是 matmul | 没走 cube（vector 模拟） |
| `engine_utilization` 整体 | cube/vec 严重失衡 | 结构问题 |
| `task.task_type` | 非 cube 但该算 matmul | 走错引擎 |
| `task.block_dim` | 远小于 40 | 核没吃满 |
| grid vs 物理核数 / `api_overhead` | grid 远大于核数 / launch 大 | 调度/核启动开销 |
| 多核耗时不均 / 计算拖尾 | 尾核空转 | 负载不均衡 |
| `aic_cube_wait_ratio`/`vec_wait_ratio` | 被阻塞但各 pipe 利用率都不高 | 跨引擎流水气泡 |
| 纯 vector 算子 `task_type=AIV` | 按 cube 核数定 grid | 向量核少用一半 |

**针对性修改（判出瓶颈 → 改什么）**:
- vec_wait_ratio 高 / mte_cflt 高 → **回 Tier4（流水线/load 独立）/ Tier3（分块）**
- cube 利用率低且任务是 matmul → 没用 tl.dot → **用 tl.dot 走 cube**（情况D）；用了仍低 → 回 Tier3/1
- cube/vec 严重失衡 → **回 Tier2 融合平衡引擎**
- grid >> 核数 / launch 大 → **grid 固定物理核数 + 核内 stride 循环**（情况G）
- 多核耗时不均/尾核空转 → **stride 切分**（情况H）
- cube/vec 被阻塞但 pipe 利用率低（流水线气泡）→ **K 循环分块 + 双缓冲 UB 预算**（情况I）
- 纯 vector 算子没吃满向量核 → **按引擎选核数 40/20**（情况J）
- 代码风格 while/动态 shape/非 math 命名空间 → **适配后端最优解析**（情况F）



## 10. diagnosis.json 完整结构与字段来源

### 10.1 数据流：msprof 产物 → diagnosis.json

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

### 10.2 diagnosis.json 顶层结构（完整展开，每字段附示例值）

> 下面是一个**真实可读的完整 JSON 样例**（两层 MLP = 3 个目标 kernel：fc1 / bias_gelu / fc2），每个字段后面注释它记录什么。`...` 仅表示“同上结构再重复若干个”，不是省略字段。

```json
{
  "meta": {
    "source": "msprof (generic) + msprof op per-kernel",   // 标记数据来自两条 msprof 链路
    "generated_at": "2026-08-11T14:30:00",                  // integrate.py 生成时刻
    "num_kernels": 3,                                       // 优化目标 kernel 数 (非 aclnn)
    "filled_kernels": 3,                                    // 其中 deep 被 msprof op 填满的个数
    "inputs": {
      "task": "round_dir/05_task/task.json",                // 骨架来源
      "boards": ["round_dir/06_diagnosis/board_1.json",     // 每 kernel 一个 deep 来源
                 "round_dir/06_diagnosis/board_2.json",
                 "round_dir/06_diagnosis/board_3.json"]
    },
    "schema_version": "4.0"
  },

  "summary": {                          // ── 全局汇总 (integrate.py 从 task.json 汇总, 喂 Tier1~6 全局摘要) ──
    "num_kernels": 3,                   // 优化目标 kernel 数 (= len(kernels[]), 排除 aclnn)
    "num_kernels_total": 7,             // 全部 distinct op 名数 (含 aclnn 框架 kernel)
    "total_ns": 5900000,                // 端到端耗时 ns = Σ 非 aclnn kernel 的 Task Duration ×1000 (与 verify 同口径)
    "num_cores": 20,                    // 实际是 launch grid (Block Dim 最大值), 不是物理核数 (910B3 固定 20 核)
    "api_overhead_total_us": 180.5,     // Σ api_overhead[].total_us = host 侧 launch/API 总开销 (大 → 融合收益)
    "l2_hit_rate": 0.42,                // 全局 L2 命中率 (0~1; 来自 l2_cache.csv, 已归一化)
    "filled_kernels": 3                 // deep 被 msprof op 填满的 kernel 数 (<num_kernels 说明有 kernel 没采到 op)
  },

  "kernels": [                          // ── 逐优化目标 kernel (每 kernel = 骨架 task + 深层 deep) ──
    {
      "kernel_name": "matmul_kernel",   // op_summary 里的 Op Name (msprof op 按它匹配 deep)
      "framework": false,               // 是否 aclnn 框架 kernel (true 的不进这里, 进 framework_kernels[])
      "launch_count": 2,                // 该 kernel 被 launch 几次 (fc1 用了 2 次 → 占比要 ×2)
      "task": {                         // ── 骨架: 来自 op_summary.csv (通用 msprof, 每 kernel 一行) ──
        "task_type": "AI_CORE:CUBE",    // Task Type 列: 引擎类型 (Cube/Vec…), 决定走计算还是访存优化
        "task_duration_us": 1850.0,     // Task Duration(us) 列: 单次 launch 耗时 (占比口径的分子)
        "block_dim": 20,                // Block Dim 列: launch grid (并行度; 太少 = 并行不足)
        "input_shapes": "(2048,2048);(2048,2048)",   // Input Shape 列: 输入张量形状 (算搬运字节用)
        "input_dtypes": "float32",      // Input Data Type 列: 输入类型 (算字节/精度)
        "output_shapes": "(2048,2048)", // Output Shape 列: 输出形状
        "output_dtypes": "float32",     // Output Data Type 列: 输出类型
        "aicore_time_us": 1820.0,       // aicore time 列: cube 核总时间
        "aiv_time_us": 0.0,             // aiv time 列: 向量核总时间
        "total_cycles": 3276000,        // total cycles 列: 总周期
        "pipes_us": {                   // per-pipe 耗时 (来自 aic_*/aiv_*_time(us) 列, 兼容列名 cube↔mac/fixpipe↔fixp)
          "aic_cube_time_us": 1820.0,   // cube 计算耗时 (Tier5 看是否计算瓶颈)
          "aic_mac_time_us": 1820.0,    // MAC 耗时 (真实列名常是这个, cube 别名)
          "aic_mte1_time_us": 410.0,    // MTE1 (L1→L0A/B) 搬运耗时 (Tier3 看 BK 是否过大)
          "aic_mte2_time_us": 320.0,    // MTE2 (GM→L1) 搬运耗时 (Tier4 看访存)
          "aic_mte3_time_us": 95.0,     // MTE3 (L0C→fixpipe→UB→GM) 写回耗时
          "aiv_vec_time_us": 0.0        // 向量耗时 (rms_norm/softmax/bias_gelu 命门)
        },
        "est_bytes_in": 33554432,       // 估算输入搬运字节 = Σ(元素数×dtype字节) (shape 推)
        "est_bytes_out": 16777216,      // 估算输出搬运字节
        "transfers": [                  // 计算出的每通路带宽 (字节 + pipe 耗时 → GB/s)
          {"path": "GM读→L1/UB(MTE2)", "bytes": 33554432, "time_us": 320.0, "bw_gb_s": 104.9},
          {"path": "L1→L0A/L0B(MTE1)", "bytes": 33554432, "time_us": 410.0, "bw_gb_s": 81.8},
          {"path": "L0C/UB→GM(写)", "bytes": 16777216, "time_us": 95.0, "bw_gb_s": 176.6},
          {"path": "Cube MAC", "macs": 17179869184, "time_us": 1820.0, "tflops": 9.4}
        ]
      },
      "deep": {                         // ── 深层: 来自 msprof op 的 8 个 CSV (board_<i>.json), 按名匹配填充 ──
        "freq_mhz": 1800,               // OpBasicInfo Current Freq: 运行频率
        "bandwidth_gb_s": {             // 各通路真实带宽 (Memory/MemoryL0/MemoryUB, MB/s→GB/s 换算)
          "main_mem_read_gb_s": 95.2,   // GM 读带宽 (Memory.csv main_mem_read_bw)
          "main_mem_write_gb_s": 45.1,  // GM 写带宽
          "l1_read_gb_s": null,         // L1 读带宽 (合法缺则 null)
          "l1_write_gb_s": null,
          "l2_read_gb_s": 320.0,        // L2 读带宽
          "l2_write_gb_s": 180.0,
          "gm_to_ub_gb_s": 95.2,        // GM→UB 读带宽 (aiv_gm_to_ub_bw)
          "ub_to_gm_gb_s": 45.1,        // UB→GM 写带宽 (aiv_ub_to_gm_bw)
          "ub_vector_read_gb_s": 210.0, // UB 向量读 (MemoryUB aiv_ub_read_bw_vector)
          "ub_vector_write_gb_s": 110.0,
          "ub_scalar_read_gb_s": null,
          "ub_scalar_write_gb_s": null,
          "ub_mte_read_gb_s": null,     // 910B3 合法缺 (仅推理产品有)
          "ub_mte_write_gb_s": null,
          "l0a_read_gb_s": 1200.0,      // L0A 读 (MemoryL0 aic_l0a_read_bw)
          "l0a_write_gb_s": null,
          "l0b_read_gb_s": 1180.0,      // L0B 读
          "l0b_write_gb_s": null,
          "l0c_read_gb_s": null,
          "l0c_write_gb_s": 600.0       // L0C 写
        },
        "engine_utilization": {         // 各引擎占比 (PipeUtilization *_ratio, 缺则 time/总时长)
          "cube": 0.71,                 // cube 占比 (高 = 计算为主)
          "vec": 0.0,                   // 向量占比 (高 = 向量型算子瓶颈)
          "mte1": 0.16,                 // MTE1 (L1→L0A/B) 占比 (高 = 内层搬运瓶颈)
          "mte2": 0.12,                 // MTE2 (GM→L1) 占比 (高 = 外层访存瓶颈)
          "mte3": 0.04,                 // MTE3 (写回) 占比
          "scalar": 0.01,               // 标量占比 (高 = 地址/循环开销)
          "fixpipe": 0.03               // fixpipe 占比 (定标)
        },
        "compute": {                    // 真实运算量/精度 (ArithmeticUtilization)
          "cube_fops": 17179869184,     // cube 浮点运算数 (FLOPs 来源, 算 TFLOPS 用)
          "cube_ratio": 0.85,           // cube 指令占比 (低 = 没走 cube, 如 conv 该 im2col)
          "cube_fp16_ratio": 0.0,       // cube fp16 占比 (精度选择依据)
          "cube_int8_ratio": 0.0,       // cube int8 占比
          "cube_instr_number": 4096,    // cube 总指令数
          "vector_fops": 0,             // 向量运算数 (非 matmul kernel 的真实计算量)
          "vec_ratio": 0.0,             // 向量占比
          "vec_fp32_ratio": 0.0,
          "vec_instr_number": 0,
          "aic_total_cycles": 3276000,  // aic 总周期
          "aiv_total_cycles": 0         // aiv 总周期
        },
        "conflict": {                   // 冲突/等待 (ResourceConflictRatio 全列, 原样保留键名)
          "bank_cflt_ratio": 0.03,      // bank 冲突 (128-bit 对齐可解, Tier5)
          "bankgroup_cflt_ratio": 0.01, // bankgroup 冲突
          "mte_cflt_ratio": 0.02,       // mte 冲突 (Tier6)
          "wait_ratio": 0.05,           // vec 被阻塞占比 (流水气泡, Tier5/6)
          "total_cflt_ratio": 0.06      // vec 总冲突
        },
        "l2_hit_rate": 0.45,            // 该 kernel 的 L2 命中率 (L2Cache, 0~1)
        "roofline": {                   //  integrate.py 计算 (不是 CSV 直读); 峰值取 hardware_peak.json
          "achieved_memory_bw_gb_s": 140.3,   // = main_mem_read + main_mem_write (读+写之和)
          "peak_memory_bw_gb_s": 1638.4,      // GM 峰值 (实测或理论 HBM2e)
          "memory_utilization": 0.086,        // = achieved/peak (Tier4 访存利用率)
          "achieved_compute_tflops": 17.18,   // = (cube_fops+vector_fops)/1e12
          "peak_compute_tflops": 294.9,       // cube fp16 峰值
          "peak_compute_fp32_tflops": 73.7,   // cube fp32 峰值 (= fp16/4)
          "compute_utilization": 0.233,       // = achieved / max(fp16峰, fp32峰) (用 max 防 fp32 被低估)
          "compute_utilization_fp16": 0.058,  // 按 fp16 峰值算
          "compute_utilization_fp32": 0.233,  // 按 fp32 峰值算
          "arithmetic_intensity": 122.5,      // = achieved_compute×1e12 / achieved_mem / 1e9 (计算/访存)
          "bottleneck_type": "compute_bound"  // mem≥0.8&comp<0.5→memory; comp≥0.8&mem<0.5→compute; 都<0.3→latency; 否则 balanced
        }
      },
      "filled_by": "msprof op"          // deep 由 msprof op 填满; "msprof only" = 没采到 op (deep=null)
    }
    // ... 其余 kernel (bias_gelu_kernel / matmul_kernel2) 同上结构
  ],

  "framework_kernels": [                // ── aclnn* 框架 kernel (torch_npu 数据准备/参考), 非优化目标, 仅观察 ──
    {
      "kernel_name": "aclnnMatmul",     // Op Name 以 aclnn 开头 → 框架 kernel
      "framework": true,
      "launch_count": 4,                // 被 launch 几次
      "task": { "task_duration_us": 12.3, "task_type": "AI_CORE:AI_VECTOR", "...": "..." },
      "deep": null,                     // 框架 kernel 不跑 msprof op (没 deep)
      "filled_by": null
    }
    // ...
  ],

  "api_overhead": [                     // ── launch/API 开销明细 (api_statistic.csv, Tier2 判断融合收益) ──
    {
      "level": "L1",                    // 调用层级
      "api_name": "aclrtLaunchKernel",  // API 名
      "total_us": 120.5,                // 该 API 总耗时
      "count": 6,                       // 调用次数
      "avg_us": 20.1,                   // 平均
      "max_us": 35.2                    // 最大
    }
    // ...
  ],

  "multi_kernel": [                     // ── 每类算子次数/耗时 (op_statistic.csv, 判断是否值得融合) ──
    {
      "op_type": "Matmul",              // 算子类型
      "core_type": "AIC",               // 核类型
      "count": 2,                       // 次数
      "total_time_us": 3700.0,          // 总耗时
      "avg_us": 1850.0,                 // 平均
      "min_us": 1800.0,                 // 最小
      "max_us": 1900.0,                 // 最大
      "ratio": 0.62                     // 占比
    }
    // ...
  ],

  "notes": [                            // ── 生成说明 (integrate.py 写的固定提示) ──
    "骨架=通用msprof(task.json); deep=msprof op 按 kernel 名填充",
    "roofline 每 kernel 一个: 带宽对1638.4GB/s, 算力对294.9/73.7TFLOPS(fp16/fp32)",
    "峰值优先取 bench_910b3/hardware_peak.json 实测, 无则回退理论值",
    "filled_by='msprof only' = 该 kernel 没跑到 op (deep=null)",
    "kernels[].task.transfers 的 bytes 为估算 (每元素每通路搬一次)"
  ]
}
```

**字段层级速记**：
- `meta` = 这次生成的元信息（来源/时刻/输入文件/计数）
- `summary` = 全局汇总（kernel 数/端到端耗时/launch 开销/L2）— Tier1~6 全局摘要都从这里取
- `kernels[i].task` = 骨架（op_summary.csv，per-kernel 耗时/形状/pipe 耗时/搬运估值）
- `kernels[i].deep` = 深层（msprof op 8 CSV，per-kernel 带宽/引擎/算力/冲突/L2）
- `kernels[i].deep.roofline` = **算出来的**（achieved vs peak → 利用率 → bottleneck_type）
- `framework_kernels` = aclnn 框架 kernel（不优化，只观察）
- `api_overhead` / `multi_kernel` = launch 开销 / 算子统计（Tier2 融合判断）

### 10.3 字段来源明细（每个字段 ← 哪个文件哪个列 / 怎么算）

#### summary.*（integrate.py 从 task.json 汇总）

| 字段 | 来源文件 → 列 | 怎么解析 |
|---|---|---|
| `num_kernels` | op_summary*.csv `Op Name` 列 | 去重非 aclnn 的 op 名数 = 优化目标 kernel 数 |
| `num_kernels_total` | op_summary*.csv `Op Name` 列 | 去重含 aclnn 的全部 op 名数 |
| `total_ns` | op_summary*.csv `Task Duration(us)` | Σ 非 aclnn 行的耗时 ×1000（与 verify 端到端口径一致） |
| `num_cores` | op_summary*.csv `Block Dim` 列 | 最大值（实际是 launch grid，910B3 固定 20 核） |
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

带宽统一换算：Ascend Memory*.csv 列是 MB/s，`>=1000` 就 ÷1000 转 GB/s（小带宽列不动，避免误除）。

#### kernels[i].deep.roofline.*（ integrate.py 计算，不是 CSV 直读）

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

---

## 11. 关键结果（讲述时讲数据，讲之前先填表）

> **运行方法**: 每个算子跑 `LLM_CLI_COMMAND='nga run' python3 main.py input/<op> --fresh --max-rounds 30` 后，自动产出 `outputs/<op>/final_output/final_summary.json`（含双口径加速比 + Event 延迟 + vs_industrial_ratio）和 `trajectory_chart.png`。
> **诚实说明**: 当前仓库 `outputs/` 内**无已完成的 v4 真实结果**（本地只有旧版弱数据/空轨迹），以下表格是讲述模板，务必在服务器跑完后用真实数据填充——**讲项目时数据是最有说服力的一页**。

| 算子 | 形状 | 初始纯kernel(us) | 优化后纯kernel(us) | 加速比 | vs PyTorch | vs 工业级最优 | 命中策略层 |
|---|---|---|---|---|---|---|---|
| matmul (2层MLP) | 2048³ | | | | | | |
| attention_mlp | 2048×2048 | | | | | | |
| matmul_relu | 2048³ | | | | | | |
| flash_attention | seq2048×8h | | | | | | |
| conv2d | 64×64 | | | | | | |
| rms_norm | 2048×2048 | | | | | | |
| ... (17 算子, 含 conv1d/batchnorm2d/maxpool2d) | | | | | | | |

**讲结果时的口径要点**（防止被追问翻车）:
- 加速比口径: 端到端（含框架 kernel），同口径对比；**绝对延迟用 Event**（无 profiler 扰动）
- vs 工业级: 我们最优纯 kernel ns / 工业级 msprof 纯 kernel 各 mode 取 min（仅真执行, ★v4.6 同尺）
- 每轮可审计: `roundN/{plan.md, diff.patch, kernel_op.py}` + `optimization_trajectory.json` history 全留痕

---

## 12. 五分钟 Demo（现场演示路径）

```bash
# ① 环境
conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd triton_agent_optimizer

# ② 证明 LLM 链路通 (可选)
echo "测试, 调用 skill triton-op-planner" | nga run

# ③ 单文件可跑 + 数值正确 (30 秒)
python3 input/matmul/kernel_op.py && MATMUL_VERIFY=1 python3 input/matmul/kernel_op.py
# 预期: [info] ... launched & synced OK + result check: PASS

# ④ 只采集+解析, 展示 diagnosis.json 的诊断能力 (2~5 分钟)
bash analyzers/run_optimize.sh input/matmul /tmp/demo_run
cat /tmp/demo_run/07_tier1_fields/tier1_fields.txt        # 每 kernel: 耗时占比+cube_util+bottleneck
cat /tmp/demo_run/06_diagnosis/diagnosis.json             # roofline: compute/memory 利用率

# ⑤ 完整优化循环 (10~30 分钟, 按需 --max-rounds)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15
# 每轮打印: 采集→07字段→planner策略→coder改动→验证→加速比→KEEP/REVERT
# 结束自动: final_summary.json + trajectory_chart.png + all/successful_strategies.md

# ⑥ 讲"为什么可信"的审计链
cat outputs/matmul/final_output/successful_strategies.md   # 每轮改了什么→多少倍
cat outputs/matmul/optimization_trajectory.json            # 全局状态+全history
```

---

## 13. 局限与未来工作（主动讲, 显专业）

**已知局限**:
1. **真机采集慢**: 每轮 run_optimize 含通用 msprof + 逐 kernel msprof op（分钟级），sweep 全量候选约 5-10 分钟 → 一轮总耗时偏长（stats/timing_stats.json 可看瓶颈阶段）
2. **LLM 输出稳定性**: planner/coder 依赖 `nga run` 输出合法 JSON/代码，虽有多层容错（extract_json 修复、Unicode 清洗、失败重试≤3），极端输出仍会 FAIL 轮
3. **同名 kernel 聚合**: attention_mlp 的 matmul_kernel 被 5 种形状复用 → msprof 同名聚合 deep 画像混合（A1 遗留, 建议拆独立函数名）
4. **无 GPU 侧对照**: 当前只针对 Ascend 910B3 + triton-ascend
5. **测量稳定性**: 双次 msprof 稳定性门控未实现（防单次采样噪声）

**未来工作**（README §10 开发计划 + 项目自身瓶颈优化）:
- verify 增加 per-kernel 期望 launch 校验（防漏记虚高）
- 采集 hash 缓存诊断（采集慢 → 结构没变跳过重采）
- 把优秀案例库（memory/tier{N}_cases.json）做成跨算子迁移学习
- 支持多卡/多算子批处理

---

## 14. 术语表（讲给不懂 Ascend 的听众）

| 术语 | 含义 |
|---|---|
| 910B3 | 华为昇腾训练/推理 NPU, 20 AI Core(cube) + 40 Vec Core @1.8GHz, UB=192KB, L0A/B=64KB, L0C=128KB, HBM 64GB |
| msprof | CANN 性能分析工具（任务级: 每 kernel 耗时/launch; `msprof op`: 单 kernel 深层 8 CSV） |
| msprof op | 逐算子深度 profiling: PipeUtilization/ArithmeticUtilization/Memory/MemoryL0/MemoryUB/L2Cache/ResourceConflictRatio/OpBasicInfo |
| HIVM | 华为中间表示 (Huawei Intermediate Representation), Triton→TTIR→HIVM→AscendC 编译链的一环; Tier2 用它的依赖分析找融合点 |
| torch.npu.Event | NPU 设备侧事件计时（对标 CUDA Event），无 profiler 扰动; ★v4.6 起为快测门粗筛 + 报告参考口径（主决策/工业级对比用 msprof 纯 kernel） |
| MTE1/2/3 | 搬运引擎: MTE1(L1→L0A/B) / MTE2(GM→L1) / MTE3(写回 GM) |
| Cube / Vec / Scalar / Fixpipe | 计算引擎: Cube(矩阵乘) / Vec(向量) / Scalar(标量) / Fixpipe(定标) |
| L0A/L0B/L0C | cube 专用缓冲: A 片/B 片/累加器; 分块调参的硬件约束 |
| UB | Unified Buffer 统一缓冲 (192KB/核) |
| roofline | 屋顶模型: 实际带宽/算力 vs 峰值 → memory/compute/latency/balanced 瓶颈分类 |
| KERNEL_LOOP | kernel_op.py 测试驱动的内部循环次数（verify 用一次 msprof 测 N 遍取平均） |
| MATMUL_VERIFY | 正确性校验开关（kernel 输出 vs torch 参考, 输出 result check: PASS） |
| nga run | 服务器本地 LLM codeagent 调用命令（保密服务器无外网 API, 用 `echo <prompt> \| nga run`） |
| AutoKernel | 对照方案: 盲目枚举调参（300~400 轮）——本项目用"真机诊断 + 6 层策略"替代盲试 |
| sweep | 程序化枚举全部 L0 合法 BLOCK 候选 + 真机 Event 实测, 替代 LLM 猜分块（分块层地基） |

---

### 15.1 工业级最优端到端耗时（industrial_time_us）

**工业级 = 昇腾上真正的工业实现，共 4 种 mode**（bench_910b3/bench_industrial.py 逐个测）：

| mode | 被测实现 | 代表什么 |
|---|---|---|
| eager | torch 直接调 CANN aclnn vendor kernel | 厂商手写算子（matmul/conv/norm 单算子天花板） |
| compile | torch.compile + TorchAir 图模式（GE 图融合） | 自动算子融合链（MLP/attention） |
| cann-fused | aclnnFusedMatmul / FusedConvBiasRelu 直接调用 | 厂商融合算子（matmul+epilogue 一次算完） |
| fa | torch_npu.npu_prompt_flash_attention | CANN FlashAttention（attention 专用） |

**每个算子测哪些 mode**：有厂商融合算子的（matmul/matmul_relu/conv_bias_relu）测 eager+compile+cann-fused 三种；
flash_attention 只测 fa；其余算子测 eager+compile 两种。

**测量工具**：torch.npu.Event（设备侧计时），被测对象 = 每种 mode 的一次真实 torch forward。

**测量步骤**（每个 mode 各测一遍）：
1. 构造该 mode 的 forward（如 matmul 的 eager = torch.matmul(F.gelu(x@w1+b1), w2)），预创建 32 组输入
2. 先跑 5 次 forward，测出单次耗时估算值 est
3. 按时间预算折算测量次数：warmup 次数 = 25ms/est，测量次数 = 100ms/est（限 5~50 次）
4. 跑 warmup 次（不计时，消化 JIT/编译/冷 cache）
5. 正式测量：对每个测量窗口，Event 记录开始 → 跑 1 次 forward（每窗口换一组输入破 L2 复用）→ Event 记录结束
6. 所有窗口取中位数 median → 该 mode 的端到端耗时

**计算步骤（怎么从各 mode 找到"工业级最优"）**：
1. 每个窗口耗时 = 结束 Event 与开始 Event 之间的设备侧耗时（含该次 forward 内全部 kernel 执行、
   kernel 间下发间隙、中间张量分配）
2. 每个 mode 的端到端 = median(全部窗口耗时)，写入 industrial_<op>_<mode>_tflops.json
3. 选优（_load_industrial_best）：各 mode 的端到端中取最小者 = 工业级最优；
   **只认真正执行的 mode**——compile 若因 torchair 不可用回退成 eager（actual_mode≠mode），
   该 mode 的重复测量不参与选优，防止回退值顶替成最优
4. 结果存 st["industrial_time_us"]（含来源文件 industrial_<op>_<mode>_tflops.json 供追溯）

### 15.2 我们优化算子的耗时（best 纯 kernel; Event 参考）

**测量工具**：torch.npu.Event（设备侧计时），被测对象 = kernel_op.py 里 KERNEL_LOOP 循环体
（我们优化后的全部 kernel launch）。

**测量步骤**：
1. 把 kernel_op.py 的 `for _ in range(LOOP): <循环体>` 整体注入进 Event 计时分支
   （KERNEL_EVENT_TIME 环境变量触发，不影响正常/msprof 路径）
2. 先跑 5 次循环体（不计时，warmup）
3. 正式测量：5 个独立窗口，每个窗口 Event 记录开始 → 跑 30 次完整循环 → Event 记录结束；
   每窗口前重建输入（新地址，破 L2 复用）
4. 5 个窗口取中位数，再除以 30 = 单次循环耗时

**计算步骤**：
1. 每个窗口耗时 = 结束与开始 Event 之间的设备侧耗时（30 次循环内全部 kernel 执行，gap≈0，无中间分配）
2. 单次端到端 us = median(5 窗口耗时) / 30
3. 每轮得到该轮耗时; **只保留纯 kernel 小于历史最小的轮次**（KEEP, ★v4.6）, 历史最小那轮的
   纯 kernel = 我们最优（best_kernel_ns）; Event 端到端并列记录作参考
4. 源码 KERNEL_LOOP 循环丢失（loop_ok=False，coder 改坏）→ 该轮不测 Event（参考值缺失）; msprof 行数 < LOOP → 欠采硬门槛不采纳（防假快; Event 漏测不影响 KEEP, 2026-08-18 修复口径）

### 15.3 验收计算

```
验收 = 工业级最优(us) ÷ 我们最优(us)   (★v4.6 双方均为 msprof 纯 kernel 口径)
     = st["industrial_time_us"] ÷ (our_best_kernel_ns / 1000)
```

配套：feedback/acceptance_report.py 批量算全部算子；feedback/remeasure_best.py 重测单个算子。
