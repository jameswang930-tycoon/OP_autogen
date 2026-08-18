# Triton Agent Optimizer — 完整架构设计 (v4.5)

> **核心差异化优势**：不靠盲试（AutoKernel 300~400 轮），用 **真机 msprof + msprof op** 精确诊断瓶颈（真实带宽/L2/算力/冲突），6 层策略按「结构影响从大到小」逐轮优化，每轮只看**当前策略需要的字段**，Planner（笨 LLM）定方向、Coder **确定性改码**、**msprof 纯 kernel 验证 + 严格最优 KEEP（纯 kernel 绝对延迟; ★v4.6 口径统一, Event 只做快测门粗筛与报告参考）**。
>
> **数据源 = 真机双源**：① 通用 `msprof`（骨架：kernel 数/耗时/形状/launch/L2）+ ② 逐 kernel `msprof op`（深层：真实带宽/引擎利用率/算力/冲突）→ `integrate.py` 按 kernel 名合并 → `diagnosis.json`（roofline 核心）。
>
> **环境**：Ascend 910B3（保密服务器，只能 paste-in）+ CANN 8.5.1 / triton-ascend 3.2.0 / torch_npu 2.9.0。
> **LLM 调用**：服务器无 Claude API，用本地 codeagent：`echo "<prompt>" | nga run`（`LLM_CLI_COMMAND="nga run"`）。
> **更新**：2026-08-13 — v4.5 记忆+效率轮（全部有回归测试守护）: ①失败案例库 `memory/failed_cases.py`（按 tier 分文件: 指纹去重/两级检索/attempted_solutions 方案收敛守卫/open→solved|stuck 状态机/负正闭环/LRU; coder 轮内重试累积上下文+库检索注入+成功 solved 回填; hist 记 error_class 四分类）②两段验证（段1 正确性+Event 快测秒级 → 不快于 best 直接 REVERT 省 msprof 分钟级; 段2 过门才全量）③Amdahl 显式编排（每轮 planner 注入 per-kernel 占比排序, 先打占比最大 kernel）④跨轮诊断快照（hist 记 top2 kernel 紧凑串: bn/cu/mu/l2/redun/引擎, planner 看"改法→指标变化→结果"; 全量写 diag_snapshots.jsonl）⑤v3 就地展开式讲演页 feedback/pipeline_diagrams_v3.html。
> **更新**：2026-08-12 — v4.4 修复轮: ①sweep 回滚内容快照(链污染) ②promote 门前置+有效轮计数(max_rounds 硬上限) ③rebaseline 同步 best Event ④diff.patch 仅成功时写 ⑤coder 清洗 4 缺陷(千分位/全角/markdown/引号两难)+报错带行内容 ⑥bench 测量方法学修复(do_bench 同款: 多窗口 median + 时间预算自适应 + 输入轮换破 L2) + FA 对齐 fp16; 全部有回归测试 `_sim_fix_regression.py` (31 项断言)。
> **更新**：2026-08-11 — v4.3：① bench 全切 Event（工业级+PyTorch，不用 msprof）② 迭代每轮补 Event e2e_event_ns ③ 严格最优 KEEP（>best_speedup）+ best_kernel 绑定 + 失败回滚 ④ sweep 每 tier3 round 都跑 ⑤ 设备污染检测+重置 ⑥ vs_industrial 比值 + strategy_summary。
> **更新**：2026-08-18 — 算子扩充 + 工程整理（细节见 §9 开发记录）：① 算子从 14 → **26 个**（08-13 复杂算子 3 个 + 08-14 工业界经典长链 9 个，与 bench_910b3 工业级基准一一对应）② batched_matmul kernel 变量复用 bug 修复（loop-carried 类型冲突，曾致新算子"采集不到 kernel"）③ 全部新算子过 **16 类禁令静态审计**（num_warps/autotune/gather 寻址/vsel 地址/mask 依赖 load/arange 非 constexpr/循环携带类型冲突等，依据 `docx/CODING_GUIDE.md`，检查器经阳性对照验证）④ 架构图定稿 `knowledge/architecture_mermaid_single_v3.html`（阶段 0-7 纵向总图，v2 式并列多箭头扇入/扇出）⑤ 文件归档：架构图 HTML/JS → `knowledge/`，调试脚本 → `test/`。
> **更新**：2026-08-18 (下午) — ★v4.6 口径统一: KEEP/REVERT 主依据从 Event 端到端改为**纯 kernel 绝对延迟**（msprof Task Duration 求和÷遍数 = verify 的 ns）+ 欠采硬门槛（行数<loop 不采纳）; speedup/best_speedup/vs_industrial 全部统一纯 kernel 口径（history 与 best 同源, 根治「显示 2.x vs best 1.x」矛盾）; Event 降为参考（快测门粗筛+报告, best_e2e_event_ns 独立维护）; 工业级对比只认 method=msprof json（`bench_industrial.py --msprof`）; verifier Event 门控改 loop_ok（msprof 漏记不株连）; 修 speedup 轮首未初始化崩溃; 回归 32/32。

---

## 0. 核心理念

| 原则 | 说明 |
|---|---|
| **真机数据优先** | 只用 msprof + msprof op（弃 hivm/simulator 主流程，按需保留 fusion） |
| **单文件驱动** | 算子 + config + 测试合成 `kernel_op.py`（①config ②kernel ③main），coder 只改它 |
| **kernel 链** | round1 读源文件，roundN 读上一轮成功输出；**源文件永不修改**；失败不提交 |
| **每轮只看该策略字段** | 6 层策略各有自己的数据段（58 字段），Planner 只喂当前层要的 |
| **确定性改码** | Planner 输出 `changes[]`（old_code→new_code 逐字符匹配），Coder 精确替换 + Unicode 清洗，找不到就报告不猜 |
| **sweep 分块地基** | round1 + **每个 tier3 round** 自动枚举全部 L0 合法 BLOCK, 在 best_kernel.py 上实测选最优 → 每轮传 planner |
| **验证 = 正确性 + msprof 纯 kernel + Event 参考** | MATMUL_VERIFY 校验 → msprof 纯 kernel ns（★v4.6 KEEP 主口径）+ Event e2e_event_ns（快测门粗筛+报告参考） |
| **严格最优 KEEP** | 本轮纯 kernel ns **< 历史最小 best_kernel_ns** 才进链（★v4.6; 欠采硬门槛 行数<loop 不采纳）; 绑定 best_kernel/best_round + 复制 best_kernel.py; 失败/回退回滚轮首快照 |
| **设备污染恢复** | verify 崩 AICore (HIVM/OOM/575) → 下轮采集前自动重置设备 (修 msprof 采不到 kernel 级联) |
| **严格晋升 + 可回退** | planner promote 必须给数据依据 (promote_evidence); 支持回退前层; 防死循环 (同路径≥3次拒绝) |
| **跳转手递** | 跨 tier 跳转时, 当前 planner 写 10_tier_handoff.json (瓶颈+方向) → 目标 tier planner 读 |
| **优秀案例** | 单轮相对上一最优 >1.3× 加速 → 自动记 memory/tier{N}_cases.json, 后续 planner 参考 |
| **失败案例库** | 失败自动记 memory/tier{N}_failed_cases.json (指纹去重/两级检索/方案收敛守卫/stuck 黑名单); coder 轮内重试读"前几次方案+报错"累积上下文, 禁止原样重试; 成功 solved 回填 |
| **两段验证** | 段1 正确性+Event 快测(秒级) → 不快于 best 直接 REVERT 省 msprof; 过门才跑全量确认+诊断 |
| **Amdahl 编排** | 每轮给 planner 注入 per-kernel 耗时占比排序, 先打占比最大的 kernel (端到端收益≈占比×加速比) |
| **跨轮诊断快照** | hist 每轮记 top2 kernel 关键指标 (bn/cu/mu/l2/redun/引擎) → planner 看"改法→指标变化→结果"因果链; 全量写 diag_snapshots.jsonl |
| **自动 bench + 图表** | 缺 PyTorch 基准自动跑 (Event); 每轮自动出策略摘要; 结束自动轨迹图 + vs_industrial 比值 |

---

## 1. 整体架构

```
main.py input/matmul [--fresh] [--resume] [--max-rounds N] [--target X]
  └─ Scheduler 循环 (默认每次初始化; --resume 续跑)
      每轮:
        ① run_optimize.sh <input_dir> <round_dir>   ← input_dir = current_kernel.parent
             ├─ 通用 msprof → task.json (骨架)
             ├─ 逐 kernel msprof op → board_<i>.json (深层)
             └─ integrate → diagnosis.json (roofline)
        ② _diagnose → 写 roundN/07_tier{N}_fields/ (当前 tier 筛字段)
        ③ (tier2) _run_fusion → roundN/08_fusion/ (HIVM MLIR + nga 融合分析)
        ④ sweep (round1 + 每个 tier3 round 都跑):
             ├─ 在 best_kernel.py (历史最优) 上程序化枚举 L0 合法 BLOCK → 09_tier3_sweep/
             ├─ 单进程 torch.npu.Event 实测 → 最优写入 round_dir/kernel_op.py
             └─ 结果持久化 st["last_sweep_result"] + history 记 sweep_ran/sweep_adopted
        ⑤ _plan (planner) → roundN/plan.md
             读: 07字段 + Amdahl优先级 + planner_context.json + 轨迹(含diag快照) + 手递 + sweep数据 + 优秀案例
             出: changes[] + promote(需evidence) + handoff(跳转给目标tier)
        ⑥ _code (coder 确定性替换 + Unicode 清洗 + ★失败库检索注入) → roundN/kernel_op.py + diff.patch
             失败 → 累积重试上下文(方案+报错全序列) + 失败库注入 → 回传 LLM 修复 (≤3 次)
        ⑦ verify — ★两段验证: 段1 正确性+Event 快测(秒级) → 不快于 best 直接 REVERT (粗筛);
             过门才跑段2 msprof 全量 (正确性+warmup+msprof 纯kernel+Event 参考) → ★严格最优 KEEP (纯kernel ns < best_kernel_ns)
        ⑧ 记录 + 决策: 严格最优/回滚/晋升(防死循环)/停止 + 优秀案例 + 失败案例 + error_class + 诊断快照 + 手递 + 设备污染检测
      每轮: strategy_summary → final_output/{all,successful}_strategies.md
      结束: 自动跑 PyTorch bench (Event) + 自动轨迹图 + vs_industrial_ratio
```

---

## 1.5 测量方法对照（msprof 诊断 + Event 工业级绝对值，双轨）

| 场景 | 工具 | 方式 | 为什么 |
|---|---|---|---|
| **sweep 分块扫描** | `torch.npu.Event` | 单进程, 每候选预热+多窗口 median | 快速筛候选(几分钟); 只需相对排序 |
| **PyTorch/工业级 bench** | `torch.npu.Event` | ★do_bench 同款: 时间预算自适应(warmup 25ms/rep 100ms) + 多窗口 median + **输入轮换破 L2 复用** | ★工业级设备侧绝对值(无 profiler 扰动, 无 L2 命中虚高) |
| **verifier 每轮验证** | **msprof + Event (两段验证)** | ★段1 `verify_fast_gate`: 正确性+Event 快测(秒级, 无 msprof) → 不快于 best 直接 REVERT; 段2 过门才 msprof KERNEL_LOOP=30 遍(op_summary 求和÷30) 给 ns/e2e_ns + Event 给 e2e_event_ns (**★2026-08-12 加每窗口输入重建破 L2, 与工业级同口径**) | msprof 纯 kernel = KEEP 主口径 (★v4.6); Event = 快测门粗筛+参考; 快测门省 REVERT 轮 msprof 分钟级 |

**★bench 测量纪律 (2026-08-12, 对齐 triton testing.do_bench)**:
- **多窗口 median**: 先 5 次估时长 → warmup/rep 次数按 ms 预算自适应 (快 kernel 自动加次) → n_rep 个**独立 Event 对** → 取 median (另报 min/mean)。旧"一次窗口包 30 次 ÷30"只有 1 个样本, 快 kernel 噪声大。
- **★输入轮换破 L2**: 连续 forward 同一批张量, 工作集 <192MB(L2) 时后 N 次全 L2 命中 → 测到 L2 带宽 (数字虚高); Ascend 无清 L2 API → 每 rep 轮换 n_buf 组输入 (组数×单组工作集 > L2) 等效 do_bench 的 clear_cache。★verify 的 Event 注入同法: 每窗口前重放 main 的张量分配 (新地址) 破 L2 — 与工业级基准同口径, vs_industrial 比值才可比。
- **口径声明**（2026-08-13 逐行核对）: 两边 Event 都是**一次完整调用的设备侧耗时**（输入均预创建不在窗口内）——
  工业级 = torch forward（多 kernel 链 + kernel 间 host 下发 gap + forward 内部**中间张量分配**，均在窗口内）;
  我们 verify Event = kernel_op.py 循环体（**融合/单遍后 kernel 数更少** + 连续 launch gap≈0 + 中间结果预分配）。
  计时方法完全一致（Event 设备侧 + 多窗口 median + 破 L2）；**kernel 数/gap/中间分配的差异 = 融合优化的真实收益，
  不是测量差异**——对比公平。大算子 (ms 级) 差异可忽略; 小算子 (逐元素/归约, us 级) 我们天然占优（这正是
  融合/单遍优化的目标）→ 报告同时给 `time_us_min/mean` + 说明。

**★v4.6 口径反转**: 主决策/主加速比/工业级对比全部用 msprof 纯 kernel（Task Duration 求和÷遍数, 与优化对象同尺）; `torch.npu.Event`（设备侧事件计时, 无 profiler 扰动）降为参考——快测门粗筛 + 报告展示。两者并存: msprof 主, Event 参考。

**口径 (★v4.6 统一)**: 主加速比与 KEEP 决策都用 msprof 纯 kernel（`baseline_ns/ns`, history speedup 与 best_speedup 同源）; Event 为参考口径（`e2e_event_ns`, 快测门+报告）; 工业级对比 = 纯 kernel 同尺（只认 method=msprof json）。

---

```
┌─ kernel 链 (★严格最优 采纳/回退) ──────────────────────────────────┐
│ round1: current_kernel = input/<op>/kernel_op.py (源, 永不改)       │
│ roundN: current_kernel = 历史最高加速比那轮 (best_kernel, 严格最优)  │
│  ★采纳 = 本轮纯 kernel ns < best_kernel_ns (历史最小, ★v4.6)        │
│  采纳时同步 best_kernel/best_round + 复制 best_kernel.py            │
│  REVERT(不达最高)/FAIL(≤3次重试) : 回滚轮首快照, 沿用历史最优       │
│  设备污染(verify 崩 AICore) → 下轮采集前 _reset_device              │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
run_optimize.sh input_dir round_dir
  ├─ 拷贝 current_kernel → roundN/input/kernel_op.py (快照)
  ├─ 通用 msprof → 骨架 → task.json
  ├─ 逐 kernel msprof op → board_<i>.json
  └─ integrate → diagnosis.json (summary + kernels[].task/deep + roofline)
           │
           ▼
_diagnose: TIER_FIELDS[tier] → roundN/07_tier{N}_fields/{txt,json}
           │
           ▼
_plan (planner via nga run):
  读: 07字段 + playbook_tier{N}.md + current_kernel + config + 历史梗概
  出: plan.md 含 changes[] (old_code→new_code) + promote
           │
           ▼
_code (coder 确定性):
  读: current_kernel + plan.changes[]
  做: 精确替换 old_code→new_code → roundN/kernel_op.py + diff.patch
  (old_code 找不到 → 报告, 不猜; NOOP 标记)
           │
           ▼
verify: ★两段验证 — 段1 正确性+Event 快测(秒级) → 不快于 best 直接 REVERT;
        过门才跑段2: msprof (warmup + KERNEL_LOOP 平均) + Event 注入计时 → e2e_event_ns
  speedup = baseline_e2e_ns / e2e_ns   (严格最优: > best_speedup 才 KEEP)
           │
           ▼
记录 hist (+ e2e_event_ns/sweep_ran/sweep_adopted/error_class/diag诊断快照) + 晋升 (planner.promote / 3轮无改进 / 达标 / tier6停)
   失败轮 → 失败案例库 (tier{N}_failed_cases.json: 指纹去重/方案收敛守卫/stuck 黑名单)
           │
           ▼
每轮产出 strategy_summary → final_output/{all,successful}_strategies.md
最终: vs_industrial_ratio = 我们最优Event / 工业级Event (优化效果终极指标)
```

---

## 3. 输入（input/<op>/）

```
input/<op>/
  kernel_op.py    ← ★单文件: ① 场景 config(M/N/K/DTYPE/BLOCK_*) ② kernel ③ 测试 main
  config.json     ← 旧式场景配置 (已不用, 并入 kernel_op.py)
  triton_kernel.py/test_matmul.py  ← 旧式三文件 (已不用, 内容并入 kernel_op.py)
```

- **kernel_op.py 是纯 triton 语法**（`@triton.jit`/`tl.dot`/BLOCK_M/N/K constexprs），跑在 **triton-ascend 后端**（910B3 NPU）。
- 约束：`num_warps`/`num_stages` 由 triton-ascend 自动管理，**不能传**（传了报 `please do not tune args`）。
- ⚠ kernel 内**同一变量名不得跨类型复用**（如 batched_matmul 曾用 `b` 既当 batch 序号 int32 又当循环内 fp32 张量块 → `CompilationError: loop-carried variable ... type stays consistent`，应用在 warmup 就崩 → msprof 采不到任何 kernel → "采集失败"）。新算子入库前建议过一遍 `docx/CODING_GUIDE.md` 禁令清单。

**算子清单（26 个，2026-08-18）**：

| 组 | 算子 |
|---|---|
| matmul 族 | `matmul`(两层MLP) / `matmul_relu` / `matmul_transpose` / `batched_matmul`(BMM) / `swiglu_mlp`(LLaMA FFN) |
| attention 族 | `attention_mlp` / `flash_attention` / `gqa_attention`(GQA+RoPE) |
| 归一化/逐元素 | `rms_norm` / `rms_norm_residual` / `layernorm` / `softmax` / `sigmoid` / `vector_add` / `fused_add_mul` |
| 卷积/池化 | `conv2d` / `conv_bias_relu` / `conv1d` / `batchnorm2d` / `maxpool2d` |
| 经典长链（08-14, 对应工业级基准） | `resnet_block` / `vit_block` / `bert_block` / `transformer_decoder_block` / `mamba_block` / `mixture_of_experts` |

---

## 4. 输出结构（每轮自包含）

```
outputs/<op>/
├── optimization.log              # 全流程运行日志 (Tee 双写: 终端 + 文件)
├── best_kernel.py                # ★历史最优 kernel (KEEP 轮更新; sweep 的实验底座)
├── baseline_verify/              # round1 基准复测产物 (与后续轮同口径)
├── <tier_name>/roundN/           # 每轮一个自包含目录 (01_algorithmic_structure ~ 06_910b3_architecture)
│   ├── input/kernel_op.py        # 本轮采集快照 (上一轮输出/源)
│   ├── kernel_op.py              # ★本轮 coder 优化输出
│   ├── diff.patch                # 本轮改动 (仅 coder 成功时写)
│   ├── plan.md                   # planner 计划 (changes[] + promote)
│   ├── event_kernel.py           # Event 计时注入版 (verify 生成)
│   ├── failed_kernel.py          # (失败时) 崩掉的中间产物, 排查留证
│   ├── 04_board/                 # msprof op 8 CSV 原始 → board_<i>.json
│   ├── 05_task/                  # 通用 msprof 原始 → task.json
│   ├── 06_diagnosis/             # diagnosis.json (骨架+deep+roofline)
│   ├── 07_tier{N}_fields/        # ★当前 tier 筛字段 (planner 读)
│   ├── 08_fusion/                # (仅 Tier2) HIVM 依赖分析 + 可融合候选
│   ├── 09_tier3_sweep/           # (仅 sweep 轮) 分块扫描: sweep_result.json + runner
│   └── msprof_0/                 # verify 阶段的 msprof 验证产物
├── optimization_trajectory.json  # ★全局状态 (state + history; --resume 续跑用它)
└── final_output/                 # 优化结束自动生成
    ├── kernel_op.py              # 最优 kernel (取 best_kernel, 可直接用)
    ├── baseline_kernel.py        # baseline 副本 (对照)
    ├── final_summary.json        # 双口径加速比 + Event 延迟 + vs_industrial_ratio
    ├── trajectory_chart.png      # 轨迹图 (加速比曲线 + tier 色带 + PyTorch 虚线 + 工业级红线)
    └── {all,successful}_strategies.md   # 每轮策略摘要 (全部 / 仅成功)
```

---

## 5. 6 层策略 × 读取字段 / 识别瓶颈 / 优化策略

| Tier | 策略 | 读取的字段 | 识别的瓶颈 | 主要优化策略 |
|---|---|---|---|---|
| 1 | 算法结构 | 算力利用率/算术强度/瓶颈类型/cube fp16·int8 占比/cube·vec 计算量/算子数/launch 开销 | 算法选错（利用率<0.3）/精度没吃满 cube/冗余访存（巨大中间张量）/重复计算+launch 开销/归约多遍扫 | fp16 输入+fp32 累加、Flash Attention（online softmax）、多个同结构小 matmul 合并单 GEMM、online 单遍归约、split-K、persistent kernel、软件 im2col 走 cube、GQA 消组内复制、MoE topk 稀疏 |
| 2 | 算子融合 | 算子数/launch 开销/每 kernel 引擎类型与次数/框架 kernel/瓶颈类型 + 08_fusion 依赖分析（含每 op 耗时占比） | matmul→逐元素→matmul 分离链有融合空间/launch 开销占比高/中间张量 GM 来回（读写带宽高但算力低） | 逐元素并入 matmul epilogue（bias/激活/残差）、同张量多次读→单次 load 复用/拼列、scale/mask/softmax 并入、UB 内直接消费；融合前先算收益（中间 kernel 耗时 > 2×launch 才融） |
| 3 | 分块配置 | 核数 block_dim/mte1·mte2·cube 引擎占比/L0A·L0B 读写带宽贴合度 | 分块太小并行不足（核数<40）/BLOCK_K 太大 MTE1 搬运瓶颈/BLOCK_M·N 太小 GM 搬运频繁/块不够大喂不饱 cube | 减 BLOCK_M/N 提核数、增 BLOCK_K 降 MTE1、增 BLOCK_M/N 降 MTE2、tile+swizzle；round1+分块轮自动 sweep 穷举 L0 合法块实测选优 |
| 4 | 访存 | GM 读写带宽/GM↔UB 通路/L2 命中率/MTE2·MTE3 耗时/访存利用率/算术强度/实际搬运量·通路利用率·冗余倍数 | GM 流量过大（memory_bound）/跨步非连续访问/L2 复用差/搬运无流水线/非 16B 对齐拆事务 | 连续化（最快维匹配布局）、128-bit 对齐+padding、L2 复用（访问序/权重预排/swizzle）、load 独立成步骤让编译器双缓冲、零散小 load 合并大块搬运、输出先 UB 内转再连续 store |
| 5 | 计算占用 | cube·标量耗时/scalar·fixpipe 占比/向量计算量/冲突占比（bank/bankgroup/等待） | 标量降级（指针 div·mod/int64 索引）/逐元素未向量化/非原生数学指令（1÷sqrt·erf·除法）/UB bank 冲突/寄存器溢出（展开过度） | 消除标量降级（2D 索引/int32）、向量 load、rsqrt/tanh 近似/FMA 融合、倒数+乘法代除法、访问 swizzle+尾轴 32B/512B 对齐、控制展开与 ILP |
| 6 | 架构专属 | 各引擎利用率分布/mte 冲突/vec·cube 等待占比/icache 缺失/核数 vs 物理核/多核耗时不均 | 跨引擎流水气泡/没走 cube（vector 模拟 matmul）/cube·vec 严重失衡/grid 远大于核数（调度开销）/尾核空转 | 回 Tier4 流水线/Tier3 分块、用 tl.dot 走 cube、grid 固定物理核数+核内 stride 循环、stride 切分消拖尾、K 循环分块+双缓冲、向量算子按引擎选核数 40/20 |

> 2026-08-12 扩充（官方文档核实，见 `docx/msprof_fields_reference.md` 四节）：Memory.csv 的 `*_datas(KB)` 实际搬运量、`*_bw_usage_rate(%)` 官方通路利用率、PipeUtilization 的 `*_active_bw(GB/s)` 活跃带宽、`ai*_icache_miss_rate`、ResourceConflictRatio 的 `ai*_wait_ratio` 规范短名（`vec_wait_ratio` 等，消除 `_get` 子串歧义）、ArithmeticUtilization 的 vec 精度细分与 cube fp/int 指令条数、OpBasicInfo 的 `Rated Freq`/`Mix Block Dim`；roofline 新增 `traffic_redundancy_read`（实际读÷理论最小，>1.5=重复搬运）。hivm fusion view 现附每 op 估算耗时占比（Tier2 排融合优先级）。

**分块调参逻辑**（用户关注点）：
- **传输瓶颈**(memory_bound, mem_util≥0.8 且 comp<0.5) → **增大 tile**（复用↑, GM流量↓）
- **计算瓶颈**(comp≥0.8 且 mem<0.5) → **不调小**（cube 已满, 看算法/精度如 fp16）
- **延迟瓶颈**(两者都<0.3) → 调小/增并行
- UB 约束：每 K 迭代 `(BLOCK_M+BLOCK_N)×BLOCK_K×dtype` ≤ 192KB

---

## 6. 文件架构（核心执行链路）

```
triton_agent_optimizer/
├── main.py                     # 入口: 解析参数 → 启动 Scheduler (--fresh/--resume/--max-rounds/--target/--stub)
├── config.py                   # 全局配置中心: 路径/阈值/.env 加载 (全链路 import)
│
├── agents/                     # ── 智能体层 (每轮调度执行) ──
│   ├── scheduler.py            # 调度状态机: 采集→诊断→规划→改码→两段验证→KEEP/REVERT 决策→晋升
│   ├── planner.py              # 规划器 (LLM): 读字段/轨迹/案例 → plan.md (changes[] + promote)
│   ├── coder.py                # 编码器: 确定性替换 changes[] + Unicode 清洗 + 失败库注入修复
│   ├── verifier.py             # 验证器: 两段验证 (快测门 + msprof 双口径 + Event 设备侧计时)
│   └── llm_client.py           # LLM 统一入口: nga run CLI / API / stub
│
├── analyzers/                  # ── 采集解析层 (每轮真机执行) ──
│   ├── run_optimize.sh         # 采集驱动: warmup → 通用 msprof → msprof op → integrate → 07 字段
│   ├── pipeline_parse_task.py  # 通用 msprof → task.json (骨架: 每算子耗时/launch/形状)
│   ├── pipeline_parse_board.py # msprof op → board_<i>.json (深层: 带宽/引擎利用率/冲突)
│   ├── integrate.py            # 骨架+深层合并 → diagnosis.json (roofline 诊断中枢)
│   ├── sweep_blocks.py         # Tier3 分块扫描: 程序化枚举 L0 合法 BLOCK + Event 实测
│   └── run_hivm_fusion.py      # Tier2 融合分析: HIVM 依赖 → 可融合候选
│
├── bench_910b3/                # ── 基准层 (Event 设备侧计时; 主循环缺基准时自动执行) ──
│   ├── bench_industrial.py     # 工业级基准 (eager/compile/cann-fused/fa 各 mode)
│   ├── bench_pytorch*.py       # PyTorch 基准 (按算子变体一组脚本)
│   └── run_bench.py            # 硬件峰值校准 → hardware_peak.json (roofline 分母)
│
├── input/                      # ── 算子源 (26 个算子, 每算子一目录) ──
│   └── <op>/kernel_op.py       #   单文件三合一: ① 场景 config(尺寸/精度/分块) ② 算子 kernel ③ 测试 main
│
├── memory/                     # ── 记忆层 (调度器/编码器调用) ──
│   ├── excellent_cases.py      # 优秀案例记录/检索 (>1.3× 轮次, planner 优化前参考)
│   └── failed_cases.py         # 失败案例库: 指纹去重/两级检索/solved 回填 (coder 修复注入)
│
├── feedback/                   # ── 结果产出 (每轮/结束时执行) ──
│   ├── strategy_summary.py     # 每轮策略摘要 → final_output/{all,successful}_strategies.md
│   └── trajectory_chart.py     # 结束轨迹图: 加速比曲线 + tier 色带 + 工业级红线
│
├── skills/                     # LLM 技能: planner/coder/fusion 三个 SKILL.md (prompt 铁律, 只读)
├── docx/                       # 知识库: 6 层 playbook + 编码规范 + 字段参考 (LLM 每轮读)
├── knowledge/                  # 技术细节文档 + 架构图 (定稿 architecture_mermaid_single_v3.html)
├── test/                       # 回归测试 (_sim_* 系列) + 归档调试脚本
├── outputs/                    # 运行产物 (运行时生成: outputs/<op>/ 轨迹 + tier/roundN/ + final_output/)
└── paper_reference/            # 论文参考资料 (只读)
```

---

## 7. 运行命令（服务器）

```bash
# 环境 (一次性)
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ① 完整优化循环 (推荐; --target 省略/0 = 不设目标跑满 --max-rounds, 看最优)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15 --target 2.0
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15   # 无目标, 跑满15轮
LLM_CLI_COMMAND='nga run' python3 main.py input/attention_mlp --fresh --max-rounds 15   # 复杂算子: 自注意力+MLP (5 kernel)
# 新算子 (08-14 经典长链, 与工业级基准一一对应; 全部 26 算子的命令见 main.py 头部注释)
LLM_CLI_COMMAND='nga run' python3 main.py input/batched_matmul --fresh --max-rounds 30   # BMM (★08-18 修复后可跑)
LLM_CLI_COMMAND='nga run' python3 main.py input/swiglu_mlp --fresh --max-rounds 30       # LLaMA FFN
LLM_CLI_COMMAND='nga run' python3 main.py input/transformer_decoder_block --fresh --max-rounds 30  # 17 kernel 大融合
LLM_CLI_COMMAND='nga run' python3 main.py input/mixture_of_experts --fresh --max-rounds 30         # MoE topk 路由

# ①.2 新算子入库前自检 (秒级; 裸跑不过 = 采集必失败, trace 会指到具体行)
for op in batched_matmul swiglu_mlp resnet_block; do timeout 180 python3 input/$op/kernel_op.py 2>&1 | tail -2; done

# ①.5 单文件能跑 + 数值校验 (跑优化前先确认 kernel_op.py 能跑)
python3 input/matmul/kernel_op.py && MATMUL_VERIFY=1 python3 input/matmul/kernel_op.py
python3 input/attention_mlp/kernel_op.py && MATMUL_VERIFY=1 python3 input/attention_mlp/kernel_op.py

# ② 只采集+解析
bash analyzers/run_optimize.sh input/matmul input/matmul/e2e_run

# ③ 逐 tier 筛字段核对 (解析完 07 自动产出)
cat input/matmul/e2e_run/07_tier1_fields/tier1_fields.txt

# ④ 工业级基准 (各 mode 真机测 → 取每算子 median 最小作为对比天花板; ★主循环缺时自动跑 bench_industrial.py,
#   无需手动; AUTO_RUN_IND_BENCH=0 关闭; 这里整批跑一次可让全部算子都有)
cd bench_910b3 && python3 bench_all.py
python3 bench_all.py --clean      # 清理 bench_910b3/outputs/ 全部产物

# ⑤ 硬件基准 (roofline 峰值) + 轨迹图
python3 run_bench.py && cd .. && python3 feedback/trajectory_chart.py outputs/matmul

# ⑥ 看诊断报告
cat <round_dir>/06_diagnosis/diagnosis.json
```

---

## 8. 910B3 硬件参数

**✅ 准确**：20 AI + 40 Vec cores @1.8GHz；UB=192KB、L1=512KB、L0A/L0B=64KB、L0C=128KB、L2=192MB、HBM=64GB；GM 峰值 **1638.4 GB/s**（HBM2e 4×409.6，联网核实 2026-08）；cube 峰值 **294.9** TFLOPS(fp16 标称) / **313**（官方）/ fp32 **73.7**。理论峰值由 `bench_910b3/bench_theory.py` 公式推导。

**微路径带宽**（GM→UB / UB→GM / L0A/L0B feed / MTE 占比）：由 `bench_910b3/run_bench.py` 的 msprof op per-path 实测填 `hardware_peak.json`（v4 已弃用 timing_estimator 静态成本模型）。

---

## 9. 开发记录

- **07-23~07-31 (v3)**：Triton→HIVM→simulator 三源 + 29 字段 dsl_merger + agents 循环。
- **08-03**：HIVM 获取打通 + msprof op 字段核实（OpBasicInfo/PipeUtilization/Memory）。
- **08-04**：v4 起步——单文件合并、run_optimize.sh 输入输出参数、scheduler 状态机、integrate v4、tier 字段提取、nga run 接入、确定性改码。
- **08-05 (本次)**：
  - kernel 链（round1 源 / roundN 上轮输出 / 源不改）
  - 修复：NA 误报、conflict 子串、pipe 归一化、coder BOM、历史 speedup bug、nga echo 转义、CLI 优先级、planner 读路径、coder 传路径、默认初始化
  - 6 层字段审计定版（50 字段）
  - 硬件基准套件 bench_910b3 + PyTorch 基准线
  - 轨迹图 v4 兼容
  - 全部提交 commit `2c53616`
- **08-10 (v4.2)**：端到端口径统一（纯kernel+E2E 双指标，主=E2E）+ 工业级基准（各 mode 取 min）+ 基准产物收纳 outputs/ + 严格晋升/回退/手递 + 优秀案例。
- **08-11 (v4.3)**：
  - ★bench 全切 Event 设备侧计时（industrial/PyTorch，不用 msprof；删 pt_msprof.py）
  - ★迭代每轮补 Event `e2e_event_ns`（verify 注入 KERNEL_LOOP，工业级绝对值；原 msprof 字段全保留）
  - ★严格最优 KEEP（`> best_speedup`）+ best_kernel/best_round 绑定 + 复制 best_kernel.py + 失败回滚（修 sweep 交叉 bug / input 链断裂）
  - sweep 每个 tier3 round 都跑（去 hash 跳过）+ 在 best_kernel 上扫 + BLOCK 匹配容错
  - 设备污染检测 + 重置（修 msprof 采不到 kernel 的采集失败级联）
  - vs_industrial_ratio（我们最优 Event / 工业级 Event）+ strategy_summary 每轮策略摘要
  - coder 制表符/header 保护 + failed_kernel.py 留证
  - main.py 注释补齐全部 14 算子运行命令
- **08-12 (v4.4 修复轮)**：
  - ★bench 测量方法学修复（对齐 triton do_bench）：多窗口 median（原单窗口 ÷N 只有 1 个样本）+ 时间预算自适应（warmup 25ms/rep 100ms）+ ★输入轮换破 L2 复用（Ascend 无清 L2 API，组数×工作集>L2 等效 clear_cache）+ 口径声明（工业级=torch 全流程 vs 我们=纯kernel，json 记 time_us_min/mean）
  - ★verifier Event 注入同步改多窗口 median（KEEP 决策依据更稳；注入产物加 compile 校验）
  - flash_attention 输入对齐 fp16（工业级 FA 即 fp16，原 fp32 vs fp16 不可比；verify 参考升 fp32）
  - 删死代码（_build_fn/_run_loop）；_sim_fix_regression.py 补 P15（bench 测量回归）+ P7 强化（Event 注入 compile 校验）
  - ★假小 Event 防毒（用户报告 "加速比突然 200x → 真实优化轮永不 KEEP"）: coder 改坏 KERNEL_LOOP 循环 → Event 窗口未跑满 → 假小值毒 best。
    ★2026-08-12 简化（用户原则: Event 测对就保留, 初始代码差几百上千倍加速比真实存在）: 防护改为
    「源码 KERNEL_LOOP 循环丢失(loop_ok=False) → verify 不测 Event(None) → 方案A 不保留; msprof 漏记(行数<loop但循环完整) → Event 照测(独立注入, ★2026-08-18 修复, 不再株连误 REVERT); 循环完整 → Event 真实,
    按绝对延迟比最小端到端, 谁小留谁, 无任何比值拦截」; best 只在 kept 时更新; rebaseline 复测同样
    由行数保证; Event 缺失原因进 hist error; 回归 P16
- **08-13 (v4.5 记忆+效率轮)**（提交 ac526da）：
  - ★失败案例库 `memory/failed_cases.py`：按 tier 分 6 个文件（与优秀案例对称）; 归一化签名指纹去重
    （去行号/地址/路径/数字）+ 两级检索（指纹精确必中 + 关键词交集近似）+ attempted_solutions
    方案历史收敛守卫（同方案不重复记）; 生命周期 open→solved（成功回填方案+补丁）| stuck
    （attempts≥3 封原方案, ★只禁原样重试不封新方案—弱模型也能继续试错）; solved 再现自动降级重计;
    负正闭环（失败方案曾在优秀案例成功过 → 提示上下文差异）; 环境性失败不入库; LRU 上限; 永不抛异常
  - scheduler 轮内重试累积上下文（每次尝试的方案+报错全序列 → 修复器知道试过什么）+ 失败库注入
    （solved/stuck/已试方案）+ 禁止原样重试; 成功轮 mark_solved 回填; hist 记 error_class 四分类
    （env/code_compile/code_numeric/code_runtime）
  - ★两段验证 TWO_PHASE_VERIFY：段1 `verify_fast_gate`（正确性+Event 快测秒级, 无 msprof）
    → Event 不快于 best 直接 REVERT（省 warmup+msprof 分钟级/轮）; 过门才段2 全量确认+诊断;
    msprof 轮 Event 缺失用快测值兜底; stub 自动禁用
  - ★Amdahl 显式编排：每轮给 planner 注入 per-kernel 耗时占比排序（×launch_count 加权）,
    先打占比最大的 kernel, 动其他 kernel 必须给理由
  - ★跨轮诊断快照：hist 记 top2 kernel 紧凑串（bn/cu/mu/l2/redun/引擎）→ planner 看
    "改法→指标变化→结果"因果链; 全量 per-kernel 快照写 diag_snapshots.jsonl（审计/讲演, 不入 context）
  - coder 旧 CodeErrorMemory（按 kernel 分/纯子串匹配）替换为 tier 级失败库注入 + solved 回填
  - verifier 抽公共 _correctness_check + 新增 verify_fast_gate
  - feedback/pipeline_diagrams_v3.html：就地展开式讲演页（点击块内 + 就地展开, CSS grid 固定主链一行, 零依赖）
  - 全部测试通过: 11 个回归脚本 + 失败案例库 7 项单测 + 诊断快照单测
- **08-14 (算子扩充)**：
  - 新增工业界经典长链 9 算子：batched_matmul / swiglu_mlp / resnet_block / vit_block / bert_block /
    gqa_attention / transformer_decoder_block / mamba_block / mixture_of_experts（与 bench_910b3 工业级基准一一对应）
  - 连同 08-13 的 batchnorm2d / maxpool2d / conv1d，算子总数 14 → 26；main.py 头部注释补齐全部运行命令
- **08-17~18 (架构图定稿 + 新算子排障 + 工程整理)**：
  - 架构图系列制作与定稿：`knowledge/architecture_mermaid_single_v3.html`（图 4 = 阶段 0-7 从上到下
    一整张纵向总图; 并列表达 = v2 式多箭头扇入/扇出 + 同级左右并排; 内置缩放控件; 阶段顺序经浏览器实测
    y 坐标逐级递增验证; 保留 v2 八张分阶段图版）
  - ★新算子"采集不到 kernel"排障定案：batched_matmul `bmm_kernel` 变量 `b` 跨类型复用
    （batch 序号 int32 ↔ 循环内 fp32 张量块）→ `CompilationError: loop-carried variable` →
    应用 warmup 即崩 → msprof op_summary 无目标 kernel 行 → run_optimize"未检测到任何目标 kernel"退出。
    修复：循环内改名 `b_tile`（顺带修了 L66 batch 偏移被覆盖的隐藏逻辑 bug）。裸跑成功 + 采集失败
    = 问题在 kernel 编译层而非环境（磁盘/缓存/设备污染假设逐一排除）
  - ★全部 12 个新算子过 16 类禁令静态审计（依据 docx/CODING_GUIDE.md + skills/triton-op-coder/SKILL.md
    "明写禁止"清单）：循环携带类型冲突 / num_warps·num_stages / autotune / TMA·erf 等禁用 API /
    tl.dot 禁参 / arange 非 constexpr·非 2 幂 / load·store 地址含 where·条件（vsel）/ gather 数据依赖寻址 /
    mask 依赖 load / kernel 内 try·dict·import·print·math.* / 非 range 循环 / BLOCK 2 幂且 ≥16 /
    launch 实参匹配 / Unicode——除已修的 batched_matmul 外全部干净; 检查器经阳性对照验证（注入 8 种违规全中）
  - 文件归档：architecture_mermaid*.html + mermaid.min.js → knowledge/；根目录 *_tmp.py 调试脚本 → test/；
    根目录只留入口与核心文档
- **08-18 下午 (v4.6 口径统一)**：
  - ★KEEP/REVERT 决策主依据从 Event 端到端改为**纯 kernel 绝对延迟**（msprof Task Duration 求和÷遍数
    = verify 的 ns）+ 欠采硬门槛（行数<loop → 求和偏小假快 → 不采纳, 防毒 best_kernel_ns）
  - speedup/best_speedup/vs_industrial 全部统一纯 kernel 口径 — history speedup 与 best_speedup 同源,
    根治用户报告的「speedup 2.x 却 REVERT（best 1.x）」口径矛盾（排查: 显示值 msprof 口径 vs 决策
    Event 口径不同源 + Event 缺失株连误 REVERT 两条路径)
  - Event 降为参考: 快测门粗筛（Event ≥ best Event → 直接 REVERT 省 msprof）+ 报告展示;
    best_e2e_event_ns 独立维护不参与决策; history 新增 event_speedup 参考字段
  - 工业级对比只认 method=msprof json（bench_industrial.py --msprof 产; Event 版跳过+提示重测）
  - verifier Event 门控改 loop_ok（源码循环在就测, msprof 漏记不再株连 → 修真实改进轮误 REVERT）
  - 修 scheduler speedup 轮首未初始化（失败轮 UnboundLocalError）; 测试 fixture 补 KERNEL_LOOP 行
    + 新增"真丢循环"对照场景; P16 断言更新（欠采提示词）; 回归 32/32 + bugfix_verify 25 项全过
  - 文档全量同步（本 README + ARCHITECTURE_DESIGN.md 26 处）到 v4.6 口径
  - ★v4.6 新增自动测量字段: 迭代前自动测工业级 **eager/compile 纯 kernel 各自单独值**
    （state: industrial_eager/compile_kernel_us; 自动跑改为逐 mode 补齐）; 每轮优化成功 (KEEP) 后记录
    我们纯 kernel 设备侧耗时（hist/state: our_kernel_us, 取本轮 verify 的 msprof 实测, 与工业级同尺）;
    final_summary 新增 our_kernel_us + vs_eager_ratio + vs_compile_ratio + 逐 mode 单独值

---

## 10. 后续开发计划

- ✅ **前置 BLOCK 扫描（已实现 `sweep_blocks.py` + `main.py --sweep-blocks`）**：
  分块层(BLOCK_M/N/K)是确定性小空间(16 倍数×UB/L0 约束)，用穷举替代笨 LLM 调参更稳。
  优化前先扫 L0 合法候选，msprof 取最快写回 config 区（固定"乘性地基"，防分块差几十倍）。
  ⚠ 每个候选=一次真机 msprof（分钟级），8 候选约半小时+，一次性成本。
- **A1 修复**：attention_mlp 的 `matmul_kernel` 被 5 种形状复用 → msprof 同名聚合 deep 画像混合，
  应拆独立函数名（对齐 matmul 算子 matmul_kernel2 的纪律）；D4 已在 coder 加同名多匹配警告。
- **新算子入库门禁**：把 08-18 的 16 类禁令静态审计固化为 `test/` 下的入库前检查脚本
  （裸跑 + 禁令扫描双门，防止 batched_matmul 类"采集失败"再入库）。
- **测量完整性**：verify 增加 per-kernel 期望 launch 校验（防漏记虚高）；
  双次 msprof 稳定性门控（成本高，待定）。
- **项目自身瓶颈分析**：`outputs/<op>/stats/timing_stats.json` 已输出各阶段耗时，
  后续可按瓶颈阶段优化流程本身（如采集太慢→hash 缓存诊断）。

---

## 11. 参考资料

- Triton 03 教程（matmul/swizzle）：https://triton-lang.org/main/_downloads/b51b68bc1c6b1a5e509f67800b6235af/03-matrix-multiplication.ipynb
- msprof op 8 CSV 字段：https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/devaids/optool/atlasopdev_16_00851.html
- triton-ascend 指南：https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/programming_guide/cube_operator.md
- ascend-dmi 硬件诊断（测带宽）：https://xiexianbin.cn/hardware/huawei-ascend/commands/index.html
