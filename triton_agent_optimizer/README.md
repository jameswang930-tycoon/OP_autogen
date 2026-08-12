# Triton Agent Optimizer — 完整架构设计 (v4.2)

> **核心差异化优势**：不靠盲试（AutoKernel 300~400 轮），用 **真机 msprof + msprof op** 精确诊断瓶颈（真实带宽/L2/算力/冲突），6 层策略按「结构影响从大到小」逐轮优化，每轮只看**当前策略需要的字段**，Planner（笨 LLM）定方向、Coder **确定性改码**、**msprof 端到端验证加速比 + Event 设备侧计时**（工业级绝对端到端）。
>
> **数据源 = 真机双源**：① 通用 `msprof`（骨架：kernel 数/耗时/形状/launch/L2）+ ② 逐 kernel `msprof op`（深层：真实带宽/引擎利用率/算力/冲突）→ `integrate.py` 按 kernel 名合并 → `diagnosis.json`（roofline 核心）。
>
> **环境**：Ascend 910B3（保密服务器，只能 paste-in）+ CANN 8.5.1 / triton-ascend 3.2.0 / torch_npu 2.9.0。
> **LLM 调用**：服务器无 Claude API，用本地 codeagent：`echo "<prompt>" | nga run`（`LLM_CLI_COMMAND="nga run"`）。
> **更新**：2026-08-12 — v4.4 修复轮: ①sweep 回滚内容快照(链污染) ②promote 门前置+有效轮计数(max_rounds 硬上限) ③rebaseline 同步 best Event ④diff.patch 仅成功时写 ⑤coder 清洗 4 缺陷(千分位/全角/markdown/引号两难)+报错带行内容 ⑥bench 测量方法学修复(do_bench 同款: 多窗口 median + 时间预算自适应 + 输入轮换破 L2) + FA 对齐 fp16; 全部有回归测试 `_sim_fix_regression.py` (31 项断言)。
> **更新**：2026-08-11 — v4.3：① bench 全切 Event（工业级+PyTorch，不用 msprof）② 迭代每轮补 Event e2e_event_ns ③ 严格最优 KEEP（>best_speedup）+ best_kernel 绑定 + 失败回滚 ④ sweep 每 tier3 round 都跑 ⑤ 设备污染检测+重置 ⑥ vs_industrial 比值 + strategy_summary。

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
| **验证 = 正确性 + msprof + Event** | MATMUL_VERIFY 校验 → msprof 端到端测速比 + Event 设备侧 e2e_event_ns (工业级绝对值) |
| **严格最优 KEEP** | 本轮 speedup **> 历史最高 best_speedup** 才进链; 绑定 best_kernel/best_round + 复制 best_kernel.py; 失败/回退回滚轮首快照 |
| **设备污染恢复** | verify 崩 AICore (HIVM/OOM/575) → 下轮采集前自动重置设备 (修 msprof 采不到 kernel 级联) |
| **严格晋升 + 可回退** | planner promote 必须给数据依据 (promote_evidence); 支持回退前层; 防死循环 (同路径≥3次拒绝) |
| **跳转手递** | 跨 tier 跳转时, 当前 planner 写 10_tier_handoff.json (瓶颈+方向) → 目标 tier planner 读 |
| **优秀案例** | 单轮相对上一最优 >1.3× 加速 → 自动记 memory/tier{N}_cases.json, 后续 planner 参考 |
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
             读: 07字段 + planner_context.json + 轨迹 + 手递 + sweep数据 + 优秀案例
             出: changes[] + promote(需evidence) + handoff(跳转给目标tier)
        ⑥ _code (coder 确定性替换 + Unicode 清洗) → roundN/kernel_op.py + diff.patch
        ⑦ verify (正确性校验 + msprof 端到端 + Event e2e_event_ns) → ★严格最优 KEEP (>best_speedup)
        ⑧ 记录 + 决策: 严格最优/回滚/晋升(防死循环)/停止 + 优秀案例 + 手递 + 设备污染检测
      每轮: strategy_summary → final_output/{all,successful}_strategies.md
      结束: 自动跑 PyTorch bench (Event) + 自动轨迹图 + vs_industrial_ratio
```

---

## 1.5 测量方法对照（msprof 诊断 + Event 工业级绝对值，双轨）

| 场景 | 工具 | 方式 | 为什么 |
|---|---|---|---|
| **sweep 分块扫描** | `torch.npu.Event` | 单进程, 每候选预热+多窗口 median | 快速筛候选(几分钟); 只需相对排序 |
| **PyTorch/工业级 bench** | `torch.npu.Event` | ★do_bench 同款: 时间预算自适应(warmup 25ms/rep 100ms) + 多窗口 median + **输入轮换破 L2 复用** | ★工业级设备侧绝对值(无 profiler 扰动, 无 L2 命中虚高) |
| **verifier 每轮验证** | **msprof + Event** | msprof KERNEL_LOOP=30 遍(op_summary 求和÷30) 给 ns/e2e_ns; 另注入 Event 给 e2e_event_ns | msprof 给纯kernel拆解+诊断; Event 给工业级绝对端到端 |

**★bench 测量纪律 (2026-08-12, 对齐 triton testing.do_bench)**:
- **多窗口 median**: 先 5 次估时长 → warmup/rep 次数按 ms 预算自适应 (快 kernel 自动加次) → n_rep 个**独立 Event 对** → 取 median (另报 min/mean)。旧"一次窗口包 30 次 ÷30"只有 1 个样本, 快 kernel 噪声大。
- **★输入轮换破 L2**: 连续 forward 同一批张量, 工作集 <192MB(L2) 时后 N 次全 L2 命中 → 测到 L2 带宽 (数字虚高); Ascend 无清 L2 API → 每 rep 轮换 n_buf 组输入 (组数×单组工作集 > L2) 等效 do_bench 的 clear_cache。
- **口径声明**: 工业级基准 = torch 全流程 (含 host 调度/内存分配); 我们 verify 的 Event = triton 纯 kernel launch 链。大算子 (ms 级) 差异可忽略; 小算子 (逐元素/归约, us 级) 我们占便宜 → 报告同时给 `time_us_min/mean` + 说明。

**为什么 Event 是工业级主测?** msprof 带 profiler 挂载开销(绝对值偏高, 但跨轮一致→相对 speedup 用它); `torch.npu.Event` 是设备侧事件计时(无扰动, 官方/NVIDIA/vLLM-Ascend 标准), 最终报告绝对 latency 用它。两者并存: msprof 拆解, Event 权威。

**口径**: 相对 speedup 用 msprof (`baseline_e2e_ns/e2e_ns`); 绝对 latency 用 Event (`e2e_event_ns`, 轨迹图/工业级对比都用 Event-Event 同口径)。

---

```
┌─ kernel 链 (★严格最优 采纳/回退) ──────────────────────────────────┐
│ round1: current_kernel = input/<op>/kernel_op.py (源, 永不改)       │
│ roundN: current_kernel = 历史最高加速比那轮 (best_kernel, 严格最优)  │
│  ★采纳 = 本轮 speedup > best_speedup (历史最高) — 优化必须超越最优   │
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
verify: 正确性校验(MATMUL_VERIFY) → msprof 跑 (warmup + KERNEL_LOOP 平均)
        + Event 注入计时 → e2e_event_ns
  speedup = baseline_e2e_ns / e2e_ns   (严格最优: > best_speedup 才 KEEP)
           │
           ▼
记录 hist (+ e2e_event_ns/sweep_ran/sweep_adopted) + 晋升 (planner.promote / 3轮无改进 / 达标 / tier6停)
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

---

## 4. 输出结构（每轮自包含）

```
outputs/<op>/<tier>/roundN/
  input/kernel_op.py      ← 本轮采集快照 (上一轮输出/源)
  kernel_op.py            ← ★本轮 coder 输出的优化 kernel
  diff.patch              ← 本轮改动
  plan.md                 ← planner 计划 (含 changes[])
  optimization_record     ← (hist 记入 trajectory)
  04_board/               ← msprof op 8 CSV 原始
  05_task/                ← 通用 msprof 原始
  06_diagnosis/           ← diagnosis.json (+task/board json)
  07_tier{N}_fields/      ← ★当前 tier 筛字段 (planner 读)
  08_fusion/              ← (仅 Tier2) HIVM MLIR + 融合分析
outputs/<op>/optimization_trajectory.json   ← ★全局状态 (tier/round/best/current_kernel/history)
outputs/<op>/final_output/trajectory_chart.png  ← 轨迹图
```

---

## 5. 6 层策略 × 每轮字段（审计定版 58 字段，2026-08-12 扩充）

| Tier | 策略 | 字段数 | 主要字段 | 晋升条件 |
|---|---|---|---|---|
| 1 | 算法结构 | 13 | cube_fops/vec_fops/cube_ratio/fp16/int8占比/cube fp指令/vec fp16占比/算力利用率/算术强度/瓶颈类型/total_ns/num_kernels | 算法已最优 |
| 2 | 算子融合 | 8(+08_fusion) | num_kernels/api_overhead/task_type/launch_count/multi_kernel/framework + HIVM 依赖分析(含每op耗时占比) | 无可融合 |
| 3 | 分块配置 | 8 | block_dim/mte1/mte2/cube_ratio/l0a读写/l0b读写 | 3轮无改进 |
| 4 | 访存 | 12 | main_mem读写/gm_to_ub/ub_to_gm/l2/mte2/3耗时/访存利用率 + **traffic_kb(实际搬运量)/bw_usage_rate(官方通路利用率)/traffic_redundancy(冗余倍数)** | 3轮无改进 |
| 5 | 计算占用 | 8 | cube耗时/标量耗时/scalar/fixpipe占比/cube_ratio/冲突 | 3轮无改进 |
| 6 | 架构专属 | 9 | engine_util/icache缺失率/cube_wait/vec_wait/mte2/3_wait/task_type/block_dim/瓶颈类型 | 3轮无改进→停 |

> 2026-08-12 扩充（官方文档核实，见 `docx/msprof_fields_reference.md` 四节）：Memory.csv 的 `*_datas(KB)` 实际搬运量、`*_bw_usage_rate(%)` 官方通路利用率、PipeUtilization 的 `*_active_bw(GB/s)` 活跃带宽、`ai*_icache_miss_rate`、ResourceConflictRatio 的 `ai*_wait_ratio` 规范短名（`vec_wait_ratio` 等，消除 `_get` 子串歧义）、ArithmeticUtilization 的 vec 精度细分与 cube fp/int 指令条数、OpBasicInfo 的 `Rated Freq`/`Mix Block Dim`；roofline 新增 `traffic_redundancy_read`（实际读÷理论最小，>1.5=重复搬运）。hivm fusion view 现附每 op 估算耗时占比（Tier2 排融合优先级）。

**分块调参逻辑**（用户关注点）：
- **传输瓶颈**(memory_bound, mem_util≥0.8 且 comp<0.5) → **增大 tile**（复用↑, GM流量↓）
- **计算瓶颈**(comp≥0.8 且 mem<0.5) → **不调小**（cube 已满, 看算法/精度如 fp16）
- **延迟瓶颈**(两者都<0.3) → 调小/增并行
- UB 约束：每 K 迭代 `(BLOCK_M+BLOCK_N)×BLOCK_K×dtype` ≤ 192KB

---

## 6. 关键技术细节

### 6.1 确定性改码（changes[]）
```json
{"strategy":"增大BLOCK_K","changes":[
  {"old_code":"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
   "new_code":"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
   "reason":"Tier3: mte1_ratio高","section":"① config"}]}
```
- Planner 必须输出 old_code 逐字符匹配当前 kernel
- Coder 精确替换**全部出现处**（old_code 须为整行）；找不到 → 报告（NOOP 标记）；绝不猜测
- 只走 LLM 的情形：有 previous_error（验证报错）需修报错

### 6.2 nga run 调用（3 个 skill）
```bash
echo "你是 Triton 优化 Planner。先调用 skill: skills/triton-op-planner/SKILL.md, 完全按 skill 指导执行。
+ 07字段 + 读 playbook_tier{N}.md + 读 current_kernel + 输出 changes[]" | nga run
```
- `skills/triton-op-planner`：判瓶颈属不属本层 + 输出 changes[]
- `skills/triton-op-coder`：确定性替换 changes[] + 报错修
- `skills/triton-op-fusion`：Tier2 读 HIVM MLIR 分析 RAW/WAR/WAW
- 调用实现：`shlex.quote(full_prompt)` 构造 `echo '<prompt>' | nga run`，`["bash","-c",...]` 执行，**实时流式打印**输出

### 6.3 历史梗概（防重复优化）
每轮 hist 记：`change`(改了啥) + `speedup`(vs初始, 累计) + `prev_speedup`(上一被采纳版) + `decision`(KEEP/REVERT/FAIL) + `result`(OK/FAIL/NOOP) + `error`(短)。Planner 读最近 5 轮 1 行梗概（REVERT 轮标「↩回退」）。

### 6.4 超时设置
| env | 默认 | 作用 |
|---|---|---|
| `LLM_CLI_TIMEOUT` | 3600 | nga run 调用 (超时自动兜底不崩循环) |
| `OPTIMIZE_TIMEOUT` | 3600 | run_optimize 采集 |
| `VERIFY_WARMUP` | 3 | 验证热身裸跑次数 |
| `VERIFY_LOOP` | 30 | 一次 msprof 内 kernel 内部循环次数 (÷N 取单次) |

### 6.5 加速比
- **加速比 = 时间比**：`baseline_time / current_time`（msprof 端到端口径）
- **端到端耗时 = 目标 kernel (非 aclnn) Task Duration 之和**（多 kernel 如 MLP: fc1+bias_gelu+fc2 求和才是总耗时; 单 kernel 即本身; baseline 与 verify 同口径）
- baseline = round1 采集的原始 kernel 端到端; current = 本轮验证 msprof 端到端
- **★严格最优 KEEP**：本轮 `speedup > best_speedup`（历史最高）才进链；`best_speedup` 绑定 `best_kernel`/`best_round` + 复制 `best_kernel.py`
- **Event 绝对端到端**：每轮另测 `e2e_event_ns`（`torch.npu.Event` 设备侧），与 msprof 并列；工业级基准对比用它
- **vs_industrial_ratio**：我们最优 Event / 工业级最优 Event（<1=快于工业级）
- 图上 TFLOPS = 诊断 cube_fops 总和 / baseline_ns（多 matmul 正确, 非 2MNK 单算）
- vs PyTorch = 我们算力 / torch.matmul 算力

---

## 7. 文件介绍

### 入口
| 文件 | 作用 |
|---|---|
| `main.py` | v4 入口。`--fresh`(清旧产物重置) / `--resume`(续跑) / `--stub`(本地) / `--max-rounds` / `--target` |

### agents（智能体层）
| 文件 | 作用 |
|---|---|
| `scheduler.py` | 状态机：kernel 链（严格最优 KEEP + best_kernel 绑定 + 失败回滚）、tier 字段提取、hist 记录、NOOP 检测、晋升决策、设备污染检测+重置、vs_industrial 比值 |
| `planner.py` | `generate_v4`：读文件路径(不嵌入内容)，输出 changes[] + promote |
| `coder.py` | 确定性 changes[] 替换 + Unicode 清洗（含制表符）+ 保 header + failed_kernel.py 留证 |
| `verifier.py` | `verify_end_to_end`：正确性校验 + msprof 双口径 + ★Event e2e_event_ns（注入 KERNEL_LOOP） |
| `llm_client.py` | `echo '<prompt>' | nga run` + 实时流式输出（API/CLI/stub 三模式，CLI 优先） |

### analyzers（采集解析层）
| 文件 | 作用 |
|---|---|
| `run_optimize.sh` | `<input_dir> <output_dir>` 采集+解析+07 产出；自包含拷贝；绝对路径 msprof |
| `pipeline_parse_task.py` | 通用 msprof → task.json（kernel_slots 去重 + 框架过滤 + pipe 归一化） |
| `pipeline_parse_board.py` | msprof op → board_<i>.json（8 CSV 全字段） |
| `integrate.py` | 按 kernel 名合并骨架+deep → diagnosis.json |
| `check_fields.py` | 缺字段精准指路（工具/文件/列名），区分「列名不匹配/合法缺」 |
| `sweep_blocks.py` | Tier3 分块扫描：程序化枚举 L0 合法 BLOCK + 真机实测 |
| `run_hivm_fusion.py` | Tier2：HIVM MLIR + nga 融合分析 |
| `merge_single_file.py` | 旧式三文件→单文件兜底 |

### skills（agent 教学）
`skills/triton-op-{planner,coder,fusion}/SKILL.md` — 每个 agent 的完整指令（读来源、输出格式、铁律）。

### docx（知识库）
`OPTIMIZATION_METHODOLOGY.md`（6层方法论）/ `msprof_fields_reference.md`（字段来源）/ `aggregation_rules.md`（聚合规则）/ `final_product_spec.md`（产物规格）/ `field_extraction_checklist.md`（缺字段核对）/ `playbook_tier1~6.md`（改码教程）。

### bench_910b3（硬件基准套件，★全 Event 设备侧计时，do_bench 同款测量）
`bench_industrial.py`（工业级基准 eager/compile/cann-fused/fa，多窗口 median + 输入轮换破 L2）/ `bench_all.py`（全算子取最优，仅真执行）/ `bench_pytorch*.py`（PyTorch 基准线，同款测量）/ `bench_common.py`（测量工具: msprof 双模式 + ★measure_event）/ `run_bench.py`（硬件峰值校准）/ `bench_kernels.py`（6个测速 kernel）。

### 其他
`feedback/trajectory_chart.py`（v4 轨迹图）/ `feedback/strategy_summary.py`（每轮策略摘要，final_output/{all,successful}_strategies.md）/ `input/matmul/kernel_op.py`（单文件源）。

---

## 8. 运行命令（服务器）

```bash
# 环境 (一次性)
conda activate triton-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# ① 完整优化循环 (推荐; --target 省略/0 = 不设目标跑满 --max-rounds, 看最优)
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15 --target 2.0
LLM_CLI_COMMAND='nga run' python3 main.py input/matmul --fresh --max-rounds 15   # 无目标, 跑满15轮
LLM_CLI_COMMAND='nga run' python3 main.py input/attention_mlp --fresh --max-rounds 15   # 复杂算子: 自注意力+MLP (5 kernel)

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

## 9. 910B3 硬件参数

**✅ 准确**：20 AI + 40 Vec cores @1.8GHz；UB=192KB、L1=512KB、L0A/L0B=64KB、L0C=128KB、L2=192MB、HBM=64GB；GM 峰值 **1638.4 GB/s**（HBM2e 4×409.6，联网核实 2026-08）；cube 峰值 **294.9** TFLOPS(fp16 标称) / **313**（官方）/ fp32 **73.7**。理论峰值由 `bench_910b3/bench_theory.py` 公式推导。

**微路径带宽**（GM→UB / UB→GM / L0A/L0B feed / MTE 占比）：由 `bench_910b3/run_bench.py` 的 msprof op per-path 实测填 `hardware_peak.json`（v4 已弃用 timing_estimator 静态成本模型）。

---

## 10. 开发记录

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
  - ★假小 Event 防毒（用户报告 "加速比突然 200x → 真实优化轮永不 KEEP"）: coder 改坏代码 → Event 窗口未跑满 → 假小值毒 best → 三层护栏 (KEEP 决策拦 / best 只在 kept 时更新 / rebaseline 复测也拦, EVENT_MIN_RATIO 默认 10 可调) + 假小提示进 hist error; 回归 P16

---

## 10. 后续开发计划

- ✅ **前置 BLOCK 扫描（已实现 `sweep_blocks.py` + `main.py --sweep-blocks`）**：
  分块层(BLOCK_M/N/K)是确定性小空间(16 倍数×UB/L0 约束)，用穷举替代笨 LLM 调参更稳。
  优化前先扫 L0 合法候选，msprof 取最快写回 config 区（固定"乘性地基"，防分块差几十倍）。
  ⚠ 每个候选=一次真机 msprof（分钟级），8 候选约半小时+，一次性成本。
- **A1 修复**：attention_mlp 的 `matmul_kernel` 被 5 种形状复用 → msprof 同名聚合 deep 画像混合，
  应拆独立函数名（对齐 matmul 算子 matmul_kernel2 的纪律）；D4 已在 coder 加同名多匹配警告。
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
