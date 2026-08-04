# Triton Agent Optimizer — 完整架构设计 (v4.0)

> **核心差异化优势**: 不靠盲试，用 **真机 msprof + msprof op** 精确诊断瓶颈（真实带宽/L2/算力/冲突），6 层策略按「结构影响从大到小」逐轮优化，每轮只看**当前策略需要的字段**，Planner 定方向、Coder 精准改码、**只用 msprof 端到端验证加速比**。
>
> **数据源 = 真机双源**: ① 通用 `msprof`（骨架：kernel 数/耗时/形状/launch/L2）+ ② 逐 kernel `msprof op`（深层：真实带宽/引擎利用率/算力/冲突）→ `integrate.py` 按 kernel 名合并 → `diagnosis.json`（roofline 核心）。
>
> **环境**: 910B3 真机（保密服务器，只能 paste-in）+ CANN 8.5.1 / triton-ascend 3.2.0 / torch_npu 2.9.0。
> **更新**: 2026-08-04 — v4 架构（单文件驱动 + 真机双源 + 6 层轮次化）。

---

## 0. 核心理念

| 原则 | 说明 |
|---|---|
| **真机数据优先** | 只用 msprof + msprof op（已弃 hivm/simulator 主流程，按需保留 fusion） |
| **单文件驱动** | 算子 + 场景 + 测试**合并成一个文件**，只运行/只读写这一个文件，杜绝多文件错位 |
| **每轮只看该策略的字段** | 6 层策略各有自己的数据段，Planner 只喂当前层要的字段 |
| **验证只跑 msprof** | 每轮改完只跑一次 msprof 端到端算加速比；不提取字段、不跑 msprof op |
| **失败就地修** | Coder 改完跑不了 → 报错直接回传 Coder 改，不新开一轮 |
| **Planner 定晋升** | 读完当前瓶颈判断是否属于本策略 → 决定停留/晋升下一策略 |

---

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               INPUT LAYER                                     │
│   input/<op>/  ──合并──►  单文件 kernel_op.py (算子 + 场景 config + 测试)      │
│   (算子/场景/测试)            ★ 后续输入输出只读写这一个文件                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │   main.py 启动
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION + ANALYSIS LAYER                           │
│   run_optimize.sh <input_dir> <output_dir>   ★支持输入/输出参数(调度器重定向) │
│      │  ① 通用 msprof ──► 骨架 task.json (kernel 数/耗时/形状/launch/L2)     │
│      │  ② 逐 kernel msprof op ──► board_<i>.json (带宽/引擎/算力/冲突)      │
│      │  ③ integrate 按 kernel 名合并 ──► diagnosis.json (roofline 核心)     │
│      ▼                                                                         │
│   outputs/<op>/<tier>/round<N>/  ← 每轮产物 (diagnosis/task/board json)      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SCHEDULER (状态机)                                 │
│   ① 读 diagnosis.json → summary.num_kernels (几个算子)                        │
│   ② 当前策略 tier (算法→融合→分块→访存→计算→架构)                             │
│   ③ 提取本 tier 要看的字段段 → 传给 Planner                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               AGENT LAYER                                     │
│   Planner (LLM): 提取字段 + 优化策略文档 + 当前单文件 + config                  │
│       → 详细计划 plan.md (哪一行/改什么/改成什么) + 晋升决策                    │
│            │                                                                   │
│            ▼                                                                   │
│   Coder (LLM): plan + 代码教程 + 纠错/改码原则文档 + 单文件                     │
│       → 修改单文件 kernel_op.py (+ diff.patch)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             VERIFY (脚本)                                     │
│   只跑一次 msprof 端到端 (整文件) → 端到端耗时 → 加速比                        │
│       ├─ 跑不了 → 报错回传 Coder 就地改 (不新开一轮)                          │
│       └─ 跑通   → 记录 + Planner 判定晋升 → 下一轮/下一 tier                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 输入（input/<op>/ → 单文件合并）

```
input/<op>/
├── triton_kernel.py      # 算子 (triton kernel)
├── config.json           # 场景信息 (M/N/K/dtype/block/优化目标)
└── test_*.py             # 测试驱动 (真机命令注释)

── 启动时合并成一个文件 ──►  kernel_op.py
   (算子 + config 常量 + 测试 main 合在一起)
   ★ 后续所有输入/输出只读写 kernel_op.py 这一个文件
```

- 合并的好处：调度器每轮只需替换/读取**一个文件**，避免多文件版本错位、import 路径错、参数不一致。
- 合并由 `main.py` 启动时做（复制算子 + 注入 config + 追加测试 main）。

---

## 3. 主循环（每轮做什么）

```
Round N (tier T):
  ├─ ① 采集+解析: run_optimize.sh <input_dir> <round_dir>
  │     通用 msprof + 逐 kernel msprof op → diagnosis.json (+task/board)
  │
  ├─ ② 调度器: 读 diagnosis.json
  │     看 summary.num_kernels → 看当前 tier → 提取该 tier 字段段
  │
  ├─ ③ Planner (LLM): 提取字段 + 策略文档 + 单文件 + config → plan.md
  │     含: 当前瓶颈是否属于本 tier? 属于→给改法; 不属于→决定晋升
  │
  ├─ ④ Coder (LLM): plan + 教程 + 纠错文档 → 修改 kernel_op.py
  │
  ├─ ⑤ 验证: 只跑一次 msprof 端到端 → 端到端耗时 → 加速比
  │     ├─ FAIL → 报错回传 Coder (同轮重改, ≤3次)
  │     └─ PASS → 记录加速比
  │
  └─ ⑥ 判定: 加速比达标? → 停; 本 tier 连续N轮无改进? → 晋升下一 tier; 否则下一轮

Round 0 (Baseline): 只采集+解析, 记基准加速比=1.0
```

---

## 4. run_optimize.sh — 采集 + 解析（支持输入输出参数）

```bash
# 用法 (支持输入/输出重定向, 调度器每轮指向不同 round_dir)
bash run_optimize.sh <input_dir> <output_dir>

# 示例:
bash analyzers/run_optimize.sh input/matmul outputs/matmul/01_algorithmic_structure/round1
```

**做什么（按顺序）：**
1. 通用 msprof 跑一遍 → 骨架 `task.json`（kernel 数/耗时/形状/引擎/launch/L2）
2. 从 task.json 取 distinct kernel 名（跳过 aclnn* 框架 kernel）→ 逐 kernel 跑 msprof op → `board_<i>.json`
3. `integrate.py` 按 **Op Name** 合并骨架+深层 → `diagnosis.json`

**产物 → 输出到 `<output_dir>/`：**
| 文件 | 内容 |
|---|---|
| `diagnosis.json` | ★最终产物（summary + kernels[].task/deep + roofline） |
| `task.json` | 骨架（通用 msprof 全字段） |
| `board_<i>.json` | 每 kernel 深层（msprof op 全字段） |

> 调度器只需换 `<input_dir>`（下一轮用哪个单文件）和 `<output_dir>`（写到哪个 round），即可驱动整个优化循环。

---

## 5. 输出结构

```
outputs/<op>/
├── 01_algorithmic_structure/round1..N/   # Tier 1 每轮产物
│   ├── diagnosis.json      ← ★每轮诊断 (roofline + kernels)
│   ├── task.json           ← 骨架
│   ├── board_<i>.json      ← 每 kernel 深层
│   ├── kernel_op.py        ← 当前单文件 (Coder 改的)
│   ├── plan.md             ← Planner 计划
│   ├── diff.patch          ← Coder 改动
│   └── optimization_record.json  ← 本轮记录 (加速比/决策)
├── 02_operator_fusion/round1..N/
├── 03_tiling_block_config/round1..N/
├── 04_memory_access/round1..N/
├── 05_compute_occupancy/round1..N/
├── 06_910b3_architecture/round1..N/
├── optimization_trajectory.json         # ★全局中枢状态 (tier/round/speedup)
└── final_output/                        # 最终: optimized kernel_op.py + summary
```

**每轮输入/执行/输出清单：**

| 环节 | 输入 | 执行文件 | 输出 | 输出到 |
|---|---|---|---|---|
| 采集解析 | `<input_dir>/kernel_op.py` | `run_optimize.sh` | diagnosis/task/board json | `<output_dir>/` |
| 提取字段 | `diagnosis.json` | scheduler | 该 tier 字段段文本 | 内存→Planner |
| Planner | 字段段 + 策略文档 + 单文件 + config | planner (LLM) | `plan.md` | `<output_dir>/` |
| Coder | `plan.md` + 教程 + 纠错文档 + 单文件 | coder (LLM) | 新 `kernel_op.py` + `diff.patch` | `<output_dir>/` |
| 验证 | 新 `kernel_op.py` | `msprof` (端到端) | 端到端耗时 → 加速比 | `optimization_record.json` |
| 判定 | 加速比 + 瓶颈 | planner/scheduler | 下一轮 or 晋升 | `optimization_trajectory.json` |

**下一个环节的输入 = 上一个环节的输出**（闭环：诊断→计划→改码→验证→下一轮诊断）。

---

## 6. 六层策略 × 每轮看哪些字段

> 字段来源详见 `docx/msprof_fields_reference.md` 第四节；完整列名见第五节。

| Tier | 策略 | 每轮提取的字段（只看这些） | 晋升条件 |
|---|---|---|---|
| **1** | 算法结构 | `deep.compute.cube_fops/vec_fops`、`deep.engine.cube_ratio/vec_ratio`、`deep.roofline.compute_utilization` | 算法已最优 |
| **2** | 算子融合 | `summary.num_kernels`、`kernels[].task.task_type`、`api_overhead`、`multi_kernel`、`framework_kernels` | 无可融合 kernel |
| **3** | 分块配置 | `task.block_dim`、`deep.engine.mte1_ratio`、`deep.bandwidth.l0a/l0b` | 连续3轮无改进 |
| **4** | 访存 | `deep.bandwidth.main_mem_read/write`、`deep.l2_hit_rate`、`task.pipes.mte2/mte3` | 连续3轮无改进 |
| **5** | 计算占用 | `deep.compute.cube_ratio`、`task.pipes.cube_time/scalar_time`、`deep.conflict.*` | 连续3轮无改进 |
| **6** | 架构专属 | `deep.engine.*`、`deep.conflict.wait_ratio`、`task.task_type` | 连续3轮无改进→停 |

**提取原则**：每轮只把**当前 tier 的字段**喂给 Planner，其他字段不注入（省 token + 聚焦 + 防弱 LLM 被无关字段误导）。

---

## 7. Planner 职责

**输入**：① 当前 tier 提取的字段段（上节）② 优化策略文档（`docx/OPTIMIZATION_METHODOLOGY.md` + `playbook_tierN_*.md`）③ 当前 `kernel_op.py` ④ `config.json`

**输出**：`plan.md`，必须**详细完整**到 Coder 能直接改：
- 改哪个函数、哪一行
- 把什么改成什么（具体参数/表达式）
- 为什么（依据哪个字段/哪个策略）
- 改完预期效果（哪个指标该变多少）

**晋升决策**：读完当前瓶颈 → 判断**是否属于本 tier 的优化范畴**：
- 属于 → 给出具体改法
- 不属于（瓶颈在别层）→ 决定晋升到下一 tier，说明理由

---

## 8. Coder 职责

**输入**：① `plan.md` ② 代码教程（改码规范）③ 纠错/改码原则文档（应对各种报错）④ 当前 `kernel_op.py`

**只做一件事**：按 plan 修改 `kernel_op.py`，不碰任何其他文件。

**出错处理**：改完跑 msprof 若报错 → 报错信息直接回传 Coder → **同轮**根据报错就地改（最多 3 次），不新开一轮。

---

## 9. 验证（只跑 msprof 端到端）

```bash
# 每轮改完后, 只跑这一下 (整文件, 端到端耗时)
msprof --output=<round_dir>/msprof --application="python3 kernel_op.py" --ai-core=on
# 从 op_summary 读 Task Duration(us) → 端到端耗时 → 加速比 = baseline / 本轮
```

- **不提取字段、不跑 msprof op**（那些只在每轮开始跑一次用于诊断）
- 加速比 = baseline 端到端耗时 / 本轮端到端耗时
- 跑不了 → 报错回传 Coder 就地改

---

## 10. 晋升 / 降级 / 停止

- **晋升**：本 tier 连续 N 轮（默认 3）无改进（加速比变化 <5%）→ 下一 tier
- **降级**：结构性变更后回退——改了算法→回 Tier2；融合了新算子→回 Tier3；改了 pipeline→回 Tier3
- **停止**：加速比达标 / 到 Tier6 且连续 3 轮无改进 / 超出最大轮数

---

## 11. 文件架构总览（v4 目标）

```
triton_agent_optimizer/
├── README.md                    # 本文档
├── main.py                      # ★入口: python main.py input/matmul [--round-dir ...]
├── config.py                    # 全局配置 (路径/硬件/阈值)
│
├── input/<op>/                  # 每算子输入: triton_kernel.py + config.json + test_*.py
├── outputs/<op>/<tier>/roundN/  # 每轮产物 (diagnosis/task/board + kernel_op.py + plan/diff/record)
│
├── analyzers/                   # 采集+解析层
│   ├── run_optimize.sh          # ★采集+解析主脚本 (支持输入/输出参数)
│   ├── pipeline_parse_task.py   #   通用 msprof → 骨架 task.json (kernel_slots + framework 过滤)
│   ├── pipeline_parse_board.py  #   msprof op → 每 kernel board_<i>.json (8 CSV 全字段)
│   ├── integrate.py             #   按 kernel 名合并 → diagnosis.json (roofline)
│   ├── check_fields.py          #   缺字段核对 (去哪个工具/文件/列名)
│   └── real_report.py           #   诊断报告渲染
│
├── agents/                      # 智能体层
│   ├── scheduler.py             #   状态机: 读 diagnosis → 提取字段 → 驱动循环
│   ├── planner.py               #   LLM: 字段+策略+单文件 → plan.md + 晋升决策
│   ├── coder.py                 #   LLM: plan → 改 kernel_op.py (+纠错循环)
│   └── llm_client.py            #   LLM API
│
├── docx/                        # 知识库
│   ├── OPTIMIZATION_METHODOLOGY.md   # 6 层策略方法论
│   ├── msprof_fields_reference.md    # 所有字段含义/列名
│   ├── field_extraction_checklist.md # 缺字段去哪核对
│   ├── aggregation_rules.md          # JSON 聚合规则
│   ├── final_product_spec.md         # 最终产物规格
│   └── playbook_tier1~6_*.md         # 每层改码教程
│
└── feedback/record_manager.py   # 加速比记录 + 晋升/停止
```

---

## 12. 关键文件读写映射

| 文件 | 写入者 | 读取者 |
|---|---|---|
| `kernel_op.py`（单文件） | Coder | run_optimize.sh / msprof / Planner / Coder |
| `diagnosis.json` | integrate.py | Scheduler → Planner |
| `task.json` / `board_<i>.json` | parse 脚本 | check_fields / 人工 |
| `plan.md` | Planner | Coder |
| `diff.patch` | Coder | 人工 review |
| `optimization_record.json` | RecordManager | 判定 |
| `optimization_trajectory.json` | RecordManager | Scheduler |

---

## 13. 与 v3 的差异（v3 → v4）

| 项 | v3（旧） | v4（本版） |
|---|---|---|
| 数据源 | 真实 HIVM + simulator + 真机 msprof op 三源 | **只 msprof + msprof op 双源**（hivm/sim 弃用主流程） |
| 文件 | kernel.py / test.py / config.json 分开 | **合并成单文件 kernel_op.py** |
| 每轮分析 | hivmir→msprof(sim)→dsl_merger→29字段 | run_optimize.sh → diagnosis.json（roofline） |
| 每轮验证 | CPU Emulator + 910B3 实测 | **只跑 msprof 端到端算加速比** |
| Planner 输入 | 全部 29 字段 | **只喂当前 tier 的字段段** |
| 失败处理 | 重开验证循环 | **报错就地回传 Coder 改** |
| 输出 | roundN 混放 | **outputs/<op>/<tier>/roundN/ 每轮含 diagnosis json** |

> **实现状态**：`run_optimize.sh` + 三个 parser + integrate + check_fields 已按 v4 落地（合成数据验证通过）；`main.py`/agents 仍为 v3 结构，需按本文档改造（单文件合并、scheduler 字段提取、验证改 msprof 端到端）。

---

## 14. 参考资料

| 内容 | 链接 |
|---|---|
| msprof op 8 CSV 字段 | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/devaids/optool/atlasopdev_16_00851.html |
| op_summary 字段 | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0008.html |
| triton-ascend profiling | https://github.com/triton-lang/triton-ascend/blob/main/docs/en/debug_guide/profiling.md |
| msopprof 模式性能数据 | https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/user_guide/msopprof_simulator_user_guide.md |
