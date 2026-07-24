# 910B3 部署交接文档 — AI Agent 执行手册

> 目标: 将本框架部署到华为 Ascend 910B3 服务器，补全所有待对齐参数，
> 确保全链路可运行，输出真实优化结果。
>
> **本文件是唯一入口。** 每个文件夹内还有详细的 `DEPLOYMENT_GUIDE_910B3.md`。

---

## 0. 强制规则 (必须遵守)

| # | 规则 | 说明 |
|---|---|---|
| 1 | **禁止使用模拟数据** | 所有 msprof/HIVMIR 数据必须来自真实编译器/硬件采集，不得用 `cost_emulator/simulator.py` 的 DSL 输出来冒充 |
| 2 | **禁止修改整体流程** | 分析链 (msprof→hivmir→merger→diagnoser→extractor) 和 Agent 链 (Plan→Code→Verify→Record) 顺序不能改 |
| 3 | **禁止修改 29 字段输出结构** | `pipeline_report.json` 和 `hivmir_report.json` 必须保持当前 JSON schema，字段名不能变 |
| 4 | **禁止修改 Tier 顺序** | 1-Algorithm→2-Fusion→3-Tiling→4-Memory→5-Compute→6-Architecture，不能跳 |
| 5 | **只能修改字段提取规则** | 当 910B3 真实数据格式与预期不同时，可以修改 `analyzers/` 中的正则/解析规则，但输出 schema 不变 |
| 6 | **必须通过完整链路测试** | 见 §3 |
| 7 | **所有路径配置在 config.py** | 不要在各文件中硬编码路径 |

---

## 1. 项目总览

### 1.1 这个框架做什么

```
输入: Triton kernel (.py) + 场景信息
  → 6 层优化 (Algorithm→Fusion→Tiling→Memory→Compute→Arch)
  → 每层多轮 Plan→Code→Verify→Record
  → 输出: 优化后 kernel + 轨迹图 + 加速比报告
```

### 1.2 核心文件架构

```
triton_agent_optimizer/
├── main.py                       # ★ 入口: python main.py kernel.py
├── config.py                     # 全局配置 (路径/硬件参数)
│
├── analyzers/                    # 分析层 (5文件, 全部完成)
│   ├── DEPLOYMENT_GUIDE_910B3.md # ← 分析层部署指南
│   ├── msprof_analyzer.py        #   ① trace.json → timing/engine
│   ├── hivmir_analyzer.py        #   ② .mlir → buffer/size/deps
│   ├── dsl_merger.py             #   ③ 合并 → 29字段报告
│   ├── bottleneck_diagnoser.py   #   ④ Tier-aware 瓶颈诊断
│   └── data_extractor.py         #   ⑤ 按需数据提取
│
├── agents/                       # 智能体层 (4文件, 全部完成)
│   ├── DEPLOYMENT_GUIDE_910B3.md # ← 智能体层部署指南
│   ├── orchestrator.py           #   调度器 (Python 状态机)
│   ├── planner.py                #   LLM Planner
│   ├── coder.py                  #   LLM Coder
│   └── verifier.py               #   验证器 (CPU仿真 + 910B3)
│
├── execution/                    # 执行层
│   ├── DEPLOYMENT_GUIDE_910B3.md # ← 执行层部署指南
│   ├── emulator_runner.py        #   Stage1: CPU 仿真验证
│   ├── compiler.py               #   编译 + HIVMIR 提取
│   └── hardware_runner.py        #   Stage2: 910B3 benchmark
│
├── feedback/                     # 反馈层
│   ├── DEPLOYMENT_GUIDE_910B3.md
│   ├── record_manager.py         #   决策引擎
│   └── trajectory_chart.py       #   优化轨迹图
│
├── memory/                       # 记忆层
│   ├── DEPLOYMENT_GUIDE_910B3.md
│   ├── experience_retriever.py   #   经验检索+记录
│   ├── context_manager.py        #   Token管理
│   └── sliding_window.py         #   滑动窗口
│
├── docx/                         # Playbook 知识库 (7文件)
│   └── playbook_tier1~6_*.md
│
├── prepare/                      # 环境准备
│   ├── setup_env.sh/.bat
│   └── env_check.py
│
├── input/                        # 输入案例
│   └── rms_norm_residual/        #   RMSNorm 真实案例
│
└── outputs/                      # 优化产物 (自动生成)
```

---

## 2. 执行步骤 (按顺序)

### Step 1: 阅读 (30min)

阅读顺序:
1. `../../claude_resume_summary/resume_2026-07-24.md` — 项目背景 + 概念澄清
2. `README.md` — 框架总览
3. `ARCHITECTURE_DESIGN.md` — 完整架构图 + 数据流
4. 各文件夹的 `DEPLOYMENT_GUIDE_910B3.md`

### Step 2: 环境准备

```bash
source prepare/setup_env.sh
python prepare/env_check.py    # 35 项检查, 必须全 PASS
```

### Step 3: 补全分析层 (最优先)

按 `analyzers/DEPLOYMENT_GUIDE_910B3.md` 操作:

| 子步骤 | 文件 | 做什么 |
|---|---|---|
| 3a | `msprof_analyzer.py` | 编译 RMSNorm kernel → `msprof op simulator` → 采集 trace.json → 验证通道名与 `PIPELINE_MAP` 一致 |
| 3b | `hivmir_analyzer.py` | 编译 kernel 带 `--mlir-print-ir-after-all` → 提取 HIVMIR .mlir → 验证格式与解析器兼容 |
| 3c | `dsl_merger.py` | **SATURATION_PARAMS Engine 3/4/5/6 实测** — 跑 matrix pipeline benchmark 获取 GM→L1/L1→L0/CubeUnit/L0→GM 真实带宽 |
| 3d | `bottleneck_diagnoser.py` | 用真实 merged_report.json 验证诊断结果合理 |
| 3e | `data_extractor.py` | 不需要修改 |

### Step 4: 补全执行层

按 `execution/DEPLOYMENT_GUIDE_910B3.md`:

| 子步骤 | 文件 | 做什么 |
|---|---|---|
| 4a | `compiler.py` | 确认编译器路径，跑通编译+HIVMIR提取 |
| 4b | `hardware_runner.py` | 实现真实 benchmark (warmup=30, repeat=200) |
| 4c | `emulator_runner.py` | 在 910B3 上验证 Stage 1 通过 |

### Step 5: 补全智能体层

按 `agents/DEPLOYMENT_GUIDE_910B3.md`:

| 子步骤 | 做什么 |
|---|---|
| 5a | 设置 `ANTHROPIC_API_KEY` |
| 5b | 验证 Planner stub → 切换到真实 LLM 模式 |
| 5c | 验证 Coder stub → 切换到真实 LLM 模式 |

### Step 6: 全链路测试

```bash
# 必须跑通这条命令, 从输入到最终输出
python main.py input/rms_norm_residual/triton_kernel.py --max-rounds 5

# 验证输出
ls outputs/rms_norm_residual/final_output/
# 必须包含: optimized_kernel.py, trajectory_chart.png, optimization_summary.md
```

**验收标准**:
- Round 0 baseline 数据来自真实 msprof + HIVMIR (不是 fallback)
- 至少跑完 2 个 Tier
- trajectory_chart.png 正确生成
- optimization_trajectory.json 每轮都有记录

---

## 3. 不懂的时候去哪里搜

| 问题 | 搜索关键词 | 来源 |
|---|---|---|
| msprof 命令用法 | `msprof op simulator Ascend910B3 --kernel-name` | [昇腾社区](https://www.hiascend.com/document) |
| HIVMIR MLIR 格式 | `hivm.gm_to_ub hivm.vadd memref AscendNPU-IR` | [AscendNPU-IR](https://gitee.com/ascend/triton-ascend) |
| SATURATION_PARAMS 实测 | 跑 `perf_test/910B3/vecadd/bench_910b3_paths.py` 拿到 bench_result.csv | 本仓库已有 |
| CANN 环境配置 | `source set_env.sh Ascend 910B3 CANN` | CANN 官方文档 |
| Triton Ascend backend | `triton-ascend kernel example 910B3` | [Triton-Ascend GitHub](https://gitee.com/ascend/triton-ascend) |
| 架构理解 | 读 `ARCHITECTURE_DESIGN.md` 的数据流图和 §5 每轮流程 | 本仓库 |

---

## 4. 关键输出文件清单

| 文件 | 路径 | 要求 |
|---|---|---|
| `pipeline_report.json` | `roundN/msprof/` | 29 字段，msprof 字段已填真实值 |
| `hivmir_report.json` | `roundN/hivmir/` | 29 字段，HIVMIR 字段已填真实值 |
| `merged_report.json` | `roundN/merged/` | 29 字段全填，bandwidth/regime 已计算 |
| `optimization_trajectory.json` | `outputs/<kernel>/` | 每轮一条记录，state 正确更新 |
| `trajectory_chart.png` | `final_output/` | 6 阶段加速比曲线 |
| `optimized_kernel.py` | `final_output/` | 通过全部验证的最终 kernel |

---

## 5. 可以改的 vs 不能改的

| 可以改 | 不能改 |
|---|---|
| `analyzers/` 中的正则/字段提取逻辑 | `analyzers/` 的输出 JSON schema |
| `config.py` 中的路径和阈值 | 29 字段的字段名和结构 |
| `compiler.py` 中的编译器路径 | 分析链 5 步的顺序 |
| `execution/` 中补充 benchmark 实现 | Tier 1→6 的顺序和晋升规则 |
| LLM model 名称 | 每轮 Plan→Code→Verify→Record 的流程 |
| Playbook 文档内容 (补充 910B3 特定知识) | Verifier 两阶段验证结构 |

---

*将本文件 + 项目代码交给接手的大模型，它应该能独立完成 910B3 部署。*
