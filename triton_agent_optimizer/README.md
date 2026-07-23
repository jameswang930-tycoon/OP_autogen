# Triton Agent Optimizer

> 基于 DSL 流水线分析 (HIVMIR + msprof Simulator) 的 Triton Kernel 自动优化框架。
> 目标硬件: **华为 Ascend 910B3 NPU**。

## 快速开始

```bash
# 1. 检查环境配置
cd triton_agent_optimizer
python config.py

# 2. (未来) 运行优化
python main.py --kernel your_kernel.py --target-speedup 1.5
```

## 目录结构

```
triton_agent_optimizer/
│
├── config.py                          # ★ 全局配置中心 (已完成)
├── main.py                            # 主入口 (待实现)
├── ARCHITECTURE_DESIGN.md             # ★ 完整架构设计文档
├── IMPLEMENTATION_PLAN.md             # ★ 逐文件实现计划 (36 个文件)
├── README.md                          # 本文档
│
├── agents/                            # 智能体层 (待实现)
│   ├── orchestrator.py                #   调度器——协调 Plan→Code→Verify→Decide→Record
│   ├── planner.py                     #   规划智能体——分析瓶颈 + 生成本轮计划
│   ├── coder.py                       #   编码智能体——按计划做最小化代码改动
│   └── verifier.py                    #   验证智能体——三阶段验证 (CPU→Sim→HW)
│
├── analyzers/                         # 分析层 (待实现)
│   ├── msprof_analyzer.py             #   Simulator 包装器: --llm (AI消费) + Gantt (人读)
│   ├── hivmir_analyzer.py             #   HIVMIR 解析: 变量名/依赖/大小/内存区域
│   ├── dsl_merger.py                  #   DSL 数据合并: simulator 输出 + HIVMIR 输出
│   ├── data_extractor.py              #   按需数据提取: 全量存文件, ~10行入 prompt
│   └── bottleneck_diagnoser.py        #   瓶颈诊断: 分类 + 可优化空间评估
│
├── optimizers/                        # 优化器 (待实现, LLM prompt 模板 + 验证逻辑)
│   ├── base_optimizer.py              #   基类
│   ├── tile_optimizer.py              #   Tier 1: BLOCK_SIZE / num_warps / num_stages
│   ├── memory_optimizer.py            #   Tier 2: 小传输合并 / coalescing / double buffer
│   ├── fusion_optimizer.py            #   Tier 3: WAR打破 / 逐元素融合
│   └── compute_optimizer.py           #   Tier 4: pipeline overlap / FP32→FP16
│
├── execution/                         # 执行层 (待实现)
│   ├── emulator_runner.py             #   Stage 1: CPU Emulator 正确性验证 (秒级)
│   ├── simulator_runner.py            #   Stage 2: Cost Simulator 性能预估 (秒级)
│   ├── compiler.py                    #   Ascend 编译器接口 + HIVMIR 提取
│   └── hardware_runner.py             #   Stage 3: 910B3 真机运行 (分钟级)
│
├── feedback/                          # 反馈与记录层 (待实现)
│   ├── round_logger.py                #   每轮记录: plan + diff + 验证 + 决策
│   ├── optimization_journal.py        #   优化日志管理: JSONL 追加/查询/导出
│   ├── stop_condition.py              #   7 条停止条件检查
│   ├── trajectory_chart.py            #   优化轨迹图: 双面板 speedup + latency
│   └── case_template.py              #   优秀案例生成: Markdown 模板填充
│
├── memory/                            # 上下文记忆层 (待实现)
│   ├── context_manager.py             #   上下文构建: prompt 组装 + token 估算 + 裁剪
│   ├── experience_retriever.py        #   经验检索: 对接项目 memory/ 模块
│   └── sliding_window.py              #   滑动窗口: 热(5轮完整)/温(15轮摘要)/冷(数据点)
│
├── playbooks/                         # 优化指导手册 (待编写, ~20章)
│   ├── optimization_playbook.md       #   总纲: 6层策略 + 晋升规则
│   ├── playbook_tiling.md             #   Tier 1: BLOCK_SIZE / UB容量约束
│   ├── playbook_memory.md             #   Tier 2: 各引擎带宽参数速查
│   ├── playbook_fusion.md             #   Tier 3: WAR打破 + 融合识别
│   ├── playbook_compute.md            #   Tier 4: VecUnit/CubeUnit 调优
│   ├── playbook_910b3_arch.md         #   Tier 5: 核数/grid/内存层级/pipeline选择
│   └── playbook_algorithmic.md        #   Tier 6: Online Softmax/Persistent/Split-K
│
├── cases/                             # 优秀案例库
│   └── template.md                    #   案例模板 (待填充)
│
├── example_output/                    # Simulator 示例输出 (7 个参考文件)
│   ├── README.md                      #   索引说明
│   ├── 01_vector_add_saturated.txt    #   --llm: 3 ops, 全饱和, 3655ns
│   ├── 02_for_loop_small_tile.txt     #   --llm: 20 ops, 全 floor, 769ns
│   ├── 03_single_load_1KB_floor.txt   #   --llm: 1KB floor 极端案例
│   ├── 04_matrix_pipeline_parallel.txt #  --llm: 12 ops, 7对并行
│   ├── 05_full_gantt_vector_add.txt   #   Gantt: 人读流水图
│   ├── 06_full_gantt_for_loop.txt     #   Gantt: 含超大展开
│   └── 07_full_gantt_matrix_pipeline.txt # Gantt: 7引擎并行视图
│
├── output/                            # 优化产物 (自动创建)
│   ├── rounds/                        #   每轮: round_NNN_plan.md + round_NNN_diff.patch
│   └── optimization_journal.jsonl     #   优化日志
│
└── tests/                             # 测试
    ├── test_kernels/
    └── test_orchestrator.py
```

## 环境要求

### 本地开发 (Windows/Linux)
- **Python 3.10+** — simulator `--llm` 模式只需标准库
- **(可选) matplotlib** — 轨迹图生成
- **(可选) networkx** — simulator `--nx` 模式

### 910B3 服务器
- **CANN Toolkit 7.0+** — `/usr/local/Ascend/ascend-toolkit/latest`
- **msprof** — `${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof`
- **Ascend C Compiler** — `${ASCEND_TOOLKIT_HOME}/compiler/bin`
- **Python 3.10+** with `torch`, `torch_npu`, `triton`

### 标准 Ascend 安装路径

| 组件 | 路径 | 环境变量 |
|------|------|----------|
| CANN 根目录 | `/usr/local/Ascend` | `ASCEND_HOME` |
| Toolkit (商用版) | `/usr/local/Ascend/ascend-toolkit/latest` | `ASCEND_TOOLKIT_HOME` |
| Toolkit (社区版) | `/usr/local/Ascend/cann` | — |
| msprof | `${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof` | `PATH` |
| 编译器 | `${ASCEND_TOOLKIT_HOME}/compiler/bin` | `PATH` |
| ascend-dmi | `${ASCEND_TOOLKIT_HOME}/bin/ascend-dmi` | `PATH` |
| npu-smi | `${ASCEND_TOOLKIT_HOME}/bin/npu-smi` | `PATH` |
| 运行时库 | `${ASCEND_TOOLKIT_HOME}/lib64` | `LD_LIBRARY_PATH` |
| OPP 算子包 | `${ASCEND_OPP_PATH}` | `ASCEND_OPP_PATH` |
| set_env.sh | `/usr/local/Ascend/ascend-toolkit/set_env.sh` | — |

## 两种 Simulator 输出

| | `--llm` 模式 | Gantt 模式 (默认) |
|---|---|---|
| **消费者** | AI (Planner/Coder Agent) | 人 (开发者 debug) |
| **频率** | 每轮优化必跑 | 按需生成 (最终报告/debug时) |
| **内容** | 7-section 结构化文本 | ASCII Gantt图 + 柱状图 |
| **体积** | ~2-8 KB | ~10-50 KB |
| **调用** | `simulator.py --llm --critical-path "..."` | `simulator.py --critical-path "..."` |
| **Windows** | 正常 (`gbk` 兼容) | 需 `PYTHONIOENCODING=utf-8` |

## 配置说明

所有配置集中在 `config.py`，按 dataclass 分为 6 组:

| 类 | 职责 | 可通过环境变量覆盖 |
|---|---|---|
| `PlatformPaths` | 所有文件路径 (本地+服务器), 自动环境检测 | ❌ |
| `HardwareParams` | 从 simulator.py 动态加载硬件参数 | ❌ (单一数据源) |
| `OptimizationParams` | 迭代控制、策略晋升阈值 | ✅ `TRITON_AGENT_*` |
| `VerificationParams` | 三阶段验证参数 | ✅ `TRITON_AGENT_HW_*` |
| `ContextParams` | 上下文窗口管理 | ✅ `TRITON_AGENT_HOT_WINDOW` 等 |
| `OutputParams` | 日志/输出控制 | ✅ `TRITON_AGENT_LOG_LEVEL` 等 |

环境变量覆盖示例:
```bash
export TRITON_AGENT_MAX_ROUNDS=300
export TRITON_AGENT_TARGET_SPEEDUP=2.0
export TRITON_AGENT_LOG_LEVEL=DEBUG
python main.py --kernel test.py
```

## 核心设计

详见 `ARCHITECTURE_DESIGN.md` 和 `IMPLEMENTATION_PLAN.md`。

- **验证三阶段**: CPU Emulator (秒, 正确性) → Cost Simulator (秒, 性能预估) → 910B3 HW (分钟, 实测)
- **6 层优化策略**: 从粗到细, 连续3轮无改进自动晋升
- **上下文管理**: 热(最近5轮完整)/温(6-15轮摘要)/冷(16+轮数据点)
- **输出格式**: 对齐 simulator `--llm` 7-section 格式, 与现有 `/triton-plan` 流水线兼容

## 相关文档

| 文档 | 定位 |
|------|------|
| `ARCHITECTURE_DESIGN.md` | 完整架构设计——流程图、数据流、设计决策 |
| `IMPLEMENTATION_PLAN.md` | 逐文件实现计划——36个文件、依赖关系、接口定义 |
| `example_output/README.md` | Simulator 示例输出索引 |
| `../claude_resume_summary/resume_2026-07-23.md` | 对话总结——项目总目标、基础知识、Gap说明 |
| `../docs/project_knowledge/` | 项目知识文档 |
