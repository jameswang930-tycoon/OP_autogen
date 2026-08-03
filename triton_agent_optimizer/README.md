# Triton Agent Optimizer — 完整架构设计 (v3.1)

> **核心差异化优势**: 不靠盲试（AutoKernel 300~400轮），而是通过 **Triton→HIVM MLIR + msprof 三源合一** 精确诊断瓶颈——知道哪个 op、哪个引擎、带宽利用率多少、为什么慢、该改什么参数。精准度比盲试高一个数量级。
>
> **数据源 = 三源合一**: ① HIVM IR（真实 hivm_try.txt，结构字段）+ ② msprof op simulator（指令级时序）+ ③ 真机 msprof op（L2/带宽/端到端校准）→ `dsl_merger` 合并 29 字段 merged_report.json → 驱动 LLM 6 层优化。
>
> **环境**: 910B3 真机（保密服务器，conda `triton-npu`）+ Windows/WSL2 开发机。triton-ascend 3.2.0 / triton 3.2.0 / torch_npu 2.9.0 / ACL (1,16,0,0)。
> **更新**: 2026-08-03 — 真实 HIVM 获取打通 + 三源产物全部核验通过（见下方「当前状态」）

---

## 当前状态（2026-08-03 定版）

### ✅ 已完成
- **真实 HIVM 获取正路打通**：`kernel.npuir.mlir` 确认不生成（根因见「关键事实」），改用**手动 bishengir 打印**（标准流程 D）成功产出 `hivm_try.txt`（真实 HIVM IR）。
- **三源产物全部采集并核验通过**：
  - HIVM：`hivm_try.txt`（32 hivm.hir op，含 **1×`hivm.hir.mmadL1`** cube op；21 sync op；19 具体 2D alloc；28 动态维；address_space={cbuf,cc,gm}）
  - simulator：24 核 cubecore instr_exe（标准表头）+ trace.json
  - 真机 msprof op：OpBasicInfo / PipeUtilization / Memory（标准字段）
- **核验方式升级**：保密服务器不能贴出内容 → 改为「网上核实标准字段 → 命令自动匹配 → PASS/ERROR」，15 项全 PASS（核验命令在 `input/matmul/test_matmul.py` 注释里，可整体粘贴跑）。

### 🔶 进行中 / 待修（下一步）
- `hivmir_analyzer.py` 3 个缺口待修：
  1. **2D alloc 不解析**（现只认 1D `memref<Nxdtype>`；真实是 `memref<256x256xf32>` 2D + `memref<?x?xf32>` 动态维）→ size_kb 需支持 2D/3D + `?` 分支
  2. **address_space 值未映射**（真实值 cbuf/cc/gm，需映射 L1/L0C/GM）
  3. **sync op 被跳过**（真实是 `hivm.hir.set_flag/wait_flag/pipe_barrier`，**前缀是 `hivm.hir.`** 非 `hivm.`；现 parser 全丢）
- `msprof_analyzer.py` 的 `parse_hardware_dir()` 真机分支**未写**（board_prof 的 OpBasicInfo/PipeUtilization 需接入，表头已验证为标准字段）
- `dsl_merger` op_id 对齐需真机验证（critical_path/parallel_pairs 曾索引错位）
- 本地跑通完整链路：`hivmir_analyzer(hivm_try.txt)` → `msprof_analyzer(sim_prof)` → `dsl_merger` → 29 字段 merged_report.json

### 关键事实（勿再踩坑）
| 项 | 事实 |
|---|---|
| npuir.mlir | **不生成**。根因：同步 pass 改名（`hivm-inject-sync`→`hivm-graph-sync-solver`）跟 CANN bishengir 走，与 triton-ascend 版本无关（3.2.0 也中招）。正路 = 手动 bishengir 打印 |
| cube op 真名 | `hivm.hir.mmadL1`（**不是** `hivm.hir.matmul`，matmul=0 是正常现象） |
| sync op 真名 | `hivm.hir.set_flag` / `hivm.hir.wait_flag` / `hivm.hir.pipe_barrier`（前缀 `hivm.hir.`） |
| `?` 字符 | 是 **MLIR 合法动态维占位**（`memref<?x?xf32>`，28 处），**不是**编码问题（文件 ASCII） |
| address_space | 真实只出现 {cbuf,cc,gm} = L1/L0C/GM 通路（无 ub/ca/cb） |
| 打印 flag | `--bishengir-print-ir-after=<pass名>`，白名单只有 `hivm-inject-sync`（**不是** `--mlir-print-ir-after-all`） |
| 保密约束 | 只能贴入不能贴出 → 核验命令只输出数字/PASS/ERROR；docstring 命令不能含反斜杠 |

---

## 0. 完整数据流（三源合一）

```
Triton Kernel (.py)
  │  triton-ascend 3.2.0 (真机编译链路)  或  triton 3.4.0 (WSL2 GPUTarget)
  ▼
TTIR MLIR (kernel.ttir.mlir)
  │  triton-ascend 适配器
  ▼
TTAdapter MLIR (kernel.ttadapter.mlir)   ←── 手动 bishengir 的正确输入
  │  bishengir-compile (标准流程 D, 打印 flag 见上)
  ▼
HIVM MLIR (hivm_try.txt) ★真实 HIVM
  │                        │
  │  hivmir_analyzer.py    │  msprof op simulator → OPPROF_xxx/simulator/
  │  结构字段               │   instr_exe.csv + trace.json
  ▼                        ▼
HIVM Report              msprof Report (指令级时序)
(结构字段)                 (duration_ns/cycles/engine/pipe/start/end)
  │                        │
  │   真机 msprof op → board_prof (L2/带宽/端到端校准)
  ▼                        ▼
  └────────── dsl_merger.py ──────────┘
              ▼
         29 字段 merged_report.json
              ▼
     BottleneckDiagnoser → DataExtractor → Planner(LLM) → Coder(LLM) → Verifier → RecordManager
```

### 环境需求

| 步骤 | 环境 | 工具 |
|---|---|---|
| Triton .py → TTIR/TTAdapter | 910B3 服务器 | triton-ascend 3.2.0（真机编译） |
| TTAdapter → 真实 HIVM | 910B3 服务器 | `bishengir-compile --bishengir-print-ir-after=hivm-inject-sync`（标准流程 D） |
| msprof simulator | 910B3 服务器 | `msprof op simulator`（需 LD_LIBRARY_PATH 指向 simulator lib） |
| msprof op 真机 | 910B3 服务器 | `msprof op`（PipeUtilization/ResourceConflictRatio 精简版已验证） |
| Analyzer + Optimizer | 任何 Python | hivmir/msprof/dsl_merger analyzer + agents |

**WSL2 侧**：保留自研兜底链路 `ttir_to_hivm.py` / `hivm_to_ascendc.py`（matmul cube 通路是近似，仅作保底，正解是服务器真实 HIVM）。

---

## 0A. 保密服务器工作方式（重要）

- **只能贴入（paste-in），不能贴出（copy-out）**。用户读服务器输出，把**数字/短字段名/PASS-ERROR** 打回对话。
- 所有核验命令必须**只输出数字 / 短分布 / PASS/ERROR**，不能输出原始 IR/CSV 内容。
- 命令写进 `input/matmul/test_matmul.py` 头注释，方便复制到服务器跑。
- 写注释的 grep **不能含反斜杠**（`\.` `\b` 会 SyntaxWarning 或变退格符）。用 `.`、`[?]`、`-F` 代替。
- 可粘贴的 shell 块：**所有非命令行必须以 `#` 开头**，否则裸 `(` 等会 syntax error。

---

## 0B. 拿真实 HIVM 的标准流程 A/B/C/D（已验证 D 成功）

```bash
# A. 找已有 ttadapter.mlir (之前拷回可复用, 跳过 B)
find . -name 'kernel.ttadapter.mlir' -o -name 'kernel.ttir.mlir' 2>/dev/null
# B. 没有就强制重编译 (nuke 整个 ~/.triton 才保险)
rm -rf ~/.triton
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
python3 test_matmul.py 2>&1 | tee run_debug.txt
find ~/.triton -type f | head -30
cp ~/.triton/dump/*/kernel.*.mlir ./    # ★ 拷回当前目录, D 才有输入文件
# C. 确认 bishengir-compile 在 PATH
which bishengir-compile || find /usr/local/Ascend -name 'bishengir-compile' 2>/dev/null
# D. ★核心: 对 ttadapter.mlir 跑官方命令打印 HIVM (已验证成功)
cd <ttadapter 所在目录>
bishengir-compile --target=Ascend910B3 --enable-auto-multi-buffer=True \
  --enable-auto-bind-sub-block=True --enable-hfusion-compile=true \
  --enable-hivm-compile=true --enable-triton-kernel-compile=true \
  --bishengir-print-ir-after=hivm-inject-sync kernel.ttadapter.mlir \
  -o /tmp/k.o 2>&1 | tee hivm_try.txt
grep -c 'hivm.hir' hivm_try.txt     # 预期 >0
```
- `aarch64-unknown-linux-gnu [-Woverride-module]` 警告**无害**，忽略。
- D 第一次报 `failed to open input file` = ttadapter 没拷回当前目录（先跑 B 末尾 cp）。

---

## 0C. 三源采集命令（全部已验证）

```bash
# ① HIVM IR（见 0B 标准流程 D）
# ② msprof op simulator（指令级, 唯一 instr_exe.csv 来源）
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
msprof op simulator --kernel-name=matmul_kernel --soc-version=Ascend910B3 \
    --output=./sim_prof python3 test_matmul.py
# ③ 真机 msprof op（L2/带宽/端到端校准; 精简版已验证通过）
msprof op --kernel-name=matmul_kernel \
    --aic-metrics=PipeUtilization,ResourceConflictRatio --output=./board_prof python3 test_matmul.py
```
**产物自动核验**（PASS/ERROR 版，[1]~[16]，写在 test_matmul.py 注释，可整体粘贴）：已 `bash -n` 验证零语法错误、零反斜杠。15 项全 PASS + `?` 判据确认。

---

## 0. Agent 实现模式

**Agent 的 Python 文件 = Prompt 编排器 + 执行框架。决策智能在 LLM 侧。**

```
  Python 骨架 (代码做的事)           LLM 大脑 (AI 做的事)
  ─────────────────────────          ─────────────────────
  1. 构建上下文 (诊断+Playbook+历史) → 1. 理解瓶颈原因
  2. 估算token用量，裁剪超限        → 2. 参考 Playbook 选择策略
  3. 调用 LLM API                   → 3. 生成具体优化计划 (JSON)
  4. 解析 JSON 输出                  → 4. 生成代码 diff
  5. 写入文件 / 执行动作             → 5. 判断优化是否有效
```

| 组件 | 实现方式 | 说明 |
|---|---|---|
| **Orchestrator** | Python 状态机 | 薄循环，不做决策 |
| **Planner** | LLM Agent | 读诊断+Playbook+历史 → 生成计划 |
| **Coder** | LLM Agent | 读计划+代码 → 最小化改动 |
| **Verifier** | Python 脚本 | CPU仿真 + 910B3实测，零推理需求 |
| **Analyzers** | Python 脚本 | 确定性函数链（hivmir/msprof/dsl_merger/diagnoser/extractor） |
| **RecordManager** | Python 规则引擎 | KEEP/REVERT + Tier + 停止条件 |

---

## 1. 完整闭环架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            INPUT LAYER                                       │
│  Triton kernel (.py)  ·  Shape/Dtype  ·  PyTorch reference  ·  Target 910B3 │
│  + 三源采集: hivm_try.txt(真实HIVM) + sim_prof(指令级) + board_prof(真机)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYSIS LAYER (5 Scripts)                          │
│                                                                              │
│    HIVMIR Analyzer              msprof Analyzer                              │
│    ┌──────────────────┐        ┌──────────────────┐                          │
│    │ hivm_try.txt     │        │ instr_exe.csv    │                          │
│    │ buffer名/size    │        │ trace.json       │                          │
│    │ RAW/WAR/WAW      │        │ timing/engine    │                          │
│    │ 操作序列         │        │ pipeline channel │                          │
│    └────────┬─────────┘        └────────┬─────────┘                          │
│             │          DSL Merger       │       真机 board_prof             │
│             └──────────┬────────────────┘        (L2/带宽校准)              │
│                        ▼                                                     │
│             ┌─────────────────────┐                                          │
│             │ 29-field Report     │  op_id 对齐, 互相填补                    │
│             │ merged_report.json  │                                          │
│             └──────────┬──────────┘                                          │
│                        ▼                                                     │
│          ┌─────────────────────────┐                                         │
│          │ BottleneckDiagnoser     │  瓶颈分类 + 优化空间评估                 │
│          └──────────┬──────────────┘                                         │
│                     ▼                                                        │
│          ┌─────────────────────────┐                                         │
│          │ DataExtractor           │  Tier x 列过滤, 聚合分析                 │
│          │ ~2KB 精简文本           │  输出 → 注入 Planner prompt              │
│          └─────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          AGENT LAYER                                           │
│  Planner (LLM) → Coder (LLM) → Verifier (Script) → Orchestrator (状态机)      │
│  → RecordManager (KEEP/REVERT + Tier + 7 停止条件 + trajectory)               │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION LAYER                                      │
│  Stage 1: CPU Emulator (emulator_runner)  ·  Stage 2: 910B3 Hardware         │
│  (本地跳过) hardware_runner + compiler.py                                    │
└──────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                        │
│  outputs/<kernel>/round0 + 01~06_tier + optimization_trajectory.json          │
│  + final_output (optimized_kernel.py / trajectory_chart.png / summary.md)     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1B. 完整数据流图 (端到端)

```
                              ┌──────────────┐
                              │ Triton Kernel │  (.py, 用户输入)
                              └──────┬───────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │ bishengir D    │    │ msprof simulator│   │ msprof op 真机 │
     │ (真实 HIVM)    │    │ (指令级)       │    │ (op 级聚合)    │
     │ hivm_try.txt   │    │ instr_exe.csv  │    │ OpBasicInfo/    │
     │                │    │ + trace.json   │    │ PipeUtil/Memory │
     └───────┬────────┘    └───────┬────────┘    └───────┬────────┘
             │                     │                     │
             ▼                     ▼                     ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │ hivmir_analyzer.py │  │ msprof_analyzer.py │  │ parse_hardware_dir │
  │ 结构字段           │  │ 指令级时序字段     │  │ (待写) 真机校准    │
  └────────┬───────────┘  └────────┬───────────┘  └────────┬───────────┘
           └───────────┬───────────┘                        │
                       ▼                                    │
        ┌──────────────────────────┐                        │
        │   dsl_merger.py          │                        │
        │   op_id 对齐 + 互相填补   │◄───────────────────────┘
        │   SATURATION_PARAMS 计算  │
        │   bw_util·regime·peak     │
        └───────────┬──────────────┘
                    ▼
        ┌──────────────────────────┐
        │   merged_report.json     │
        │   ★ 29 fields            │
        └───────────┬──────────────┘
                    ▼
     BottleneckDiagnoser → DataExtractor → Planner(LLM) → Coder(LLM)
     → Verifier(CPU仿真+910B3) → RecordManager(KEEP/REVERT) → 循环
```

### 关键文件读/写映射

| 文件 | 写入者 | 读取者 |
|---|---|---|
| `roundN/kernel.py` | Coder | Emulator, Hardware, Analyzers(下轮) |
| `roundN/plan.md` | Planner | Coder, 人工debug |
| `roundN/diff.patch` | Coder | 人工review |
| `roundN/optimization_record.json` | RecordManager | 人工, 后续分析 |
| `roundN/verification.json` | Verifier | RecordManager |
| `sim_prof/OPPROF_*/simulator/*_instr_exe.csv` | msprof op simulator | msprof_analyzer |
| `hivm_try.txt` (真实 HIVM) | bishengir-compile (流程 D) | hivmir_analyzer |
| `board_prof/OPPROF_*/OpBasicInfo.csv` 等 | msprof op | msprof_analyzer.parse_hardware_dir |
| `merged/merged_report.json` | dsl_merger | diagnoser, extractor, 人工 |
| `merged/final_report_llm.txt` | dsl_merger | Planner (LLM prompt) |
| `optimization_trajectory.json` | RecordManager | Orchestrator, Planner, trajectory_chart |
| `memory/experiences/tier*.json` | RecordManager | Planner (经验检索) |

---

## 2. 每轮执行流程

```
Round N:
  Orchestrator._run_one_round()
  │
  ├─ ① Analyzers.run()         [脚本, 5步链, 每轮重跑]
  │   hivmir(hivm_try.txt) → msprof(sim_prof+board_prof) → merger → diagnoser → extractor
  │   → merged_report.json + BottleneckDiagnosis + extracted_text
  │
  ├─ ② Planner.generate()      [LLM]
  │   输入: diagnosis + extracted + playbook_tier_N + history[-5:] + kernel_code
  │   输出: round_N_plan.md
  │
  ├─ ③ Coder.apply()           [LLM]
  │   输入: plan_text + kernel_code (+ previous_error if retry)
  │   输出: round_N/kernel.py + round_N/diff.patch
  │
  ├─ ④ Verifier.verify()       [脚本, 两阶段]
  │   Stage1: CPU Emulator → PASS/FAIL
  │     FAIL → error回传Coder, 重试最多3次
  │   Stage2: 910B3 → actual_speedup (服务器实测)
  │   输出: round_N/verification.json
  │
  └─ ⑤ RecordManager.evaluate() [规则引擎]
      decide KEEP/REVERT
      save round_N/optimization_record.json
      update optimization_trajectory.json
      check stop → CONTINUE / STOP

Round 0 (Baseline):
  只跑 Analyzers, 写入 trajectory.json baseline
```

---

## 3. 六层优化策略

**原则: 从结构影响最大到最小。改了上层, 下层要重做。**

| Tier | 名称 | Playbook | 做什么 | 晋升条件 |
|---|---|---|---|---|
| **1** | Algorithmic Structure | `playbook_tier1_algorithm.md` | Online Softmax / Split-K / Persistent Kernel | 算法已最优 |
| **2** | Operator Fusion | `playbook_tier2_fusion.md` | 逐元素融合 / WAR打破 / 激活融合 | 无可融合op |
| **3** | Tiling & Block Config | `playbook_tier3_tiling.md` | BLOCK_SIZE / num_warps / num_stages | 连续3轮无改进 |
| **4** | Memory Access | `playbook_tier4_memory.md` | 小传输合并 / double buffer / coalescing | 连续3轮无改进 |
| **5** | Compute & Occupancy | `playbook_tier5_compute.md` | 计算-传输重叠 / 向量化 / 精度取舍 | 连续3轮无改进 |
| **6** | 910B3 Architecture | `playbook_tier6_architecture.md` | Grid / Pipeline / L2驻留 / 混合精度 | 连续3轮无改进→停止 |

**降级**: 融合新算子→回到Tier3; 改了算法→回到Tier2

---

## 4. 文件架构总览（实际现状）

```
triton_agent_optimizer/
│
├── README.md                    # 本文档
├── ARCHITECTURE_DESIGN.md       # 架构设计
├── IMPLEMENTATION_PLAN.md       # 实现计划
├── OUTPUT_STRUCTURE.md          # 输出结构说明
├── config.py                    # 全局配置 (路径/硬件参数/阈值)
├── main.py                      # 入口: python main.py input/<op>/triton_kernel.py
│
├── prepare/                     # 环境准备 (setup_env.sh/.bat + env_check.py 35项检查)
│
├── analyzers/                   # 分析层
│   ├── hivmir_analyzer.py       #   真实 HIVM → 结构字段 (3 缺口待修: 2D alloc/address_space/sync)
│   ├── msprof_analyzer.py       #   instr_exe.csv+trace.json → 指令级时序 (parse_hardware_dir 待写)
│   ├── dsl_merger.py            #   合并 → 29 字段报告 + LLM 文本
│   ├── bottleneck_diagnoser.py  #   Tier-aware 瓶颈诊断
│   ├── data_extractor.py        #   Tier×列过滤精简提取
│   ├── timing_estimator.py      #   (估算用) SAT 公式估算时序
│   ├── ttir_to_hivm.py          #   (兜底) WSL2 近似转换, 非真实 HIVM
│   ├── hivm_to_ascendc.py       #   (兜底) WSL2 AscendC 生成, cube 通路近似
│   └── triton_to_hivm/          #   (兜底) 自研转换工具
│
├── agents/                      # 智能体层
│   ├── orchestrator.py          #   状态机调度
│   ├── planner.py               #   LLM 规划
│   ├── coder.py                 #   LLM 改代码, 只改 kernel.py
│   ├── verifier.py              #   CPU仿真 + 910B3 实测
│   └── llm_client.py            #   LLM API 客户端
│
├── execution/                   # 执行层
│   ├── emulator_runner.py       #   Stage1: CPU 正确性验证
│   ├── compiler.py              #   910B3 Ascend 编译
│   └── hardware_runner.py       #   910B3 benchmark
│
├── feedback/                    # 反馈层
│   ├── record_manager.py        #   KEEP/REVERT + Tier + Stop
│   └── trajectory_chart.py      #   6 阶段加速比曲线
│
├── memory/                      # 记忆层
│   ├── sliding_window.py        #   热/温/冷窗口
│   ├── context_manager.py       #   Token 估算+裁剪
│   ├── experience_retriever.py  #   分 Tier 检索
│   └── experiences/             #   经验库 JSON
│
├── docx/                        # Playbook 知识库
│   ├── OPTIMIZATION_METHODOLOGY.md
│   └── playbook_tier1~6_*.md
│
├── input/                       # 输入算子 (每 op 一个目录)
│   ├── matmul/                  #   test_matmul.py (真机驱动+注释命令) + triton_kernel.py
│   ├── softmax/  ├── rms_norm/  #   其他算子
│   ├── rms_norm_residual/  ├── fused_add_mul/  └── vector_add/
│
├── outputs/                     # 优化产物 (自动生成, 不入 git)
├── example_output/              # 参考示例
├── scripts/                     # 工具 (init_output_structure.py)
├── question/                    # 问答/临时笔记 (scratch)
└── paper_reference/             # 论文参考
```

---

## 5. 数据流: HIVM + msprof → 完整报告

```
  hivm_try.txt                instr_exe.csv + trace.json        board_prof csv
        │                              │                              │
  hivmir_analyzer                msprof_analyzer              parse_hardware_dir
  (结构字段)                      (指令级时序)                  (真机校准, 待写)
        │                              │                              │
        └──────────┬───────────────────┘──────────────────────────────┘
                   ▼
             dsl_merger
        (op_id 对齐, 互相填补)
                   │
                   ▼
         merged_report.json
         (29 fields)
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  bottleneck_diagnoser   data_extractor
  (瓶颈分类+评估)         (Tier×列过滤)
        │                     │
        └──────────┬──────────┘
                   ▼
           Planner (LLM)
      (诊断+Playbook+历史→计划)
```

**当前字段覆盖**（三源可覆盖 27/29 字段；`peak_bw_gb_s`/`regime` 需真机 Memory.csv 校准 + size 扫描曲线）：
- `hivmir_analyzer` 产：op_id/op_type/instruction/dst/src/src2/size_kb/memory_region/dependencies/buffers/dtype/attrs（**3 缺口待修**）
- `msprof_analyzer` 产：duration_ns/start_ns/end_ns/time_ratio/cycles/engine/pipeline_channel/core_id + 聚合 total_ns/execution_mode/critical_path/parallel_pairs/engine_utilization
- `dsl_merger` 产：29 字段完整报告（bw_utilization/regime/peak_bw 由 SATURATION_PARAMS + 真机校准）

---

## 6. 输出目录

```
outputs/<kernel_name>/
│
├── round0/                                    # ★ 基准分析 (仅 Analyzer, 无优化)
│   │  ┌─ 分析产物 ───────────────────────────┐
│   │  │ kernel.py            原始 Triton kernel        │
│   │  │ hivmir/  hivm_try.txt + hivmir_report.json    │
│   │  │ msprof/  sim_prof + pipeline_report.json      │
│   │  │ board_prof/ 真机 OpBasicInfo/PipeUtilization  │
│   │  │ merged/   merged_report.json(29字段全填)      │
│   │  │          final_report_llm.txt / human.txt     │
│   │  └──────────────────────────────────────────────┘
│
├── 01_algorithmic_structure/round1..N/        # Tier 1 (roundN 有 kernel.py/plan.md/diff.patch/optimization_record.json/verification.json)
├── 02_operator_fusion/round1..N/              # Tier 2
├── 03_tiling_block_config/round1..N/          # Tier 3 ← 主要加速阶段
├── 04_memory_access/round1..N/                # Tier 4
├── 05_compute_occupancy/round1..N/            # Tier 5
├── 06_910b3_architecture/round1..N/           # Tier 6
│
├── optimization_trajectory.json               # ★ 全局中枢状态 (RecordManager)
│
└── final_output/                              # ★ 最终产物
    ├── optimized_kernel.py
    ├── trajectory_chart.png
    ├── optimization_summary.md
    ├── final_merged_report.json
    └── final_report_llm.txt / human.txt
```

### optimization_trajectory.json

```json
{
  "state":  {"tier": 3, "round": 12, "best_speedup": 1.52,
             "consecutive_reverts": 0, "consecutive_no_improvement": 0},
  "baseline": {"total_ns": 3655.6, "num_ops": 3,
               "bottleneck_type": "memory_bandwidth"},
  "history": [
    {"round": 1, "tier": 1, "strategy": "...",
     "actual_speedup": 1.0, "cumulative_speedup": 1.0,
     "decision": "KEEP", "decision_reason": "..."}
  ]
}
```

---

## 7. 设计决策速查

| 决策 | 选择 | 理由 |
|---|---|---|
| 数据源 | 三源合一（真实 HIVM + simulator + 真机 msprof op） | 单一来源字段凑不齐；用户坚持"真实正确" |
| 瓶颈诊断 | 脚本规则引擎 | 确定性, 压缩到 ~2KB |
| 策略规划 | LLM (Planner) | 需要上下文推理 |
| 代码修改 | LLM (Coder) | 语义理解, 最小化改动 |
| 验证 | 脚本 (CPU+Hardware) | 确定性, 不推理 |
| 调度 | Python 状态机 | 管循环, 不做决策 |
| 决策/记录 | RecordManager | 集中管理 Tier+Stop+轨迹 |
| 优化顺序 | Algorithm→Fusion→Tile→Memory→Compute→Arch | 结构影响从大到小 |
| 记忆 | 分Tier JSON+3级匹配 | 简单, LLM可读, 无需向量库 |
| 保密服务器核验 | 标准字段自动匹配 PASS/ERROR | 只能贴入不能贴出，减少往返 |

---

## 8. 参考资料

| 内容 | 链接 |
|---|---|
| instr_exe.csv 字段（instr/addr/pipe/call_count/cycles/running_time/detail） | https://gitcode.com/mengguangxin/msopprof/blob/dev_0226/docs/zh/msopprof_simulator_performance_data.md |
| OpBasicInfo.csv 字段 | https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/opdev/optool/atlasopdev_16_0107.html |
| PipeUtilization.csv 字段 | https://www.hiascend.com/document/detail/zh/canncommercial/80RC1/devaids/auxiliarydevtool/atlasopdev_16_0079.html |
| triton-ascend address_space 扩展（ub/cbuf/ca/cb/cc 映射） | https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/triton_api_extention/al/al.ascend_address_space.md |
| HIVM Dialect 文档（cube/sync/op 语法） | https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/developer_guide/dialects/HIVMDialect.md |
| HIVMSynchronizationOps.td（sync op 定义） | https://gitcode.com/Ascend/AscendNPU-IR/blob/661afcc31a55ecfa08e3a2bc035ccdae60dcbcec/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td |
| bishengir 打印 flag 用法（FAQ） | https://gitcode.com/Ascend/AscendNPU-IR/blob/master/docs/source/zh_cn/faq/faq.md |
| bishengir 打印白名单（PassManagerOptions.cpp） | https://gitcode.com/Ascend/AscendNPU-IR/blob/af5499b3b9f3dbab50b2834bcfff5da5c2a1d920/bishengir/lib/Pass/PassManagerOptions.cpp |
| AscendNPU-IR issue #216（官方完整 bishengir 命令） | https://gitcode.com/Ascend/AscendNPU-IR/issues/216 |
| msopprof simulator 用户指南（cubecore/instr_exe 结构） | https://gitcode.com/mengguangxin/msopprof/blob/master/docs/zh/msopprof_simulator_user_guide.md |
| simulator 调优指南（指标/场景/限制） | https://gitcode.com/Ascend/msagent/blob/master/skills/msot-msopprof-operator-profiler/references/simulator-tuning-guide.md |
| triton-ascend 环境变量与编译选项参考 | https://github.com/triton-lang/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md |
| triton-ascend compiler.py（bishengir 调用） | https://gitee.com/ascend/triton-ascend/blob/95463d7f9c76fb090d4557030e1ea21191682f07/ascend/backend/compiler.py |
