# 执行手册增补：T10 – T12（确定性编排器收尾）

> **读者：Claude Code + GLM 5.2**（非保密开发机）
> **前置**：T1–T9 已完成并全部门禁通过（81 tests green，HEAD `7277b90`）。
> **本文档取代原 T10**。任务重新编号如下，请同步更新 `PROGRESS.md`：
>
> | 新编号 | 任务 | 原编号 |
> |---|---|---|
> | **T10** | 三个 skill 双模改造（dual-mode） | 新增 |
> | **T11** | 确定性编排器 orchestrator | 新增 |
> | **T12** | 交接包 `HANDOFF_GLM47.md` | 原 T10 |
>
> **架构变更说明**：目标形态从「agent 驱动的工作流」收紧为「**确定性编排器驱动的流水线**」。用户输入一个 job 文件，得到最终 kernel 与 report，全程无自然语言交互。LLM 从「流程驱动者」降级为「**被编排器调用的决策函数**」，只在圈定范围内做决策。

---

## 0. 本次收紧的三条设计原则

**① 流程正确性归代码，不归 LLM。** 编排器决定每一步调什么；LLM 只回答被问到的问题。最终执行环境是 GLM 4.7，把流程判断交给它是主要故障源。

**② LLM 的每个输出都要过确定性闸门。** 解析闸门 → pre-sim gate → 正确性闸门 → 回退保护 → 循环边界。任何一关不过，不许进入下一步。

**③ 每次 LLM 调用无状态。** 不累积会话历史；历史在文件里（runlog / best-so-far / report）。这样第 8 轮与第 1 轮的上下文占用相同——这是 128K 弱模型能跑长循环的前提，也是「状态只走文件、不进会话」不变量的极致形式。

---

## T10. 三个 skill 双模改造（dual-mode）

**目标**：让 `sim-analyze` / `triton-gen` / `extension-guide` 同时服务两种调用方式——**编排器直接读正文当 prompt 模板**（主用），**保留 frontmatter 使其仍可被 agent 模式手动触发**（人工调试后路）。成本极低，但保留了人工介入的能力。

**动作**：

1. **frontmatter 原样保留**（description 已写得很好，无 cross-trigger，不要动）。
2. **正文改造为可参数化的模板**：把正文中依赖运行时数据的位置，改为**显式占位符**，统一用 `{{VAR}}` 形式。各 skill 的占位符：
   - `triton-gen`：`{{OP}}`、`{{SHAPES}}`、`{{DTYPE}}`、`{{BASELINE_SRC}}`（可为空）、`{{VERDICT_JSON}}`（首轮为空）、`{{FEEDBACK_SUMMARY}}`（首轮为空）、`{{RETRIEVED_EXPERIENCE}}`、`{{EXTENSION_INDEX}}`
   - `sim-analyze`：`{{VERDICT_JSON}}`、`{{FEEDBACK_SUMMARY}}`、`{{CANDIDATE_LEVERS}}`
   - `extension-guide`：无需占位符（作为索引被读取）
3. **每个模板末尾追加一段「输出契约」**，规定返回格式必须机器可解析（见 T11 §3 受约束输出）。
4. **占位符在 agent 模式下的降级说明**：正文中加一句——若以 agent 模式手动触发（占位符未被替换），则按用户在对话中提供的信息填充。这保证双模都能工作。

**槽位**：无。

**验收**：扩展 `tests/test_t10_skills_dualmode.py`——
- 三个 SKILL.md 的 frontmatter 仍合法、description 未变（回归保护）；
- `triton-gen` / `sim-analyze` 正文中的占位符集合与 T11 编排器实际注入的变量名**完全一致**（这条必须自动检查，占位符与注入变量对不上是最易犯且最难查的错）；
- 每个模板含「输出契约」段。

---

## T11. 确定性编排器 `control/orchestrator.py`

**目标**：一条命令跑完全流程。

```bash
.venv/bin/python -m control.orchestrator --job jobs/matmul.yaml
# → outputs/<op>_<timestamp>/{recommended.py, final_round.py, report.json, log/}
```

### 1. Job 规格（`control/job_spec.py` + YAML）

支持三种输入形态，**在边界处归一化**，不允许多态性泄漏进循环：

```yaml
op: matmul
input:
  form: triton_file        # triton_file | pytorch | shape_only
  path: ./baseline.py      # form=shape_only 时省略
shapes: [1024, 1024, 1024]
dtype: fp16
budget:
  max_rounds: 6
  epsilon: 0.03
  llm_retries: 3           # 解析失败重试，不计入轮数
  presim_retries: 2        # 未过静态闸门重试，不计入轮数
  sim_retries: 3           # 仿真设施故障退避重试，不计入轮数
```

**归一化产出统一的 `NormalizedJob`**，字段至少含：`op`、`shapes`、`dtype`、`baseline_src: Optional[str]`、`reference_src: Optional[str]`、`has_baseline: bool`。

三种形态的处理差异**只在归一化这一层**：

| form | baseline_src | reference | 第 0 轮 |
|---|---|---|---|
| `triton_file` | 用户提供的 Triton | 从文件提取或由 gen 生成 | **实测 baseline**，seed best-so-far |
| `pytorch` | 无 | 由 PyTorch 代码提供 | **无 baseline 轮**，直接进第 1 轮生成 |
| `shape_only` | 无 | 由 gen 一并生成 | 同上 |

**关键：循环必须支持 no-baseline 模式**（`has_baseline=False` 时 best-so-far 从空开始）。T3 的 loop_controller 已能处理 `best is None`（其测试 `test_consecutive_fail_counts_toward_rounds` 已覆盖），无需改动它。

本次**只需完整实现 `triton_file`**；`pytorch` 与 `shape_only` 留出归一化函数骨架 + `NotImplementedError`，并在 job schema 中已声明——保证后续接入不改架构。

### 2. 主循环（基于已有真实接口，不要臆造）

已验证可用的真实接口，直接调用：

```python
from control.contracts import Event, Verdict, SimResult
from control.feedback_adapter import parse_raw, adapt        # adapt(events) -> AdapterOutput(.verdict, .summary)
from control.loop_controller import ...                       # StopReason: EPSILON/MAX_ROUNDS/IRREDUCIBLE/OSCILLATION/NUMERICAL_FAIL
from control.launch_template import launch                    # 槽位
from control.presim_gate import check_extension_calls, ...    # 通用检查已实现
from control import vocabulary
from memory.writeback import record_attempt                   # (log, store, fp, retrieved_ids, passed, cycles, kernel_ref, extension_used, stage)
```

流程：

```
normalize(job) -> NormalizedJob
if has_baseline:  round0: launch(baseline) -> SimResult -> seed best-so-far
loop:
  1. retrieve   : memory 检索 -> retrieved_experience
  2. lever      : 首轮无 verdict -> 跳过; 否则 vocabulary 查表
                  查表唯一 -> 直接用（不调 LLM）
                  查表多候选 -> llm_choose_lever(verdict, candidates)   [LLM 调用点 2]
  3. generate   : llm_generate(模板注入)  -> kernel_src + meta          [LLM 调用点 1]
                  解析失败 -> 重试(llm_retries), 不计轮数
  4. presim     : presim_gate 全部检查; 不过 -> 回到 3(presim_retries), 不计轮数
  5. launch     : launch(kernel) -> raw; 设施故障 -> 退避重试(sim_retries), 不计轮数
  6. parse      : parse_raw(raw) -> events;  SimResult 取 correct/cycles
  7. adapt      : adapt(events) -> AdapterOutput(.verdict, .summary)
                  verdict.bottleneck 不在词表 -> 停止并记录（不许猜，见 §5）
  8. record     : record_attempt(..., passed=correct, cycles=..., extension_used=...)
  9. controller : 送入 loop_controller -> should_stop / reason / best / rolled_back
                  should_stop -> break
emit report
```

**LLM 调用点全流程仅两个**，且第 2 个只在查表出现多候选时才触发。这是「在圈定范围内做决策」的精确落点：**候选集由代码圈定，选择由模型做**。

### 3. 受约束输出（constrained output）与解析闸门

`llm_generate` 的返回必须机器可解析，约定格式：

````
```python
<kernel 源码：多段式模块 kernel / reference / compare>
```
```json
{"lever": "<lever id or null>", "extension_used": "<primitive name or null>", "notes": "<=100 chars"}
```
````

解析器要求：提取到**恰好一个 python 块与一个 json 块**，且 json 字段齐全、`extension_used`（若非 null）在速查表内。任一不满足 → 视为解析失败 → 重试。**不允许自由散文进入流程。**

### 4. 重试预算按失败类型分离（鲁棒性核心）

这是最易踩的坑：若所有失败共用一个预算，一次网络抖动就能吃光整轮优化。必须分开：

| 失败类型 | 归因 | 预算 | 是否计入轮数 |
|---|---|---|---|
| LLM 输出解析失败 | 工具没配合好 | `llm_retries` | ❌ |
| 未过 pre-sim gate | 工具没配合好 | `presim_retries` | ❌ |
| 仿真设施故障（超时/断连） | 基础设施 | `sim_retries`（指数退避） | ❌ |
| **数值 FAIL（correct=false）** | **模型真的做错了** | 不重试 | ✅ **计入**（T3 已定口径） |

各类预算耗尽 → 停止并在 report 中记录明确原因。

### 5. 词表闭包（unknown 必停，不许猜）

`adapt` 产出的 `bottleneck` 若不在 `vocabulary.yaml` 中 → **立即停止，记录该未知类别**。不许回退到"最接近的类别"或让 LLM 猜。理由：未知类别本身是有价值的信号——它说明词表该扩了；猜过去会把问题掩盖。

### 6. 输出产物（report 是一等交付物）

```
outputs/<op>_<timestamp>/
  recommended.py     # best-so-far；no-baseline 且从未 PASS 时可为空并在 report 说明
  final_round.py     # 最后一轮产物（即使比 baseline 慢，也保留）
  report.json
  log/round_<n>_{prompt,response,raw_sim,summary}.txt
```

`report.json` 至少包含：

```json
{
  "job": {...},
  "baseline": {"cycles": 12000, "present": true},
  "recommended": {"round": 3, "cycles": 8100, "speedup_vs_baseline": 1.48},
  "final_round": {"round": 6, "cycles": 13500, "speedup_vs_baseline": 0.89},
  "stop": {"reason": "EPSILON", "rounds_used": 6},
  "rounds": [
    {"n": 1, "correct": true, "cycles": 11000, "bottleneck": "memory_underfilled",
     "lever": "...", "extension_used": null, "rolled_back": false, "kernel": "log/..."}
  ]
}
```

**设计要点**：`recommended` 与 `final_round` **并列输出**。前者保证下游不会误用比 baseline 慢的 kernel；后者保留"为什么变差"的证据。对自动算子生成而言，失败样本与成功样本同等重要——`rounds[]` 必须完整留痕，含失败轮（`correct=false` 的轮次也要记，其 cycles 置 null）。

### 7. 验收（`tests/test_t11_orchestrator.py`）

用 monkeypatch 注入假 `launch`（返回 fixture）与假 LLM（返回预置的合规/不合规响应），**全部离线可跑，不需任何保密信息**：

1. **端到端**：`triton_file` 形态跑通 N 轮 → 产出 `report.json`、`recommended.py`、`final_round.py`，字段齐全。
2. **兜底行为**：构造"每轮都比 baseline 慢" → `recommended` 指向 baseline（speedup=1.0），`final_round` 指向最后一轮（speedup<1），二者都存在。
3. **重试隔离**：注入 2 次解析失败 + 1 次成功 → 轮数只增加 1（验证解析失败不计入轮数）。
4. **数值 FAIL 计入轮数**：连续 correct=false → 达 `max_rounds` 停止（复用 T3 口径）。
5. **词表闭包**：adapt 产出未知 bottleneck → 立即停止，report 中 `stop.reason` 标明未知类别。
6. **no-baseline 模式**：`has_baseline=false` 时正常跑，best-so-far 从空开始。
7. **占位符一致性**：编排器注入的变量名与 T10 模板中的占位符集合完全一致。

---

## T12. 交接包 `HANDOFF_GLM47.md`

**目标**：产出给 GLM 4.7 直接执行的文档。**最后写**，因为它必须反映编排器落地后的最终形态与真实签名。

**编写原则**：不要让 4.7 读架构文档或执行手册（读不完也不需要）；交接包必须**自包含且极短**；每项写成**填空题 + 自检命令**；中文、句子短、一项一段。

**必须包含五节**：

**① 五个槽位任务表**（4.7 全部要动笔的地方，路径与签名须与 `control/` 下真实代码逐一核对）：

| 任务 | 文件 | 形式 | 自检 |
|---|---|---|---|
| 填词表内容 | `control/vocabulary.yaml` | 3 条示例替换为真实 stall 类型，5–8 条 | `python -m control.check_vocab_consistency` |
| 写 `parse_raw()` | `control/feedback_adapter.py` | 单函数，`Event` 字段已冻结 | `pytest tests/test_t5_adapter.py` |
| 填 extension 速查表 | `.claude/skills/extension-guide/references/` | 照样例条目格式逐条填 | `python -m control.check_extension_cheatsheet` |
| 写 `launch()` | `control/launch_template.py` | 单函数，发射并取回原始输出 | `pytest tests/test_t6_launch.py` |
| 加 extension 合法性检查 | `control/presim_gate.py` | 补 `check_extension_calls()` | `pytest tests/test_t7_gate.py` |

**② 三条纪律**（同时已在 `AGENTS.md`）：不改签名/schema/词表结构；不做架构判断（遇到设计问题停下上报）；每改一处跑对应自检。

**③ 运行说明**：如何写 job 文件、如何跑 `python -m control.orchestrator --job ...`、产物在哪、怎么读 `report.json`。

**④ 验证清单**：小 kernel 跑通完整闭环；停止逻辑三用例；记忆冷启动价值（清空经验库 vs 带经验，对比首轮质量与轮数）；词表一致性三方同源。

**⑤ 保密纪律**：4.7 填入的真实 stall 类型名、原语名、仿真字段名**不得回流公开分支**；公开分支保持占位，测试靠 fixture（预置测试数据）跑。

---

## 执行顺序与停止点

```
T10 ── T11  ║停止点④║  T12  ║停止点⑤║
（自动执行，门禁把关）   （人工复核）
```

- **T10、T11 连续自动执行**，门禁逐个把关；完成后停下等复核。
- **T12 单独开新会话**：先重新扫一遍 `control/` 下实际代码，确认五个槽位真实路径与函数签名，再写交接包。

沿用原协议：每任务先写测试 → 实现 → 门禁 → `git commit`（`T<n>: <名> [gate passed]`）→ 更新 `PROGRESS.md`。门禁连续失败 3 次停止上报，**不许放宽测试断言**。

**最终硬判据**：无任何保密信息的情况下 `pytest tests/ -v` 全绿。

---

## 禁令（延续原手册）

- 不改 `costModel/`；不把 memory 做成 skill；不臆造硬件细节（**臆造比留空危险得多**）。
- 不改动 T1–T9 已冻结的四份契约与三个 skill 的 description（如确需变更，停下上报）。
- 编排器**不得**把流程判断委托给 LLM——判停、归类、重试计数全部在确定性代码里。
