# 本地适配框架改造执行手册（面向 Claude Code + GLM 5.2）

> **本文档的读者是你**——运行在非保密开发机上的 Claude Code + GLM 5.2。
> **嵌套结构**：本文档指导你改造框架；你的**最后一项任务（T10）是编写一份《交接包》**，用于指导保密环境上的 OpenCode + GLM 4.7 完成适配、运行与验证。你不只是执行者，也是下一级指导的作者。
> **设计依据**：架构结论见配套的《本地定制硬件生态适配：架构演进指导》（下称"架构文档"），本文档不重复论证，只讲怎么做。
> **版本对齐**：本手册针对 `wsx` 分支 commit `79b4da1` 编写。开工前先 `git log -1` 核对；若已前进，重点复查 `.claude/skills/` 与 `memory/` 是否有变化。

---

## 0. 你必须理解的三件事

**第一，这次改造的实质是流水线因果方向反转。** 旧：cost model 预测流水 → 生成 kernel。新：生成 kernel → 远端真实仿真实测 → 分析瓶颈 → 再生成。cost model 的解析式流水预测**退役**，真实仿真成为测量真值。

**第二，你看不到、也不需要看到任何真实硬件/仿真信息。** 那些只存在于保密环境。你的职责是**把一切与硬件无关的部分做完、做扎实**，并为需要真实信息的地方留出**定义清晰的槽位（slot）**。

**第三，最终执行环境是 OpenCode + GLM 4.7（128K 上下文、能力弱于你）。** 因此你留下的每个槽位，必须让 4.7 能以"填空"的方式完成——有明确签名、有格式规范、有自检命令、不需要跨文件推理、不需要做架构判断。**凡是你能替 4.7 做掉的判断，都要在这里做掉。** 这是本次改造的第一原则。

---

## 1. 前置动作

```bash
git log -1                      # 核对 HEAD 是否 79b4da1
git checkout -b local-adapt     # 单独分支演进，便于回滚
```

环境提醒：本仓 system `python3` 是 3.7，凡执行脚本一律用 `.venv/bin/python`（沿用现有约定，见 `docs/project_knowledge/environment_and_running.md`）。

---

## 2. 目标结构（改造后）

```
.claude/skills/
  sim-analyze/SKILL.md          # 由 triton-plan 改造而来（职责反转）
  triton-gen/SKILL.md           # 保留并改造（目标改为真实 Triton + extension）
  extension-guide/              # 新增：extension 速查表 skill（正文由你写，内容留空给 4.7）
    SKILL.md
    references/                 # 详细条目，按需读
control/                        # 新增：确定性控制面（非 skill，纯 Python）
  contracts.py                  # 四份冻结契约的定义与校验
  vocabulary.yaml               # 瓶颈类别词表（格式由你定，内容留空）
  vocabulary.py                 # 词表加载/校验
  feedback_adapter.py           # 骨架完整，parse_raw() 留空
  loop_controller.py            # 你全量实现
  presim_gate.py                # 通用检查完整，extension 检查留空
  launch_template.py            # 模板完整，launch() 留空
  fixtures/                     # 合成夹具（你写，使上述在无保密信息下可测）
memory/                         # 演进：cycles 打分 + extension_used
tests/                          # 你写的单元测试，全部可在本机跑通
HANDOFF_GLM47.md                # T10 产出：给 4.7 的交接包
opencode.json                   # OpenCode 配置（skill 权限）
AGENTS.md                       # 项目规则（镜像 CLAUDE.md + 4.7 纪律）
```

---

## 3. 任务清单（按顺序执行）

每项任务的格式统一为：**目标 / 动作 / 留给 4.7 的槽位 / 验收**。

---

### T1. 退役与保留

**目标**：清掉不再适用的部分，避免 4.7 被过时资产误导。

**动作**：
- 删除 skill：`.claude/skills/triton-convert/`（gen 直接产出真实 Triton，转换步骤消失）。
- 删除 skill：`.claude/skills/triton-verify/`、`.claude/skills/triton-fix/`。理由见架构文档 §1.0：远端仿真是 functional + timing 兼具，正确性比对由发射脚本内的 reference 承担，二者职责溶解进"发射脚本 + loop-controller"。
- `emulators/` 目录：**不要删除**，改为标记退役。在 `emulators/README.md` 写明"CPU 侧 emulator 已退役，仅作历史参考，新流水线不使用"。保留的理由是其中的 `common/__init__.py` 记录了 `tl.*` 的实现语义，改造 gen 时你可能需要参考。
- `costModel/`：**只读保留，不要修改**（协作方仓库）。其 `Skills/bottleneck-analysis/SKILL.md` 中"关键路径优先 + 瓶颈分类→优化杠杆"的分析骨架仍然有效，你在 T8 改造 `sim-analyze` 时应当继承这套方法论，只替换其数据来源。

**槽位**：无。

**验收**：`.claude/skills/` 下只剩 `sim-analyze`、`triton-gen`、`extension-guide` 三个；`git status` 显示删除项符合上述清单。

---

### T2. 冻结四份契约 + 词表注册表（地基，必须最先做）

**目标**：定义两个环境之间的全部接口。契约冻结后，4.7 只填内容、绝不改形状。

**动作**：在 `control/contracts.py` 中定义并实现校验函数，共四份：

1. **`Event`（仿真事件中间结构）**——feedback_adapter 的输入。字段建议：`name: str`、`start: int`、`end: int`、`duration: int`、`unit: str`（执行单元）、`stall_class: str`（须取自词表）、`bytes: int | None`。**字段名由你冻结，4.7 不得更改。**
2. **`Verdict`（分析结论）**——adapter 的输出，loop_controller 与 memory 的输入。建议：`{"bottleneck": str, "lever": str, "cycles": int, "expected_gain": float}`。`bottleneck` 必须是词表中的合法 id。
3. **`SimResult`（发射脚本输出）**——建议：`{"correct": bool, "max_abs_err": float, "cycles": int, "pipeline": {...}}`。**正确性与性能必须是两个可分辨字段**（架构文档 §3.6），不得混在自由文本里。
4. **`vocabulary.yaml`（瓶颈词表格式）**——每条含 `id`、`desc`（一句话语义）、`lever`（对应优化杠杆）、`primitives`（关联 extension 原语，留空数组）。

**同时实现** `control/vocabulary.py`：加载词表、校验格式、提供 `all_ids()`。并写一个**一致性检查脚本**，确认三处引用同一份词表、无孤儿标签：adapter 输出的 `stall_class`/`bottleneck`、memory 的 `Fingerprint.bottleneck`、extension 速查表的索引。

**槽位**：`control/vocabulary.yaml` 的**词条内容**留空，只放 2–3 条**示例条目**（用通用名如 `compute_bound_at_peak`、`memory_underfilled`、`stall_dependency`）并注明"示例，待保密环境按真实 stall 类型替换/增补，总数控制在 5–8 条"。

**验收**：一致性检查脚本可运行；用示例词表能通过；故意写一个不在词表中的标签，校验应报错。

---

### T3. loop-controller（你全量实现，4.7 一行不碰）

**目标**：把停止决策完全放进确定性代码，不依赖 4.7 的判断力。这是对弱模型最重要的一层保护。

**动作**：按架构文档 §3.5 实现 `control/loop_controller.py`：

- **主判据**：本轮相对 best-so-far 的 cycles 改善 < ε → 停（ε 默认 0.03–0.05）。
- **兜底**：轮数达 N → 停（N 默认 5–8）。
- **不可约**：瓶颈类别判定 irreducible → 停。
- **振荡/无进展**：瓶颈类别连续 2 轮不变，或在两个变体间来回（以 kernel 变体指纹检测）→ 停。
- **数值 FAIL 且重生成仍不过** → 停并上报。
- **回退保护**：本轮 cycles 变大 → 回滚上一版，可再试一个备选杠杆，仍无改善则停。
- **best-so-far 底线**：全程保留历史最优，**停止时返回实测 cycles 最优的那一版，不是最后一轮那一版**。
- **两条计算口径**：判停只在 `correct=true` 的轮次上计算改善；但 `correct=false` 的轮次**仍计入轮数预算**。

ε 与 N 必须是**可配置参数**（配置文件或环境变量），不得硬编码。

**槽位**：仅配置数值，4.7 可按仿真成本调整，无需写码。

**验收**：写三个单元测试并全部通过——
1. 每轮仅 1% 改善 → 在 ε 处停，而非跑满 N 轮；
2. 第 3 轮 cycles 变大 → 回滚，且最终返回的是历史最优那版；
3. 连续 `correct=false` → 不参与改善计算，但计入轮数、最终达上限停止。

---

### T4. memory 模块演进（你全量实现）

**目标**：把价值信号从"正确性通过"切换到"性能改善"。现有 `memory/` 八个文件与 `memory_cli.py` 的结构保留，只改语义。

**动作**：
- `memory/schema.py`：`AttemptRecord.latency_us` **替换为 `cycles: Optional[int]`**（安全：`memory_cli.py` 的 `cmd_record` 当前并未传 `latency_us`，无数据依赖）。`Experience` 与 `AttemptRecord` 各增加可选字段 `extension_used: Optional[str]`（架构文档 §5.4），使经验能携带"用了哪个 extension 原语"。
- **新增每 fingerprint 的历史最优 cycles 记录**（存于 `ExperienceStore` 或并列的小文件）。`score()` 中 helped 的语义改为：**该经验在场，且这一轮刷新了该 fingerprint 的历史最优 cycles**。原 `(helped+1)/(used+2)` 的平滑形式保留。
- `memory/writeback.py` 的 `record_attempt`：入参增加 `cycles`；`correct=false` 的尝试**不参与性能打分**（不 bump helped），但**仍写入 runlog**——失败经验有价值。
- `memory_cli.py`：`record` 子命令增加 `--cycles` 参数；`stats` 输出中展示 cycles 相关统计。inject/record 的**调用位置与 CLI 形态保持不变**（仍是确定性控制面，不做成 skill）。

**槽位**：无。

**验收**：写单元测试——同一 fingerprint 连续三次尝试（第二次刷新最优、第三次未刷新），确认 `score()` 只在第二次上升；`correct=false` 的尝试不影响 score 但出现在 runlog 中。

---

### T5. feedback-adapter 骨架 + 合成夹具（本次改造技术含量最高的一项）

**目标**：把"海量真实仿真反馈 → 弱模型能消化的短摘要"这条链路，除最后一步解析外**全部做完**。

**动作**：实现 `control/feedback_adapter.py`，包含三段（架构文档 §3.3）：

1. **Reduce（削减）**：从事件列表中只保留关键路径 + top-k 主导代价。路径外的一切都是噪声。
2. **Classify（归类）**：给主导代价打词表中的标签。
3. **Render（渲染）**：产出与旧 raw_llm **同构的 7 段摘要**（沿用旧格式的理由：gen 已"会读"这个结构，不必重学），并在**顶部附一个机器可读的 `Verdict` JSON 头**。严格限定输出体量（建议关键路径 + top-5 代价 + 一行结论），因为 4.7 的 128K 还要装 extension 速查表、记忆与 kernel 本身。

**写合成夹具**：在 `control/fixtures/` 手写至少三份符合 `Event` 契约的假数据，覆盖典型形态（compute-bound、传输欠填充、依赖 stall）。**它们的作用是让上述全部逻辑在你这台无保密信息的机器上就能跑通、测过**，4.7 接手时只需替换数据来源。

**槽位**：只留一个函数——

```python
def parse_raw(raw_sim_output) -> list[Event]:
    """槽位：把真实仿真输出解析为 Event 列表。
    由保密环境的 GLM 4.7 实现。Event 字段已冻结（见 control/contracts.py），不得更改。
    """
    raise NotImplementedError("待保密环境实现")
```

**验收**：三份夹具分别喂入，均能产出结构正确的 7 段摘要 + 合法 Verdict（`bottleneck` 在词表内）；输出体量在设定上限内。

---

### T6. 发射脚本模板

**目标**：给出"发到远端仿真器执行"的完整文件模板。

**动作**：实现 `control/launch_template.py`。注意架构文档 §1.0 的结论——发射到仿真器的文件**不止 kernel**，还必须自带 reference 与比对代码才构成可发射的完整单元。因此模板须为**多段式**：kernel / reference（numpy 或 torch 金标准）/ 比对与结果输出。输出严格遵循 `SimResult` 契约。

**槽位**：

```python
def launch(kernel_file) -> raw_sim_output:
    """槽位：把 kernel 发射到远端仿真器并取回原始输出。
    由保密环境的 GLM 4.7 实现。
    """
    raise NotImplementedError("待保密环境实现")
```

**验收**：用一个本地假 `launch`（返回夹具数据）跑通全链路，确认能产出合法 `SimResult`。

---

### T7. pre-sim gate（静态预筛）

**目标**：在花掉一次昂贵仿真前，挡掉明显坏掉的 kernel（架构文档 §1.3、§3.6）。大 kernel 场景下，因低级错误浪费一次仿真是最亏的。

**动作**：实现 `control/presim_gate.py` 的通用检查：能否编译/语法合法、shape 与 dtype 是否自洽。**不做数值仿真。**

**槽位**：

```python
def check_extension_calls(kernel_src) -> list[str]:
    """槽位：检查 extension 原语调用是否合法。返回问题列表，空表示通过。
    由保密环境的 GLM 4.7 实现（需真实原语签名）。
    """
    return []  # 占位：保密环境实现前恒通过
```

**验收**：构造一个 shape 不自洽的 kernel，确认被挡下。

---

### T8. 改造两个 skill 正文

**通用要求**：SKILL.md **全部用英文**（沿用本仓已验证的约定：body 与 description 均英文，触发短语中可保留少量中文 token 如 `瓶颈分析`、`上板` 以提高中英混用指令的召回）。description 中用**文件态前置条件**区分各 skill，防 cross-trigger。

**T8a. `triton-plan` → `sim-analyze`（职责反转）**
- 删除：写抽象 7-engine DSL、跑 `simulator.py` 做解析式预测的全部步骤。
- 保留并迁移：`costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md` 中的分析方法论骨架——**关键路径优先**（只优化关键路径上的算子）、**主导代价归入小而可操作的分类**、**每类对应明确的优化杠杆**。这套骨架合理，只是数据来源从解析式模型换成真实仿真。
- 新职责：读 adapter 产出的 7 段摘要 + Verdict → 选优化杠杆 → 输出下一轮改进方向。**"瓶颈类别 → 杠杆"尽量做成查表**，减少弱模型的自由推理。

**T8b. `triton-gen` 改造**
- 目标产物：**真实 Triton + extension**，不再是 emulator 形态。
- 删除：正文中"baseline triton → emulator 反向映射"整段（`from common import tl`、去 `@triton.jit`、`tl.load(ptr, offsets, ...)` 逗号形式等）。
- 改写：原"NPU-compatible coding rules"改为 **extension 使用规则**，核心一条——**默认写标准 Triton，仅当 Verdict 的瓶颈类别明确要求时才引入对应 extension 原语**（架构文档 §4.4）。好处是基线永远是合法 vanilla Triton，extension 是局部叠加；即使 4.7 没吃透 extension，最坏情况也只是产出正确但未优化的代码，而非坏掉的代码。
- 产物形态：**多段式模块**（kernel / reference / 比对），不是裸 kernel（理由同 T6）。
- 增加：读取 `retrieved_experience`（memory inject 注入）作为生成参考——这条现有正文已有，保留。

**T8c. 新建 `extension-guide` skill**
- SKILL.md 正文：由你写完整——**按瓶颈类别索引原语**的组织方式（架构文档 §4.2），正文只放"原语名 + 所解决的瓶颈类别"的**短索引**，详细条目放 `references/`，按需读（渐进披露）。
- **必须写一份完整的样例条目**作模板：用公开类比（Triton-Ascend / torch-npu 这类公开扩展包的通用形态）写，字段为：名称 / 一句话语义 / 签名 / 解决哪个瓶颈类别 / 一个最小示例 / 常见坑。**这份样例对 4.7 极其重要——它照着填就行，不必自己设计组织方式。**
- 写一个格式校验脚本：每条的瓶颈类别必须在词表内。

**槽位**：`references/` 下的**真实原语内容**留空，只放你写的样例条目 + `TODO` 说明。

**验收**：`sim-analyze` 与 `triton-gen` 的 description 满足文件态前置条件、无 cross-trigger；`extension-guide` 的格式校验脚本能跑通样例条目。

---

### T9. OpenCode 配置

**目标**：目标环境是 OpenCode，需补两样 Claude Code 不需要的东西。

**动作**：
- `opencode.json`：OpenCode **会忽略** `disable-model-invocation` 字段（它只认 `name`/`description`/`license`/`compatibility`/`metadata`）。因此用 `permission.skill` 复刻治理，例如把重量级或高代价的 skill 设为 `"ask"`。已知 `.claude/skills/` 是 OpenCode 的合法项目级发现路径（本仓已实测触发成功），**skill 文件不需要移动**。
- `AGENTS.md`：OpenCode 的原生项目规则文件。镜像 `CLAUDE.md` 的行为准则 + landing，**并加入 T10 中给 4.7 的三条纪律**。
- 确保 skill 名字在所有搜索路径下唯一（不要在 `.opencode/skills/` 再放一份同名的，会撞名）。

**验收**：`opencode.json` 为合法 JSON；`AGENTS.md` 含三条纪律。

---

### T10. 编写《交接包》`HANDOFF_GLM47.md`（你的最后一项任务，也是最重要的一项）

**目标**：产出一份**给 GLM 4.7 直接执行**的文档。它不是本手册的副本——本手册是给你的，那份是给一个能力更弱、上下文更小的模型的。

**编写原则**（务必遵守）：
- **不要让 4.7 读架构文档或本手册**。它读不完，也不需要读。交接包必须**自包含且极短**。
- 每项任务写成**填空题 + 自检命令**，不写设计论证。
- 用中文，句子短，一项一段，避免嵌套条件。
- 明确标注每个槽位的**文件路径 + 函数签名 + 冻结的数据结构**。

**交接包必须包含以下五节**：

**① 五个槽位任务表**（这是 4.7 全部要动笔的地方）：

| 任务 | 文件 | 形式 | 自检命令 |
|---|---|---|---|
| 填词表内容 | `control/vocabulary.yaml` | 按真实 stall 类型补 5–8 条 | 跑一致性检查脚本 |
| 写 `parse_raw()` | `control/feedback_adapter.py` | 单函数，Event 字段已冻结 | 跑 adapter 测试，能出 7 段 + Verdict |
| 填 extension 速查表 | `.claude/skills/extension-guide/references/` | 照样例条目格式逐条填 | 跑格式校验，类别须在词表内 |
| 写 `launch()` | `control/launch_template.py` | 单函数，发射并取回 | 跑一个小 kernel，能拿到合法 SimResult |
| 加 extension 合法性检查 | `control/presim_gate.py` | 补 `check_extension_calls()` | 用故意写错的调用，确认被挡下 |

**② 三条纪律**（同时写进 `AGENTS.md`）：
1. 不改任何函数签名、schema、词表结构——那是非保密环境冻结的契约，只填内容。
2. 不做架构判断——遇到"这里该怎么设计"，停下来上报，不要自行发挥。
3. 每改一处，跑对应自检，通过再做下一处。

**③ 运行说明**：如何跑通一轮完整闭环（gen' → pre-sim gate → launch → adapter → sim-analyze → loop-controller 判停 → memory 写回），含具体命令。

**④ 验证清单**（4.7 完成适配后必须自证）：
1. 小 kernel 走通一轮完整闭环，产物为实测 cycles 最优的 kernel。
2. 停止逻辑三个用例：每轮 1% 改善 → 在 ε 处停；某轮 cycles 变大 → 回滚且返回历史最优版；连续 `correct=false` → 不参与改善计算但计入轮数。
3. 记忆冷启动价值：清空经验库跑一遍、再带经验跑一遍同类算子，确认第二次首轮 kernel 更优、仿真轮数更少。
4. 词表一致性：adapter 标签、memory fingerprint 键、速查表索引三者同源。

**⑤ 保密纪律**：4.7 填入的真实 stall 类型名、extension 原语名、仿真字段名**均不得回流到公开分支**。公开分支中这四处保持占位，测试靠合成夹具跑。

---

## 4. 槽位总表（你留、4.7 填，共五处）

| # | 文件 | 槽位 | 占位实现 |
|---|---|---|---|
| 1 | `control/vocabulary.yaml` | 词条内容 | 2–3 条通用示例 + TODO |
| 2 | `control/feedback_adapter.py` | `parse_raw()` | `raise NotImplementedError` |
| 3 | `.claude/skills/extension-guide/references/` | 原语内容 | 一份完整样例条目 + TODO |
| 4 | `control/launch_template.py` | `launch()` | `raise NotImplementedError` |
| 5 | `control/presim_gate.py` | `check_extension_calls()` | `return []`（恒通过） |

**除这五处外，其余一切由你完成。** 如果你在改造中发现某处"必须等 4.7 才能决定"，先停下来判断：这真的需要保密信息吗？还是可以用一个契约或参数把它推迟？**能推迟的都要推迟，槽位越少越好。**

---

## 5. 禁令（DO NOT）

- **不要修改 `costModel/`**（协作方仓库，只读）。其分析方法论可以继承，代码不要动。
- **不要**把 memory 模块做成 skill——它是确定性控制面，经 CLI 调用，延续"控制面与 LLM 生成 agent 分离、状态只走文件"的不变量。
- **不要**在骨架里写任何真实硬件/仿真的猜测内容。不知道就留槽位，不要臆造字段名、原语名、stall 类型。**臆造比留空危险得多**——4.7 会当成事实照做。
- **不要**让 4.7 承担需要跨文件推理或架构判断的工作。发现这类工作，说明骨架没搭到位，回来补。
- **不要**把停止判断、瓶颈归类交给 LLM 的散文判断——它们必须在确定性代码里，读机器可读的 Verdict。
- **不要**一次性删除旧资产而不留说明。`emulators/` 标记退役而非删除。

---

## 6. 完成定义（DoD）

- [ ] 分支 `local-adapt` 建立；T1 的退役与保留符合清单。
- [ ] 四份契约在 `control/contracts.py` 中定义并可校验；词表格式与一致性检查脚本可运行。
- [ ] loop-controller 全量实现，三个单元测试通过（ε 停、回退+best-so-far、FAIL 计数口径）。
- [ ] memory 演进完成（cycles 打分、`extension_used`、正确性前置闸门），单元测试通过。
- [ ] feedback-adapter 骨架完成；三份合成夹具可产出合法 7 段 + Verdict。
- [ ] 发射脚本模板、pre-sim gate 完成，槽位以标准占位实现留出。
- [ ] `sim-analyze`、`triton-gen`、`extension-guide` 三个 SKILL.md 完成，英文，description 无 cross-trigger；extension 样例条目已写。
- [ ] `opencode.json` 与 `AGENTS.md` 完成。
- [ ] **`HANDOFF_GLM47.md` 完成，含五节，自包含、极短、可由 GLM 4.7 直接执行。**
- [ ] 全部改动在 `local-adapt` 分支，可一键回滚；`git diff` 复核无意外改动（`costModel/` 无变更）。
- [ ] **在无任何保密信息的情况下，`tests/` 全部通过**——这是骨架是否扎实的最终判据。

---

## 7. 推进顺序建议

1. T1、T2（退役 + 契约地基）——后续一切依赖契约，最先冻结。
2. T3、T4（loop-controller + memory）——零保密依赖，纯收益，可完整验收。
3. T5（adapter 骨架 + 夹具）——消解"既需设计又需保密信息"的关键动作，做完后 4.7 的任务才真正降级为填空。
4. T6、T7（发射模板 + pre-sim gate）。
5. T8、T9（skill 正文 + OpenCode 配置）。
6. T10（交接包）——**最后写**，因为它要准确反映前面所有槽位的真实签名与路径。

---

## 8. 需要上报而非自行决定的情形

遇到以下情况，**停下来向用户确认，不要自行发挥**：

- 某个槽位似乎需要你猜测真实硬件语义才能设计接口。
- 词表的类别粒度无法在不知道真实 stall 类型的前提下确定（此时应把格式定死、内容留空，而非猜测条目）。
- 发现某项改造会破坏"状态只走文件、不进会话"的不变量。
- 发现契约需要变更，而变更会影响已冻结的四份之一。
