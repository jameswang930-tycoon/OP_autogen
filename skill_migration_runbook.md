# OP_autogen：将 slash command 迁移为 Agent Skill 的整改指导

> **执行者**：本地 Claude Code + GLM 5.2
> **目标仓库**：`OP_autogen`（`wsx` 分支）
> **交付物**：`.claude/skills/` 下 5 个 Skill，触发（trigger）由 `description` 治理；`docs/` 保持单一来源；`CLAUDE.md` 瘦身。
> **执行原则**：这是**机械迁移 + 声明式改写**，不是重写。除本文件明确指出的改动外，命令正文（body）逐字保留。**禁止发明**新的函数签名、DSL 语义、路径或流程细节——凡不确定的，读真实文件确认，不要臆造。
>
> **版本对齐**：本 runbook 已针对 `wsx` 分支 commit **`57913bb`**（"docs: memory_skeleton 并入 docs/project_knowledge/"）核验。该 commit 引入了确定性记忆模块（`memory/` 包 + `memory_cli.py`）及两份 memory 文档，本文已把相关影响并入（见 §2、§5、§6 及附录 A）。执行前先 `git log -1` 确认本地 HEAD 是否 = `57913bb`；若已前进，重点复查 `.plan.json` schema 与 `triton-gen` 正文是否又有变化。

---

## 0. 迁移前必读：为什么这么改（一句话）

当前 `.claude/commands/*.md` 是 **slash command（用户手动触发，`/triton-plan`）**；改成 **Skill（model-invoked，模型每轮读 `description` 自行判断是否触发）** 之后，"何时 trigger" 就由 `description` 这个声明式字段治理，而不再依赖用户记住流程顺序。**token 不是本次整改的动机**（实测 session 启动只自动加载 `CLAUDE.md` ≈ 1000 tokens，其余全部按需读），**trigger 机制**才是。

三级渐进披露（progressive disclosure）在迁移后依然成立：
1. 启动时只加载每个 skill 的 `name`+`description`（≈每个 ~100 tokens）；
2. 命中触发时才把该 `SKILL.md` 正文读入；
3. 正文引用的 `docs/*.md` 只在真正需要时才读。

**关键决策（本 runbook 采用）**：`docs/` **原地不动**，各 `SKILL.md` 用**仓库相对路径**引用它，而不是把 doc 拷进每个 skill 的 `references/`。原因：`test_conventions.md` / `emulator_error_coverage.md` / `emulator_improvements_done.md` / `plan_code_contract.md` 是**多个 skill 共享**的，拷贝会导致 drift（内容漂移）。相对路径引用同样享受"按需读"，token 经济性完全一致。Claude Code 对整个仓库有文件系统访问权，跨目录引用可正常读取。

---

## 1. 目标目录结构（迁移后）

```
.claude/
  skills/
    triton-plan/SKILL.md
    triton-gen/SKILL.md
    triton-verify/SKILL.md
    triton-fix/SKILL.md
    triton-convert/SKILL.md
  commands/                 # 迁移并验证通过后，在“单独一次 commit”里删除（见 §7）
docs/                       # 原地不动，作为 skill 的按需引用来源（single source of truth）
  project_knowledge/*.md
  emulator_observations/*.md
  dev_plan/*.md             # 开发【记录】，禁止被任何 SKILL.md 引用（见 §6 禁令）
CLAUDE.md                   # 瘦身（见 §5）
```

> ⚠️ 新建**顶层 skills 目录**后，Claude Code 需要**重启**才能开始 watch 它（这是它能被发现的前提）。对已存在 `SKILL.md` 的**后续编辑**则实时生效、无需重启。

---

## 2. 逐个 Skill 的改写规范

> **语言约定（重要）**：`SKILL.md` 的**全部内容用英文**——`description` frontmatter（本节给定）与正文 body 都是英文。原 command body 本就是全英文，"逐字保留"即自动保持英文；`CLAUDE.md` 也已是英文，无需翻译。**唯一例外**：`description` 触发短语中刻意保留的少量中文 token（如 `上板`/`瓶颈分析`/`修复`），用于提高中英混用指令的召回。本 runbook 文档本身（中文）只是给你/执行者看的操作说明，**不写进任何 SKILL.md**。

每个 skill 的改写 = **① 换 frontmatter（用本节给定的英文 description 逐字替换）** + **② 正文做机械改写（见 §2.1）** + **③ 修正引用路径**。正文的技术内容（步骤、DSL 示例、import 白名单、5 条改写规则等）**逐字保留原命令的英文内容，不得改写、翻译、精简或"优化"**。

### 2.1 正文的 3 处机械改写（对全部 5 个 skill 适用）

1. **`$ARGUMENTS` 处理**：`$ARGUMENTS` 是 slash command 的模板变量，在 Skill（model-invoked）里**不会被替换**。凡正文出现 `$ARGUMENTS`（如 `User input: $ARGUMENTS`、`Input: $ARGUMENTS (<op> name)`），改写为自然语言指示（**保持英文**，与正文一致）：
   > `Input: extract the operator name <op> (and any optional input type / feedback) from the user's request.`
   保留其后对 `<op>` 的一切引用不变。

2. **引用路径确认为仓库相对路径**：正文中所有 `docs/...`、`emulators/...`、`costModel/...` 引用**保持仓库根相对路径**（例如 ``docs/project_knowledge/input_detection.md``）。不要改成 `references/`，不要拷贝文件。

3. **Python 解释器一致性修正**：`plan` 与 `gen` 已明确"system `python3` 是 3.7，simulator 需 3.10+，必须用 `.venv/bin/python`"。`triton-verify` 与其它 stage 的 run 代码块若出现**裸 `python3`**，统一改为 `.venv/bin/python`（保持 `cd emulators && ../.venv/bin/python ...` 这种相对层级）。**这是修 bug，不是加特性**——仅改解释器路径，run 命令的其余部分不动。
   > 已核实（commit `57913bb`）：`triton-verify.md` 第 20 行 `cd emulators && python3 -c ...` 即为此处待修目标。

4. **（仅 triton-gen）保留 `retrieved_experience` 段**：`triton-gen.md` 正文 Step 1 现含一段 **"Optional `retrieved_experience`"**（说明 memory 模块经 `memory_cli.py inject <op>` 注入的历史经验、缺省则按无 memory 生成）。它是当前 body 的一部分，**必须原样保留**，不要因为"与 command→skill 迁移无关"而删除。

### 2.2 五个 Skill 的 `description`（英文，逐字替换，勿改）

> 设计说明（供执行者理解，不写进文件）：`description` **全部用英文**（触发原语更贴合模型语料）。它同时承担两件事——**触发短语略 "pushy"** 以对抗 skill 的欠触发（undertrigger），**同时用文件态前置条件**（`.plan.json` / `__init__.py` / `triton_real.py` 是否存在）把 5 个 stage 互相隔开、防 cross-trigger。触发短语里**刻意保留少量中文 token**（`上板`/`瓶颈分析`/`修复` 等），因为实际下指令常中英混用，可提高召回；这与"描述主体用英文"并不矛盾。

**`triton-plan/SKILL.md` frontmatter：**
```yaml
---
name: triton-plan
description: >
  Pipeline stage 1 (cost-model planning). Use when the user wants to "plan",
  "estimate cost", or do bottleneck analysis (瓶颈分析) for an operator that has
  NO emulators/test/<op>/.plan.json yet. Input may be NL / PyTorch / ONNX /
  baseline triton / a fixed shape. This skill writes a cost_emulator DSL program,
  runs the simulator directly (--verify + --llm --critical-path), and dumps the
  raw_llm output verbatim to .plan.json. It ONLY plans: it does NOT interpret
  raw_llm (that is triton-gen's job) and does NOT generate a kernel. Trigger this
  for any "plan / estimate / bottleneck / 瓶颈分析 / 规划" request even if the
  user does not literally say "plan".
---
```

**`triton-gen/SKILL.md` frontmatter：**
```yaml
---
name: triton-gen
description: >
  Pipeline stage 2 (emulator kernel generation). Use when
  emulators/test/<op>/.plan.json already exists and a kernel must be generated
  from it into emulators/test/<op>/__init__.py (4-part: kernel / emulate /
  reference / test); also use when the user pastes a baseline triton kernel to be
  turned into emulator form. After generating, run ONE inline verification and
  report PASS/FAIL only: do NOT enter a repair loop (that is triton-fix) and do
  NOT do real-triton conversion (that is triton-convert). Trigger for any
  "generate kernel / 生成算子 / turn the plan into a kernel" request.
---
```

**`triton-verify/SKILL.md` frontmatter：**
```yaml
---
name: triton-verify
description: >
  Pipeline verification stage (READ-ONLY). Use when
  emulators/test/<op>/__init__.py already exists and the user wants to
  "verify / check <op>", "is <op> correct", or "run <op>'s test" (跑测试). Runs
  run_with_feedback or the module's test() and reports PASS (with error
  magnitudes) or FAIL (with the deduplicated feedback string). This skill NEVER
  edits code, NEVER writes to disk, and NEVER repairs: for repair use triton-fix.
  Trigger for any "verify / check / is it correct / 是否正确 / run the test"
  request.
---
```

**`triton-fix/SKILL.md` frontmatter：**
```yaml
---
name: triton-fix
description: >
  Pipeline repair stage. Use after triton-verify or triton-gen reports FAIL and
  emulators/test/<op>/__init__.py exists and needs fixing. Runs an internal loop
  (max 5 rounds): classify the failure (EmulatorError / Shape / Numerical) → make
  the smallest fix → re-verify → until PASS or the budget is exhausted. Edits ONLY
  emulators/test/<op>/__init__.py. Trigger for any "fix / debug / 修复 / the op is
  wrong / results don't match" request. Note: the emulator has a ~30-40% silent
  numerical blind spot — do not loop forever on errors it cannot catch; surface
  them to the user instead.
---
```

**`triton-convert/SKILL.md` frontmatter：**
```yaml
---
name: triton-convert
description: >
  Pipeline final stage (emulator kernel → deployable real triton). PRECONDITION:
  emulators/test/<op>/__init__.py must already have PASSED verification. Use when
  the user explicitly asks to "convert to real triton / 上板 / generate
  triton_real". Applies 5 mechanical rewrites (the kernel's compute logic stays
  identical), then runs an NPU constraint self-check. If <op> has not PASSED yet,
  tell the user to verify/fix first and do NOT convert. This skill only does the
  mechanical rewrite plus warnings; it does NOT do two-level tiling.
---
```

### 2.3 每个 Skill 的引用清单（迁移后必须原样保留、路径正确）

以下是从真实命令正文中抽取的引用关系。改写后**逐条核对**，确保路径存在且未被误删：

| Skill | 读/写的真实产物 | 按需引用的 docs（仓库相对路径） | 外部/代码引用（原地读，勿拷贝） |
|---|---|---|---|
| triton-plan | 写 `emulators/test/<op>/.plan.json` = `{op,shapes,dtype,dsl,raw_llm}`（失败则 `{mock:true,...}`）；跑 `costModel/cost_emulator/simulator.py` | `docs/project_knowledge/input_detection.md` | `costModel/cost_emulator/Skills/bottleneck-analysis/SKILL.md`（协作方仓库，**只读，勿改**） |
| triton-gen | 读 `.plan.json`（含可选 `retrieved_experience`）→ 写 `emulators/test/<op>/__init__.py`；跑 inline test | `docs/emulator_observations/implementation_patterns.md`、`docs/project_knowledge/test_conventions.md`、（深引用）`docs/project_knowledge/plan_code_contract.md`；**仅当 `retrieved_experience` 存在时**才可再深读 `docs/project_knowledge/memory_integration.md`（229 行，勿无条件读入） | `emulators/common/__init__.py`（权威 `tl.*` 签名，原地读） |
| triton-verify | 读 `emulators/test/<op>/__init__.py`；`run_with_feedback` 返回 `{"passed","feedback",...}` | `docs/project_knowledge/emulator_improvements_done.md`、`docs/project_knowledge/emulator_error_coverage.md`、`docs/emulator_observations/{api_gaps,error_accumulation,precision_gaps,missing_coverage}.md`、`docs/project_knowledge/test_conventions.md` | — |
| triton-fix | 编辑 `emulators/test/<op>/__init__.py`；`run_with_feedback` | `docs/project_knowledge/emulator_error_coverage.md`、`docs/project_knowledge/emulator_improvements_done.md` | — |
| triton-convert | 读 `__init__.py` → 写 `emulators/test/<op>/triton_real.py` | `docs/project_knowledge/emulator_to_triton_conversion.md` | 样例真值：`emulators/test/{matmul,resnet18,resnet34,mobilenetv3_small}/triton_real.py`（原地读） |

> `docs/project_knowledge/{project_overview,emulator_next_steps}.md` **不归属任何 stage skill**——它们是 landing / roadmap，留在 `docs/`，由 `CLAUDE.md` 的落地段落轻量指向即可（见 §5）。

---

## 3. 触发治理（trigger governance）：model-invocation 策略

这是需要**产品负责人拍板**的一处策略。本 runbook 给出推荐默认值，执行者按此设置；若负责人另有决定，仅调整对应 frontmatter 字段。

| Skill | 推荐 model-invocation | 理由 |
|---|---|---|
| triton-plan | **允许**（默认，不加字段） | 入口；最坏情况只是多产出一个 `.plan.json`，代价低 |
| triton-gen | **允许**（默认） | description 已用"`.plan.json` 存在"做前置守卫 |
| triton-verify | **允许**（默认） | 只读，误触发无副作用；可选加 `allowed-tools` 限制为只读（见下） |
| triton-fix | **允许**（默认） | 只改 `__init__.py`，风险低。**注**：这是未来最先要隔离进 sub-agent 的 stage（修复循环噪声污染主 session），届时可改 `disable-model-invocation: true` 或交给子 agent 预加载 |
| triton-convert | **建议加 `disable-model-invocation: true`**（改为仅手动/显式调用） | 它产出的是**要上板实测**的 `triton_real.py`；在 verify 未 PASS 时误触发是最贵的错误，且 emulator 有 30–40% 盲区。让"上板转换"始终是一次**刻意的人工动作**更稳妥。若负责人希望它也能自动触发，则**不加**该字段，但必须依赖 description 里的"必须已 PASS"前置条件 |

**可选的字段（按需使用，勿臆造其它字段名）**：
- `disable-model-invocation: true` — 该 skill 不再被模型自动触发，只能显式按名调用。
- `allowed-tools:` — 限制该 skill 可用工具，实现最小权限。`triton-verify` 契合"绝不写盘"的约定，可限制为只读 + 运行命令、不含文件写/编辑工具。
- `paths:` — 用 glob 把 skill 限定到匹配文件才自动激活。本流水线以**算子名**驱动、非以文件类型驱动，`paths` 收益有限，**默认不设**。

> 有效的 frontmatter 字段仅限：`name`、`description`（必需）、`disable-model-invocation`、`allowed-tools`、`paths`。**不要**发明其它字段。

---

## 4. 防 cross-trigger 的核心机制（执行者须理解）

5 个 skill 的触发短语不可避免地都含"triton / kernel / 算子"。防止互相抢触发，靠的是 **description 里的文件态前置条件**，形成一条按产物推进的状态机：

```
无 .plan.json ─plan→ 有 .plan.json ─gen→ 有 __init__.py ─verify→ PASS ─convert→ triton_real.py
                                              │
                                          FAIL └─fix→ 回到 verify
```

改写完成后，务必**自检**：每个 description 是否都写清了"**在什么文件已存在 / 未存在时**才用我"。这一条比触发短语更能决定路由正确性。

---

## 5. `CLAUDE.md` 瘦身

当前 `CLAUDE.md` 末尾有一张"Project Knowledge / Owner skill / When to Read"表——它是**手动版**的渐进披露，靠模型"读表并听话"生效，是约定而非机制。知识归入各 skill 的引用后，这张表大部分冗余。

改法：
1. **删除**那张 owner-skill 映射表（每个 doc 何时读，已分散进各 `SKILL.md` 的正文引用里）。
2. **保留** `CLAUDE.md` 顶部的行为准则（Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution）——这些是全局行为，与 skill 无关。
3. 在末尾补**一小段 landing**（≤6 行），指向这几份"按需读"的顶层文档：
   - `docs/project_knowledge/project_overview.md`（项目结构与 5-skill 工作流）
   - `docs/project_knowledge/emulator_next_steps.md`（roadmap）
   - `docs/project_knowledge/memory_architecture.md`（记忆模块两层结构 experience/runlog、fingerprint、retrieve/record）
   - `docs/project_knowledge/memory_integration.md`（记忆模块如何接入流水线：inject/record hook、AB test）
   > 删除 owner 表时，表内新增的两行 memory 文档一并删除；它们的"何时读"由这段 landing 承接。

目标：`CLAUDE.md` 保持在行为准则 + 一段 landing，不再承担"路由 doc"的职责。

---

## 6. 禁令清单（DO NOT）

- ❌ **不要**把 `docs/*.md` 拷进任何 skill 的 `references/`（共享 doc 会 drift）。用仓库相对路径引用。
- ❌ **不要**让任何 `SKILL.md` 引用 `docs/dev_plan/*.md`（`resnet18_conv_dev_plan.md` ≈5.5K tokens、`vadd_pipeline_dev_record.md`）——它们是开发**记录/artifact**，不是流程知识，误引用即 context 污染源。
- ❌ **不要**修改 `costModel/cost_emulator/`（协作方仓库，只读）。
- ❌ **不要**改写、精简、"优化"命令正文里的技术内容（DSL 引擎表、import 白名单、NPU 编码规则、5 条转换规则、失败分类 A/B/C 等）——逐字保留。
- ❌ **不要**发明新的函数签名或路径。`run_with_feedback` 返回 `{"passed","feedback","details"}`；`.plan.json` 结构为 `{op,shapes,dtype,dsl,raw_llm}`，外加**可选** `mock`（simulator 失败时）与**可选** `retrieved_experience`（memory 模块 inject 后追加，非 `/triton-plan` 产出）——以真实文件为准，不确定就读 `emulators/common/__init__.py`、`docs/project_knowledge/plan_code_contract.md` 与既有 `.plan.json`。**注意 schema 不是闭集**：见到 `retrieved_experience` 不要当作多余字段剥除。
- ❌ **不要**在本步同时删除 `.claude/commands/`。先建 skills、验证通过，再单独一次 commit 删除（保证可回滚）。
- ❌ **不要**把 memory 模块 skill 化。`memory/` 包与 `memory_cli.py` 是**确定性控制面（deterministic control plane）**，经 CLI 手动/编排调用（`inject` 在 plan 后 gen 前、`record` 在 verify 后），**不是** LLM skill——这与项目已锁定的"控制面与 LLM 生成 agent 分离、状态只走文件"不变量一致。本次 5-skill 迁移**不触碰** `memory/`、`memory_cli.py`。
- ❌ **不要**让任何流水线 SKILL.md **无条件**引用 `docs/project_knowledge/memory_integration.md`（229 行）或 `memory_architecture.md`。仅 `triton-gen` 在 `.plan.json` 里**确实出现 `retrieved_experience` 时**才可深读 `memory_integration.md`；两份 memory 文档整体属于 landing 层（见 §5），不归属任何 stage skill。

---

## 7. 分步执行顺序

```
1. git checkout -b skillify   # 单独分支，便于回滚
2. mkdir -p .claude/skills/{triton-plan,triton-gen,triton-verify,triton-fix,triton-convert}
3. 对 5 个 stage 逐个：
     - 新建 .claude/skills/<name>/SKILL.md
     - frontmatter 用 §2.2 给定文本逐字写入
     - body 从对应 .claude/commands/<name>.md 拷入，应用 §2.1 的 3 处机械改写
     - 按 §2.3 核对引用路径
     - 按 §3 决定是否加 disable-model-invocation / allowed-tools
4. 按 §5 瘦身 CLAUDE.md
5. 【暂不删除 commands/】
6. 重启 Claude Code（让新 skills 目录被 watch）
7. 执行 §8 全部校验
8. 全绿后：git rm -r .claude/commands/  （单独一次 commit）
9. 再跑一遍 §8.3 端到端，确认删除 commands 后 skill 路径仍正常
```

---

## 8. 校验：怎么确认整改生效

> 校验分四层：**结构 → 触发 → 功能端到端 → 回归**。每层给出可判定的通过标准。

### 8.1 结构校验（脚本可判定）

对 5 个 `SKILL.md` 逐个确认：
- [ ] 文件存在于 `.claude/skills/<name>/SKILL.md`。
- [ ] frontmatter 是合法 YAML，含 `name` 与 `description` 两个必需字段；`name` 与目录名一致。
- [ ] 正文**无残留 `$ARGUMENTS`** 字面量。
- [ ] 正文行数 < 500（现有 body 都在 38–78 行，天然满足；若超说明误把 doc 塞进了 body）。
- [ ] 正文中出现的每个 `docs/...`、`emulators/...`、`costModel/...` 路径**真实存在**（逐条 `test -f`）。
- [ ] 无任何 `SKILL.md` 引用 `docs/dev_plan/`。
- [ ] `triton-verify` 与其它 stage 的 run 代码块已无裸 `python3`（统一 `.venv/bin/python`）。

参考脚本（在仓库根运行）：
```bash
for d in .claude/skills/*/; do
  f="$d/SKILL.md"
  echo "== $f =="
  grep -q '^name:' "$f" && grep -q '^description:' "$f" && echo "  frontmatter OK" || echo "  !! frontmatter MISSING"
  grep -q '\$ARGUMENTS' "$f" && echo "  !! \$ARGUMENTS leftover" || echo "  no \$ARGUMENTS OK"
  grep -q 'dev_plan' "$f" && echo "  !! references dev_plan (forbidden)" || echo "  no dev_plan OK"
  echo "  lines: $(wc -l < "$f")"
  # 引用路径存在性
  grep -oE '(docs|emulators|costModel)/[A-Za-z0-9_/.-]+\.(md|py|json)' "$f" | sort -u | while read p; do
    [ -e "$p" ] && echo "    ref OK: $p" || echo "    !! ref MISSING: $p"
  done
done
# 裸 python3 检查（应无输出；.venv/bin/python 不算）
grep -rn 'python3' .claude/skills/ | grep -v '\.venv/bin/python' || echo "no bare python3 OK"
```

### 8.2 触发校验（trigger matrix，最关键）

**先重启 Claude Code**，然后：
- [ ] 运行 `/skills`，确认 5 个 skill 全部被发现并列出。
- [ ] 若某 skill 不出现或不触发，运行 `/doctor` 诊断（它会报告 description 预算溢出、关键词被静默丢弃等问题）。

用下面这张矩阵做**行为测试**：给出 prompt，观察是否触发了**期望的 skill**、且**没有误触发其它 skill**。执行前先准备一个测试算子的中间态（例如让某个 `<op>` 只到 `.plan.json` 阶段、另一个已有 `__init__.py`），以便前置条件生效。

| 输入 prompt（自然语言，不敲 `/`） | 期望触发 | 期望**不**触发 |
|---|---|---|
| "帮我为 softmax 做个瓶颈分析和 plan"（无 .plan.json） | triton-plan | gen / verify / convert |
| "softmax 的 plan 有了，生成 kernel"（.plan.json 存在） | triton-gen | plan / convert |
| "检查一下 softmax 算得对不对"（__init__.py 存在） | triton-verify | fix / convert |
| "softmax 跑挂了，结果不对，修一下" | triton-fix | verify（只读，不应改代码）/ convert |
| "softmax 验过了，转成真实 triton 准备上板" | triton-convert（若设了 disable-model-invocation，则应提示需显式调用） | 其余 |
| "softmax 还没验过，直接上板"（未 PASS） | 期望被 convert 的前置条件拦下、提示先 verify | convert 直接执行 |

通过标准：**每行的"期望触发"命中、"期望不触发"均未命中**；最后一行能被前置条件正确拦截。若出现 cross-trigger，按 §4 收紧对应 description 的**文件态前置条件**（而非仅调触发短语），改完实时生效、无需重启。

### 8.3 功能端到端校验（用文档化的简单算子 vadd）

`vadd` 是仓库里已有开发记录的最简算子，适合做冒烟测试。全程走一遍：
```bash
# 1) plan：应产出 emulators/test/vadd/.plan.json，含 raw_llm 的 7 个 section
.venv/bin/python costModel/cost_emulator/simulator.py --verify "<vadd DSL>"
.venv/bin/python costModel/cost_emulator/simulator.py --llm --critical-path "<vadd DSL>"
# 2) gen：应产出 emulators/test/vadd/__init__.py（四段式）
# 3) verify：应报 PASS + 误差量级
cd emulators && ../.venv/bin/python -c "from test.vadd import test; test()"
# 4) convert：应产出 emulators/test/vadd/triton_real.py，并跑 NPU 约束自检
```
通过标准：
- [ ] `.plan.json` 至少含 `{op,shapes,dtype,dsl,raw_llm}`（非 mock），`raw_llm` 含 7 段（execution summary / time breakdown / per-op / engine util / bandwidth util / parallelism / critical path）；若 memory 已接入并跑过 `inject`，可能**额外**含 `retrieved_experience`——此为合法可选字段，存在即通过，不得视为异常。
- [ ] `__init__.py` 四段齐全（kernel/emulate/reference/test），import 只用白名单。
- [ ] verify 报 PASS 并打印 `max_abs_err` / `max_rel_err`。
- [ ] `triton_real.py` 生成，5 条机械改写就位，NPU 自检（grid_size ≤ 65535 / 对齐 / UB ≤ 192KB）无阻断性告警。
- [ ] **提醒**：emulator PASS ≠ 正确（30–40% 静默数值盲区）；上板实测仍是正确性经验的最终准入，A/B 不得只看 emulator pass rate。

### 8.4 回归校验

- [ ] 删除 `.claude/commands/` 后（§7 第 8 步），重跑 §8.3 端到端，确认无路径断裂。
- [ ] `git diff` 复核：改动只落在 `.claude/skills/`（新增）、`.claude/commands/`（删除）、`CLAUDE.md`（瘦身）三处；`docs/`、`emulators/`、`costModel/` 无意外改动。

---

## 9. 最终验收清单（Definition of Done）

- [ ] `.claude/skills/` 下 5 个 SKILL.md，frontmatter 合法、description 为 §2.2 文本、无 `$ARGUMENTS` 残留。
- [ ] 所有引用路径真实存在；无 skill 引用 `dev_plan/`；无裸 `python3`。
- [ ] model-invocation 策略按 §3 设定（尤其 convert 的处理已确认）。
- [ ] `CLAUDE.md` 已瘦身，owner-skill 表已删，仅余行为准则 + landing。
- [ ] `/skills` 列出全部 5 个；`/doctor` 无 trigger 告警。
- [ ] §8.2 触发矩阵全部符合预期，无 cross-trigger。
- [ ] `vadd` 端到端 plan→gen→verify→convert 走通，产物与判定标准一致。
- [ ] `.claude/commands/` 已在单独 commit 删除，删除后回归通过。
- [ ] 全部改动在 `skillify` 分支，可一键回滚。

---

## 附录 A：与 memory-driven 记忆架构的衔接

> 状态更新（commit `57913bb`）：记忆模块**已部分落地**——`memory/` 包 + `memory_cli.py` 已在仓库，且落成了**确定性 CLI**（`inject` 在 plan 后 gen 前、`record` 在 verify 后），与"控制面与 LLM 生成 agent 分离、状态只走文件"的不变量一致。以下衔接点因此从"设想"变为"已印证方向"：

- 记忆读写已经由 `memory_cli.py` 封装 memory 包纯函数对外，模型侧只看到 `.plan.json` 里被追加的 `retrieved_experience`（精简经验文本），而非把整个 experience/runlog 读进 context——正是"脚本执行不读入 context"的渐进披露。**本次迁移不动这套 CLI**（见 §6 禁令）。
- 本次 skill 化仍是后续 sub-agent 化的前置铺垫：三个触发信号里"`triton-fix` 修复循环噪声污染主 session"对应的隔离，靠把 `triton-fix` 交给子 agent 预加载即可实现，不动主流水线。
- 因此：**先 skill 化流水线（本次），memory CLI 保持确定性控制面（已落地），二者正交推进——skill 化本身就为 sub-agent 化铺好了轨道。**
