# Skill 迁移：重启后「验证生效 + 续作」指南

> **本文档用途**：重启 Claude Code 之后，把这份文档交给 Claude（或自己照做）——
> 先做 **§1 判断是否生效**（全绿才往下），再按 **§2 续作**把剩余任务做完。
> 配套权威规范：`skill_migration_runbook.md`（不要改它，以它为准）。
> 状态日期：2026-07-20 · 分支：`wsx` · **已完成并提交**（删 commands = commit `40f8fea`；skills + CLAUDE.md + 本文档 = 后续 commit）。

---

## 0. 一句话现状

✅ **迁移已完成并验证通过**（2026-07-20）。5 个 slash command 迁为 Agent Skill，commands 已删（commit `40f8fea`），skills + CLAUDE.md + 本文档已提交。验证链：§1 生效 ✅ / §2.2 `vadd_fp16` verify PASS ✅ / §2.1 真触发实测通过（`softmax` verify，2026-07-20）✅ / §2.3 删 commands ✅。正文 §1–§4 保留作过程记录。

---

## 1. 判断是否生效（重启后先做这一节）

### 1.1 文件层校验（确定性，不依赖模型 —— 最可靠的判定）

在仓库根跑这段脚本，期望**全部绿**：

```bash
for d in .claude/skills/*/; do
  f="$d/SKILL.md"
  echo "== $f =="
  grep -q '^name:' "$f" && grep -q '^description:' "$f" && echo "  frontmatter OK" || echo "  !! frontmatter MISSING"
  grep -q '\$ARGUMENTS' "$f" && echo "  !! \$ARGUMENTS leftover" || echo "  no \$ARGUMENTS OK"
  grep -q 'dev_plan' "$f" && echo "  !! references dev_plan (forbidden)" || echo "  no dev_plan OK"
  echo "  lines: $(wc -l < "$f")"
  grep -oE '(docs|emulators|costModel)/[A-Za-z0-9_/.-]+\.(md|py|json)' "$f" | sort -u | while read p; do
    [ -e "$p" ] && echo "    ref OK: $p" || echo "    !! ref MISSING: $p"
  done
done
echo "=== bare python3 check (唯一允许命中是 triton-plan 里那条注释) ==="
grep -rn 'python3' .claude/skills/ | grep -v '\.venv/bin/python' || echo "no bare python3 OK"
```

**通过标准**：5 个文件 frontmatter OK / 无 `$ARGUMENTS` / 无 `dev_plan` / 行数 < 500 / 所有 ref OK / 裸 `python3` 只剩 triton-plan 第 41 行那条**注释**（`# ...system python3 is 3.7, so use .venv`，非执行命令）。

### 1.2 harness 层校验（重启后才看得到）

- **`/skills`** 应列出 5 个；其中 `triton-plan`/`gen`/`verify`/`fix` 显示的是**新**英文 description（含文件态前置条件："Use when ... has NO .plan.json yet" / "...already exists" 等）。
- **`/doctor`** 无 trigger 告警（描述预算溢出 / 关键词被静默丢弃等）。
- **`triton-convert` 的 `disable-model-invocation: true` 表现**：convert 不应被模型自动触发——自然语言说"转成真实 triton/上板"时，模型应**提示需显式调用**，而不是自动 fire。
- **过渡期重复（正常，别慌）**：因为 `.claude/commands/` 还在，plan/gen/verify/fix 在列表里可能出现**新旧两条**。这是 §7 预期的中间态，删 commands（§2.3）后只剩新的 5 个。

> 判定结论：§1.1 全绿 ⇒ 文件正确无误；§1.2 新 description 出现 + convert 不自动 fire ⇒ 迁移已生效。两者都满足即可进入 §2。

---

## 2. 续作（剩余任务，按顺序）

### 2.1 §8.2 触发矩阵（最关键）

先准备**处于不同阶段**的测试算子（一个只有 `.plan.json`、一个已有 `__init__.py`），让 description 里的文件态前置条件能生效。然后逐行测，期望只命中"期望触发"、不误触发其它：

| 输入 prompt（自然语言，不敲 `/`） | 期望触发 | 期望**不**触发 |
|---|---|---|
| "帮我为 softmax 做个瓶颈分析和 plan"（无 .plan.json） | triton-plan | gen / verify / convert |
| "softmax 的 plan 有了，生成 kernel"（.plan.json 存在） | triton-gen | plan / convert |
| "检查一下 softmax 算得对不对"（__init__.py 存在） | triton-verify | fix / convert |
| "softmax 跑挂了，结果不对，修一下" | triton-fix | verify（只读不改码）/ convert |
| "softmax 验过了，转成真实 triton 准备上板" | convert 应**提示显式调用**（非自动 fire） | 其余 |
| "softmax 还没验过，直接上板"（未 PASS） | 被 convert 前置条件拦下、提示先 verify | convert 直接执行 |

出现 cross-trigger 时，按 runbook §4 收紧对应 description 的**文件态前置条件**（而非只调触发短语），改完实时生效、无需再重启。

### 2.2 §8.3 功能端到端（算子 = **`vadd_fp16`**，不是 `vadd`）

> runbook §8.3 写的 `vadd` 是笔误，仓库里实际叫 `vadd_fp16`。

走一遍 plan→gen→verify→convert，核对产物：
- [ ] `emulators/test/vadd_fp16/.plan.json` 含 5 字段 `{op,shapes,dtype,dsl,raw_llm}`（非 mock），`raw_llm` 含 7 段；若跑过 memory `inject` 可能**额外**含 `retrieved_experience`（合法可选，勿当异常）。
- [ ] `__init__.py` 四段齐全（kernel/emulate/reference/test），import 只用白名单。
- [ ] verify 报 PASS 并打印 `max_abs_err` / `max_rel_err`。
- [ ] `triton_real.py` 生成，5 条机械改写就位，NPU 自检（grid_size ≤ 65535 / 对齐 / UB ≤ 192KB）无阻断告警。
- [ ] **提醒**：emulator PASS ≠ 正确（30–40% 静默盲区），上板实测才是最终准入。

### 2.3 §8.4 回归 + 删 commands（全绿后才做）

- [ ] `git rm -r .claude/commands/`（**单独一次 commit**，保证可回滚）。
- [ ] 删除后重跑 §2.2 端到端，确认 skill 路径无断裂、不再有新旧重复。
- [ ] `git diff` 复核：改动只在 `.claude/skills/`（新增）、`.claude/commands/`（删除）、`CLAUDE.md`（瘦身）三处；`docs/`、`emulators/`、`costModel/` 无意外改动。

### 2.4（可选）提交

若用户要求提交：建议把 `.claude/skills/`、`CLAUDE.md`、以及 `skill_migration_runbook.md` + 本文档一起纳入。是否切 `skillify` 分支由用户定（目前留在 `wsx`）。

---

## 3. 续作所需上下文（给新会话，避免重新摸索）

**已锁定决策**：
1. `SKILL.md` 全英文（description + body）；description 用 runbook §2.2 英文文本逐字写入，仅在触发短语保留少量中文 token（`上板`/`瓶颈分析`/`修复`/`是否正确`/`生成算子`/`跑测试`/`规划`）。
2. `triton-convert` 加了 `disable-model-invocation: true`。

**契约（已核实，勿臆造）**：
- `run_with_feedback(emulate_fn, reference_fn, op_name="unknown", rtol=1e-3, atol=1e-5) -> dict`，返回 `{"passed","feedback","details"}`。定义于 `emulators/common/__init__.py:1027`。
- `.plan.json` = `{op,shapes,dtype,dsl,raw_llm}`，外加**可选** `mock`（simulator 失败）、**可选** `retrieved_experience`/`retrieved_ids`（`memory_cli.py inject` 追加）。**schema 非闭集，勿剥除**这些可选字段。

**关键事实**：
- `.venv/bin/python` = CPython **3.13.13**（uv 管理）；系统 `python3` 为 3.7，simulator 需 3.10+，故 run 命令一律 `.venv/bin/python`（`cd emulators && ../.venv/bin/python ...`）。
- 本仓库之前会话里的 5 个"可用 skill"其实是 `.claude/commands/*.md` 被直接呈现；**无外部 `~/.claude/skills/`**，新建 `.claude/skills/` 不存在命名冲突。
- `memory/` 包 + `memory_cli.py`（`inject`/`record`/`add`/`stats`）是**确定性控制面，本次不触碰、不 skill 化**（runbook §6）。
- 唯一既有 `.plan.json`（`emulators/test/vadd_fp16/.plan.json`）带 schema drift（多余 `supported`/`tile`/`plan`），gen 会忽略——新 plan 仍按 5 字段写。

---

## 4. 已完成清单（备查）

- [x] 5 个 `.claude/skills/<name>/SKILL.md` 建好（frontmatter 英文 + body 逐字拷自命令 + 3 处机械改写）。
- [x] 3 处改写：`$ARGUMENTS`→自然语言；路径保持仓库相对；`triton-verify` 裸 `python3`→`.venv/bin/python`。
- [x] 保留 `triton-gen` 的 "Optional `retrieved_experience`" 段。
- [x] 额外修一处路径：`triton-fix` 裸 `emulator_error_coverage.md` → `docs/project_knowledge/emulator_error_coverage.md`。
- [x] `CLAUDE.md` 瘦身：删 owner-skill 表，留 4 条行为准则 + 4 行 landing。
- [x] `.claude/commands/` 保留未删（待 §2.3）。
- [x] §8.1 文件层校验通过（§1.1 脚本全绿）。

**迁移完成时 `git status`（历史快照）**：本批 commit 后，`M CLAUDE.md` / `?? .claude/skills/` / `?? skill_migration_*.md` 全部清零。

---

## 5. 完成记录（2026-07-20）

迁移验证链全部闭合，skill 迁移完成：

- [x] §1 生效（文件层 §1.1 全绿 + harness 层新英文 description + convert `disable-model-invocation`）
- [x] §2.2 端到端：`vadd_fp16` `.plan.json` 5 字段（+ 已知 drift）/ `__init__.py` 四段 / verify PASS / `triton_real.py` 5 改写 + NPU 合规
- [x] §2.1 触发矩阵：静态论证 + **真触发实测**（新窗口发「检查一下 softmax 算得对不对」→ 命中 `triton-verify`、四个 case PASS、未误触 plan/gen、convert 仅建议显式调用）
- [x] §2.3 删 commands（commit `40f8fea`，删后 skill 列表即时刷新、无新旧重复）

**遗留（非阻断）**：`run_with_feedback` 无参闭包契约与 gen 产物 `emulate_X(x,...)` 系统性不兼容 → triton-verify 偏好路径对全算子 TypeError、实际走 fallback `test()`，triton-fix 的 feedback 链路受影响。非迁移引入（commands 时代既有），本次搁置。
