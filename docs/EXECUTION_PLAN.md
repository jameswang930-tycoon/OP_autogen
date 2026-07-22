# 自动执行计划（EXECUTION_PLAN）

> 本文件与《执行手册》(`docs/glm52_execution_runbook.md`) 配套：手册说**每个任务怎么做**，本文件说**按什么顺序自动跑、每步怎么自证、什么时候必须停**。
> 使用方式：放进仓库根，GLM 5.2 全程读它并**实时更新 §4 的进度表**。

---

## 1. 自动化边界（先明确）

自动执行的前提是**可机器验证**。按此把 10 个任务分三类：

| 类别 | 任务 | 能否自动 | 判定方式 |
|---|---|---|---|
| **A. 全自动** | T1–T7 | ✅ 连续跑 | 门禁命令返回 0 即通过 |
| **B. 自动执行 + 人工复核** | T8、T9 | ⚠️ 跑完停下 | 结构可自动查，**内容质量需人看** |
| **C. 必须人工复核** | T10 | ⛔ 跑完必停 | 交接包是给弱模型的唯一依据，无法自动判定可读性 |

**为什么 T8/T10 不能全自动**：SKILL.md 的 description 是否会 cross-trigger、交接包对 GLM 4.7 是否真的"自包含且极短"，都没有可执行的判据。这类质量问题在保密环境才暴露的话，返工成本极高。

---

## 2. 自动执行的三个前提

**① 进度文件（`PROGRESS.md`）**——GLM 5.2 每完成一个任务就更新它。作用是**上下文耗尽或会话中断后能续跑**：新会话只需读 `PROGRESS.md` 就知道做到哪、下一步是什么，不依赖会话记忆（延续"状态只走文件、不进会话"的不变量）。

**② 每任务门禁（gate）**——每个任务结束时跑一条命令，返回 0 才算完成。**门禁不过不许进入下一任务。** 门禁失败时的处理协议见 §5。

**③ 检查点提交（checkpoint commit）**——每个任务门禁通过后立即 `git commit`，消息格式 `T<n>: <任务名> [gate passed]`。作用是任一环节出错可精确回滚到上一个已验证状态，不必从头再来。

---

## 3. 完整任务计划

| 任务 | 依赖 | 产出 | 门禁命令 | 复核 |
|---|---|---|---|---|
| **T1** 退役与保留 | — | skill 目录清理、`emulators/README.md` 退役说明 | `pytest tests/test_t1_structure.py` | 自动 |
| **T2** 契约 + 词表注册表 | T1 | `control/contracts.py`、`vocabulary.yaml`、`vocabulary.py`、一致性检查脚本 | `pytest tests/test_t2_contracts.py` | 自动 |
| **T3** loop-controller | T2 | `control/loop_controller.py` | `pytest tests/test_t3_controller.py` | 自动 |
| **T4** memory 演进 | T2 | `memory/` 改造、`memory_cli.py` 增 `--cycles` | `pytest tests/test_t4_memory.py` | 自动 |
| **T5** adapter 骨架 + 夹具 | T2 | `control/feedback_adapter.py`、`control/fixtures/` | `pytest tests/test_t5_adapter.py` | 自动 |
| **T6** 发射脚本模板 | T2,T5 | `control/launch_template.py` | `pytest tests/test_t6_launch.py` | 自动 |
| **T7** pre-sim gate | T2 | `control/presim_gate.py` | `pytest tests/test_t7_gate.py` | 自动 |
| **T8** 三个 skill 正文 | T2,T5 | `sim-analyze`、`triton-gen`、`extension-guide` | `pytest tests/test_t8_skills.py` | ⚠️ **人工** |
| **T9** OpenCode 配置 | T8 | `opencode.json`、`AGENTS.md` | `pytest tests/test_t9_config.py` | ⚠️ **人工** |
| **T10** 交接包 | 全部 | `HANDOFF_GLM47.md` | `pytest tests/test_t10_handoff.py` | ⛔ **必须人工** |

**门禁的测试由 GLM 5.2 自己编写**（手册各任务的"验收"段已给出要测什么）。要求：测试**先于或同步于**实现落地，不允许"先写完实现再补测试"——后者容易写出只验证自己实现的空测试。

### 各门禁必须覆盖的内容

- **T1**：`.claude/skills/` 下只剩三个 skill；`triton-convert/verify/fix` 已删；`costModel/` 无改动。
- **T2**：四份契约可校验；合法数据通过、非法数据报错；词表中不存在的标签被拒绝。
- **T3**：三个用例——① 每轮仅 1% 改善 → 在 ε 处停而非跑满 N；② 第 3 轮 cycles 变大 → 回滚且**最终返回历史最优版**；③ 连续 `correct=false` → 不参与改善计算但计入轮数。
- **T4**：同 fingerprint 三次尝试（第二次刷新最优、第三次未刷新）→ `score()` 只在第二次上升；`correct=false` 不影响 score 但出现在 runlog。
- **T5**：三份夹具（compute-bound / 传输欠填充 / 依赖 stall）均产出合法 7 段 + Verdict；输出体量在上限内；`parse_raw()` 保持 `NotImplementedError`。
- **T6**：用假 `launch`（返回夹具）跑通全链路 → 合法 `SimResult`；`launch()` 保持 `NotImplementedError`。
- **T7**：shape 不自洽的 kernel 被挡下；`check_extension_calls()` 保持恒通过占位。
- **T8**：三个 SKILL.md 的 frontmatter 合法、`name` 与目录一致、无 `$ARGUMENTS` 残留、正文引用路径真实存在、正文为英文、行数 < 500。（**注意：这些只是结构检查，description 是否 cross-trigger 需人工判断**）
- **T9**：`opencode.json` 是合法 JSON 且含 `permission.skill`；`AGENTS.md` 含三条纪律。
- **T10**：`HANDOFF_GLM47.md` 五节齐全；**五个槽位的文件路径与函数签名与 `control/` 下实际代码逐一对得上**（这条必须是自动检查，签名对不上是最常见也最致命的错误）。

### 全局门禁（每个检查点都要跑）

```bash
.venv/bin/python -m pytest tests/ -v      # 全部测试
git diff --stat HEAD~1                     # 复核改动范围，确认 costModel/ 无变更
```

**最终硬判据**：在**没有任何保密信息**的情况下 `tests/` 全绿。若某测试必须等真实数据才能跑，说明合成夹具没做到位，回去补。

---

## 4. 进度文件模板（`PROGRESS.md`）

GLM 5.2 在仓库根创建并实时更新：

```markdown
# 改造进度

分支: local-adapt    基线 commit: 79b4da1

| 任务 | 状态 | 门禁 | checkpoint commit | 备注 |
|---|---|---|---|---|
| T1 | ⬜ 未开始 | — | — | |
| T2 | ⬜ | — | — | |
| T3 | ⬜ | — | — | |
| T4 | ⬜ | — | — | |
| T5 | ⬜ | — | — | |
| T6 | ⬜ | — | — | |
| T7 | ⬜ | — | — | |
| T8 | ⬜ | — | — | 需人工复核 |
| T9 | ⬜ | — | — | 需人工复核 |
| T10 | ⬜ | — | — | 必须人工复核 |

状态: ⬜未开始 / 🔄进行中 / ✅门禁通过 / ⛔阻塞待确认

## 待确认事项
（GLM 5.2 遇到需上报的问题时记在这里，不要自行发挥）
```

---

## 5. 运行协议

### 5.1 三个强制停止点

```
T1 ─ T2 ─ T3 ─ T4 ─ T5 ─ T6 ─ T7  ║停止点①║  T8 ─ T9  ║停止点②║  T10  ║停止点③║
        （连续自动执行，门禁逐个把关）      （人工复核）        （人工复核）
```

- **停止点①（T7 后）**：全部代码骨架完成。你复核测试是否真的在测有意义的东西（不是自证式空测试）。
- **停止点②（T9 后）**：skill 正文与配置完成。你**必须人工读一遍三个 description**，判断是否会 cross-trigger、extension 样例条目是否足够清晰到能让 4.7 照着填。
- **停止点③（T10 后）**：交接包完成。**这是最关键的复核**——它是 GLM 4.7 的唯一依据。

### 5.2 门禁失败的处理协议

**最多自修 2 次，之后必停。** 具体：

1. 门禁失败 → 读错误、修复、重跑。
2. 第 2 次仍失败 → 再修一次。
3. 第 3 次仍失败 → **停止，在 `PROGRESS.md` 的"待确认事项"写明失败任务、错误信息、已尝试的两种修法，等待人工介入。**

不许无限重试，也不许为了让门禁通过而放宽测试断言——**改测试让它通过是最严重的违规**（这等于自己给自己发合格证）。

> 这套"有限重试 + 明确停止条件 + 不许放宽判据"的结构，与你让它实现的 loop-controller 是同一个设计思路。

### 5.3 续跑协议（上下文耗尽或会话中断）

新会话开场只需：

> 读 `PROGRESS.md` 和 `docs/execution_runbook.md`，从第一个非 ✅ 的任务继续。

因为状态全在文件里，不依赖会话记忆。**这也是为什么每个任务必须提交 checkpoint commit**——续跑时 `git log` 就是可信的进度真相。

---

## 6. 三个高风险点与对策

| 风险 | 后果 | 对策 |
|---|---|---|
| **臆造硬件细节** | 4.7 会当成事实照做，错误直达保密环境 | 门禁中加静态检查：`control/` 与 skill 正文里出现的 stall 类型、原语名，必须**要么在词表/样例中、要么是显式 TODO**。发现疑似臆造 → 停止上报 |
| **T2 契约有细微错误** | 后续 8 个任务全建在错误地基上 | 停止点①之前，T2 的门禁要格外严；契约一旦冻结，后续任务不得修改（要改必须回退重来） |
| **写出自证式空测试** | 门禁全绿但实际没验证任何东西 | 停止点①由你人工抽查 2–3 个测试；要求测试**先于实现**落地 |

---

## 7. 给用户的操作清单

**首次启动**（开发机仓库根，Claude Code）：

> 读 `EXECUTION_PLAN.md` 和 `docs/execution_runbook.md`。
> 建分支 `local-adapt`，创建 `PROGRESS.md`，然后连续执行 T1 到 T7。
> 每个任务：先写测试 → 实现 → 跑门禁 → 通过后 `git commit` → 更新 `PROGRESS.md` → 进入下一任务。
> 门禁连续失败 3 次则停止上报，不要放宽测试断言。
> **完成 T7 后停下，不要继续 T8。**

**停止点①之后**：

> 继续执行 T8、T9，完成后停下等待复核。

**停止点②之后**：

> 重新扫一遍 `control/` 下的实际代码（确认五个槽位的真实路径与函数签名），然后执行 T10 编写 `HANDOFF_GLM47.md`。完成后停下。

**每个停止点你要做的**：
1. 跑 `.venv/bin/python -m pytest tests/ -v` 确认全绿
2. 看 `PROGRESS.md` 的"待确认事项"有没有内容
3. 按 §6 的对策做对应抽查
