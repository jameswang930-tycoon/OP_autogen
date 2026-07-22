# AGENTS.md

OpenCode 的原生项目规则文件（镜像 `CLAUDE.md` 的行为准则 + landing，并加上给保密环境
GLM 4.7 的三条纪律）。Claude Code 读 `CLAUDE.md`；OpenCode 读本文件。两者内容保持一致。

## 给保密环境 GLM 4.7 的三条纪律（最高优先级，先读这节）

你在保密环境接手一个**非保密环境已搭好的骨架**。只填内容，绝不改形状。

1. **不改任何函数签名、schema、词表结构**——那是非保密环境冻结的契约（见
   `control/contracts.py`、`control/vocabulary.yaml`、`memory/schema.py`），只填内容。
2. **不做架构判断**——遇到"这里该怎么设计"，停下来上报，不要自行发挥。
3. **每改一处，跑对应自检**，通过再做下一处（自检命令见各槽位与 `HANDOFF_GLM47.md`）。

## 行为准则（镜像 CLAUDE.md）

**先想再写。** 别假设、别藏起困惑、把权衡摆出来。多种解读就列出来，别默默挑一个；
有更简单的做法就说；不清楚就停下问。

**简单优先。** 用解决问题的最少代码；不为单次用途造抽象；不加未要求的"灵活性"。
若 200 行能压成 50 行，重写。

**外科手术式改动。** 只动该动的；不"顺手改进"相邻代码；匹配既有风格；发现无关死代码
只提一下、不删；自己改动产生的孤儿（未用 import/变量）才清理。

**目标驱动。** 把任务转成可验证目标（"修 bug"→"先写复现测试再让它过"）；多步任务先列
简短计划 + 每步验证。强成功标准让你能独立循环；弱标准（"让它能用"）会反复返工。

## 环境与执行

- 凡执行脚本一律用 `.venv/bin/python`（系统 `python3` 是 3.7，工具需 3.10+）。
- **绝不修改 `costModel/`**（协作方只读仓库）。其分析方法论可继承，代码不动。
- **绝不臆造硬件/仿真细节**——stall 类型、原语名、仿真字段名不知道就留槽位或 TODO，
  不要编。臆造比留空危险得多（会被当成事实照做）。
- 控制面（`control/`、`memory/`）是确定性代码、经 CLI 调用，**不做成 skill**；状态只走
  文件、不进会话。

## skill 治理（OpenCode）

OpenCode 只认 skill frontmatter 的 `name`/`description`/`license`/`compatibility`/`metadata`，
忽略 `disable-model-invocation`。故用 `opencode.json` 的 `permission.skill` 复刻治理：
重量级/高代价的 `triton-gen`、`sim-analyze` 设为 `ask`；只读查阅的 `extension-guide` 设为
`allow`。skill 文件留在 `.claude/skills/`（OpenCode 的合法项目级发现路径，无需移动），
不要在 `.opencode/skills/` 再放同名副本。

## Project Knowledge（按需读）

- `docs/project_knowledge/project_overview.md` — 项目结构与 skill 工作流
- `docs/project_knowledge/memory_architecture.md` — 记忆模块
- `control/contracts.py` — 四份冻结契约（Event / Verdict / SimResult + 词表）
- `HANDOFF_GLM47.md` — 给保密环境的交接包（五个槽位任务 + 自检命令）
