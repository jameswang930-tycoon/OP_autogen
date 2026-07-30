# HANDOFF_GLM52 — 保密环境交接/适配参照（GLM52 框架优化批）

> 读者：保密环境接手者。本批是 `local-adapt` 分支上的 **GLM52 框架优化（P1–P5）**。
> 两条主线：① **怎么把 `ext_distill` / `remote_dsl` 两条接缝接上去**（§1，handoff 的核心）；② **本批优化改了什么、要不要重新适配**（§2–§3，答案：两接缝零适配）。
> 配套：`docs/GLM52_OPTIMIZATION_GUIDE.md`（为什么改）、`PROGRESS.md`「GLM52 关键决定」（决定/gate）。

---

## 0. 绝不可破坏的两个接缝（本批零改动）

`ext_distill`（extension 语料/签名表）和 `remote_dsl`（远端仿真调用）是保密环境花很久适配的两个接缝。
**本批改动没碰它们的任何契约**——`git diff` 确认下列文件全部零改动：

| 冻结文件 | 冻结内容 |
|---|---|
| `control/contracts.py` | `Event` / `Verdict` / `SimResult` 字段 |
| `control/launch_template.py` | `launch(kernel_file)->dict`、`raw_sim_output` schema、`new_run_id`、`build_sim_result`、E4 异常体系 |
| `control/feedback_adapter.py` | `parse_raw(raw_sim_output) -> list[Event]` |
| `control/presim_gate.py` | `check_extension_calls` / `load_signature_table` / `extract_extension_calls` |
| `control/build_signature_table.py` | `api_inventory.txt → 签名表` 输入输出格式 |
| `control/check_extension_cheatsheet.py` / `check_vocab_consistency.py` | 校验契约 |
| `control/orchestrator.py` | `EXT_REFS` 路径值（行号 40→41，值不变）、extension 作为 index 读取 |
| 环境变量 | 所有 `SIM_*` / `PRESIM_*` / `NGA_*` / `LAUNCHABLE_*`（含 `EXTENSION_NAMESPACE`） |

---

## 1. 两条接缝怎么对接（保密环境填/实现什么）

### 接口 A — `ext_distill`（extension 语料 / 签名表）

保密环境准备 **3 样**：

1. **真实 extension 原语 yaml** —— 放 `EXT_REFS`（`.claude/skills/extension-guide/references/`，**路径固定**）。用真实原语替换 `sample_entry.yaml`；每条必填 `name/semantics/signature/category/example/pitfalls`，`category` 必须是 `control/vocabulary.yaml` 里的 id。**可选**加 `module`（真实模块/命名空间）+ `applies_to`（适用算子列表，红利见 §3）。
   - 自检：`.venv/bin/python -m control.check_extension_cheatsheet`
2. **真实签名表** —— 跑一次：
   `.venv/bin/python -m control.build_signature_table <api_inventory.txt> <signature_table.yaml>`
   inventory 行格式 `模块路径 | 名称 | signature | doc首行`（**脚本只用 `name`+`signature`，忽略模块列** —— `build_signature_table.py:70`）。
   把产出的 yaml 路径配到环境变量 **`PRESIM_SIGNATURE_TABLE`**（`load_signature_table` 读它）。
3. **`EXTENSION_NAMESPACE`** 环境变量 —— kernel 里 extension 调用的命名空间（`<namespace>.name`），默认 `ext`。`check_extension_calls` **只提取并校验这个命名空间**的调用（`presim_gate.py:85`）。
   - 若填了 `module` 字段，**务必与 `EXTENSION_NAMESPACE` 一致**（见 §3.1a），否则模型按 index 写的调用不被校验。
   - 自检：`.venv/bin/python -m control.bringup extcheck --kernel <含 extension 调用的 kernel>`
   - 一致性：`.venv/bin/python -m control.check_vocab_consistency`

> 本批对这套接缝的唯一"邻域改动"是 orchestrator 现在会**额外**读 yaml 里的可选 `module`/`applies_to`（§3）。读取路径、index 形式、签名表/校验全部原样。

### 接口 B — `remote_dsl`（远端仿真调用）

保密环境实现/配置 **4 样**：

1. **`launch(kernel_file: str) -> dict`**（`control/launch_template.py`，当前是 `NotImplementedError` 槽位）—— 发射组装好的可发射文件到远端仿真器、取回原始输出。**必须**按规范返回 `raw_sim_output`：
   `{"correct": bool, "max_abs_err": float, "cycles": int|None, "pipeline": {unit: cycles}, "compiled": bool, "compile_log": str}`
   须遵守三条「目录式调用约束」（`launch_template.py` 模块 docstring）：
   - ① 输入/输出目录、远程脚本路径全走配置或环境变量（**`SIM_*`**：`SIM_ROOT/SIM_SCRIPT/SIM_INPUT_DIR/SIM_RESULT_DIR/SIM_TIMEOUT` 等，由 `env.sh` 配置、`launch()` 读取，**不硬编码**）；
   - ② 每次用 `new_run_id()` 生成唯一 id，按 run id 区分文件/子目录，读结果时**校验 run id 属于本次**（不匹配→`ResultMismatch`）；
   - ③ 异步轮询等待 + 超时，故障抛可识别异常。
2. **`parse_raw(raw_sim_output) -> list[Event]`**（`control/feedback_adapter.py`）—— 把真实 raw 转成 `Event` 列表（`Event` 字段冻结）。
   - **分工**（别踩坑）：`launch()` 负责**组装并取回** raw dict（correctness 来自 compare 段、编译状态/流水来自仿真器产物——**两路来源**）；`parse_raw()` **只把 raw 里的 pipeline 部分转成 Events**，不重算正确性。先确认两路来源（落注释）再实现。
3. **失败归类（E4，异常类型框架已定义）**—— 按情况抛：
   连不上→`RemoteConnectionError(endpoint, original)`；超时→`RemoteTimeout(timeout_s, run_id)`；脚本非零退出→`RemoteScriptError(exit_code, stderr)`（保留 stderr 原文）；超时内无结果→`ResultNotFound(expected_path, run_id, wait_s)`；run id 不匹配→`ResultMismatch(expected_run_id, actual_run_id)`。
   **前四类=基础设施**（编排器按 `sim_retries` 退避重试、不计轮）；**最后=框架 bug**（立即停、不重试）。只扩展不改语义。
4. **真实可发射模板**（通常要）—— 把真实 triton.py 模板放到 **`LAUNCHABLE_TEMPLATE_PATH`** 指向的路径。占位符集合冻结 `LAUNCHABLE_PLACEHOLDERS`=`{OP,SHAPES,DTYPE,KERNEL_BODY,REFERENCE}`；compare 段必须吐 raw_sim_output 字段。
   - 自检：`bringup launch --kernel <k>`（raw 键齐全）、`bringup parse --raw <raw.json>`（parse+adapt 出合法 Verdict）、`bringup template`（模板占位符+组装语法）。

> 本批对这条接缝**零邻域改动**。

---

## 2. 本批 GLM52 改了什么 / 要不要重新适配

| 优先级 | 改动（文件） | 接缝 | 重新适配？ |
|---|---|---|---|
| **P1** prompt 重构 | `triton-gen/SKILL.md`：规则区去内联占位符→指向性引用 | gen-prompt | **零**。模板文本，placeholder 集合不变 |
| **P2** memory 两 bug | `orchestrator.py`：retrieve 用已知瓶颈、record 回传真实 ID | 内部 | **零**。`memory/` 零改动，修复自动生效 |
| **P3** extension 索引 | `orchestrator.py`（按场景检索 + 模块全限定名）；`sample_entry.yaml`+`README.md` 加可选 `module`/`applies_to` | ext_distill 邻域 | **可选**（见 §3） |
| **P4** 静默失效 | `orchestrator.py`：store/log 缺失即 `warnings.warn` | 内部 | **零**。接好 store+log 就不报 |
| **P5** 文档/测试 | `PROGRESS.md`（只追加）；删 4 个旧 HANDOFF 文档内容测试，契约守卫抽到 `tests/test_contract_alignment.py` | — | **零** |

---

## 3. P3 适配细节（唯一可选工作）

`module` / `applies_to` 是**可选字段**，不填也照常（检索空退回全量、`module` 缺省渲染裸名）。

**三条一致性提醒**：
- **(a) `module` 对齐 `EXTENSION_NAMESPACE`**：`check_extension_calls` 只校验 `<EXTENSION_NAMESPACE>.name`；index 渲染 `module.name` 让模型照写。单命名空间→module 填它（或留空只设 `EXTENSION_NAMESPACE`）；多命名空间→当前校验只覆盖一个（**既有**限制，P3 没引入）。
- **(b) 签名表本不带模块**：`build_signature_table.py` 忽略 inventory 模块列；`module` 只服务 prompt 呈现，与签名表是两套独立数据。
- **(c) 取证优先级**：真实 kernel 用法 > 手册 > api_inventory（inventory 模块归属不可信）；拿不准就**留空**。

字段写法见 `references/README.md`「Optional fields」。

---

## 4. 故障定位（联调 triage：症状 → 看哪里 → 怎么办）

联调任一步炸，按此表先锁范围（不必通读全局）：

| 症状 | 先看 | 怎么办 |
|---|---|---|
| `RemoteTimeout` / `RemoteConnectionError` / `ResultNotFound` | 远端连通性、`SIM_*` 配置、输出目录 | 基础设施问题，编排器退避重试；查 `env.sh`/网络/超时 |
| `RemoteScriptError`（非零退出） | `log/round_N/05_launch_input.py` + 远端 stderr | 环境问题（非 kernel）；看 stderr 原文 |
| `ResultMismatch`（run id 不匹配） | `new_run_id` 隔离、共享目录 | **框架 bug，立即停**；查 run-id 隔离 |
| `UNKNOWN_BOTTLENECK` | `parse_raw` 字段映射、`vocabulary.yaml` | parse_raw 映射错，或 bottleneck 不在词表 |
| 编译失败（`compiled=False`） | `log/round_N/compile_fail_*.log` | compile_log 回喂 gen 重生成（不计轮）；看原文 |
| 数值 FAIL（`correct=False`） | `log/round_N/06_raw_sim.json`（`max_abs_err`） | 计入轮；看 reference 比对 |
| 比 baseline 慢 | —— | **正常**，best-so-far + rollback 兜底，非 bug |
| 任何 FAIL 想降维 | `log/round_N/` 全量落盘 | 先 E2 单轮重放（只读）定位是哪一环 |

---

## 5. 拉回 + bring-up 步骤

```
1. git pull（本批改动）
2. .venv/bin/python -m pytest -q            # 期望: 190 passed, 0 failed, 0 errors
3. python -m control.preflight              # 配置自检（四态表）
4. python -m control.bringup template       # 模板占位符 + 组装语法
   python -m control.bringup llm            # nga 调通 + 解析闸门（需真实 LLM 后端）
   python -m control.bringup launch --kernel <k>   # 真实 launch + raw_sim_output 键齐全
   python -m control.bringup parse --raw <raw.json> # parse_raw + adapt 出合法 Verdict
   python -m control.bringup extcheck --kernel <k>  # extension 调用签名校验
5.（可选）真实原语 yaml 加 module/applies_to（见 §3）
6. 两条接缝的既有适配原样不动，直接跑编排器
```

任一环 FAIL → 按 §4 锁定单接缝（看 `log/round_N/` 或 E2 重放降维）。
（bring-up **代码**在 `control/bringup.py`；旧 GLM4.7 步骤文档 `HANDOFF_GLM47.md` 已随版本退役删除。）

---

## 6. 自检命令一览

```bash
.venv/bin/python -m pytest -q                              # 全量 gate（190 passed）
.venv/bin/python -m control.check_extension_cheatsheet     # 速查表 category ∈ vocabulary
.venv/bin/python -m control.check_vocab_consistency        # 词表/速查表一致
.venv/bin/python -m control.bringup <template|llm|launch|parse|extcheck>
```

---

## 7. 回退

本批全是加性/局部改动，无数据迁移、无契约变形。回退即 `git revert` 本批 commit（或 checkout 改动文件）。P2 只改 orchestrator 接线、不动 `memory/`；P3 新字段对校验器透明。
