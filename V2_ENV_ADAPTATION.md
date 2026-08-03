# V2 收尾：环境侧适配交接（GLM 5.2 → GLM 5.1）

> v2 框架侧（P1-P5）已在 local-adapt-v2 完成、验收通过（224 passed，冻结接缝零改动）。
> 现在产出一份面向目标环境 GLM 5.1 的适配清单。请 GLM 5.2 把下面内容整理成
> `docs/dev_plan/V2_ENV_ADAPTATION.md`（新建，供拉回环境后 5.1 读），**不要改动已完成的框架代码**。
> 进展更新位置：环境侧适配的进度，5.1 追加到环境本地的 `PROGRESS.md`（只追加，不删历史）。

## 交接原则

- 框架侧（P1-P5 逻辑）已冻结，环境侧只做"对接真实资源 + 配置 + 语料"，不改框架逻辑。
- 冻结接缝零改动：`contracts.py` / `launch_template.py`（launch 实现除外，那本就是环境槽位）/
  `feedback_adapter.py` / `presim_gate.py` / `build_signature_table.py` / `check_vocab_consistency.py`。
- 证据优先级：真实源码(operator.py) > 手册 > api_inventory / 现有 yaml。

## 环境侧适配清单（5.1 执行）

### A. 配置对接（env.sh）
1. **P1 配套**：`NGA_CHOOSE_LEVER_MODEL=<小模型，如 MiniMax-M2.7>`（choose_lever 只返回一行 json，不用大模型）；`NGA_CHOOSE_LEVER_TIMEOUT_S=180`（余量调宽，实测 85s、默认 120 太紧）。
2. `NGA_GENERATE_MODEL` / `NGA_GENERATE_TIMEOUT_S` 按环境模型能力设。
3. **P5.2 双模式开关**：`GEN_PROMPT_MODE=agent`（走 agent 精简 prompt，靠 skill lazy-load）或留空/`nga`（哑后端塞全量 index）。先决定环境用哪种模式（见 C）。
4. SIM_* 系列（SIM_ROOT/SIM_SCRIPT/SIM_INPUT_DIR/SIM_RESULT_DIR/SIM_TIMEOUT）、PRESIM_SIGNATURE_TABLE、LAUNCHABLE_TEMPLATE_PATH——沿用环境现有稳定值，不动。

### B. 稳定接缝对接（沿用，仅确认）
1. 远端仿真：launch 实现、raw_sim_output schema、结果目录约定、SSH/conda 激活——沿用现状，仅确认路径有效。
2. 编译手册原语 / 转置 txt / triton 扩展包接口与调用范例——环境准备好，框架按契约消费。

### C. P5 agent 模式落地（本次 v2 的架构升级，环境侧核心工作）
1. **AgentBackend 接真实 agent CLI**：框架已留 `AgentBackend`（接口同 NgaBackend，runner 可注入）。5.1 把 runner 换成真实 agent 的命令行格式（`agent "query"` 形式）、配置模型名/variant/skill 目录路径。
2. **验证 skill 真实触发**（关键，不能只看代码）：用 agent 命令行描述一个符合某 ext-* skill description 的任务，确认终端显示触发了**预期的那一个** skill（如 reduce_sum 任务触发 ext-reduction，不触发 ext-activation）。这是"按场景过滤治 softmax"能否从根上生效的验证。
3. **决定生产模式**：若环境 agent 稳定触发 skill → 用 `GEN_PROMPT_MODE=agent`（prompt 精简、原语按需 lazy-load）；若暂不稳 → 先用 nga 哑后端模式（塞 index），两种框架都支持。

### D. extension 语料（治本，5.1 重点）
1. **P5.3 拆分落地**：框架已建 5 个 `ext-*` skill 目录（ext-reduction/activation/matmul/shape/quant），references/ 为空待填。把真实原语 yaml 按场景归入对应 skill 的 references/。
2. **每个原语 yaml 必须**：`module`（全限定名，杜绝 tlext1.add 幻觉）、`applies_to`（适用算子，专用原语必标——治 softmax 混入的根本）、`signature`（对齐源码 operator.py，过滤 _builder/_generator 内部参数）、`example`（从真实 kernel 提取或按源码手写，**绝不从 LLM 生成的错误代码提取**）、`category`（vocabulary id）。
3. **强制校验**：跑 `check_yaml_signatures.py`（AST 对源码校验签名）+ `check_extension_cheatsheet.py`（多目录已适配）。所有 yaml 签名必须过校验。
4. **P5.4 extension-guide 退役**：确认 nga 模式不再需要单一 extension-guide 后，可退役旧的单目录 index（拆分后的 ext-* 是新源）。
5. `EXT_REFS` 读取路径：拆成多目录后，环境侧的读取逻辑相应适配（框架 check 已支持多目录遍历）。

### E. verify_method prompt 精简（P3 环境侧部分）
launchable 模板的 ref_func / operator 两种验证示范，当前 prompt 两种都附。改为：组装时读 `job.verify_method`，只附对应那一种示范 kernel + case。这是环境侧 build_gen_prompt / 模板改动（公开分支无此套）。

### F. memory 端到端验证（拉回环境后）
框架 memory 三职责已验证（mock）。环境上真实跑，确认：round1 后 experience.json 有内容、round2 `has_exp=True`、best-so-far 作迭代基准、失败原语进避坑清单、多轮 used/helped 分数更新。

## 进展更新位置

- **环境侧适配进度**：5.1 追加到环境本地 `PROGRESS.md`（只追加，不删历史；不新建中间 md）。
- **框架侧（本 v2）**：已记在 local-adapt-v2 的 PROGRESS.md「V2 迭代」节，不再改动。

## 交接边界

- 框架逻辑（P1-P5）不改，环境侧只做 A-F。
- 若环境侧发现框架 bug（非配置/语料问题），记进 PROGRESS.md 待确认区，不擅自改冻结接缝。
