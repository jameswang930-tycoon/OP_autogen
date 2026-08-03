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
3. **backend 配置（无双模式）**：V2 backend 重构后只剩**一个** backend（`NgaBackend`，配置驱动），gen prompt 统一精简（场景提示 + ext-* skill lazy-load）——不存在"哑后端/双模式"开关（`GEN_PROMPT_MODE` 已删）。环境侧按 C.0 探测结果填 backend config（cmd 前缀、options、extra_args、是否结构化输出）。
4. SIM_* 系列（SIM_ROOT/SIM_SCRIPT/SIM_INPUT_DIR/SIM_RESULT_DIR/SIM_TIMEOUT）、PRESIM_SIGNATURE_TABLE、LAUNCHABLE_TEMPLATE_PATH——沿用环境现有稳定值，不动。

### B. 稳定接缝对接（沿用，仅确认）
1. 远端仿真：launch 实现、raw_sim_output schema、结果目录约定、SSH/conda 激活——沿用现状，仅确认路径有效。
2. 编译手册原语 / 转置 txt / triton 扩展包接口与调用范例——环境准备好，框架按契约消费。

### C. backend 落地（V2 重构：配置驱动，**无哑后端/双模式**）

框架侧只剩**一个** backend（`NgaBackend`，`control/llm_backend.py`）——配置驱动 + 可扩展，不预设 agent 能力；generate/choose_lever 对 orchestrator 不变。环境侧只填配置，不改框架。

- **C.0 探测 agent 能力（适配前必做，环境侧）**：用 `agent --help`、官方文档、实际试调，摸清目标 agent 支持的命令行选项（指定模型 / 思考开关 / 输出格式 / skill 触发方式 / ...）及确切写法，产出**能力清单**。这是后续填配置的依据，框架侧不参与、不假设 agent 支持哪些。
- **C.1 填 backend config（按 C.0 清单）**：`NgaBackend(config)` 的 config 结构——`cmd`（命令前缀，如 `["xxx","run"]`）、`generate`/`choose_lever` 各自 `{model, options:{...}, extra_args:[...], timeout_s}`、可选 `structured`。框架只提供 options→命令行的**通用映射**（`{key:value}→--key value`、`True→--key`、`False/None→省略`）+ `extra_args` 原样透传；**具体 key/value 由环境侧填真实值**。agent 新增选项时只加 config、不改框架代码。**真实 agent 命令行、模型名、语料、路径均为环境侧保密信息，不出现在开源仓库。**
- **C.2 验证 skill 真实触发**（关键，不能只看代码）：用配置好的 backend 命令行描述一个符合某 ext-* skill description 的任务，确认终端显示触发了**预期的那一个** skill（如 reduce_sum 任务触发 ext-reduction，不触发 ext-activation）。这是"按场景过滤治 softmax"能否从根上生效的验证。
- **C.3 结构化输出（可选）**：若 C.0 探明 agent 支持结构化输出（如 json），在 config 声明 `structured={enabled, request, kernel_key, meta_key}`——backend 会要求结构化输出并把结果规范成 fenced block（比从自由文本抠可靠、解决 kernel/json 混排）。未声明则用自由文本解析（默认降级）。是否启用由配置决定，框架不假设 agent 一定支持。

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
