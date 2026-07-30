# 改造进度

分支: local-adapt    基线 commit: e1dc19d

> 说明：计划/手册以 `79b4da1` 为基线，实际 HEAD 为 `e1dc19d`（多一个 `requirements.txt` 提交，未触及 `.claude/skills/` 与 `memory/`，符合手册"若已前进"协议）。`local-adapt` 分支于开工前已存在且与 `wsx` 同点（无差异），直接复用。

| 任务 | 状态 | 门禁 | checkpoint commit | 备注 |
|---|---|---|---|---|
| T1 | ✅ 门禁通过 | `pytest tests/test_t1_structure.py` (4 passed) | 7a430ee | 删 triton-convert/verify/fix；emulators 标退役；costModel 未改动 |
| T2 | ✅ 门禁通过 | `pytest tests/test_t2_contracts.py` (22 passed) | d8740cd | contracts(Event/Verdict/SimResult)+vocabulary.yaml/.py+一致性脚本；内容留空 |
| T3 | ✅ 门禁通过 | `pytest tests/test_t3_controller.py` (9 passed) | 508f89b | loop-controller 全量实现；ε/轮数/不可约/振荡/无进展/numerical-fail；best-so-far；env 可配 |
| T4 | ✅ 门禁通过 | `pytest tests/test_t4_memory.py` (7 passed) | 853a71f | latency_us→cycles；extension_used；每 fp 历史最优；score 价值=性能改善；FAIL 不影响 score；CLI --cycles/stats |
| T5 | ✅ 门禁通过 | `pytest tests/test_t5_adapter.py` (7 passed) | 3c73783 | feedback_adapter（reduce/classify/render→7段+Verdict 头）；3 份夹具；parse_raw 留槽；体量受限 |
| T6 | ✅ 门禁通过 | `pytest tests/test_t6_launch.py` (8 passed) | 26867a3 | launch_template：多段式模板(kernel/ref/compare)+launch()槽+build_sim_result+run()；假 launch 跑通出 SimResult |
| T7 | ✅ 门禁通过 | `pytest tests/test_t7_gate.py` (9 passed) | b0f6e31 | presim_gate：语法+shape/dtype 静态校验(matmul/elementwise/reduce)；挡下不自洽 kernel；check_extension_calls 占位恒通过 |
| T8 | ✅ 门禁通过（待人工复核 description cross-trigger） | `pytest tests/test_t8_skills.py` (12 passed) | 39d2181 | triton-plan→sim-analyze；triton-gen 改真实 Triton+extension；新建 extension-guide+样例+校验脚本；3 skill 英文正文 |
| T9 | ✅ 门禁通过（待人工复核） | `pytest tests/test_t9_config.py` (3 passed) | 7277b90 | opencode.json(permission.skill: gen/analyze=ask, guide=allow)；AGENTS.md 镜像 CLAUDE.md + 三条纪律；skill 名跨路径唯一 |
| T10 | ✅ 门禁通过 | `pytest tests/test_t10_skills_dualmode.py` (6 passed) | （见 T10 commit） | skill 双模：triton-gen/sim-analyze 正文改 `{{VAR}}` 占位符+输出契约；frontmatter 不动；control/placeholders.py 单一来源 |
| T11 | ✅ 门禁通过 | `pytest tests/test_t11_orchestrator.py` (10 passed) | （见 T11 commit） | 确定性编排器：job_spec(normalize triton_file；pytorch/shape_only 留槽)+主循环+解析/pre-sim/正确性闸门+重试预算分离+词表闭包+report；全离线可测 |
| T12 | ✅ 门禁通过 | `pytest tests/test_t12_handoff.py` (7 passed) | （见 T12 commit） | 交接包 HANDOFF_GLM47.md：推导程序式（材料自举+5 槽位+四节）；路径/签名已与 control/ 实际代码逐一核对；自包含、不引 spec |

### T13 增补（六项，逐项门禁）

| 子项 | 状态 | 门禁 | commit | 备注 |
|---|---|---|---|---|
| T13-1 LLM 后端 | ✅ 门禁通过 | `pytest tests/test_t13_llm_backend.py` (5 passed) | （见下） | control/llm_backend.py(ConfigurableLLMBackend，env 可配，不硬编码)；HANDOFF 加槽位 0；test_t12 同步至 6 槽位 |
| T13-2 launch 目录式 | ✅ 门禁通过 | `pytest tests/test_t13_launch_dirmode.py` (5 passed) | （见下） | launch_template docstring 三细节(可配置目录/run-id 隔离/等待超时)+new_run_id()；HANDOFF 槽位 4 同步 |
| T13-3 编译信号（契约修订，已授权） | ✅ 门禁通过 | `pytest tests/test_t13_compile_signal.py` (8 passed) | （见下） | SimResult+=compiled/compile_log；raw schema 同步；orchestrator COMPILE_FAIL(不计轮/compile_retries/compile_log 回喂/{{COMPILE_ERROR}})；triton-gen 加占位符；report rounds+=compiled；HANDOFF 槽位 4 加字段来源；test_t6/t10/t11 同步 |
| T13-4 槽位 4 分工表述 | ✅ 门禁通过 | `pytest tests/test_t13_launch_split.py` (4 passed) | （见下） | HANDOFF 槽位 4：删旧"不要解析"句；加 launch() 组装 / parse_raw() 只转换流水 的分工 + "先确认两路来源写注释"步 |
| T13-5 extension 四渠道（含 extension-forward，已授权） | ✅ 门禁通过 | `pytest tests/test_t13_extension_channels.py` (9 passed) | （见下） | ①sample example 改完整 kernel+.pyi 转录；④record_attempt 传 extension_used；⑤triton-gen vanilla-first→extension-forward(四规则,反向回归)；⑥pick_lever 探索偏好(off/mild/aggressive)；负面经验分类(compile/semantic) |
| T13-6 requirements.txt | ✅ 门禁通过 | `pytest tests/test_t13_requirements.py` (2 passed) | （见下） | 补 pyyaml+pytest；标注旧 emulator 遗留(numpy/networkx) vs 新框架所需 |

### GLM52 收尾（任务 A/B/C，逐项门禁）

| 子项 | 状态 | 门禁 | commit | 备注 |
|---|---|---|---|---|
| 任务 A / T13-7 NgaBackend | ✅ 门禁通过 | `pytest tests/test_t13_7_nga_backend.py` (7 passed) | （见下） | subprocess 调 nga run；无状态/只认 fenced block/忽略>头/失败抛异常不重试；模型配置驱动；mock 测试；HANDOFF 槽位 0 改为"框架已实现" |
| 任务 B 槽位 5 逻辑下推 | ✅ 门禁通过 | `pytest tests/test_slot5_logic.py` (8 passed) | （见下） | check_extension_calls 拆逻辑(AST 解析+比对签名)+数据(签名表文件)；ExtensionSignature+extract_extension_calls+load_signature_table；control/build_signature_table.py(inventory→签名表)；HANDOFF 槽位 5 改"只生成签名表" |
| 任务 C 槽位 6 模板文件化 | ✅ 门禁通过 | `pytest tests/test_slot6_template.py` (5 passed) | （见下） | LAUNCHABLE_TEMPLATE 改文件加载(load_launchable_template/assemble_launchable)；冻结 LAUNCHABLE_PLACEHOLDERS；triton-gen 格式跟随加载模板；HANDOFF 加槽位 6；test_t12 同步至 7 槽位 |

### 任务 E（可观测性与联调安全网，E1–E8，逐项门禁）

| 子项 | 状态 | 门禁 | commit | 备注 |
|---|---|---|---|---|
| E1 每轮全量落盘 | ✅ 门禁通过 | `pytest tests/test_e1_transcript.py` (2 passed) | （见下） | log/round_N/ 编号文件 01-11 + meta.txt；失败轮落盘部分并标 fail_stage；_RoundTranscript |
| E2 单轮重放入口 | ✅ 门禁通过 | `pytest tests/test_e2_replay.py` (5 passed) | （见下） | feedback_adapter(replay/adapt-only)、launch_template(assemble)、loop_controller(replay) CLI 重放；只读、坏输入清晰报错 |
| E3 组件边界断言 | ✅ 门禁通过 | `pytest tests/test_e3_boundary.py` (6 passed) | （见下） | feedback_adapter 加 ParseError + validate_events(duration==end-start/词表/索引定位) + validate_output(词表/cycles/summary 非空)；adapt 入口/出口强制 |
| E4 launch 失败五分类 | ✅ 门禁通过 | `pytest tests/test_e4_launch_errors.py` (6 passed) | （见下） | launch_template 定义 5 类异常(带证据)+launch() docstring 归类约定；前四类+SimInfraError 退避重试、ResultMismatch 立即停；证据入 report.detail/log |
| E5 HANDOFF 故障定位节 | ✅ 门禁通过 | `pytest tests/test_e5_handoff_fault.py` (3 passed) | （见下） | HANDOFF 增"故障定位"表（症状→看哪个文件→怎么办）+ 指引先用 E2 重放降维 |
| E6 preflight + bringup | ✅ 门禁通过 | `pytest tests/test_e6_preflight_bringup.py` (11 passed) | （见下） | control/preflight.py(四态 OK/STUB/MISSING/EXAMPLE 状态表，纯本地)；control/bringup.py(template/llm/launch/parse/extcheck/all 单点验证，launch/parse 解耦)；mock 测试 |
| E7 实时进度 | ✅ 门禁通过 | `pytest tests/test_e7_progress.py` (3 passed) | （见下） | _Progress 旁路事件：stderr 人可读 + progress.jsonl 机器可读；baseline/retrieve/generate/launch/result/best/stop；quiet 仅关键节点、normal 各阶段；不承载状态 |
| E8 HANDOFF 逐点联调节 | ✅ 门禁通过 | `pytest tests/test_e8_handoff_bringup.py` (3 passed) | （见下） | HANDOFF 增"逐点联调"节：preflight→bringup llm/template/launch/parse/extcheck/all→真实编排器；FAIL 锁定单接缝 |

> **重编号说明（2026-07-23）**：依 `docs/T10_T12_orchestrator_spec.md`，原 T10（交接包）改为 **T12**；新增 **T10**（skill 双模改造）与 **T11**（确定性编排器）。目标形态收紧为「确定性编排器驱动的流水线」。本批执行 T10、T11，完成后停于停止点④。

状态: ⬜未开始 / 🔄进行中 / ✅门禁通过 / ⛔阻塞待确认

## 待确认事项
（GLM 5.2 遇到需上报的问题时记在这里，不要自行发挥）

- **停止点①已到达（T7 完成）**：T1–T7 全部门禁通过（66 tests green），`costModel/` 自 fork 点 e1dc19d 起零改动。按指令停下，**未执行 T8/T9/T10**。
- **门禁判据说明（非阻塞）**：计划里 T1 的"`.claude/skills/` 下只剩三个 skill"描述的是 T8 之后的最终态（sim-analyze、extension-guide 在 T8 才产生）。T1 门禁因此只校验 T1 自身的产出——删 triton-convert/verify/fix、emulators 标退役、costModel 未动；"三 skill 最终态"留给 T8。已在 test_t1_structure.py 注释中写明。
- **停止点②已到达（T9 完成）**：T8、T9 门禁通过（累计 81 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T10**。停止点②复核要点：人工读三个 skill 的 description 判断是否 cross-trigger；确认 extension-guide 样例条目清晰到能让 4.7 照填。
- **停止点④已到达（T11 完成）**：T10、T11 门禁通过（累计 97 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T12**（交接包须单独会话：先重扫 control/ 实际签名再写）。
- **停止点⑤已到达（T12 完成）**：交接包 HANDOFF_GLM47.md 写成推导程序式；五个槽位的路径/签名已与 `control/` 实际代码逐一核对（test_t12_handoff 自动检查）；累计 **104 tests green**，`costModel/` 自 fork 点起零改动。T1–T12 全部完成。

## GLM52 框架优化（local-adapt 分支，按 docs/GLM52_OPTIMIZATION_GUIDE.md 优先级 1-5）

> 硬约束：绝不破坏 ext_distill / remote_dsl 两个接缝契约（第零部分"契约冻结清单"）。每项改完跑全量 pytest，本节只追加。
> 开工前 gate 基线：`pytest --continue-on-collection-errors` = **179 passed / 0 failed / 4 collection errors**。4 个 errors 全来自有意删除的 `HANDOFF_GLM47.md`（用户确认故意删，非代码问题），对应 4 个读它的 test（E5/E8/t12/t13）变孤立。gate 判据：**不新增 failed、不新增 collection error**（passed 数 ≥ 179）。

| 优先级 | 状态 | 改动 | 契约检查 | 备注（为什么这么改） |
|---|---|---|---|---|
| P1 prompt 组织 | ✅ gate 通过 | `triton-gen/SKILL.md`：去掉规则区(Step0/Step1)里内联的数据占位符引用——`{{VERDICT_JSON}}`(原 3 处)、`{{RETRIEVED_EXPERIENCE}}(原 2 处)、`{{COMPILE_ERROR}}`(原 2 处)——改指向性引用("the Verdict input"/"the Retrieved-experience input"/"the compile-error input")。每个占位符现只在 Inputs 区出现一次 | 只改 skill 模板文本。未碰冻结契约：placeholder **集合**不变(`test_placeholder_consistency` 过)；`EXT_REFS` 路径与"index 形式读取"不动；无 dataclass/launch/parse/env 改动 | 实现 instruction(稳定规则)/data(Inputs 数据段)分离，消除"数据被内联进规则、出现两次"的污染（指导第一部分核心红利）。保留测试要求的不变式 `do not hedge`/`structural base`/`First-round`，且不重新引入被禁短语。sim-analyze 模板本就干净(每占位符一处)，无需改 |
| P2 memory 两个 bug | ✅ gate 通过 | `control/orchestrator.py`：①`_retrieve_experience(bottleneck)` 改用上一轮 verdict 的已知瓶颈构造 fingerprint（旧 `bottleneck=None` → key "op\|unknown"，与 record 存的 "op\|真实瓶颈" 错位、by_key 永远 miss）；②返回 `(文本, 命中ID)`，`_record_attempt` 把真实命中 ID 传给 `record_attempt`（旧硬编码 `retrieved_ids=[]` → 分数永不迭代）。新增 `tests/test_glm52_memory_wiring.py`(2 测，一 bug 一测) | 只改 orchestrator 的 memory 接线。未碰冻结契约：`EXT_REFS`/index 读取、contracts/launch/parse/presim/env 全不动；`memory/`(retrieve/writeback/store/schema) 零改动，只是调用方传对了参数 | 修好"核心欠债"：经验检索恢复瓶颈区分度、好坏经验分数能随成败迭代。两测分别独立锁两 bug（Bug2：单经验单轮 used>0；Bug1：同瓶颈≥n 时 round2 不回退、排除异瓶颈），互不掩盖 |
| P3 extension 原语呈现 | ✅ gate 通过 | `control/orchestrator.py`：拆出 `_load_extension_entries()`(读 name/module/category/signature/applies_to 轻量字段)；`extension_index_text(op_kind,bottleneck)` 按 (算子场景,瓶颈) 只给相关子集 + 带模块全限定名(`module.name`)+ applies_to 标注，空则退回全量；`_resolve_lever` 候选改用 `_relevant_entries`(去误归类污染)；`_allowed_extensions()` 保持全量(parse 校验不受影响)。`sample_entry.yaml` 加可选 `module`/`applies_to` 示范；`references/README.md` 增"Optional fields"节 + 证据优先级教训。新增 `tests/test_glm52_extension_index.py`(4 测) | 未碰冻结契约：`EXT_REFS` 路径与"index 形式读取"不变(仍 yaml glob、无占位符)；`check_extension_cheatsheet.py` **零改动**(校验契约原样，sample 仍过)；contracts/launch/parse/presim/env 不动。新字段对校验器是透明的可选字段 | 解决幻觉与误用两根因：模块归属杜绝 `tlext1.add` 式猜模块；applies_to 让 conv-only 原语不污染 elementwise 候选。设计上保守——缺 applies_to 视为通用、检索空退回全量，绝不因标注缺失而隐藏原语 |
| P4 静默失效防御 | ✅ gate 通过 | `control/orchestrator.py`：新增 `_warn_if_memory_disabled()`——store/log 缺失时首次用到 memory 即 `warnings.warn`（每实例一次，绑真正用到 memory 的时机而非构造时），从 `_retrieve_experience`/`_record_attempt` 调用。新增 `tests/test_glm52_silent_failure.py`(3 测：未接线报 warn / 每实例一次 / 接好不报) | 纯加性：只在 no-op 路径加 warning，retrieve/record 的返回值与行为不变；未碰任何冻结契约。`bringup`/example 的"PASS 只验语法"属保密环境侧（公开模板 `_compare()` 本就是待填充的形状），未强改 | 把"memory 全程没工作却无人发现"的静默失效亮成显式 warning（suite 里 store=None 的用例现会报， truthful 非失败）。example 模板语义对齐留保密环境（公开模板无法跑真实 `_compare()`） |
| P5 文档卫生 | ✅ 维持 | 全程未产生临时 md（分析/审计直接落本表）；PROGRESS.md **严格只追加**（本批 +12 行 / 0 删除，未动既有决定与踩坑章节）；"为什么这么改"在各行备注列。`tmp/` 下两份 md（DESIGN_dual_runner / emulator_error_coverage，5 月旧档、已 tracked）非本批产物，按 surgical 原则保留不删（仅在此备注） | — | 关键决定见下节"GLM52 关键决定" |

### GLM52 关键决定（只增不减）

1. **P1 用"prompt 内 instruction/data 分离"，而非 system+user 双消息**：双消息要改 `LLMProvider.generate(prompt: str)` 签名，会波及所有测试的 FakeLLM 与真实后端。改为单 prompt 内把数据集中在 Inputs 段、规则只做指向性引用（"the Verdict input"等），同样消除"数据内联进规则、出现两次"的污染，且零签名/契约改动。placeholder **集合**不变（`test_placeholder_consistency` 仍是单一事实源）。
2. **P3 场景检索带保守兜底**：缺 `applies_to` 视为通用、检索结果为空退回全量——绝不因标注缺失而隐藏原语。这是保密环境把 `applies_to` 填全前的安全网（指导第二部分"语料的有≠对"）。
3. **P4 用 `warnings.warn` 而非抛异常**：memory 未接线是"失效但可继续"，非致命错误；故 warning 不 raise，每实例一次防刷屏。
4. **gate 判据（开工即锁定）**：基线 `pytest --continue-on-collection-errors` = 179 passed / 0 failed / **4 collection errors**（全来自有意删除的 `HANDOFF_GLM47.md`，E5/E8/t12/t13 四测试成孤立、非代码问题）。每项优化后判据 = "不新增 failed、不新增 collection error"；passed 随回归测试单调递增 179→181(P2)→185(P3)→188(P4)。
5. **环境**：本机（macOS）够不到 `/workspace/triton_env`（保密机路径）；用本仓 `.venv`(uv, py3.13.13, 已装 pytest/networkx/yaml/torch)，按需 `uv pip install`。triton 未装但不进测试路径。
6. **孤立测试处理（P5 收尾后追加）**：读已删 `HANDOFF_GLM47.md` 的 4 个测试文件（`test_e5_handoff_fault` / `test_e8_handoff_bringup` / `test_t13_launch_split` / `test_t12_handoff`）—— 全是文档内容类（验"故障定位"/"逐点联调"/槽位措辞等节），文档已故意删故退役；`test_t12` 里两条**不依赖文档**的契约守卫（冻结签名在源码中真实存在、sample 标注）抽到新文件 `tests/test_contract_alignment.py` 保留。gate 现**全绿**：`pytest -q` = **190 passed / 0 failed / 0 errors**（不再需要 `--continue-on-collection-errors`）。决定 4 记的"4 collection errors"为处理前历史状态。
7. **保密环境适配参照（交付文档）**：`HANDOFF_GLM52.md`（仓库根，旧 `HANDOFF_GLM47.md` 的 GLM52 继任）。含：冻结接缝零适配清单 + P1–P5 逐项适配表 + P3 可选字段（`module`/`applies_to` 与 `EXTENSION_NAMESPACE` 一致性、签名表不带模块、证据优先级）+ bring-up 步骤 + 自检命令 + 回退。**纯参照用、不挂内容测试**（避免旧 handoff 被"文档内容测试"绑死、删文档即孤立的重蹈）。
8. **提交前 code-review 修正（pre-commit，medium effort）**：review 发现 P3 引入一个回归——index 渲染全限定名 `module.name`，但 `parse_generate_response` 只按裸名校验 `extension_used`，弱模型照 index 抄全限定名会被拒 → 每轮 `BudgetExhausted("llm_retries")`（FakeLLM 测不出，因其写裸名/null）。修：parse 把 `extension_used` 规范化（`module.name`→裸名）再校验/存档，两种写法都接受；+回归测 `test_parse_accepts_module_qualified_extension_used`。另修两处自身测试鲁棒性：① P2 memory 测试改用 `extension_used=null`（与 sample 速查表解耦，sample 改名不再误判 memory 回归）；② `test_bug1` 同瓶颈经验 3→5 条（留 retrieve `n=3` 余量，n 上调也不会误触发回退）。review 后 gate：**191 passed**。
