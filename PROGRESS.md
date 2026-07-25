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
| E5 HANDOFF 故障定位节 | ⬜ | — | — | |
| E6 preflight + bringup | ⬜ | — | — | |
| E7 实时进度 | ⬜ | — | — | |
| E8 HANDOFF 逐点联调节 | ⬜ | — | — | |

> **重编号说明（2026-07-23）**：依 `docs/T10_T12_orchestrator_spec.md`，原 T10（交接包）改为 **T12**；新增 **T10**（skill 双模改造）与 **T11**（确定性编排器）。目标形态收紧为「确定性编排器驱动的流水线」。本批执行 T10、T11，完成后停于停止点④。

状态: ⬜未开始 / 🔄进行中 / ✅门禁通过 / ⛔阻塞待确认

## 待确认事项
（GLM 5.2 遇到需上报的问题时记在这里，不要自行发挥）

- **停止点①已到达（T7 完成）**：T1–T7 全部门禁通过（66 tests green），`costModel/` 自 fork 点 e1dc19d 起零改动。按指令停下，**未执行 T8/T9/T10**。
- **门禁判据说明（非阻塞）**：计划里 T1 的"`.claude/skills/` 下只剩三个 skill"描述的是 T8 之后的最终态（sim-analyze、extension-guide 在 T8 才产生）。T1 门禁因此只校验 T1 自身的产出——删 triton-convert/verify/fix、emulators 标退役、costModel 未动；"三 skill 最终态"留给 T8。已在 test_t1_structure.py 注释中写明。
- **停止点②已到达（T9 完成）**：T8、T9 门禁通过（累计 81 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T10**。停止点②复核要点：人工读三个 skill 的 description 判断是否 cross-trigger；确认 extension-guide 样例条目清晰到能让 4.7 照填。
- **停止点④已到达（T11 完成）**：T10、T11 门禁通过（累计 97 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T12**（交接包须单独会话：先重扫 control/ 实际签名再写）。
- **停止点⑤已到达（T12 完成）**：交接包 HANDOFF_GLM47.md 写成推导程序式；五个槽位的路径/签名已与 `control/` 实际代码逐一核对（test_t12_handoff 自动检查）；累计 **104 tests green**，`costModel/` 自 fork 点起零改动。T1–T12 全部完成。
