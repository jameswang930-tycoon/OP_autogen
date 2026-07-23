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
| T12 | ⬜ 未开始 | — | — | 交接包 `HANDOFF_GLM47.md`（原 T10，重编号为 T12）—— 单独会话写 |

> **重编号说明（2026-07-23）**：依 `docs/T10_T12_orchestrator_spec.md`，原 T10（交接包）改为 **T12**；新增 **T10**（skill 双模改造）与 **T11**（确定性编排器）。目标形态收紧为「确定性编排器驱动的流水线」。本批执行 T10、T11，完成后停于停止点④。

状态: ⬜未开始 / 🔄进行中 / ✅门禁通过 / ⛔阻塞待确认

## 待确认事项
（GLM 5.2 遇到需上报的问题时记在这里，不要自行发挥）

- **停止点①已到达（T7 完成）**：T1–T7 全部门禁通过（66 tests green），`costModel/` 自 fork 点 e1dc19d 起零改动。按指令停下，**未执行 T8/T9/T10**。
- **门禁判据说明（非阻塞）**：计划里 T1 的"`.claude/skills/` 下只剩三个 skill"描述的是 T8 之后的最终态（sim-analyze、extension-guide 在 T8 才产生）。T1 门禁因此只校验 T1 自身的产出——删 triton-convert/verify/fix、emulators 标退役、costModel 未动；"三 skill 最终态"留给 T8。已在 test_t1_structure.py 注释中写明。
- **停止点②已到达（T9 完成）**：T8、T9 门禁通过（累计 81 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T10**。停止点②复核要点：人工读三个 skill 的 description 判断是否 cross-trigger；确认 extension-guide 样例条目清晰到能让 4.7 照填。
- **停止点④已到达（T11 完成）**：T10、T11 门禁通过（累计 97 tests green），`costModel/` 自 fork 点起零改动。按指令停下，**未执行 T12**（交接包须单独会话：先重扫 control/ 实际签名再写）。
