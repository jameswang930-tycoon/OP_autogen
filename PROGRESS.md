# 改造进度

分支: local-adapt    基线 commit: e1dc19d

> 说明：计划/手册以 `79b4da1` 为基线，实际 HEAD 为 `e1dc19d`（多一个 `requirements.txt` 提交，未触及 `.claude/skills/` 与 `memory/`，符合手册"若已前进"协议）。`local-adapt` 分支于开工前已存在且与 `wsx` 同点（无差异），直接复用。

| 任务 | 状态 | 门禁 | checkpoint commit | 备注 |
|---|---|---|---|---|
| T1 | ✅ 门禁通过 | `pytest tests/test_t1_structure.py` (4 passed) | 7a430ee | 删 triton-convert/verify/fix；emulators 标退役；costModel 未改动 |
| T2 | ✅ 门禁通过 | `pytest tests/test_t2_contracts.py` (22 passed) | （见 T2 commit） | contracts(Event/Verdict/SimResult)+vocabulary.yaml/.py+一致性脚本；内容留空 |
| T3 | ⬜ | — | — | |
| T4 | ⬜ | — | — | |
| T5 | ⬜ | — | — | |
| T6 | ⬜ | — | — | |
| T7 | ⬜ | — | — | |
| T8 | ⬜ | — | — | 需人工复核（本批不执行） |
| T9 | ⬜ | — | — | 需人工复核（本批不执行） |
| T10 | ⬜ | — | — | 必须人工复核（本批不执行） |

状态: ⬜未开始 / 🔄进行中 / ✅门禁通过 / ⛔阻塞待确认

## 待确认事项
（GLM 5.2 遇到需上报的问题时记在这里，不要自行发挥）
