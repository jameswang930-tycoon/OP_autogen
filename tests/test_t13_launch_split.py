"""T13-4 gate: 修正 HANDOFF 槽位 4 的分工表述。

旧表述"不要在这里解析"与真实两路数据来源冲突（correctness 来自 compare 段、
编译状态/流水来自仿真器产物）。改为明确的分工，并加"先确认两路来源再实现"一步。
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HANDOFF = (REPO / "HANDOFF_GLM47.md").read_text(encoding="utf-8")


def _slot4() -> str:
    for c in re.split(r"(?=^## 槽位 4)", HANDOFF, flags=re.MULTILINE):
        if c.startswith("## 槽位 4"):
            # stop at the next slot/section header
            return c.split("\n## ", 1)[0]
    return ""


def test_slot4_states_launch_assembles_raw():
    s = _slot4()
    assert "launch()" in s and "组装" in s, "must say launch() assembles the raw dict"


def test_slot4_states_parse_raw_only_converts_pipeline():
    s = _slot4()
    assert "parse_raw()" in s and ("只负责" in s or "转换" in s), (
        "must say parse_raw() only converts the pipeline part to Events"
    )


def test_slot4_removed_misleading_do_not_parse_sentence():
    s = _slot4()
    assert "不要在这里解析" not in s, "the misleading old sentence must be gone"


def test_slot4_has_two_source_mapping_step():
    s = _slot4()
    assert "两路" in s and "注释" in s, (
        "must require confirming both sources (as comments) before implementing"
    )
