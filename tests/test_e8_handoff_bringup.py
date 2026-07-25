"""E8 gate: HANDOFF 增"逐点联调（bring-up）"节。"""
from pathlib import Path

HANDOFF = (Path(__file__).resolve().parent.parent / "HANDOFF_GLM47.md").read_text(encoding="utf-8")
LOW = HANDOFF.lower()


def test_handoff_has_bringup_section():
    assert "逐点联调" in HANDOFF or "bring-up" in LOW or "bringup" in LOW


def test_bringup_section_lists_ordered_steps():
    for kw in (
        "preflight",
        "bringup llm",
        "bringup template",
        "bringup launch",
        "bringup parse",
        "bringup extcheck",
    ):
        assert kw in LOW, f"bring-up section missing step {kw!r}"


def test_bringup_section_states_fail_locks_to_one_seam():
    assert "fail" in LOW or "锁定" in HANDOFF
