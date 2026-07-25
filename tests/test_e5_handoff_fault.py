"""E5 gate: HANDOFF 增"故障定位"节（症状 → 看哪个文件 → 怎么办）。

让定位不需要理解框架全局：读一个文件、看一个组件的输入输出就找到病灶。
"""
from pathlib import Path

HANDOFF = (Path(__file__).resolve().parent.parent / "HANDOFF_GLM47.md").read_text(encoding="utf-8")


def test_handoff_has_fault_location_section():
    assert "故障定位" in HANDOFF


def test_fault_table_covers_key_symptoms_and_actions():
    # each key symptom + its "look here" / action must appear
    for kw in (
        "RemoteTimeout", "RemoteScriptError",   # infra classification
        "ResultMismatch",                        # framework bug
        "UNKNOWN_BOTTLENECK",                    # vocab/parse_raw
        "compile",                               # compile-fail loop
        "replay",                                # E2 replay as the定位 lever
        "baseline",                              # worse-than-baseline is normal
    ):
        assert kw in HANDOFF, f"fault-location section missing key term {kw!r}"


def test_fault_section_points_at_log_artifacts():
    # the table must reference concrete log/ artifacts to look at
    for artifact in ("meta.txt", "log/round_"):
        assert artifact in HANDOFF, f"fault section should point at {artifact!r}"
