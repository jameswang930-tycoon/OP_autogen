"""T13-6 gate: requirements.txt 补齐新框架依赖并标注遗留。"""
from pathlib import Path

REQ = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")


def test_has_new_framework_deps():
    low = REQ.lower()
    assert "pyyaml" in low, "missing pyyaml (job / vocabulary yaml parsing)"
    assert "pytest" in low, "missing pytest (tests/ gates)"


def test_keeps_legacy_deps_and_marks_them():
    low = REQ.lower()
    assert "numpy" in low and "networkx" in low, "legacy emulator deps must remain"
    assert ("遗留" in REQ) or ("legacy" in low), "legacy deps must be marked"
    assert ("新框架" in REQ) or ("framework" in low), "new deps must be marked as new"
