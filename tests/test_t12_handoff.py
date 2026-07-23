"""T12 gate: 交接包 HANDOFF_GLM47.md。

Covers T12_spec §5:
  - exists; has bootstrap section + 5 slots + 4 other sections
  - every slot's file path & function signature matches control/ actual code
  - every slot has: numbered steps + a self-check command + a 'stop-and-escalate' branch
  - every self-check command parses and points at a real path
  - self-contained: does not reference the arch doc / runbook / specs
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HANDOFF = (REPO / "HANDOFF_GLM47.md").read_text(encoding="utf-8")

# (slot label, substrings that MUST appear — file path + frozen signature)
SLOTS = {
    "vocabulary": ["control/vocabulary.yaml"],
    "parse_raw": ["control/feedback_adapter.py", "parse_raw(raw_sim_output)"],
    "extension_refs": [".claude/skills/extension-guide/references/"],
    "launch": ["control/launch_template.py", "launch(kernel_file)"],
    "check_extension_calls": ["control/presim_gate.py", "check_extension_calls(kernel_src)"],
}


def test_handoff_exists_and_has_structure():
    assert HANDOFF.strip()
    assert "材料自举" in HANDOFF and "第 0 步" in HANDOFF
    for sec in ("三条纪律", "运行说明", "验证清单", "保密纪律"):
        assert sec in HANDOFF, f"missing section {sec!r}"


def test_every_slot_path_and_signature_present():
    for slot, subs in SLOTS.items():
        for s in subs:
            assert s in HANDOFF, f"slot {slot!r}: missing {s!r}"


def test_slot_signatures_match_actual_code():
    # signatures referenced in the handoff must exist in the real source
    checks = {
        "control/feedback_adapter.py": "def parse_raw(raw_sim_output)",
        "control/launch_template.py": "def launch(kernel_file",
        "control/presim_gate.py": "def check_extension_calls(kernel_src",
    }
    for path, sig in checks.items():
        src = (REPO / path).read_text(encoding="utf-8")
        assert sig in src, f"{path}: signature {sig!r} not found in actual code"


def _slot_chunks():
    # each chunk starts at a '## 槽位 N' header
    return [c for c in re.split(r"(?=^## 槽位 \d)", HANDOFF, flags=re.MULTILINE)
            if re.match(r"## 槽位 \d", c)]


def test_five_slots_each_with_steps_selfcheck_and_stop_branch():
    chunks = _slot_chunks()
    assert len(chunks) == 5, f"expected 5 slot sections, got {len(chunks)}"
    for c in chunks:
        assert re.search(r"^\s*\d+\.", c, flags=re.MULTILINE), "slot missing numbered steps"
        assert ".venv/bin/python" in c, "slot missing self-check command"
        assert "停下上报" in c, "slot missing 'stop-and-escalate' branch"


def test_self_check_commands_resolve_to_real_paths():
    modules = re.findall(r"python -m (control\.\w+)", HANDOFF)
    assert modules, "no control.* self-check commands found"
    for mod in set(modules):
        p = REPO / (mod.replace(".", "/") + ".py")
        assert p.exists(), f"self-check module path does not exist: {p}"

    tests = re.findall(r"pytest (tests/[\w/]+\.py)", HANDOFF)
    for t in set(tests):
        assert (REPO / t).exists(), f"self-check test path does not exist: {t}"


def test_self_contained_no_doc_references():
    forbidden = [
        "local_adaptation_guidance.md",
        "glm52_execution_runbook.md",
        "EXECUTION_PLAN.md",
        "T10_T12_orchestrator_spec.md",
        "T12_handoff_spec.md",
    ]
    for f in forbidden:
        assert f not in HANDOFF, f"handoff is not self-contained: references {f!r}"


def test_sample_entry_marked_as_sample():
    # the example entry in the cheatsheet must be explicitly marked as a sample to replace
    sample = (REPO / ".claude" / "skills" / "extension-guide" / "references" / "sample_entry.yaml")
    txt = sample.read_text(encoding="utf-8")
    assert "SAMPLE" in txt or "示例" in txt or "sample" in txt.lower()
