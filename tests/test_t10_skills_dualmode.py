"""T10 gate: 三个 skill 双模改造（dual-mode）.

Covers plan T10:
  - frontmatter still valid, descriptions unchanged (regression)
  - triton-gen / sim-analyze body placeholder set == exactly the names the orchestrator
    injects (placeholder/injection mismatch is the easiest and hardest-to-find bug)
  - each template has an Output Contract section
  - bodies still English (no CJK), no $ARGUMENTS (T8 invariants hold)
"""
import re
from pathlib import Path

from control import placeholders

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".claude" / "skills"


def _split(skill: str):
    md = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", md, re.DOTALL)
    assert m, f"{skill}: missing frontmatter"
    return m.group(1), m.group(2)


def _placeholders(body: str) -> frozenset:
    return frozenset(re.findall(r"{{([A-Z][A-Z0-9_]*)}}", body))


def test_triton_gen_placeholders_match_orchestrator():
    _, body = _split("triton-gen")
    assert _placeholders(body) == placeholders.TRITON_GEN_PLACEHOLDERS, (
        f"triton-gen placeholders {_placeholders(body)} != orchestrator set "
        f"{set(placeholders.TRITON_GEN_PLACEHOLDERS)}"
    )


def test_sim_analyze_placeholders_match_orchestrator():
    _, body = _split("sim-analyze")
    assert _placeholders(body) == placeholders.SIM_ANALYZE_PLACEHOLDERS, (
        f"sim-analyze placeholders {_placeholders(body)} != orchestrator set "
        f"{set(placeholders.SIM_ANALYZE_PLACEHOLDERS)}"
    )


def test_extension_guide_has_no_placeholders():
    _, body = _split("extension-guide")
    assert _placeholders(body) == placeholders.EXTENSION_GUIDE_PLACEHOLDERS


def test_descriptions_unchanged():
    # regression: the T8 descriptions must be preserved verbatim by dual-mode
    expected = {
        "sim-analyze": ("analyze stage", "分析瓶颈"),
        "triton-gen": ("generate stage", "生成算子"),
        "extension-guide": ("cheatsheet", "原语查询"),
    }
    for skill, (en, zh) in expected.items():
        fm, _ = _split(skill)
        assert en in fm, f"{skill}: description lost English phrase {en!r}"
        assert zh in fm, f"{skill}: description lost trigger token {zh!r}"


def test_output_contract_present_on_templates():
    for skill in ("triton-gen", "sim-analyze"):
        _, body = _split(skill)
        assert "Output Contract" in body, f"{skill}: missing Output Contract section"


def test_dual_mode_bodies_still_english_no_args():
    # T8 invariants must survive the re-templating
    for skill in ("triton-gen", "sim-analyze", "extension-guide"):
        _, body = _split(skill)
        assert "$ARGUMENTS" not in body, f"{skill}: leftover $ARGUMENTS"
        assert not re.search(r"[一-鿿]", body), f"{skill}: body must stay English (no CJK)"
