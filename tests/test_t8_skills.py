"""T8 gate: 三个 SKILL.md 正文（sim-analyze / triton-gen / extension-guide）。

Structural checks (plan §3 T8). NOTE: whether a description cross-triggers is a
human judgment (manual review at stop-point ②); here we only assert structure.
  - frontmatter valid, name == dir
  - no $ARGUMENTS leftover, body English (no CJK), < 500 lines
  - every backtick path in the body exists in the repo
  - exactly the 3 skills present
  - descriptions are pairwise distinct
  - extension cheatsheet validator runs green on the sample entry
"""
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS = REPO_ROOT / ".claude" / "skills"
EXPECTED = ["sim-analyze", "triton-gen", "extension-guide"]


def _parse(skill: str):
    md = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", md, re.DOTALL)
    assert m, f"{skill}: SKILL.md missing frontmatter delimiters"
    fm = yaml.safe_load(m.group(1))
    return fm, m.group(2)


def test_exactly_three_skills():
    dirs = {p.name for p in SKILLS.iterdir() if p.is_dir()}
    assert dirs == set(EXPECTED), f"unexpected skills: {sorted(dirs)}"


@pytest.mark.parametrize("skill", EXPECTED)
def test_frontmatter_and_name(skill):
    fm, _ = _parse(skill)
    assert isinstance(fm, dict), f"{skill}: frontmatter not a mapping"
    assert fm.get("name") == skill, f"{skill}: name {fm.get('name')!r} != dir"
    desc = (fm.get("description") or "").strip()
    assert desc, f"{skill}: empty description"


@pytest.mark.parametrize("skill", EXPECTED)
def test_body_english_no_args_short(skill):
    _, body = _parse(skill)
    assert "$ARGUMENTS" not in body, f"{skill}: leftover $ARGUMENTS"
    assert not re.search(r"[一-鿿]", body), f"{skill}: body must be English (no CJK)"
    assert body.count("\n") < 500, f"{skill}: body too long ({body.count(chr(10))} lines)"


@pytest.mark.parametrize("skill", EXPECTED)
def test_body_backtick_paths_exist(skill):
    _, body = _parse(skill)
    paths = re.findall(r"`([a-zA-Z0-9_./-]+/[a-zA-Z0-9_./-]+)`", body)
    for p in paths:
        assert (REPO_ROOT / p).exists(), f"{skill}: referenced path {p!r} does not exist"


def test_descriptions_pairwise_distinct():
    descs = [_parse(s)[0]["description"] for s in EXPECTED]
    assert len(set(descs)) == len(EXPECTED), "skill descriptions are not distinct"


def test_extension_cheatsheet_validator_passes_on_sample():
    from control import check_extension_cheatsheet as cec
    assert cec.main() == 0
