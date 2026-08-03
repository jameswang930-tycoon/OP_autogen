"""V2-P5.3 回归：extension-guide 按场景拆分的 ext-* skill（agent 隐式触发的场景分类）。

公开分支：建 5 个按场景的 ext-* skill（精准 description + references/ 结构），保留
extension-guide 作 nga 模式的 index 源；check_extension_cheatsheet 遍历多目录。原语真实
内容 + orchestrator 多目录读取 + extension-guide 退役 = 环境侧 P5.4（指南明确）。
"""
import re
from pathlib import Path

import yaml

from control import check_extension_cheatsheet as cec

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".claude" / "skills"
EXT_SKILLS = ["ext-reduction", "ext-activation", "ext-matmul", "ext-shape", "ext-quant"]


def _frontmatter(skill):
    md = (SKILLS / skill / "SKILL.md").read_text(encoding="utf-8")
    return yaml.safe_load(re.match(r"^---\n(.*?)\n---\n", md, re.DOTALL).group(1))


def test_ext_scene_skills_exist_with_precise_descriptions():
    """5 个 ext-* skill 各有 SKILL.md + name==dir + 说明适用场景的 description。"""
    for s in EXT_SKILLS:
        fm = _frontmatter(s)
        assert fm["name"] == s
        desc = fm["description"]
        assert "Use when" in desc, f"{s}: description 应写清适用场景（供 agent 隐式触发）"


def test_ext_skill_descriptions_pairwise_distinct():
    descs = [_frontmatter(s)["description"] for s in EXT_SKILLS]
    assert len(set(descs)) == len(EXT_SKILLS), "ext-* description 应两两不同"


def test_ext_skills_have_references_dir():
    """每个 ext-* skill 有 references/（结构同 extension-guide，env 侧填真实原语）。"""
    for s in EXT_SKILLS:
        assert (SKILLS / s / "references").is_dir(), f"{s}: 缺 references/"


def test_validator_traverses_multiple_references_dirs():
    """check_extension_cheatsheet 遍历 extension-guide + ext-*/references（多目录）。"""
    dirs = cec.all_references_dirs()
    names = {d.parent.name for d in dirs}  # parent = skill 名
    assert "extension-guide" in names
    assert set(EXT_SKILLS).issubset(names), f"多目录应含全部 ext-*，实得 {names}"
    assert cec.main() == 0  # sample 在 extension-guide，校验仍过
