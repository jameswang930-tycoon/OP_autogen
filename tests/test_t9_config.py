"""T9 gate: OpenCode 配置。

Covers plan §3 T9:
  - opencode.json is valid JSON and contains permission.skill
  - AGENTS.md contains the three disciplines for GLM 4.7
  - skill names are unique across discovery paths (no .opencode/skills collisions)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_opencode_json_valid_with_permission_skill():
    data = json.loads((REPO / "opencode.json").read_text(encoding="utf-8"))
    assert "permission" in data and "skill" in data["permission"]
    skills = data["permission"]["skill"]
    for s in ("sim-analyze", "triton-gen", "extension-guide"):
        assert s in skills, f"opencode.json permission.skill missing {s}"


def test_agents_md_contains_three_disciplines():
    text = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    # 1) do not change frozen signatures / schema / vocabulary
    assert "签名" in text and "词表" in text
    # 2) no architecture judgment — escalate instead
    assert "架构判断" in text and "上报" in text
    # 3) run the matching self-check after each change
    assert "自检" in text


def test_skill_names_unique_across_discovery_paths():
    def names(rel):
        d = REPO / rel
        return {p.name for p in d.iterdir() if p.is_dir()} if d.exists() else set()
    dup = names(".claude/skills") & names(".opencode/skills")
    assert not dup, f"duplicate skill names across discovery paths: {sorted(dup)}"
