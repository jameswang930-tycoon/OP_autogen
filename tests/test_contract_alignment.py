"""契约对齐守卫（从 test_t12_handoff.py 保留的两条**不依赖 HANDOFF 文档**的检查）。

`HANDOFF_GLM47.md` 已有意删除（GLM4.7→5.2 迁移），读它的文档内容类测试随之退役；
这两条检查的是**代码本身**、与文档无关，故独立保留：

  - 交接包/契约引用的冻结签名在真实源码里确实存在（防签名被悄悄改掉）；
  - extension 速查表的 sample 条目仍显式标注为 sample（防被当成真实原语）。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_handoff_referenced_signatures_exist_in_source():
    """交接包/契约引用的关键签名必须在真实源码里（冻结契约守卫）。"""
    checks = {
        "control/feedback_adapter.py": "def parse_raw(raw_sim_output)",
        "control/launch_template.py": "def launch(kernel_file",
        "control/presim_gate.py": "def check_extension_calls(",
    }
    for path, sig in checks.items():
        src = (REPO / path).read_text(encoding="utf-8")
        assert sig in src, f"{path}: signature {sig!r} not found in actual code"


def test_sample_entry_marked_as_sample():
    sample = (REPO / ".claude" / "skills" / "extension-guide" / "references"
              / "sample_entry.yaml")
    txt = sample.read_text(encoding="utf-8")
    assert "SAMPLE" in txt or "示例" in txt or "sample" in txt.lower()
