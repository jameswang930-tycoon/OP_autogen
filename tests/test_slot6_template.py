"""任务 C gate: 槽位 6 可发射模板改为文件加载 + 占位符契约冻结。

真实 kernel 内容涉密，但模板的参数化结构不涉密。框架提供 load_launchable_template
+ 冻结占位符契约；保密环境把真实 triton.py 按契约挖成模板文件即可，不改框架逻辑。
"""
import re
from pathlib import Path

from control.launch_template import (
    LAUNCHABLE_PLACEHOLDERS, assemble_launchable, load_launchable_template,
)

REPO = Path(__file__).resolve().parent.parent
TRITON_GEN = (REPO / ".claude" / "skills" / "triton-gen" / "SKILL.md").read_text(encoding="utf-8")
RAW_FIELDS = ("correct", "max_abs_err", "cycles", "pipeline", "compiled", "compile_log")


def test_load_launchable_template_returns_text():
    tmpl = load_launchable_template()
    assert isinstance(tmpl, str) and tmpl.strip()


def test_template_placeholders_frozen_and_consistent():
    tmpl = load_launchable_template()
    found = set(re.findall(r"{{([A-Z][A-Z0-9_]*)}}", tmpl))
    assert found == set(LAUNCHABLE_PLACEHOLDERS), (
        f"template placeholders {found} != frozen set {set(LAUNCHABLE_PLACEHOLDERS)}"
    )


def test_assemble_produces_valid_python_with_all_raw_fields():
    tmpl = load_launchable_template()
    values = {
        "OP": "matmul",
        "SHAPES": [1024, 1024, 1024],
        "DTYPE": "fp16",
        "KERNEL_BODY": "def kernel(a, b, c):\n    return a @ b",
        "REFERENCE": "def reference(a, b):\n    return a @ b",
    }
    assembled = assemble_launchable(tmpl, values)
    compile(assembled, "<launchable>", "exec")  # syntactically valid python
    for field in RAW_FIELDS:
        assert field in assembled, f"compare section missing raw_sim_output field {field!r}"


def test_assemble_no_leftover_placeholders():
    tmpl = load_launchable_template()
    values = {k: "x" for k in LAUNCHABLE_PLACEHOLDERS}
    assembled = assemble_launchable(tmpl, values)
    assert "{{" not in assembled, "unsubstituted placeholders remain"


def test_triton_gen_format_follows_loaded_template():
    # regression: format requirement is "as the loaded template dictates", not hardcoded
    low = TRITON_GEN.lower()
    assert "load_launchable_template" in low or "loaded template" in low or "加载到的模板" in TRITON_GEN
