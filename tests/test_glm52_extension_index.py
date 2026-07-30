"""GLM52 优化指导 P3 回归：extension 原语呈现（模块归属 + 适用场景 + 按场景检索）。

针对公开分支唯一的 sample 条目（sample_entry.yaml，带 module=ext_copy、
applies_to=[matmul,conv]）验证四点：
  - 模块归属：索引渲染全限定名 module.name（杜绝 tlext1.add 式猜模块幻觉）；
  - 场景过滤：applies_to 限定后，不在场景内的算子检索不到该原语（去 img2col 式误归类污染）；
  - 保守退回：场景检索为空时索引退回全量，永不因缺标注/不匹配而隐藏全部原语；
  - 全量合法集：parse 闸门用的 _allowed_extensions 是全量、不受场景检索影响。
读取路径仍是 EXT_REFS 下的同一份 index，校验契约（check_extension_cheatsheet.py）未动。
"""
from control.orchestrator import (
    _allowed_extensions, _relevant_entries, extension_index_text)


def test_index_renders_module_attribution():
    """有 module 字段 → 索引渲染全限定名 module.name。"""
    assert "ext_copy.sample_async_copy_template" in extension_index_text()


def test_scene_filter_excludes_non_applicable_op():
    """applies_to 限定后，不在场景内的算子检索不到该原语。"""
    out_of_scene = [e["name"] for e in _relevant_entries("elementwise", "memory_underfilled")]
    assert "sample_async_copy_template" not in out_of_scene
    in_scene = [e["name"] for e in _relevant_entries("matmul", "memory_underfilled")]
    assert "sample_async_copy_template" in in_scene


def test_index_falls_back_when_scene_yields_nothing():
    """场景检索为空 → 索引退回全量，不隐藏原语。"""
    assert "sample_async_copy_template" in extension_index_text(
        "elementwise", "memory_underfilled")


def test_allowed_extensions_is_full_set_unfiltered():
    """parse 闸门的合法原语集是全量、不受场景检索影响。"""
    assert "sample_async_copy_template" in _allowed_extensions()


def test_parse_accepts_module_qualified_extension_used():
    """index 渲染 module.name；模型回写全限定名时 parse 应接受并规范化为裸名。

    否则弱模型照 index 抄 `ext_copy.sample_async_copy_template` 会被 parse 以
    "not in cheatsheet" 拒掉 → 每轮 BudgetExhausted("llm_retries")。
    """
    from control.orchestrator import parse_generate_response
    resp = ('```python\ndef k(a, b, c):\n    return a @ b\n```\n'
            '```json\n{"lever": null, "extension_used": "ext_copy.sample_async_copy_template", "notes": "x"}\n```')
    _py, meta = parse_generate_response(resp, allowed_extensions={"sample_async_copy_template"})
    assert meta["extension_used"] == "sample_async_copy_template"
