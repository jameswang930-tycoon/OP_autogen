"""V2-P2 回归：choose_lever 候选池严格场景过滤。

漏标专用原语（缺 applies_to）在 strict 候选池里被排除——治 softmax/img2col 那类
"靠通用论据误选"。index 展示仍宽松（strict=False，缺标注视为通用、不隐藏）。"""
import control.orchestrator as orch

_ENTRIES = [
    {"name": "p_annotated", "module": None, "category": "memory_underfilled",
     "signature": None, "applies_to": ["matmul"]},
    {"name": "p_unannotated", "module": None, "category": "memory_underfilled",
     "signature": None, "applies_to": []},
]


def test_strict_candidate_pool_excludes_unannotated(monkeypatch):
    monkeypatch.setattr(orch, "_load_extension_entries", lambda: list(_ENTRIES))
    # strict=True（choose_lever 候选池）：缺 applies_to 的原语不进候选
    strict = [e["name"] for e in
              orch._relevant_entries("matmul", "memory_underfilled", strict=True)]
    assert strict == ["p_annotated"], "strict 候选池应排除缺 applies_to 的原语"


def test_lenient_index_keeps_unannotated(monkeypatch):
    monkeypatch.setattr(orch, "_load_extension_entries", lambda: list(_ENTRIES))
    # strict=False（index 展示）：缺 applies_to 视为通用、保留
    loose = [e["name"] for e in orch._relevant_entries("matmul", "memory_underfilled")]
    assert set(loose) == {"p_annotated", "p_unannotated"}


def test_strict_excludes_off_scene_annotated(monkeypatch):
    """标了 applies_to 但当前算子不在内：两种模式都排除它；未标注的在 strict 排除、lenient 保留。"""
    monkeypatch.setattr(orch, "_load_extension_entries", lambda: list(_ENTRIES))
    # strict：标 [matmul] 的（op=elementwise 不在内）排除；未标注的也被 strict 排除 -> 全空
    assert orch._relevant_entries("elementwise", "memory_underfilled", strict=True) == []
    # lenient：标 [matmul] 的排除（场景不符）；未标注的视为通用 -> 仍保留
    loose = [e["name"] for e in orch._relevant_entries("elementwise", "memory_underfilled")]
    assert loose == ["p_unannotated"]
