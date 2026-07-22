"""瓶颈类别词表：加载、格式校验、标签断言。

词表是 single source of truth（架构文档 §5.6），被 adapter / memory /
extension-cheatsheet 三方引用。本模块只做加载与校验，不做内容定义——
内容由 control/vocabulary.yaml 给出（示例条目，待保密环境替换）。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent / "vocabulary.yaml"

_REQUIRED_FIELDS = ("id", "desc", "lever", "primitives")


@lru_cache(maxsize=8)
def load(path: Optional[str | Path] = None) -> list[dict]:
    """加载词表为条目列表。path 缺省取 DEFAULT_PATH。"""
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if isinstance(raw, dict) and "entries" in raw:
        raw = raw["entries"]
    if not isinstance(raw, list):
        raise ValueError(f"vocabulary must be a list of entries, got {type(raw).__name__}")
    return raw


def reload(path: Optional[str | Path] = None):
    """清掉缓存重新加载（测试或词表热替换时用）。"""
    load.cache_clear()


def validate_format(entries: list[dict]) -> None:
    """校验词表格式：每条含 id/desc/lever/primitives，类型正确，id 唯一。"""
    if not isinstance(entries, list):
        raise ValueError("vocabulary entries must be a list")
    seen: set[str] = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise ValueError(f"vocab entry #{i} is not a mapping: {e!r}")
        for fld in _REQUIRED_FIELDS:
            if fld not in e:
                raise ValueError(f"vocab entry #{i} missing field {fld!r}: {e!r}")
        if not isinstance(e["id"], str) or not e["id"]:
            raise ValueError(f"vocab entry #{i} has empty/non-str id: {e!r}")
        if not isinstance(e["desc"], str):
            raise ValueError(f"vocab entry {e['id']!r}: desc must be str")
        if not isinstance(e["lever"], str):
            raise ValueError(f"vocab entry {e['id']!r}: lever must be str")
        if not isinstance(e["primitives"], list):
            raise ValueError(f"vocab entry {e['id']!r}: primitives must be a list")
        if e["id"] in seen:
            raise ValueError(f"duplicate vocab id: {e['id']!r}")
        seen.add(e["id"])


def all_ids(path: Optional[str | Path] = None) -> set[str]:
    """返回词表中全部合法 id。"""
    entries = load(path)
    validate_format(entries)
    return {e["id"] for e in entries}


def assert_label(label: str, path: Optional[str | Path] = None) -> None:
    """断言 label 是词表中的合法 id，否则 ValueError。"""
    if label not in all_ids(path):
        raise ValueError(
            f"label {label!r} is not a known bottleneck category "
            f"(allowed: {sorted(all_ids(path))})"
        )


def lever_for(label: str, path: Optional[str | Path] = None) -> str:
    """返回某瓶颈类别对应的优化杠杆（供 sim-analyze 查表）。"""
    for e in load(path):
        if e["id"] == label:
            return e["lever"]
    raise ValueError(f"label {label!r} not in vocabulary")
