"""extension 速查表格式校验（T8c）：每条原语 entry 的瓶颈类别必须在词表内。

扫描 extension-guide/references/ 下的 YAML 条目，逐条校验：
  - 必填字段齐全：name / semantics / signature / category / example / pitfalls
  - category 必须是 control/vocabulary.yaml 中的合法 id
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import vocabulary

REQUIRED_FIELDS = ("name", "semantics", "signature", "category", "example", "pitfalls")
DEFAULT_REFS = (
    Path(__file__).resolve().parent.parent
    / ".claude" / "skills" / "extension-guide" / "references"
)


def validate_entry(entry: dict, source: str) -> list[str]:
    problems: list[str] = []
    for fld in REQUIRED_FIELDS:
        if fld not in entry:
            problems.append(f"{source}: missing field {fld!r}")
    category = entry.get("category")
    if category is not None:
        try:
            vocabulary.assert_label(category)
        except Exception as exc:  # noqa: BLE001 - report any vocab rejection
            problems.append(f"{source}: {exc}")
    return problems


def main(refs_dir: str | Path | None = None) -> int:
    refs = Path(refs_dir) if refs_dir else DEFAULT_REFS
    if not refs.is_dir():
        print(f"FAIL: references dir not found: {refs}")
        return 1
    files = sorted([*refs.glob("*.yaml"), *refs.glob("*.yml")])
    if not files:
        print(f"FAIL: no YAML entries under {refs}")
        return 1
    problems: list[str] = []
    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{f.name}: yaml parse error: {exc}")
            continue
        if not isinstance(data, dict):
            problems.append(f"{f.name}: entry must be a mapping")
            continue
        problems.extend(validate_entry(data, f.name))
    if problems:
        print("FAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: {len(files)} extension entries valid (categories in vocabulary)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
