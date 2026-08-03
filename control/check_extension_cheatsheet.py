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
# V2-P5.3：extension 原语按场景拆到 ext-* skill，校验需遍历多个 references 目录。
SKILLS_DIR = DEFAULT_REFS.parent.parent


def all_references_dirs() -> list[Path]:
    """extension-guide/references + ext-*/references（按场景拆分后多目录）。"""
    dirs = [DEFAULT_REFS]
    dirs += sorted(p for p in SKILLS_DIR.glob("ext-*/references") if p.is_dir())
    return dirs


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
    refs_dirs = [Path(refs_dir)] if refs_dir else all_references_dirs()
    files: list[Path] = []
    for refs in refs_dirs:
        if refs.is_dir():
            files.extend(sorted([*refs.glob("*.yaml"), *refs.glob("*.yml")]))
    if not files:
        print(f"FAIL: no YAML entries under {refs_dirs}")
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
