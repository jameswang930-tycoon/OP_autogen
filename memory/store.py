"""经验库:JSON 文件后端。

当前是纯文件 + 精确/放宽键匹配;
语义/向量检索是预留位,不在此实现。
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import Experience


class ExperienceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: dict[str, Experience] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._items = {k: Experience.from_dict(v) for k, v in data.items()}

    def save(self) -> None:
        data = {k: v.to_dict() for k, v in self._items.items()}
        self.path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, exp: Experience) -> str:
        self._items[exp.id] = exp
        self.save()
        return exp.id

    def get(self, exp_id: str) -> Experience | None:
        return self._items.get(exp_id)

    def all(self) -> list[Experience]:
        return list(self._items.values())

    def by_key(self, key: str) -> list[Experience]:
        """精确匹配某个算子特征主键。"""
        return [e for e in self._items.values() if e.applies_to == key]

    def by_op_kind(self, op_kind: str) -> list[Experience]:
        """放宽匹配:只按算子类型(主键形如 'op_kind|bottleneck')。"""
        return [
            e for e in self._items.values()
            if e.applies_to.split("|", 1)[0] == op_kind
        ]

    def bump(self, ids: list[str], passed: bool) -> None:
        """写回:用过次数加一;通过则 helped+1,否则 failed+1(中性负向)。

        harmed(被证实有害)预留:第一版不在此自动判定,需归因机制。
        """
        for i in ids:
            e = self._items.get(i)
            if e is None:
                continue
            e.used += 1
            if passed:
                e.helped += 1
            else:
                e.failed += 1
        self.save()
