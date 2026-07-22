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
        self.best_path = self.path.parent / "best_cycles.json"
        self._items: dict[str, Experience] = {}
        self._best_cycles: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            self._items = {k: Experience.from_dict(v) for k, v in data.items()}
        if self.best_path.exists():
            self._best_cycles = json.loads(self.best_path.read_text(encoding="utf-8") or "{}")

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

    def update_best(self, key: str, cycles: int) -> bool:
        """登记某 fingerprint 的实测 cycles，返回「是否刷新历史最优」。

        - 首次记录（无先验）→ 记为基线，返回 False（无先验可比，不计为「帮上忙」）。
        - 严格优于历史最优 → 更新并返回 True（价值信号：性能改善，§5.2）。
        - 否则 → 不更新，返回 False。
        """
        prior = self._best_cycles.get(key)
        if prior is None:
            self._best_cycles[key] = cycles
            self._save_best()
            return False
        if cycles < prior:
            self._best_cycles[key] = cycles
            self._save_best()
            return True
        return False

    def best_cycles_for(self, key: str) -> int | None:
        return self._best_cycles.get(key)

    def best_cycles_all(self) -> dict[str, int]:
        return dict(self._best_cycles)

    def _save_best(self) -> None:
        self.best_path.write_text(
            json.dumps(self._best_cycles, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def bump(self, ids: list[str], *, helped: bool = False, failed: bool = False) -> None:
        """写回价值信号（性能驱动，§5.2）。

        - pass 轮：used+1；若 helped（在场且本轮刷新该 fingerprint 历史最优 cycles）则 helped+1。
        - FAIL 轮：failed+1（中性负向）；不碰 used/helped，故 score 不受影响。

        harmed(被证实有害)预留:第一版不在此自动判定,需归因机制。
        """
        for i in ids:
            e = self._items.get(i)
            if e is None:
                continue
            if failed:
                e.failed += 1
            else:
                e.used += 1
                if helped:
                    e.helped += 1
        self.save()
