"""运行日志:只追加的 JSONL 文件。

对应《最小架构.md》第 1、4 节。它是唯一的事实来源;经验统计都可从它重放。
"""

from __future__ import annotations

import json
from pathlib import Path

from .schema import AttemptRecord


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: AttemptRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict]:
        out = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
