#!/usr/bin/env python3
"""滑动窗口 — 热/温/冷三层上下文管理。"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class WindowEntry:
    """单轮记录。"""
    round: int; tier: int; tier_name: str
    strategy: str; decision: str
    actual_speedup: float; cumulative_speedup: float
    bottleneck_type: str = ""; bottleneck_op_id: int = -1
    decision_reason: str = ""
    full_context: str = ""     # 完整上下文 (仅热层保留)


class SlidingWindow:
    """三层滑动窗口。

    Hot  (最近 5 轮): 完整上下文 (代码+diff+结果+决策)
    Warm (6~15 轮前): 摘要 (策略+加速比+瓶颈)
    Cold (16+ 轮):    仅关键数据点 (round, speedup, bottleneck_type)

    用于构建注入 Planner prompt 的历史上下文。
    """

    def __init__(self, hot_size: int = 5, warm_size: int = 15):
        self.hot_size = hot_size
        self.warm_size = warm_size
        self._entries: List[WindowEntry] = []

    @property
    def total_rounds(self) -> int:
        return len(self._entries)

    def add(self, entry: WindowEntry):
        """添加新轮次, 自动分类到热/温/冷。"""
        self._entries.append(entry)

    def get_hot(self) -> List[WindowEntry]:
        """热层: 最近 N 轮完整上下文。"""
        return self._entries[-self.hot_size:] if self._entries else []

    def get_warm(self) -> List[dict]:
        """温层: N+1 ~ M 轮摘要。"""
        start = max(0, len(self._entries) - self.warm_size)
        end = max(0, len(self._entries) - self.hot_size)
        return [self._summarize(e) for e in self._entries[start:end]]

    def get_cold(self) -> List[dict]:
        """冷层: M+1 以前仅关键数据点。"""
        end = max(0, len(self._entries) - self.warm_size)
        return [{"round": e.round, "speedup": e.cumulative_speedup,
                 "bottleneck_type": e.bottleneck_type}
                for e in self._entries[:end]]

    def get_recent(self, n: int = 5) -> List[WindowEntry]:
        """获取最近 N 轮 (不分层)。"""
        return self._entries[-n:] if self._entries else []

    def get_context_for_prompt(self) -> str:
        """构建可注入 LLM prompt 的上下文文本。"""
        lines = []

        hot = self.get_hot()
        if hot:
            lines.append("## Recent Rounds (full detail)")
            for e in hot:
                lines.append(
                    f"- **Round {e.round}** (Tier {e.tier} {e.tier_name}): "
                    f"`{e.strategy}` → {e.decision} "
                    f"({e.actual_speedup:.2f}x, cumulative {e.cumulative_speedup:.2f}x)"
                )
                if e.decision_reason:
                    lines.append(f"  Reason: {e.decision_reason[:120]}")
            lines.append("")

        warm = self.get_warm()
        if warm:
            lines.append("## Earlier Rounds (summary)")
            for e in warm:
                lines.append(
                    f"- R{e['round']}: `{e['strategy'][:40]}` → "
                    f"{e['decision']} ({e['speedup']:.2f}x) "
                    f"bottleneck: {e['bottleneck_type']}"
                )
            lines.append("")

        return "\n".join(lines) if lines else "(no history)"

    def _summarize(self, entry: WindowEntry) -> dict:
        return {
            "round": entry.round,
            "tier": entry.tier,
            "strategy": entry.strategy,
            "decision": entry.decision,
            "speedup": entry.actual_speedup,
            "cumulative_speedup": entry.cumulative_speedup,
            "bottleneck_type": entry.bottleneck_type,
        }
