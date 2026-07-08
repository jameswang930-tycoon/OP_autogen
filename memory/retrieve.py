"""检索:按算子特征匹配,按可解释分数排序,取前 n 条。

排序 = 帮上忙比例(已平滑)+ 新近程度并列打破。
不引入任何学习或价值函数。
"""

from __future__ import annotations

from .schema import Experience, Fingerprint
from .store import ExperienceStore


def retrieve(store: ExperienceStore, fp: Fingerprint, n: int = 3) -> list[Experience]:
    # 1. 先精确匹配「算子类型 + 瓶颈类别」
    hits = store.by_key(fp.key())

    # 2. 不足则放宽到只匹配算子类型,补齐(去重)
    if len(hits) < n:
        seen = {e.id for e in hits}
        for e in store.by_op_kind(fp.op_kind):
            if e.id not in seen:
                hits.append(e)

    # 3. 排序:分数高者优先,并列时新的优先
    hits.sort(key=lambda e: (e.score(), e.created_at), reverse=True)
    return hits[:n]


def format_context(experiences: list[Experience]) -> str:
    """把检索到的经验拼成一段可注入提示词的文本(注入点用)。"""
    if not experiences:
        return ""
    lines = ["以下是相关的历史经验,供参考:"]
    for e in experiences:
        lines.append(f"- [{e.id}] {e.text}")
    return "\n".join(lines)
