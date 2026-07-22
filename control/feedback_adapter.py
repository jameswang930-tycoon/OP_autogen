"""feedback-adapter —— 海量仿真反馈 → 弱模型能消化的短摘要（T5）。

确定性三段流水（架构文档 §3.3）：
  Reduce    事件列表 -> 关键路径 + top-k 主导代价（路径外皆为噪声）
  Classify  主导代价 -> 词表标签（bottleneck = 占总时长最大的 stall_class）
  Render    产出与旧 raw_llm 同构的 7 段摘要 + 顶部 Verdict JSON 头

7 段（与旧 simulator --llm 输出同构，triton-gen 已会读）：
  Execution Summary / Time Breakdown / Per-Op / Engine Util /
  Bandwidth Util / Parallelism / Critical Path

唯一槽位 parse_raw() 留给保密环境：真实仿真输出 -> list[Event]。
Event 字段已冻结（见 control/contracts.py），4.7 不得更改。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .contracts import Event, Verdict
from . import vocabulary

TOP_K = 5
MAX_OUTPUT_CHARS = 8192  # token 预算上限（§3.3）：4.7 的 128K 还要装速查表/记忆/kernel

SECTIONS = (
    "Execution Summary",
    "Time Breakdown",
    "Per-Op",
    "Engine Util",
    "Bandwidth Util",
    "Parallelism",
    "Critical Path",
)


def parse_raw(raw_sim_output) -> list[Event]:
    """槽位：把真实仿真输出解析为 Event 列表。由保密环境的 GLM 4.7 实现。

    Event 字段已冻结（见 control/contracts.py），不得更改。
    """
    raise NotImplementedError(
        "待保密环境实现：把真实仿真输出解析为 Event 列表"
    )


# ---------------- Reduce ----------------

@dataclass
class Reduced:
    makespan: int                  # max end - min start（= 实测总 cycles）
    total_duration: int            # 所有事件 duration 之和（含并行重叠）
    n_events: int
    dominant: list[Event]          # 按 duration 降序的 top-k 事件
    critical_path: list[Event]     # 最长「不重叠」时长链（串行关键路径）
    by_class: dict[str, int]       # stall_class -> 总 duration
    by_unit: dict[str, int]        # unit -> 总 duration
    mem_events: list[Event]        # 带 bytes 的事件（带宽视角）


def _critical_path(events: list[Event]) -> list[Event]:
    """最长不重叠时长链（加权区间调度 DP）：串行关键路径的代理。

    事件无显式依赖边，故用「时间不重叠」近似串行依赖：j.end <= i.start 才能接续。
    """
    order = sorted(events, key=lambda e: (e.end, e.start))
    n = len(order)
    if n == 0:
        return []
    best = [e.duration for e in order]
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if order[j].end <= order[i].start and best[j] + order[i].duration > best[i]:
                best[i] = best[j] + order[i].duration
                parent[i] = j
    end = max(range(n), key=lambda i: best[i])
    chain = []
    k = end
    while k != -1:
        chain.append(order[k])
        k = parent[k]
    chain.reverse()
    return chain


def reduce_events(events: list[Event], k: int = TOP_K) -> Reduced:
    if not events:
        raise ValueError("reduce_events: empty event list (nothing to analyze)")
    starts = min(e.start for e in events)
    ends = max(e.end for e in events)
    makespan = ends - starts
    total = sum(e.duration for e in events)
    by_class: dict[str, int] = {}
    by_unit: dict[str, int] = {}
    for e in events:
        by_class[e.stall_class] = by_class.get(e.stall_class, 0) + e.duration
        by_unit[e.unit] = by_unit.get(e.unit, 0) + e.duration
    dominant = sorted(events, key=lambda e: e.duration, reverse=True)[:k]
    mem = [e for e in events if e.bytes is not None]
    return Reduced(
        makespan=makespan,
        total_duration=total,
        n_events=len(events),
        dominant=dominant,
        critical_path=_critical_path(events),
        by_class=by_class,
        by_unit=by_unit,
        mem_events=mem,
    )


# ---------------- Classify ----------------

@dataclass
class Classification:
    bottleneck: str               # 占总时长最大的 stall_class（词表内）
    lever: str                    # 该类别对应的优化杠杆（vocabulary.lever_for）
    cycles: int                   # = makespan（实测总 cycles）
    expected_gain: float          # 该类时长占比 = 消除它的上界 headroom（§3.4）
    class_shares: dict[str, float]


def classify(r: Reduced) -> Classification:
    bn = max(r.by_class, key=lambda c: r.by_class[c])
    share = (r.by_class[bn] / r.total_duration) if r.total_duration else 0.0
    shares = (
        {c: round(d / r.total_duration, 3) for c, d in r.by_class.items()}
        if r.total_duration else {}
    )
    return Classification(
        bottleneck=bn,
        lever=vocabulary.lever_for(bn),
        cycles=r.makespan,
        expected_gain=round(min(share, 1.0), 3),
        class_shares=shares,
    )


def to_verdict(c: Classification) -> Verdict:
    return Verdict(
        bottleneck=c.bottleneck, lever=c.lever,
        cycles=c.cycles, expected_gain=c.expected_gain,
    )


# ---------------- Render ----------------

def _pct(x: float) -> str:
    return f"{round(x * 100)}%"


def render(r: Reduced, c: Classification) -> str:
    v = to_verdict(c)
    lines: list[str] = []
    # 顶部机器可读 Verdict JSON 头（供确定性控制面 / 人读备份；loop_controller 直接读 Verdict 对象）
    lines.append("<!-- VERDICT " + json.dumps({
        "bottleneck": v.bottleneck, "lever": v.lever,
        "cycles": v.cycles, "expected_gain": v.expected_gain,
    }, ensure_ascii=False) + " -->")

    lines.append("# Execution Summary")
    lines.append(
        f"makespan={r.makespan} cycles over {r.n_events} events. "
        f"bottleneck={c.bottleneck} (lever: {c.lever}); expected_gain={c.expected_gain}."
    )

    lines.append("# Time Breakdown")
    for cls, dur in sorted(r.by_class.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]:
        lines.append(f"  {cls}: {dur} cyc ({_pct(dur / r.total_duration) if r.total_duration else 0})")

    lines.append("# Per-Op")
    for unit, dur in sorted(r.by_unit.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]:
        lines.append(f"  {unit}: {dur} cyc")

    lines.append("# Engine Util")
    for unit, dur in sorted(r.by_unit.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]:
        lines.append(f"  {unit}: {_pct(dur / r.makespan) if r.makespan else 0}")

    lines.append("# Bandwidth Util")
    bw = sorted(r.mem_events, key=lambda e: e.duration, reverse=True)[:TOP_K]
    if bw:
        for e in bw:
            thr = (e.bytes / e.duration) if e.duration else 0
            lines.append(f"  {e.name}: {e.bytes}B / {e.duration} cyc = {round(thr, 2)} B/cyc")
    else:
        lines.append("  (no bandwidth-relevant events)")

    lines.append("# Parallelism")
    ratio = (r.total_duration / r.makespan) if r.makespan else 0.0
    lines.append(f"  sum_durations={r.total_duration}, makespan={r.makespan} -> ratio={round(ratio, 2)}")

    lines.append("# Critical Path")
    cp = r.critical_path[:TOP_K]
    if cp:
        names = " -> ".join(f"{e.name}({e.duration})" for e in cp)
        more = "" if len(r.critical_path) <= TOP_K else f"  (+{len(r.critical_path) - TOP_K} more)"
        chain_dur = sum(e.duration for e in r.critical_path)
        lines.append(f"  {names}  | total={chain_dur} cyc{more}")
    else:
        lines.append("  (empty)")

    return "\n".join(lines)


# ---------------- top-level ----------------

@dataclass
class AdapterOutput:
    summary: str        # 7 段摘要 + Verdict JSON 头（喂给 sim-analyze）
    verdict: Verdict    # 机器可读结论（喂给 loop-controller / memory）


def adapt(events: list[Event], k: int = TOP_K) -> AdapterOutput:
    """全链路：events -> (7 段摘要, Verdict)。"""
    r = reduce_events(events, k=k)
    c = classify(r)
    summary = render(r, c)
    if len(summary) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"adapter output {len(summary)} chars exceeds budget {MAX_OUTPUT_CHARS}; "
            "reduce TOP_K or fixture size"
        )
    return AdapterOutput(summary=summary, verdict=to_verdict(c))
