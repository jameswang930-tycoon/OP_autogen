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
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
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


class ParseError(Exception):
    """组件边界断言失败（E3）：坏数据在产生它的那一步被拦下，带可操作的定位信息。"""


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
    class_capacity: dict[str, dict]  # 预留(V2): stall_class -> {bytes, capacity=Σduration·unit_peak}（仅有 unit_peak 的事件计入；空→降级纯占比）


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
    # 预留(V2)：单元能力上限聚合到 class（仅有 bytes+unit_peak 的事件计入；缺省空→判定降级）
    class_capacity: dict[str, dict] = {}
    for e in events:
        if e.bytes and e.unit_peak and e.duration:
            cap = class_capacity.setdefault(e.stall_class, {"bytes": 0, "capacity": 0.0})
            cap["bytes"] += e.bytes
            cap["capacity"] += e.duration * e.unit_peak
    return Reduced(
        makespan=makespan,
        total_duration=total,
        n_events=len(events),
        dominant=dominant,
        critical_path=_critical_path(events),
        by_class=by_class,
        by_unit=by_unit,
        mem_events=mem,
        class_capacity=class_capacity,
    )


# ---------------- Classify ----------------

@dataclass
class Classification:
    bottleneck: str               # 占总时长最大的 stall_class（词表内）
    lever: str                    # 该类别对应的优化杠杆（vocabulary.lever_for）
    cycles: int                   # = makespan（实测总 cycles）
    expected_gain: float          # 该类时长占比 = 消除它的上界 headroom（§3.4）
    class_shares: dict[str, float]
    constraints: list[str] = field(default_factory=list)  # 预留(V2):已饱和(占比大但无空间)的类——约束而非真瓶颈


def classify(r: Reduced, *, saturation_threshold: Optional[float] = None) -> Classification:
    """判定瓶颈。默认纯占比（max share）。

    V2 预留利用率分支（仅当事件带 unit_peak 且给出 saturation_threshold）：
    占比大但利用率≥阈值 = 已饱和(约束，标记不死磕)；改选"占比次大但未饱和、有空间"的类作真瓶颈。
    无 unit_peak 数据 / 无阈值 → 降级为纯占比，行为与现状完全一致。"""
    ranked = sorted(r.by_class.items(), key=lambda kv: kv[1], reverse=True)
    bn = ranked[0][0]
    constraints: list[str] = []
    has_capacity = any((r.class_capacity.get(c) or {}).get("capacity", 0) > 0 for c in r.by_class)
    if saturation_threshold is not None and has_capacity:
        for c, _dur in ranked:
            cap = r.class_capacity.get(c) or {}
            capacity = cap.get("capacity", 0.0)
            if capacity > 0 and (cap.get("bytes", 0) / capacity) >= saturation_threshold:
                constraints.append(c)   # 饱和 → 约束，跳过不死磕
            else:
                bn = c                  # 未饱和（或无上限数据无法判饱和）→ 真瓶颈
                break
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
        constraints=constraints,
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
    if c.constraints:
        lines.append(
            f"  (saturated constraints, not targeted: {', '.join(c.constraints)})")

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


def adapt(events: list[Event], k: int = TOP_K,
          saturation_threshold: Optional[float] = None) -> AdapterOutput:
    """全链路：events -> (7 段摘要, Verdict)。

    saturation_threshold 缺省读 env SATURATION_THRESHOLD；未设 → None → classify 降级纯占比。"""
    if saturation_threshold is None:
        env = os.environ.get("SATURATION_THRESHOLD")
        saturation_threshold = float(env) if env else None
    validate_events(events)
    r = reduce_events(events, k=k)
    c = classify(r, saturation_threshold=saturation_threshold)
    summary = render(r, c)
    if len(summary) > MAX_OUTPUT_CHARS:
        raise ValueError(
            f"adapter output {len(summary)} chars exceeds budget {MAX_OUTPUT_CHARS}; "
            "reduce TOP_K or fixture size"
        )
    out = AdapterOutput(summary=summary, verdict=to_verdict(c))
    validate_output(out)
    return out


# ---------------- component boundary assertions (E3) ----------------

def validate_events(events) -> None:
    """parse_raw 输出边界：每条 Event 字段自洽、stall_class 在词表内、duration==end-start。

    报错信息可操作：带索引、字段、原值，让 4.7 知道去改哪个映射。
    """
    ids = vocabulary.all_ids()
    for i, e in enumerate(events):
        if e.start > e.end:
            raise ParseError(
                f"Event[{i}].start={e.start} > end={e.end} (name={e.name!r})")
        if e.duration != (e.end - e.start):
            raise ParseError(
                f"Event[{i}].duration={e.duration} but end-start={e.end - e.start} "
                f"(name={e.name!r}); fix the mapping in parse_raw")
        if e.stall_class not in ids:
            raise ParseError(
                f"Event[{i}].stall_class={e.stall_class!r} not in vocabulary {sorted(ids)} "
                f"(name={e.name!r}); map it to a vocab id or extend the vocabulary")


def validate_output(out: "AdapterOutput") -> None:
    """adapt 输出边界：Verdict.bottleneck 在词表内、cycles>=0、summary 非空。"""
    v = out.verdict
    ids = vocabulary.all_ids()
    if v.bottleneck not in ids:
        raise ParseError(
            f"Verdict.bottleneck={v.bottleneck!r} not in vocabulary {sorted(ids)}")
    if v.cycles < 0:
        raise ParseError(f"Verdict.cycles={v.cycles} must be >= 0")
    if not (out.summary and out.summary.strip()):
        raise ParseError("adapt produced an empty summary")


# ---------------- single-round replay (E2, read-only) ----------------

def _event_to_dict(e) -> dict:
    return {"name": e.name, "start": e.start, "end": e.end, "duration": e.duration,
            "unit": e.unit, "stall_class": e.stall_class, "bytes": e.bytes}


def main(argv: Optional[list] = None) -> int:
    """命令行重放入口（只读）：replay <raw.json> | adapt-only <events.json>。

    不发射、不调 LLM、不写 outputs；读落盘文件，跑该组件，打印结果或清晰错误。
    """
    import argparse
    ap = argparse.ArgumentParser(prog="control.feedback_adapter")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="parse_raw on a saved raw_sim_output json")
    rp.add_argument("raw_json")
    ao = sub.add_parser("adapt-only", help="adapt on a saved Event list json")
    ao.add_argument("events_json")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "replay":
            raw = json.loads(Path(args.raw_json).read_text(encoding="utf-8"))
            events = parse_raw(raw)
            print(json.dumps([_event_to_dict(e) for e in events], ensure_ascii=False, indent=2))
        else:  # adapt-only
            from .contracts import Event
            data = json.loads(Path(args.events_json).read_text(encoding="utf-8"))
            events = [Event(**d) for d in data]
            out = adapt(events)
            v = out.verdict
            print("VERDICT " + json.dumps({
                "bottleneck": v.bottleneck, "lever": v.lever,
                "cycles": v.cycles, "expected_gain": v.expected_gain,
            }, ensure_ascii=False))
            print("\n# --- 7-section summary ---\n" + out.summary)
    except Exception as exc:  # noqa: BLE001 - 重放要把任何错误清晰打印，不静默
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
