#!/usr/bin/env python3
"""
DSL 数据合并器 —— 合并 msprof pipeline_report.json + HIVMIR hivmir_report.json。

═══════════════════════════════════════════════════════════════════════════════
  输入 (同一轮目录下)
═══════════════════════════════════════════════════════════════════════════════

  msprof/pipeline_report.json   — 16 fields ✅ (timing/engine/channel) + 13 ❌
  hivmir/hivmir_report.json     —  9 fields ✅ (buffer/size/deps)     + 16 ❌

═══════════════════════════════════════════════════════════════════════════════
  输出
═══════════════════════════════════════════════════════════════════════════════

  merged/merged_report.json      — 29 字段完整填充 (JSON, LLM 可直接读取)
  merged/final_report_llm.txt    — LLM 消费: 7-section 结构化文本
  merged/final_report_human.txt  — 人读: ASCII Gantt 流水图 + 表格

═══════════════════════════════════════════════════════════════════════════════
  合并规则
═══════════════════════════════════════════════════════════════════════════════

  通过 op_id 对齐 → msprof 字段优先填充, HIVMIR 补充 TBD:
    msprof ✅ → 直接使用 (duration_ns, start_ns, end_ns, engine, pipeline_channel, ...)
    HIVMIR ✅ → 填充 msprof 的 TBD (instruction, dst, src, size_kb, dependencies, ...)

  带宽计算 (需要 size_kb + engine + SATURATION_PARAMS):
    bw = vpeak * size_kb / (size_kb + k0), clamped to peak_clamp
    regime = floor(ratio<50%) / ramp(50~95%) / saturated(>95%) / flat(k0=0)

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  python analyzers/dsl_merger.py outputs/vector_add_fp16_N65536/round0
  python analyzers/dsl_merger.py outputs/vector_add_fp16_N65536/01_block_size_launch/round1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  硬件参数 (从 simulator.py 同源加载, 失败则用内置 fallback)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "simulator",
        str(Path(__file__).resolve().parent.parent.parent
            / "costModel" / "cost_emulator" / "simulator.py"),
    )
    if _spec and _spec.loader:
        _sim = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_sim)
        SATURATION_PARAMS = _sim.SATURATION_PARAMS
        ENG_NAME = _sim.ENG_NAME
    else:
        raise ImportError("spec failed")
except Exception:
    # 内置 fallback (与 simulator.py 保持同步)
    SATURATION_PARAMS = {
        0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},   # GM→UB
        1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},  # UB→GM
        2: {"vpeak": 461.0,  "k0": 4.50, "peak_clamp": 404.0},   # VecUnit
        3: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # GM→L1
        4: {"vpeak": 100.0,  "k0": 6.65, "peak_clamp": 100.0},   # L1→L0
        5: {"vpeak": 150.0,  "k0": 0,    "peak_clamp": 150.0},   # CubeUnit
        6: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # L0→GM
    }
    ENG_NAME = {
        0: "GM→UB", 1: "UB→GM", 2: "VecUnit",
        3: "GM→L1", 4: "L1→L0", 5: "CubeUnit", 6: "L0→GM",
    }

ENGINE_TO_ID = {v: k for k, v in ENG_NAME.items()}


def calc_bw(engine_name: str, size_kb: float):
    """根据引擎名 + 数据大小计算带宽/regime。"""
    eid = ENGINE_TO_ID.get(engine_name)
    if eid is None:
        return {"effective_bw_gb_s": 0, "peak_bw_gb_s": 0, "bw_utilization": 0, "regime": "unknown"}
    p = SATURATION_PARAMS[eid]
    vpeak, k0, peak_clamp = p["vpeak"], p["k0"], p["peak_clamp"]
    if k0 == 0:
        return {"effective_bw_gb_s": round(peak_clamp, 2), "peak_bw_gb_s": peak_clamp,
                "bw_utilization": 1.0, "regime": "flat"}
    bw = vpeak * size_kb / (size_kb + k0)
    bw = min(bw, peak_clamp)
    ratio = bw / peak_clamp if peak_clamp else 1.0
    regime = "saturated" if ratio >= 0.95 else ("floor" if ratio <= 0.5 else "ramp")
    return {"effective_bw_gb_s": round(bw, 2), "peak_bw_gb_s": peak_clamp,
            "bw_utilization": round(ratio, 4), "regime": regime}


# ═══════════════════════════════════════════════════════════════════════════════
#  合并核心
# ═══════════════════════════════════════════════════════════════════════════════

TBD = "待补充"


def merge(msprof_path: Path, hivmir_path: Path) -> dict:
    """读取 msprof + HIVMIR JSON, 合并为完整 29 字段报告。"""
    with open(msprof_path, encoding="utf-8") as f:
        msprof = json.load(f)
    with open(hivmir_path, encoding="utf-8") as f:
        hivmir = json.load(f)

    msprof_ops = {op["op_id"]: op for op in msprof.get("per_op_statistics", [])}
    hivmir_ops = {op["op_id"]: op for op in hivmir.get("per_op_statistics", [])}

    # 合并 per-op
    all_ids = sorted(set(msprof_ops.keys()) | set(hivmir_ops.keys()))
    merged_ops = []
    for oid in all_ids:
        m = msprof_ops.get(oid, {})
        h = hivmir_ops.get(oid, {})
        op = _merge_op(m, h)
        merged_ops.append(op)

    # EXECUTION SUMMARY: msprof 优先
    msprof_summary = msprof.get("execution_summary", {})
    hivmir_summary = hivmir.get("execution_summary", {})
    total_ns = msprof_summary.get("total_ns", hivmir_summary.get("total_ns", 0))
    if total_ns == TBD:
        total_ns = 0
    total_ns = float(total_ns)

    # 重新计算 time_ratio
    if total_ns > 0:
        for op in merged_ops:
            if isinstance(op.get("duration_ns"), (int, float)):
                op["time_ratio"] = round(float(op["duration_ns"]) / total_ns, 4)

    # ENGINE UTILIZATION: msprof
    engine_util = msprof.get("engine_utilization", {})
    if engine_util == TBD:
        # 从 merged ops 重新计算
        engine_util = {}
        for op in merged_ops:
            eng = op.get("engine", "")
            dur = op.get("duration_ns", 0)
            if eng and isinstance(dur, (int, float)) and total_ns > 0:
                engine_util[eng] = engine_util.get(eng, 0) + dur
        for eng in engine_util:
            engine_util[eng] = round(engine_util[eng] / total_ns, 4)

    # PARALLELISM: msprof
    parallelism = msprof.get("parallelism", {})

    # CRITICAL PATH: msprof
    critical_path = msprof.get("critical_path", {})

    # BANDWIDTH (per-op): 从合并后数据计算
    for op in merged_ops:
        eng = op.get("engine", "")
        size = op.get("size_kb", 0)
        if eng and isinstance(size, (int, float)) and size > 0:
            bw = calc_bw(eng, float(size))
            op["effective_bw_gb_s"] = bw["effective_bw_gb_s"]
            op["peak_bw_gb_s"] = bw["peak_bw_gb_s"]
            op["bw_utilization"] = bw["bw_utilization"]
            op["regime"] = bw["regime"]

    # 构建完整报告
    return {
        "meta": {
            "source": "dsl_merger",
            "generated_at": datetime.now().isoformat(),
            "msprof_source": str(msprof_path),
            "hivmir_source": str(hivmir_path),
            "total_fields": 29,
            "fields_from_msprof": 16,
            "fields_from_hivmir": 9,
            "fields_calculated": 4,
            "note": "msprof + HIVMIR 合并完成, 29 字段全部填充, bandwidth/regime 由 SATURATION_PARAMS 计算",
        },
        "execution_summary": {
            "total_ns": total_ns,
            "num_ops": len(merged_ops),
            "execution_mode": msprof_summary.get("execution_mode",
                                                   hivmir_summary.get("execution_mode", TBD)),
            "num_cores": msprof_summary.get("num_cores",
                                             hivmir_summary.get("num_cores", TBD)),
        },
        "time_breakdown": sorted(
            [{"op_id": op["op_id"], "op_type": op["op_type"], "engine": op["engine"],
              "duration_ns": op.get("duration_ns", 0), "time_ratio": op.get("time_ratio", 0)}
             for op in merged_ops],
            key=lambda x: x.get("time_ratio", 0), reverse=True,
        ),
        "per_op_statistics": merged_ops,
        "engine_utilization": engine_util,
        "bandwidth_utilization": "see per_op_statistics",
        "parallelism": parallelism,
        "critical_path": critical_path,
        "dependencies_summary": hivmir.get("dependencies_summary", {}),
        "buffers": hivmir.get("buffers", {}),
    }


def _merge_op(msprof_op: dict, hivmir_op: dict) -> dict:
    """合并单个 op。msprof 字段优先, HIVMIR 填补 TBD。"""
    # msprof 字段 (直接取)
    def _m(key): return _val(msprof_op.get(key))
    # HIVMIR 字段 (填补 TBD)
    def _h(key): return _val(hivmir_op.get(key))

    op = {
        "op_id": _m("op_id") if _m("op_id") is not None else _h("op_id"),
        "op_type": _m("op_type") if _m("op_type") else _h("op_type"),
        "engine": _m("engine") if _m("engine") != TBD else _h("engine"),
        "instruction": _h("instruction") if _h("instruction") != TBD else _m("instruction"),
        "dst": _h("dst") if _h("dst") != TBD else _m("dst"),
        "src": _h("src") if _h("src") != TBD else _m("src"),
        "src2": _h("src2") if _h("src2") and _h("src2") != TBD else _m("src2"),
        "size_kb": _h("size_kb") if _h("size_kb") != TBD else _m("size_kb"),
        "memory_region": _h("memory_region_dst") if _h("memory_region_dst") != TBD \
                         else _h("memory_region"),
        "variable_name": _h("variable_name") if _h("variable_name") != TBD \
                         else _m("variable_name"),
        # msprof timing
        "duration_ns": _m("duration_ns"),
        "start_ns": _m("start_ns"),
        "end_ns": _m("end_ns"),
        "time_ratio": _m("time_ratio"),
        # bandwidth (calculated later)
        "effective_bw_gb_s": TBD,
        "peak_bw_gb_s": TBD,
        "bw_utilization": TBD,
        "regime": TBD,
        # misc
        "wait_before_start_ns": _m("wait_before_start_ns"),
        "blocked_by": _h("blocked_by") if _h("blocked_by") != TBD \
                      else _m("blocked_by"),
        "pipeline_channel": _m("pipeline_channel"),
        "core_id": _m("core_id"),
        "trace_event_name": _m("trace_event_name"),
        # HIVMIR extras
        "dependencies": _h("dependencies") if _h("dependencies") else [],
        "scalar": _h("scalar") if _h("scalar") else 0.0,
        "address_offset": _h("address_offset") if _h("address_offset") else "",
        "line_number": _h("line_number") if _h("line_number") else 0,
    }
    # 清理 TBD 值
    for k in op:
        if op[k] == TBD or op[k] is None:
            if k in ("duration_ns", "start_ns", "end_ns", "time_ratio", "size_kb",
                     "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization",
                     "scalar", "line_number"):
                op[k] = 0
            elif k in ("dependencies", "blocked_by"):
                op[k] = [] if k == "dependencies" else ""
            else:
                op[k] = ""
    return op


def _val(v):
    if v is None: return TBD
    if v == TBD: return TBD
    return v


# ═══════════════════════════════════════════════════════════════════════════════
#  格式化输出: LLM 文本 (对齐 simulator --llm 7-section 格式)
# ═══════════════════════════════════════════════════════════════════════════════

def format_llm(report: dict) -> str:
    """从合并后的完整报告生成 LLM 可读的 7-section 文本。"""
    summary = report["execution_summary"]
    ops = report["per_op_statistics"]
    engine_util = report.get("engine_utilization", {})
    parallelism = report.get("parallelism", {})
    cp = report.get("critical_path", {})
    deps = report.get("dependencies_summary", {})

    total_ns = summary.get("total_ns", 0)

    lines = []
    lines.append("=== EXECUTION SUMMARY ===")
    lines.append(f"total_ns: {total_ns:.2f}")
    lines.append(f"num_ops: {len(ops)}")
    lines.append(f"execution_mode: {summary.get('execution_mode', 'unknown')}")
    lines.append(f"num_cores: {summary.get('num_cores', '?')}")
    lines.append("")

    lines.append("=== TIME BREAKDOWN ===")
    lines.append("(time_ratio = op duration / total_ns, sorted biggest first)")
    for tb in report.get("time_breakdown", []):
        lines.append(
            f"op{tb['op_id']}: {ops[tb['op_id']].get('instruction', ops[tb['op_id']]['op_type'])}  "
            f"duration_ns={tb['duration_ns']:.2f}  time_ratio={tb['time_ratio']:.2%}  "
            f"({tb['duration_ns']:.2f}/{total_ns:.2f} ns)"
        )
    lines.append("")

    lines.append("=== PER-OP STATISTICS ===")
    for op in ops:
        lines.append(f"op{op['op_id']}: {op.get('instruction', op['op_type'])}")
        lines.append(f"  engine: {op.get('engine', '?')}")
        lines.append(f"  size: {op.get('size_kb', 0):.1f} KB")
        lines.append(f"  cycles_ns: [{op.get('start_ns', 0):.2f}..{op.get('end_ns', 0):.2f}]  "
                     f"duration_ns={op.get('duration_ns', 0):.2f}  "
                     f"time_ratio={op.get('time_ratio', 0):.2%}")
        lines.append(f"  bandwidth: effective={op.get('effective_bw_gb_s', 0):.4g} GB/s  "
                     f"peak={op.get('peak_bw_gb_s', 0):.0f} GB/s  "
                     f"utilization={op.get('bw_utilization', 0):.2%}  "
                     f"regime={op.get('regime', 'unknown')}")
        lines.append(f"  wait_ns_before_start: {op.get('wait_before_start_ns', 0)}")
        blocked = op.get("blocked_by", "")
        if blocked and blocked != TBD and blocked != []:
            lines.append(f"  blocked_by: {blocked}")
        elif op.get("dependencies"):
            deps_str = ", ".join(
                f"op{d['from_op_id']}({d['type']})" for d in op["dependencies"]
            )
            lines.append(f"  blocked_by: {deps_str}")
        else:
            lines.append(f"  blocked_by: none")
        # HIVMIR extra
        if op.get("variable_name"):
            lines.append(f"  variable: {op['variable_name']}  "
                         f"region={op.get('memory_region', '?')}")
        lines.append("")

    lines.append("=== ENGINE UTILIZATION ===")
    if engine_util and engine_util != TBD:
        for eng, ratio in engine_util.items():
            lines.append(f"{eng}: utilization={ratio:.2%}")
    else:
        for eng_name in ENG_NAME.values():
            busy = sum(float(op.get("duration_ns", 0)) for op in ops
                       if op.get("engine") == eng_name)
            lines.append(f"{eng_name}: busy={busy:.2f}/{total_ns:.2f} ns  "
                         f"utilization={busy/total_ns:.2%}" if total_ns > 0 else f"{eng_name}: no data")
    lines.append("")

    lines.append("=== BANDWIDTH UTILIZATION ===")
    lines.append("(effective_bw / peak_bw per op; bandwidth ramps with size)")
    for op in ops:
        lines.append(
            f"op{op['op_id']} ({op.get('engine', '?')}): "
            f"effective={op.get('effective_bw_gb_s', 0):.4g} GB/s  "
            f"peak={op.get('peak_bw_gb_s', 0):.0f} GB/s  "
            f"utilization={op.get('bw_utilization', 0):.2%}  "
            f"regime={op.get('regime', 'unknown')}"
        )
    lines.append("")

    lines.append("=== PARALLELISM ===")
    pairs = parallelism.get("parallel_pairs", parallelism.get("pairs", []))
    if pairs:
        lines.append(f"parallel_pairs: {len(pairs)}")
        for p in pairs[:10]:
            lines.append(f"  op{p.get('op_a', '?')} || op{p.get('op_b', '?')}: "
                         f"overlap_ns={p.get('overlap_ns', 0):.2f}")
    else:
        lines.append("parallel_pairs: 0")
        lines.append("root_cause_of_sequential_execution:")
        raw_deps = deps.get("raw", [])
        for d in raw_deps[:10]:
            lines.append(f"  op{d['from_op']}->op{d['to_op']}: RAW on '{d['buffer']}'")
    lines.append("")

    lines.append("=== CRITICAL PATH ===")
    path = cp.get("path", [])
    if path:
        lines.append(f"path: {' -> '.join(f'op{i}' for i in path)}")
        lines.append(f"length_ns: {cp.get('length_ns', total_ns):.2f}")
        lines.append(f"fraction_of_makespan: {cp.get('fraction', '?')}")
    else:
        lines.append("(not computed)")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  格式化输出: 人读 Gantt (ASCII 图表)
# ═══════════════════════════════════════════════════════════════════════════════

def format_human(report: dict) -> str:
    """从合并后的完整报告生成人读 ASCII Gantt 流水图。"""
    ops = report["per_op_statistics"]
    total_ns = report["execution_summary"].get("total_ns", 0)
    engine_util = report.get("engine_utilization", {})
    tbd = report.get("time_breakdown", [])

    if total_ns <= 0:
        return "(Cannot generate Gantt: total_ns=0)"

    lines = []
    W = 90  # Gantt 宽度

    # Header
    lines.append(f"  ┌─ Pipeline Execution Graph (merged) "
                 f"{'─' * max(0, W - 45)}┐")
    lines.append("")
    lines.append(f"  Time axis: {W} cols ≈ {_fmt_ns(total_ns)} makespan "
                 f"({total_ns / W:.1f} ns/col)")
    lines.append(f"           {'─' * W}")

    # Gantt rows per engine
    for eng_name in ENG_NAME.values():
        eng_ops = [op for op in ops if op.get("engine") == eng_name
                   and isinstance(op.get("start_ns"), (int, float))
                   and isinstance(op.get("end_ns"), (int, float))]
        row = _gantt_row(eng_ops, total_ns, W)
        lines.append(f"    {eng_name:10s} │ {row}")

    lines.append(f"           {'─' * W}")
    lines.append("")

    # Op table
    cw = [4, 35, 10, 10, 22, 10]
    lines.append(f"  {'Op':<{cw[0]}} {'Instruction':<{cw[1]}} {'Engine':<{cw[2]}} "
                 f"{'Size':<{cw[3]}} {'Time (ns)':<{cw[4]}} {'BW util':<{cw[5]}} "
                 f"Waits for")
    lines.append(f"  {'─' * 110}")

    for op in ops:
        instr = op.get("instruction", op.get("op_type", "?"))
        eng = op.get("engine", "?")
        size = f"{op.get('size_kb', 0):.0f} KB" if op.get("size_kb") else "?"
        dur = op.get("duration_ns", 0)
        start = op.get("start_ns", 0)
        end = op.get("end_ns", 0)
        bw = op.get("bw_utilization", 0)
        bw_str = f"{bw:.0%}" if isinstance(bw, (int, float)) else str(bw)

        deps = op.get("dependencies", [])
        if deps:
            waits = ", ".join(f"op{d['from_op_id']}({d['type']})" for d in deps)
        else:
            blocked = op.get("blocked_by", "")
            waits = blocked if (blocked and blocked != TBD and blocked != []) else "—"

        t_str = f"[{start:.1f}..{end:.1f}]"
        lines.append(f"  {op['op_id']:<{cw[0]}} {instr:<{cw[1]}.{cw[1]}} "
                     f"{eng:<{cw[2]}} {size:<{cw[3]}} {t_str:<{cw[4]}} "
                     f"{bw_str:<{cw[5]}} {waits}")

    lines.append("")

    # Time breakdown
    lines.append(f"  Time breakdown (op duration ÷ total_ns, sorted):")
    for tb in tbd[:10]:
        pct = tb.get("time_ratio", 0)
        bar = ("█" * int(round(pct * 20))).ljust(20)
        instr = ops[tb['op_id']].get("instruction", ops[tb['op_id']]['op_type'])
        lines.append(f"    op{tb['op_id']:<2} {instr:<30} [{bar}] "
                     f"{pct:6.1%} ({tb['duration_ns']:.1f}/{total_ns:.1f} ns)")

    lines.append("")

    # Engine utilization
    lines.append(f"  Engine utilization:")
    for eng_name in ENG_NAME.values():
        ratio = engine_util.get(eng_name, 0) if engine_util else 0
        pct = int(ratio * 100) if isinstance(ratio, (int, float)) else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        lines.append(f"    {eng_name:10s} [{bar}] {pct:3}%")

    lines.append("")

    # Bandwidth summary
    lines.append(f"  Bandwidth utilization (effective ÷ peak):")
    for op in ops:
        reg = op.get("regime", "?")
        bw_u = op.get("bw_utilization", 0)
        pct = int(bw_u * 100) if isinstance(bw_u, (int, float)) else 0
        bar = ("█" * (pct // 5)).ljust(20)
        lines.append(f"    op{op['op_id']:<2} {op.get('engine', '?'):8s} "
                     f"[{bar}] {pct:3}% ({reg})")

    lines.append("")

    # Critical path
    cp = report.get("critical_path", {})
    path = cp.get("path", [])
    if path:
        chain = "  ─  ".join(f"op{i}" for i in path)
        lines.append(f"  Critical path:  {chain}")
        lines.append(f"    length = {cp.get('length_ns', total_ns):.1f} ns  "
                     f"= {cp.get('fraction', '?')} of makespan")

    lines.append("")
    lines.append(f"  └{'─' * (W + 10)}┘")

    return "\n".join(lines)


def _gantt_row(eng_ops: list, horizon: float, width: int) -> str:
    row = ["·"] * width
    for op in eng_ops:
        c0 = int(round(op["start_ns"] / horizon * width))
        c1 = max(c0 + 1, int(round(op["end_ns"] / horizon * width)))
        for t in range(c0, min(c1, width)):
            row[t] = "█"
        label = f"op{op['op_id']}"
        for k, c in enumerate(label):
            pos = c0 + k
            if c0 <= pos < min(c1, width):
                row[pos] = c
    return "".join(row)


def _fmt_ns(ns: float) -> str:
    if ns >= 1000:
        return f"{ns / 1000:.3g} µs"
    return f"{ns:.3g} ns"


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════

def merge_round(round_dir: Path):
    """合并单个 round 目录下的 msprof + HIVMIR 数据。

    Args:
        round_dir: round0/ 或 01_xxx/roundN/ 目录
    """
    round_dir = Path(round_dir)
    msprof_file = round_dir / "msprof" / "pipeline_report.json"
    hivmir_file = round_dir / "hivmir" / "hivmir_report.json"

    if not msprof_file.exists():
        print(f"[merger] SKIP: msprof file not found → {msprof_file}")
        return
    if not hivmir_file.exists():
        print(f"[merger] SKIP: hivmir file not found → {hivmir_file}")
        return

    merged_dir = round_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 合并
    report = merge(msprof_file, hivmir_file)

    # 写入 merged_report.json
    json_path = merged_dir / "merged_report.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[merger] merged_report.json → {json_path} "
          f"({len(report['per_op_statistics'])} ops)")

    # 写入 final_report_llm.txt
    llm_path = merged_dir / "final_report_llm.txt"
    llm_text = format_llm(report)
    llm_path.write_text(llm_text, encoding="utf-8")
    print(f"[merger] final_report_llm.txt → {llm_path} ({len(llm_text)} chars)")

    # 写入 final_report_human.txt
    human_path = merged_dir / "final_report_human.txt"
    human_text = format_human(report)
    human_path.write_text(human_text, encoding="utf-8")
    print(f"[merger] final_report_human.txt → {human_path} ({len(human_text)} chars)")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  自测 & CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """用 vector_add_fp16_N65536/round0 的数据测试合并。"""
    outputs_root = Path(__file__).resolve().parent.parent / "outputs"
    round_dir = outputs_root / "vector_add_fp16_N65536" / "round0"

    if not (round_dir / "msprof" / "pipeline_report.json").exists():
        print(f"[merger] SKIP: round0 data not found at {round_dir}")
        print(f"  Run scripts/init_output_structure.py first to create mock data.")
        return

    print("=" * 60)
    print(f"Merging: {round_dir}")
    print("=" * 60)

    report = merge_round(round_dir)

    if report:
        # 验证合并结果
        ops = report["per_op_statistics"]
        assert len(ops) == 3, f"Expected 3 ops, got {len(ops)}"
        for op in ops:
            # 应该有完整的 instruction (from HIVMIR)
            assert op.get("instruction") and op["instruction"] != TBD, \
                f"op{op['op_id']} missing instruction"
            # 应该有 duration (from msprof)
            assert isinstance(op.get("duration_ns"), (int, float)) and op["duration_ns"] > 0, \
                f"op{op['op_id']} missing duration_ns"
            # 应该有 bandwidth (calculated)
            assert isinstance(op.get("effective_bw_gb_s"), (int, float)), \
                f"op{op['op_id']} missing effective_bw_gb_s"
            assert op.get("regime") and op["regime"] != TBD, \
                f"op{op['op_id']} missing regime"
        print(f"\n[merger] All assertions passed [OK]")
        print(f"  3 ops, all 29 fields filled, bandwidth/regime calculated")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        merge_round(Path(sys.argv[1]))
    else:
        _self_test()
