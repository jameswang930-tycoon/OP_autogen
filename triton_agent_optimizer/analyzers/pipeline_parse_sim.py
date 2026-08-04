#!/usr/bin/env python3
"""源2: 解析 msprof op simulator 输出目录 → 统一格式 JSON。

输入: sim_prof 目录 (含 OPPROF_xxx/simulator/core*/...instr_exe.csv + trace.json)
指令级时序真实; 按 (指令名, pipe) 跨核聚合 (sum cycles/duration, 计核数)。
trace.json 提供 summary (total_ns/execution_mode/parallelism/critical_path)。
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from msprof_analyzer import MsprofParser  # noqa: E402
from pipeline_schema import empty_op, make_report, write_json  # noqa: E402

PIPE_TO_ENGINE = {
    "MTE2": "GM→UB", "MTE3": "UB→GM", "VECTOR": "VecUnit",
    "CUBE": "CubeUnit", "MTE1": "L1→L0",
}


def find_sim_dir(base):
    for opprof in sorted(Path(base).glob("OPPROF_*")):
        sim = opprof / "simulator"
        if sim.is_dir():
            return sim
    return None


# AI Core 频率 (MHz) — duration 用 cycles 换算的时钟; 910B3 ≈ 1900 MHz (可用 AIC_FREQ_MHZ 覆盖)
AIC_FREQ_MHZ = 1900.0


def parse_instr_csv(path):
    """鲁棒解析 instr_exe.csv: 表头列名可能随版本变, 按子串匹配列位置。

    真实数据 (Ascend910B3) 常见: 指令的 running_time(us)=0, 只有 cycles 有值。
    因此每条指令的 duration 优先 running_time(us), 为 0 时用 cycles/freq 换算。
    """
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        col = {}
        for i, h in enumerate(header):
            hl = (h or "").strip().lower()
            for key in ("instr", "addr", "pipe", "call_count", "cycles", "running", "detail"):
                if key in hl and key not in col:
                    col[key] = i

        def g(r, k):
            i = col.get(k)
            return r[i].strip() if i is not None and i < len(r) else ""

        for r in reader:
            if len(r) < 3:
                continue
            cyc = g(r, "cycles")
            rt = g(r, "running")
            dur_ns = 0.0
            try:
                rt_us = float(rt)
                if rt_us > 0:
                    dur_ns = rt_us * 1000.0
            except ValueError:
                pass
            if dur_ns == 0.0:
                try:
                    dur_ns = int(cyc) * 1000.0 / AIC_FREQ_MHZ
                except ValueError:
                    dur_ns = 0.0
            rows.append({
                "instr": g(r, "instr"), "pipe": g(r, "pipe"),
                "call_count": g(r, "call_count") or "1",
                "cycles": cyc, "duration_ns": dur_ns,
                "detail": g(r, "detail"),
            })
    return rows


def parse_trace_events(trace_path):
    """解析 trace.json (Chrome trace) 的 ph=X 事件 → 按 (名称,通道) 聚合 start/end。

    官方: trace.json 是指令流水图, 每条指令有 ts/dur → 补 instr_exe 缺失的 start/end。
    返回 { (name, cat): {name, cat, start_ns, end_ns, count} }
    """
    import json as _json
    data = _json.loads(Path(trace_path).read_text(encoding="utf-8"))
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    agg = {}
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        ts = float(e.get("ts", 0))
        dur = float(e.get("dur", 0))
        key = (name, cat)
        a = agg.setdefault(key, {"name": name, "cat": cat,
                                 "start_ns": ts, "end_ns": ts + dur, "count": 0})
        a["start_ns"] = min(a["start_ns"], ts)
        a["end_ns"] = max(a["end_ns"], ts + dur)
        a["count"] += 1
    return agg


def parse_detail(detail_str):
    """从 detail 提取搬运数据块大小 (字节) 与 dtype。detail 逗号分隔的 K:V。"""
    size, dtype = 0, ""
    for part in str(detail_str).split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k in ("XD:X3", "XM:X2", "XD:X2", "X2", "X3"):
                try:
                    size = int(v, 16)
                except ValueError:
                    pass
            elif k.strip() == "Dtype":
                dtype = v.strip()
    return size, dtype


def parse(base):
    sim = find_sim_dir(base)
    if sim is None:
        raise SystemExit(f"[sim] 找不到 {base}/OPPROF_*/simulator/")

    agg = {}
    n_cores = 0
    for core_dir in sorted(sim.glob("core*")):
        n_cores += 1
        for csvf in core_dir.glob("*_instr_exe.csv"):
            for rec in parse_instr_csv(csvf):
                inst = rec.get("instr", "").strip()
                pipe = rec.get("pipe", "").strip()
                if not inst or not pipe:
                    continue
                key = (inst, pipe)
                a = agg.setdefault(key, {
                    "instr": inst, "pipe": pipe, "call_count": 0,
                    "cycles": 0, "duration_ns_total": 0.0, "cores": set(), "detail": "",
                })
                a["call_count"] += int(rec["call_count"] or 1)
                a["cycles"] += int(rec["cycles"] or 0)
                a["duration_ns_total"] += rec["duration_ns"]
                a["cores"].add(core_dir.name)
                if not a["detail"]:
                    a["detail"] = rec["detail"]

    ops = []
    for key in sorted(agg.keys()):
        a = agg[key]
        data_bytes, dtype = parse_detail(a["detail"])
        n = max(1, a["call_count"])
        per_call_ns = a["duration_ns_total"] / n
        per_call_cyc = a["cycles"] // n
        o = empty_op()
        o.update({
            "op_id": len(ops),
            "op_name": a["instr"],                     # 可对齐名 = 指令名
            "op_type": a["instr"],
            "engine": PIPE_TO_ENGINE.get(a["pipe"], a["pipe"]),
            "instruction": f"{a['instr']} (pipe={a['pipe']})",
            "pipeline_channel": a["pipe"],
            # duration_ns = 单次调用平均耗时 (running_time 优先, 0 时用 cycles/freq 换算)
            "duration_ns": round(per_call_ns, 2),
            "cycles": per_call_cyc,
            "call_count": a["call_count"],
            "total_duration_ns": round(a["duration_ns_total"], 2),
            "total_cycles": a["cycles"],
            "core_id": f"{len(a['cores'])} cores",
            "data_size_bytes": data_bytes,
            "dtype": dtype,
            "size_kb": round(data_bytes / 1024.0, 3) if data_bytes else None,
        })
        ops.append(o)

    # trace.json → summary + 逐指令 start/end (关键路径/并行/时序)
    total_ns = exec_mode = n_trace_cores = None
    parallelism, critical_path, trace_events = {}, {}, {}
    trace = sim / "trace.json"
    if trace.exists():
        ti = MsprofParser.parse_trace_json(trace)
        total_ns = ti.get("total_ns")
        exec_mode = ti.get("execution_mode")
        parallelism = {"parallel_pairs": ti.get("parallel_pairs"),
                       "total_pairs": len(ti.get("parallel_pairs") or [])}
        critical_path = {"path": ti.get("critical_path"),
                         "length_ns": ti.get("critical_path_length_ns")}
        trace_events = parse_trace_events(trace)

    summary = {"total_ns": total_ns, "num_ops": len(ops),
               "execution_mode": exec_mode, "num_cores": n_cores or n_trace_cores,
               "kernel_name": None}
    report = make_report("sim", [str(sim)], summary, ops,
                         parallelism=parallelism, critical_path=critical_path,
                         notes=[f"{n_cores} 核 instr_exe 按指令名聚合; 未匹配 HIVM op 的指令在 merge 时做引擎/pipe 对齐",
                                "SCALAR 指令已保留 (engine=SCALAR), 如需排除在 merge 里过滤",
                                "trace_events = trace.json 逐指令 start/end (Chrome trace, ph=X)"])
    report["trace_events"] = trace_events
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_sim.py <sim_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[sim] {sys.argv[1]} -> {sys.argv[2]}")
