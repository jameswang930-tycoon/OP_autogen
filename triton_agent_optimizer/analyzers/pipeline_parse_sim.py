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


def parse_instr_csv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            rows.append(rec)
    return rows


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
                    "cycles": 0, "running_time_us": 0.0, "cores": set(), "detail": "",
                })
                a["call_count"] += int(rec.get("call_count") or 0)
                a["cycles"] += int(rec.get("cycles") or 0)
                a["running_time_us"] += float(rec.get("running_time(us)") or 0)
                a["cores"].add(core_dir.name)
                if not a["detail"]:
                    a["detail"] = rec.get("detail", "")

    ops = []
    for key in sorted(agg.keys()):
        a = agg[key]
        data_bytes, dtype = parse_detail(a["detail"])
        per_call_us = a["running_time_us"] / max(1, a["call_count"])
        per_call_cyc = a["cycles"] // max(1, a["call_count"])
        o = empty_op()
        o.update({
            "op_id": len(ops),
            "op_name": a["instr"],                     # 可对齐名 = 指令名
            "op_type": a["instr"],
            "engine": PIPE_TO_ENGINE.get(a["pipe"], a["pipe"]),
            "instruction": f"{a['instr']} (pipe={a['pipe']})",
            "pipeline_channel": a["pipe"],
            # duration_ns = 单次调用平均耗时 (对齐语义 op 用); 总量在 total_duration_ns
            "duration_ns": round(per_call_us * 1000.0, 2),
            "cycles": per_call_cyc,
            "call_count": a["call_count"],
            "total_duration_ns": round(a["running_time_us"] * 1000.0, 2),
            "total_cycles": a["cycles"],
            "core_id": f"{len(a['cores'])} cores",
            "data_size_bytes": data_bytes,
            "dtype": dtype,
            "size_kb": round(data_bytes / 1024.0, 3) if data_bytes else None,
        })
        ops.append(o)

    # trace.json → summary
    total_ns = exec_mode = n_trace_cores = None
    parallelism, critical_path = {}, {}
    trace = sim / "trace.json"
    if trace.exists():
        ti = MsprofParser.parse_trace_json(trace)
        total_ns = ti.get("total_ns")
        exec_mode = ti.get("execution_mode")
        parallelism = {"parallel_pairs": ti.get("parallel_pairs"),
                       "total_pairs": len(ti.get("parallel_pairs") or [])}
        critical_path = {"path": ti.get("critical_path"),
                         "length_ns": ti.get("critical_path_length_ns")}

    summary = {"total_ns": total_ns, "num_ops": len(ops),
               "execution_mode": exec_mode, "num_cores": n_cores or n_trace_cores,
               "kernel_name": None}
    return make_report("sim", [str(sim)], summary, ops,
                       parallelism=parallelism, critical_path=critical_path,
                       notes=[f"{n_cores} 核 instr_exe 按指令名聚合; 未匹配 HIVM op 的指令在 merge 时做引擎/pipe 对齐",
                              "SCALAR 指令已保留 (engine=SCALAR), 如需排除在 merge 里过滤"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_sim.py <sim_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[sim] {sys.argv[1]} -> {sys.argv[2]}")
