#!/usr/bin/env python3
"""源4: 解析通用 msprof (真机任务级) → task.json — 完整提取全部文件的所有字段。

文件 (官网核实):
  op_summary / op_statistic / task_time / api_statistic / l2_cache / msprof*.json
输出:
  - raw[]         每文件 所有列 + 所有行 (不遗漏)
  - normalized    每kernel耗时/核数/多kernel分解/launch开销/L2
用法: python pipeline_parse_task.py <task_prof目录> <out.json>
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import write_json  # noqa: E402


def find_prof_dir(base):
    """找 mindstudio_profiler_output 目录 (目录名拼写可能不一, 宽找)"""
    for f in Path(base).rglob("op_summary*.csv"):
        return f.parent
    return None


def read_csv_all(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    columns = [c.strip() for c in rows[0]]
    data = []
    for r in rows[1:]:
        if r:
            data.append({columns[i]: r[i].strip()
                         for i in range(min(len(r), len(columns)))})
    return columns, data


def _f(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("n/a", "nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _first(row, *keys):
    if not row:
        return None
    for k, v in row.items():
        if all(x.lower() in k.lower() for x in keys):
            return v
    return None


def parse(base):
    prof_out = find_prof_dir(base)
    if prof_out is None:
        raise SystemExit(f"[task] 找不到 {base} 下 op_summary*.csv\n"
                         f"  需先跑: msprof --output=<dir> --application='python3 test_matmul.py' --ai-core=on")

    # ── raw: 全部文件全字段 (文件名带时间戳, 按前缀匹配) ──
    def _find(prefix):
        for p in sorted(prof_out.glob(f"{prefix}*.csv")):
            return p
        return None

    raw = {}
    for prefix in ("op_summary", "op_statistic", "task_time", "api_statistic", "l2_cache"):
        p = _find(prefix)
        if p:
            cols, rows = read_csv_all(p)
            raw[prefix] = {"file": p.name, "columns": cols, "rows": rows}

    # ── normalized ──
    op_sum = raw.get("op_summary", {}).get("rows", [])
    op_stat = raw.get("op_statistic", {}).get("rows", [])
    api_stat = raw.get("api_statistic", {}).get("rows", [])
    l2 = raw.get("l2_cache", {}).get("rows", [])

    # 每 kernel 耗时 (op_summary 每行一个 kernel)
    kernels = []
    for r in op_sum:
        kernels.append({
            "op_name": _first(r, "Op", "Name") or _first(r, "op", "name"),
            "task_duration_us": _f(_first(r, "Task", "Duration")),
            "task_start_us": _f(_first(r, "Task", "Start")),
            "task_wait_us": _f(_first(r, "Task", "Wait")),
            "block_dim": _f(_first(r, "Block", "Dim")),
            "task_type": _first(r, "Task", "Type"),
            "input_shapes": _first(r, "Input", "Shape"),
            "output_shapes": _first(r, "Output", "Shape"),
            "aicore_time_us": _f(_first(r, "aicore", "time")),
            "aiv_time_us": _f(_first(r, "aiv", "time")),
            "total_cycles": _f(_first(r, "total", "cycle")),
        })

    # 多 kernel 分解 (op_statistic: 每类算子 次数/总耗时)
    multi_kernel = []
    for r in op_stat:
        multi_kernel.append({
            "op_type": _first(r, "OP", "Type") or _first(r, "top", "type"),
            "core_type": _first(r, "Core", "Type"),
            "count": _f(_first(r, "Count")),
            "total_time_us": _f(_first(r, "Total", "Time")),
            "avg_us": _f(_first(r, "Avg")), "min_us": _f(_first(r, "Min")),
            "max_us": _f(_first(r, "Max")), "ratio": _f(_first(r, "Ratio")),
        })

    # launch/API 开销
    api_overhead = []
    for r in api_stat:
        api_overhead.append({
            "level": _first(r, "Level"), "api_name": _first(r, "API", "Name"),
            "total_us": _f(_first(r, "Time")), "count": _f(_first(r, "Count")),
            "avg_us": _f(_first(r, "Avg")), "max_us": _f(_first(r, "Max")),
        })

    # L2
    l2_hit = None
    if l2:
        for k, v in l2[0].items():
            if "hit" in k.lower() and _f(v) is not None:
                l2_hit = round(_f(v), 4)
                break

    # summary
    def _max(k):
        vals = [x.get(k) for x in kernels if x.get(k)]
        return max(vals) if vals else None
    total_ns = _max("task_duration_us")
    total_ns = total_ns * 1000 if total_ns else None
    kernel = next((k["op_name"] for k in kernels if k["op_name"]), None)

    report = {
        "meta": {"source": "task", "generated_at": datetime.now().isoformat(),
                 "input_files": [str(prof_out)], "schema_version": "2.0"},
        "execution_summary": {"total_ns": total_ns,
                              "num_cores": _max("block_dim"),
                              "kernel_name": kernel,
                              "num_kernels": len(kernels)},
        "raw": raw,                       # ★ 全文件全字段 (不遗漏)
        "normalized": {
            "kernels": kernels,
            "multi_kernel": multi_kernel,
            "api_overhead": api_overhead,
            "l2_hit_rate": l2_hit,
        },
        "notes": ["task.json = 通用 msprof 全字段; normalized 是 LLM 用关键字段",
                  "op_summary 每 kernel 一行; api_overhead 判断 launch 开销; multi_kernel 判断是否值得 kernel 融合"],
    }
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_task.py <task_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[task] {sys.argv[1]} -> {sys.argv[2]}")
