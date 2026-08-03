#!/usr/bin/env python3
"""源4: 解析通用 msprof (真机任务级) 的 op_summary_*.csv → task.json。

这是真机每 op 的 REAL 指标来源 (比 msprof op 丰富得多):
  - 每 op: Task Duration/aicore_time/aiv_time/total_cycles/Block Num/Input-Output shapes
  - PMU (需 --aic-mode=task-based --task-time=l1 --aic-metrics=...):
      main_mem/ub/l1/l2 read-write bw (每通路真实带宽), L2 hit,
      cube/vec fops (算力), pipe ratios (mte1/2/3, mac, vec, scalar)

输入: task_prof 目录 (含 PROF_xxx/mindstudio_profiler_output/op_summary_*.csv)
列名随版本变 → 全部按子串匹配, 不遗漏字段。
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_op_summary(base):
    """在 base 下找 mindstudio_profiler_output/op_summary_*.csv"""
    for prof in sorted(Path(base).glob("PROF_*")):
        for f in (prof / "mindstudio_profiler_output").glob("op_summary*.csv"):
            return f
    # 也直接找 op_summary*.csv
    for f in sorted(Path(base).rglob("op_summary*.csv")):
        return f
    return None


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def _f(v):
    """float 或 None"""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("n/a", "nan", "none"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_row(row):
    """从一行 op_summary 提取所有已知指标 (按子串匹配列名)。"""
    def col(*keys):
        for k, v in row.items():
            kl = (k or "").lower()
            if all(kk.lower() in kl for kk in keys):
                return v
        return None

    def colf(*keys):
        return _f(col(*keys))

    # 基础
    rec = {
        "op_name": (col("op", "name") or col("kernel", "name") or "").strip(),
        "op_type": (col("op", "type") or "").strip(),
        "task_type": (col("task", "type") or "").strip(),
        "task_duration_us": colf("task", "duration"),
        "task_start_us": colf("task", "start"),
        "task_wait_us": colf("task", "wait"),
        "aicore_time_us": colf("aicore", "time"),
        "aiv_time_us": colf("aiv", "time"),
        "total_cycles": colf("total", "cycle"),
        "block_num": colf("block", "num"),
        "input_shapes": col("input", "shape"),
        "output_shapes": col("output", "shape"),
        "data_type": col("data", "type"),
    }
    # PMU: 各通路带宽 (bw)
    bw_map = {
        "main_mem_read_bw": ("main", "mem", "read", "bw"),
        "main_mem_write_bw": ("main", "mem", "write", "bw"),
        "ub_read_bw": ("ub", "read", "bw"),
        "ub_write_bw": ("ub", "write", "bw"),
        "l1_read_bw": ("l1", "read", "bw"),
        "l1_write_bw": ("l1", "write", "bw"),
        "l2_read_bw": ("l2", "read", "bw"),
        "l2_write_bw": ("l2", "write", "bw"),
        "l0a_read_bw": ("l0a", "read", "bw"),
        "l0a_write_bw": ("l0a", "write", "bw"),
        "l0b_read_bw": ("l0b", "read", "bw"),
        "l0b_write_bw": ("l0b", "write", "bw"),
        "l0c_read_bw": ("l0c", "read", "bw"),
        "l0c_write_bw": ("l0c", "write", "bw"),
    }
    for k, keys in bw_map.items():
        rec[k] = colf(*keys)
    # L2 cache
    rec["l2_hit_rate"] = colf("cache", "hit", "rate") or colf("hit", "rate")
    rec["l2_victim_rate"] = colf("victim", "rate")
    # 算力
    rec["cube_fops"] = colf("cube", "fops")
    rec["vector_fops"] = colf("vector", "fops")
    # 搬运量
    rec["gm_to_l1_data"] = colf("gm", "to", "l1", "data")
    rec["l0c_to_gm_data"] = colf("l0c", "to", "gm", "data")
    # pipe 占比/耗时
    rec["pipe_ratios"] = {}
    rec["pipe_times"] = {}
    for pipe in ("vec", "mac", "scalar", "mte1", "mte2", "mte3", "fixpipe", "icache"):
        r = colf(pipe, "ratio")
        if r is not None:
            rec["pipe_ratios"][pipe] = r
        t = colf(pipe, "time")
        if t is not None:
            rec["pipe_times"][pipe] = t
    return rec


def parse(base):
    csv_path = find_op_summary(base)
    if csv_path is None:
        raise SystemExit(f"[task] 找不到 {base} 下 op_summary_*.csv\n"
                         f"  需先跑: msprof --output=<dir> --application='python bench_matmul.py' "
                         f"--aic-mode=task-based --task-time=l1 --aic-metrics=PipeUtilization,"
                         f"ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio")
    rows = read_rows(csv_path)
    ops = [extract_row(r) for r in rows]
    ops = [o for o in ops if o["op_name"] or o["op_type"]]

    # summary: 取最大 task_duration 作为端到端 (主 kernel)
    def _max(k):
        vals = [o[k] for o in ops if o.get(k)]
        return max(vals) if vals else None

    total_ns = _max("task_duration_us")
    total_ns = total_ns * 1000.0 if total_ns else None
    kernel = next((o["op_name"] for o in ops if "matmul" in o["op_name"].lower() or o["op_name"]),
                  ops[0]["op_name"] if ops else None)
    return {
        "meta": {
            "source": "task", "generated_at": datetime.now().isoformat(),
            "input_files": [str(csv_path)], "schema_version": "1.0",
        },
        "execution_summary": {
            "total_ns": total_ns, "num_ops": len(ops),
            "num_cores": _max("block_num"), "kernel_name": kernel,
        },
        "per_op": ops,
        "notes": ["真机任务级 op_summary: 每 kernel 调用一行, 含真实带宽/L2/算力/pipe",
                  "带宽单位按 op_summary (通常 MB/s 或 GB/s, 见列值量级); total_ns 取最大 Task Duration"],
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_task.py <task_prof目录> <out.json>")
        sys.exit(1)
    from pipeline_schema import write_json
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[task] {sys.argv[1]} -> {sys.argv[2]}")
