#!/usr/bin/env python3
"""源3: 解析真机 msprof op 输出目录 (board_prof) → 统一格式 JSON。

op 级聚合 (整个 kernel 一行):
  - OpBasicInfo.csv            → total_ns (Task Duration us→ns), num_cores (Block Dim), kernel_name
  - PipeUtilization.csv        → 各引擎耗时 (aic_cube_time/aiv_vec_time/mte1/2/3/scalar) → per-engine pseudo-ops
  - ArithmeticUtilization.csv  → cube/vec 周期占比 + FLOPs + 指令数
  - Memory*.csv                → 各级带宽 (峰值校准用)

列名随 CANN 版本变 (ai*_ 前缀 = aic_ 或 aiv_), 用子串匹配, 未识别列进 notes。
"""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_schema import empty_op, make_report, write_json  # noqa: E402


def find_opprof(base):
    for opprof in sorted(Path(base).glob("OPPROF_*")):
        if opprof.is_dir():
            return opprof
    return None


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find_col(row, *substrings):
    """在行里找包含任一子串的列, 返回首个匹配 (值/None)。"""
    for k, v in row.items():
        kl = (k or "").lower()
        for s in substrings:
            if s.lower() in kl:
                return v
    return None


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "n/a":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse(base):
    opprof = find_opprof(base)
    if opprof is None:
        raise SystemExit(f"[board] 找不到 {base}/OPPROF_*/")
    notes = []

    # ── OpBasicInfo.csv ──
    total_ns = num_cores = kernel_name = None
    obi = opprof / "OpBasicInfo.csv"
    if obi.exists():
        rows = read_rows(obi)
        if rows:
            r = rows[0]
            dur = to_float(find_col(r, "Task Duration"))
            total_ns = round(dur * 1000.0, 2) if dur else None
            num_cores = to_float(find_col(r, "Block Dim"))
            kernel_name = find_col(r, "Op Name")
        else:
            notes.append("OpBasicInfo.csv 空")
    else:
        notes.append("无 OpBasicInfo.csv (需 --aic-metrics=... 全指标)")

    # ── PipeUtilization.csv → per-engine pseudo-ops ──
    ops = []
    engine_util = {}
    pu = opprof / "PipeUtilization.csv"
    if pu.exists():
        rows = read_rows(pu)
        if rows:
            r = rows[0]
            eng_cols = [
                ("CubeUnit", "aic_cube_time", "aic_cube_ratio"),
                ("VecUnit", "aiv_vec_time", "aiv_vec_ratio"),
                ("GM→UB", "mte2_time", "mte2_ratio"),
                ("UB→GM", "mte3_time", "mte3_ratio"),
                ("L1→L0", "mte1_time", "mte1_ratio"),
                ("Scalar", "scalar_time", "scalar_ratio"),
                ("FixPipe", "fixpipe_time", "fixpipe_ratio"),
            ]
            for eng, time_sub, ratio_sub in eng_cols:
                t = to_float(find_col(r, time_sub))
                if t is None:
                    continue
                o = empty_op()
                o.update({
                    "op_id": len(ops),
                    "op_name": eng,
                    "op_type": eng,
                    "engine": eng,
                    "instruction": f"PipeUtilization {eng}",
                    "duration_ns": round(t * 1000.0, 2),
                    "memory_region": "chip",
                    "core_id": str(find_col(r, "block_id") or num_cores),
                })
                ops.append(o)
                rat = to_float(find_col(r, ratio_sub))
                if rat is not None:
                    engine_util[eng] = round(rat, 4)
    else:
        notes.append("无 PipeUtilization.csv")

    # ── ArithmeticUtilization.csv → cube/vec 计算特征 ──
    arith = {}
    au = opprof / "ArithmeticUtilization.csv"
    if au.exists():
        rows = read_rows(au)
        if rows:
            r = rows[0]
            arith = {
                "aic_cube_ratio": to_float(find_col(r, "aic_cube_ratio")),
                "aic_cube_fops": to_float(find_col(r, "aic_cube_fops")),
                "aic_cube_instr": to_float(find_col(r, "total_instr_number")),
                "aiv_vec_ratio": to_float(find_col(r, "aiv_vec_ratio")),
                "aiv_vec_fops": to_float(find_col(r, "aiv_vec_fops")),
            }
            if arith.get("aic_cube_ratio") is not None and "CubeUnit" not in engine_util:
                engine_util["CubeUnit"] = arith["aic_cube_ratio"]
            if arith.get("aiv_vec_ratio") is not None and "VecUnit" not in engine_util:
                engine_util["VecUnit"] = arith["aiv_vec_ratio"]
    else:
        notes.append("无 ArithmeticUtilization.csv")

    # ── Memory*.csv → 带宽 (峰值校准源) ──
    bandwidth = {}
    for mf in sorted(opprof.glob("Memory*.csv")):
        rows = read_rows(mf)
        if not rows:
            continue
        r = rows[0]
        bw = {}
        for k, v in r.items():
            if v is not None and to_float(v) is not None:
                bw[k.strip()] = to_float(v)
        if bw:
            bandwidth[mf.name] = bw
    if not bandwidth:
        notes.append("无 Memory*.csv (需全指标 --aic-metrics 含 Memory)")

    # ── L2Cache.csv → L2 命中率 ──
    l2 = {}
    l2f = opprof / "L2Cache.csv"
    if l2f.exists():
        rows = read_rows(l2f)
        if rows:
            r = rows[0]
            l2 = {k.strip(): to_float(v) for k, v in r.items()
                  if to_float(v) is not None}
    else:
        notes.append("无 L2Cache.csv (910B3 A2 应支持; 若缺看是否需显式指标)")

    summary = {"total_ns": total_ns, "num_ops": len(ops),
               "execution_mode": None, "num_cores": num_cores,
               "kernel_name": kernel_name}
    report = make_report("board", [str(opprof)], summary, ops,
                         engine_util=engine_util,
                         notes=notes + ["真机 op 级聚合: 整个 kernel 一行, per-op 只有引擎级伪 op (PipeUtilization)",
                                        "total_ns 来自 Task Duration(us); 带宽/FLOPs/L2 见 bandwidth_utilization"])
    report["bandwidth_utilization"] = {"memory_bandwidth_gb_s": bandwidth,
                                       "arithmetic": arith,
                                       "l2_cache": l2}
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_board.py <board_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[board] {sys.argv[1]} -> {sys.argv[2]}")
