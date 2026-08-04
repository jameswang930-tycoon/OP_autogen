#!/usr/bin/env python3
"""源3: 解析 msprof op (真机单算子) → board.json — 完整提取全部 8 个 CSV 的所有字段。

msprof op 默认产出 8 个 CSV (官网核实):
  OpBasicInfo / PipeUtilization / ArithmeticUtilization / Memory / MemoryL0 /
  MemoryUB / L2Cache / ResourceConflictRatio

输出结构:
  - raw[]        每 CSV 的 所有列 + 所有行 (不遗漏任何字段)
  - normalized    关键字段标准化: engine_util(各pipe占比) / bandwidth(每通路) /
                   compute(cube/vec fops) / l2 / conflict
用法: python pipeline_parse_board.py <board_prof目录> <out.json>
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_schema import write_json  # noqa: E402

# msprof op 的 8 个 CSV 文件名 (官网核实)
CSV_FILES = ["OpBasicInfo", "PipeUtilization", "ArithmeticUtilization",
             "Memory", "MemoryL0", "MemoryUB", "L2Cache", "ResourceConflictRatio"]


def find_opprof(base):
    """base 可直接是 OPPROF 目录 (含 OpBasicInfo.csv) 或它的父目录。"""
    if (Path(base) / "OpBasicInfo.csv").exists():
        return Path(base)
    for opprof in sorted(Path(base).glob("OPPROF_*")):
        if opprof.is_dir():
            return opprof
    return None


def read_csv_all(path):
    """读 CSV: 返回 (columns, rows), 所有列都保留。"""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    columns = [c.strip() for c in rows[0]]
    data = []
    for r in rows[1:]:
        if len(r) == len(columns):
            data.append({columns[i]: r[i].strip() for i in range(len(columns))})
        elif r:
            data.append({f"col{i}": v.strip() for i, v in enumerate(r)})
    return columns, data


def _f(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("n/a", "nan", "none", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _first(rows, *keys):
    """在 rows[0] 找含所有子串的列值"""
    if not rows:
        return None
    for k, v in rows[0].items():
        kl = k.lower()
        if all(x.lower() in kl for x in keys):
            return v
    return None


def parse(base):
    opprof = find_opprof(base)
    if opprof is None:
        raise SystemExit(f"[board] 找不到 {base}/OPPROF_*/")
    notes = []

    # ── raw: 全部 CSV 全字段 ──
    raw = {}
    for name in CSV_FILES:
        p = opprof / f"{name}.csv"
        if p.exists():
            cols, rows = read_csv_all(p)
            raw[name] = {"columns": cols, "rows": rows}
        else:
            notes.append(f"缺 {name}.csv")

    # ── normalized ──
    obi = raw.get("OpBasicInfo", {}).get("rows", [])
    pu = raw.get("PipeUtilization", {}).get("rows", [])
    au = raw.get("ArithmeticUtilization", {}).get("rows", [])
    mem = raw.get("Memory", {}).get("rows", [])
    meml0 = raw.get("MemoryL0", {}).get("rows", [])
    memub = raw.get("MemoryUB", {}).get("rows", [])
    l2 = raw.get("L2Cache", {}).get("rows", [])
    rcr = raw.get("ResourceConflictRatio", {}).get("rows", [])

    # kernel 级
    dur_us = _f(_first(obi, "Task", "Duration"))
    num_cores = _f(_first(obi, "Block", "Dim"))
    kernel_name = _first(obi, "Op", "Name")
    freq = _f(_first(obi, "Current", "Freq"))

    # 引擎利用率 (PipeUtilization) — 多键子串匹配, 与 check_fields 键一致; fixp 兼容 fixpipe
    engine_util = {}
    for eng, time_k, ratio_k in [
        ("cube", ("aic", "cube", "time"), ("aic", "cube", "ratio")),
        ("vec", ("aiv", "vec", "time"), ("aiv", "vec", "ratio")),
        ("mte1", ("mte1", "time"), ("mte1", "ratio")),
        ("mte2", ("mte2", "time"), ("mte2", "ratio")),
        ("mte3", ("mte3", "time"), ("mte3", "ratio")),
        ("scalar", ("scalar", "time"), ("scalar", "ratio")),
        ("fixpipe", ("fixp", "time"), ("fixp", "ratio")),
    ]:
        t = _f(_first(pu, *time_k))
        r = _f(_first(pu, *ratio_k))
        if r is not None:
            engine_util[eng] = round(r, 4)
        elif t is not None and dur_us:
            engine_util[eng] = round(t / dur_us, 4)

    # 每通路带宽 (Memory/MemoryL0/MemoryUB) — 全列
    def _bw(rows, *keys):
        return _f(_first(rows, *keys))

    bandwidth = {
        "main_mem_read_gb_s": _bw(mem, "main", "mem", "read", "bw"),
        "main_mem_write_gb_s": _bw(mem, "main", "mem", "write", "bw"),
        "l1_read_gb_s": _bw(mem, "l1", "read", "bw"),
        "l1_write_gb_s": _bw(mem, "l1", "write", "bw"),
        "l2_read_gb_s": _bw(mem, "l2", "read", "bw"),
        "l2_write_gb_s": _bw(mem, "l2", "write", "bw"),
        # UB↔GM 真实搬运带宽 (MTE2 load / MTE3 store) — Memory.csv 真实列名:
        #   aiv_gm_to_ub_bw (GM→UB 读) / aiv_ub_to_gm_bw (UB→GM 写)
        "gm_to_ub_gb_s": _bw(mem, "gm", "to_ub", "bw"),
        "ub_to_gm_gb_s": _bw(mem, "ub", "to_gm", "bw"),
        # MemoryUB.csv 真实列名: aiv_ub_read/write_bw_vector / aiv_ub_read/write_bw_scalar
        #   (ub_read_bw_mte 仅推理产品有, 910B3 合法缺)
        "ub_vector_read_gb_s": _bw(memub, "vector", "read", "bw"),
        "ub_vector_write_gb_s": _bw(memub, "vector", "write", "bw"),
        "ub_scalar_read_gb_s": _bw(memub, "scalar", "read", "bw"),
        "ub_scalar_write_gb_s": _bw(memub, "scalar", "write", "bw"),
        "ub_mte_read_gb_s": _bw(memub, "mte", "read", "bw"),
        "ub_mte_write_gb_s": _bw(memub, "mte", "write", "bw"),
        # MemoryL0.csv 真实列名 (A2 系用 aic_ 前缀): aic_l0a_read_bw / l0c_read_bw_cube ...
        "l0a_read_gb_s": _bw(meml0, "l0a", "read", "bw"),
        "l0a_write_gb_s": _bw(meml0, "l0a", "write", "bw"),
        "l0b_read_gb_s": _bw(meml0, "l0b", "read", "bw"),
        "l0b_write_gb_s": _bw(meml0, "l0b", "write", "bw"),
        "l0c_read_gb_s": _bw(meml0, "l0c", "read", "bw"),
        "l0c_write_gb_s": _bw(meml0, "l0c", "write", "bw"),
    }
    # 带宽单位换算: 官网值常为 MB/s, 统一转 GB/s (按量级推断)
    for k in list(bandwidth):
        v = bandwidth[k]
        if v is not None and v >= 1e4:
            bandwidth[k] = round(v / 1000.0, 3)  # MB/s → GB/s

    # 计算 (ArithmeticUtilization)
    compute = {
        "cube_fops": _f(_first(au, "cube", "fops")),
        "cube_ratio": _f(_first(au, "cube", "ratio")),
        "cube_fp16_ratio": _f(_first(au, "cube", "fp16", "ratio")),
        "cube_int8_ratio": _f(_first(au, "cube", "int8", "ratio")),
        "cube_instr_number": _f(_first(au, "cube", "total", "instr")),
        "vector_fops": _f(_first(au, "vec", "fops")),  # 真实列名 aiv_vec_fops
        "vec_ratio": _f(_first(au, "aiv", "vec", "ratio")),
        "vec_fp32_ratio": _f(_first(au, "vec", "fp32", "ratio")),
        "vec_instr_number": _f(_first(au, "vec", "total", "instr")),
        "aic_total_cycles": _f(_first(au, "aic", "total", "cycle")),
        "aiv_total_cycles": _f(_first(au, "aiv", "total", "cycle")),
    }

    # L2 (列名 aic_total_hit_rate(%) 是百分数, 归一化为 0~1)
    l2_hit = None
    for k, v in (l2[0].items() if l2 else []):
        kl = k.lower()
        if "hit" in kl and _f(v) is not None:
            val = _f(v)
            l2_hit = round(val / 100.0, 4) if val > 1 else round(val, 4)
            break

    # 冲突
    conflict = {}
    for k, v in (rcr[0].items() if rcr else []):
        if _f(v) is not None:
            conflict[k.strip()] = round(_f(v), 4)

    summary = {"total_ns": round(dur_us * 1000, 2) if dur_us else None,
               "num_cores": num_cores, "kernel_name": kernel_name,
               "freq_mhz": freq}
    report = {
        "meta": {"source": "board", "generated_at": datetime.now().isoformat(),
                 "input_files": [str(opprof)], "schema_version": "2.0"},
        "execution_summary": summary,
        "raw": raw,                       # ★ 8 CSV 全字段 (不遗漏)
        "normalized": {
            "engine_utilization": engine_util,
            "bandwidth_gb_s": bandwidth,
            "compute": compute,
            "l2_hit_rate": l2_hit,
            "conflict": conflict,
        },
        "notes": notes + ["board.json = msprof op 全字段; normalized 是 LLM 用关键字段",
                          "带宽单位已统一转 GB/s (原可能 MB/s); 8 CSV 缺哪个在 raw 里可见"],
    }
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_board.py <board_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[board] {sys.argv[1]} -> {sys.argv[2]}")
