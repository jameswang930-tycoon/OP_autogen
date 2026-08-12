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
import re
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
    if not s or s.lower() in ("n/a", "na", "nan", "none", "-", "null"):
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


def _norm_col(c):
    """列名规范化: 去括号块(单位/estimate 后缀) + 去空格 + 小写. 用于官方列名精确匹配.

    read_main_memory_datas(KB) → read_main_memory_datas
    GM_to_L1_bw_usage_rate(%)  → gm_to_l1_bw_usage_rate
    L1_to_GM_datas(KB)(estimate) → l1_to_gm_datas
    """
    return re.sub(r"\([^)]*\)", "", str(c)).replace(" ", "").lower()


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
    num_cores = _f(_first(obi, "Block", "Dim"))   # ★实际是 launch grid (Block Dim), 非物理核数
    if num_cores is None and obi:
        # ★防误匹配 Mix Block Dim (官方列序 Block Dim 在前, 但版本列序不同时兜底精确匹配)
        for k, v in obi[0].items():
            if _norm_col(k) == "blockdim":
                num_cores = _f(v)
                break
    kernel_name = _first(obi, "Op", "Name")
    freq = _f(_first(obi, "Current", "Freq"))
    # ★新: Rated Freq (理论频率, 对比 Current Freq 检测降频) / Mix Block Dim (Mix 融合算子从核数, N/A=非Mix)
    rated_freq = _f(_first(obi, "Rated", "Freq"))
    mix_block_dim = _f(_first(obi, "Mix", "Block", "Dim"))

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

    # ★新: 活跃带宽 (PipeUtilization, 官网单位 GB/s — 列名带 (GB/s) 后缀, 与 Memory.csv 的 MB/s 口径不同, 不除 1000)
    #   aic/aiv_mte2/mte3_active_bw = 真实搬运带宽 (不需字节估算); aic_mte1/mte2_active_bw 需 --aic-metrics=MemoryDetail, 合法缺
    active_bw = {
        "mte2_aiv_gb_s": _f(_first(pu, "aiv", "mte2", "active", "bw")),
        "mte3_aic_gb_s": _f(_first(pu, "aic", "mte3", "active", "bw")),
        "mte3_aiv_gb_s": _f(_first(pu, "aiv", "mte3", "active", "bw")),
        "fixpipe_aic_gb_s": _f(_first(pu, "aic", "fixpipe", "active", "bw")),
    }
    # ★新: ICache 缺失率 (ai*_icache_miss_rate, 数值越小越好 — Tier6 指令取指判据)
    icache_miss = {}
    if pu:
        for k, v in pu[0].items():
            if "icache_miss_rate" in _norm_col(k):
                key = "cube" if _norm_col(k).startswith("aic") else (
                    "vec" if _norm_col(k).startswith("aiv") else _norm_col(k))
                val = _f(v)
                if val is not None:
                    icache_miss[key] = round(val, 4)

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
    # 带宽单位换算: Ascend Memory.csv 全为 MB/s, 统一转 GB/s.
    # ★bug 修复: 原 `>= 1e4` 阈值漏掉 [1000,10000) MB/s (即 1~10 GB/s, 中小 kernel 常见) →
    #   记成 1000× 高, mem_util 变 490%+, 误判 memory_bound. 改 `>= 1000`:
    #   (本硬件原生 GB/s 值 <1000 的列不会被误除; ≥1000 MB/s 才是真实搬运带宽)
    for k in list(bandwidth):
        v = bandwidth[k]
        if v is not None and v >= 1000:
            bandwidth[k] = round(v / 1000.0, 3)  # MB/s → GB/s

    # ★新: 实际搬运量 (Memory.csv, 官方列名带 (KB) 后缀, 单位 KB) — ★比 op_summary est 估算精确 10 倍.
    #   用规范化列名精确匹配 (L1_to_GM_datas(KB)(estimate) 的 estimate 后缀会被去掉).
    #   traffic_redundancy: 实际搬了 N 次 vs 理论最小 1 次 → 分块复用/冗余搬运判据 (Tier3/4).
    mem_cols = {_norm_col(k): v for k, v in (mem[0].items() if mem else [])}
    traffic_kb = {}
    for _norm_name, _key in [
        ("read_main_memory_datas", "main_mem_read_kb"),
        ("write_main_memory_datas", "main_mem_write_kb"),
        ("gm_to_l1_datas", "gm_to_l1_kb"),
        ("l1_to_gm_datas", "l1_to_gm_kb"),
        ("l0c_to_l1_datas", "l0c_to_l1_kb"),
        ("l0c_to_gm_datas", "l0c_to_gm_kb"),
        ("gm_to_ub_datas", "gm_to_ub_kb"),
        ("ub_to_gm_datas", "ub_to_gm_kb"),
    ]:
        val = _f(mem_cols.get(_norm_name))
        if val is not None:
            traffic_kb[_key] = round(val, 1)
    # ★新: 官方通路带宽利用率 (Memory.csv, 列名带 (%) 后缀 → 归一化 0~1).
    #   GM_to_L1_bw_usage_rate / L1_to_GM_bw_usage_rate / L0C_to_L1 / L0C_to_GM / GM_to_UB / UB_to_GM
    bw_usage = {}
    for k, v in (mem[0].items() if mem else []):
        kl = _norm_col(k)
        if kl.endswith("bw_usage_rate"):
            val = _f(v)
            if val is not None:
                key = kl.replace("_bw_usage_rate", "")
                bw_usage[key] = round(val / 100.0, 4) if val > 1 else round(val, 4)

    # 计算 (ArithmeticUtilization)
    compute = {
        "cube_fops": _f(_first(au, "cube", "fops")),
        "cube_ratio": _f(_first(au, "cube", "ratio")),
        "cube_fp16_ratio": _f(_first(au, "cube", "fp16", "ratio")),
        "cube_int8_ratio": _f(_first(au, "cube", "int8", "ratio")),
        "cube_instr_number": _f(_first(au, "cube", "total", "instr")),
        "cube_fp_instr_number": _f(_first(au, "cube", "fp", "instr")),      # ★新: fp/int 指令细分 (冗余计算判断)
        "cube_int_instr_number": _f(_first(au, "cube", "int", "instr")),
        "vector_fops": _f(_first(au, "vec", "fops")),  # 真实列名 aiv_vec_fops
        "vec_ratio": _f(_first(au, "aiv", "vec", "ratio")),
        "vec_fp32_ratio": _f(_first(au, "vec", "fp32", "ratio")),
        "vec_fp16_ratio": _f(_first(au, "vec", "fp16", "ratio")),           # ★新: vec 精度细分 (vec fp32 高 → 可降 fp16)
        "vec_int32_ratio": _f(_first(au, "vec", "int32", "ratio")),
        "vec_int16_ratio": _f(_first(au, "vec", "int16", "ratio")),
        "vec_misc_ratio": _f(_first(au, "vec", "misc", "ratio")),
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
    # ★P1 修复: 加规范短名键 (planner 精确匹配, 消除 _get 子串匹配歧义).
    #   原名键保留 (向后兼容 check_fields / 旧 07 字段); wait 系列 aic/aiv 同名 → 优先 aic (cube 侧, 官方列序在前)
    _CFL_NORM = {
        "aic_cube_wait_ratio": "cube_wait_ratio",
        "aiv_vec_wait_ratio": "vec_wait_ratio",
        "aic_mte1_wait_ratio": "mte1_wait_ratio",
        "aiv_mte1_wait_ratio": "mte1_wait_ratio",
        "aic_mte2_wait_ratio": "mte2_wait_ratio",
        "aiv_mte2_wait_ratio": "mte2_wait_ratio",
        "aic_mte3_wait_ratio": "mte3_wait_ratio",
        "aiv_mte3_wait_ratio": "mte3_wait_ratio",
        "aiv_vec_total_cflt_ratio": "total_cflt_ratio",
        "aiv_vec_bank_cflt_ratio": "bank_cflt_ratio",
        "aiv_vec_bankgroup_cflt_ratio": "bankgroup_cflt_ratio",
        "aiv_vec_resc_cflt_ratio": "resc_cflt_ratio",
        "aiv_vec_mte_cflt_ratio": "mte_cflt_ratio",
    }
    for _raw, _norm in _CFL_NORM.items():
        if _norm in conflict:
            continue
        for _orig in list(conflict):
            if _orig.strip().lower() == _raw:
                conflict[_norm] = conflict[_orig]
                break

    summary = {"total_ns": round(dur_us * 1000, 2) if dur_us else None,
               "num_cores": num_cores, "kernel_name": kernel_name,
               "freq_mhz": freq, "rated_freq_mhz": rated_freq,
               "mix_block_dim": mix_block_dim}
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
            "traffic_kb": traffic_kb,             # ★新: 各通路实际搬运量 (KB)
            "bw_usage_rate": bw_usage,            # ★新: 官方通路带宽利用率 (0~1)
            "active_bw_gb_s": {k: v for k, v in active_bw.items() if v is not None},   # ★新: 活跃带宽 (GB/s)
            "icache_miss_rate": icache_miss or None,  # ★新: ICache 缺失率 (cube/vec)
        },
        "notes": notes + ["board.json = msprof op 全字段; normalized 是 LLM 用关键字段",
                          "带宽单位已统一转 GB/s (原可能 MB/s); 8 CSV 缺哪个在 raw 里可见",
                          "traffic_kb 官方实际搬运量 (Memory.csv *_datas); bw_usage_rate 官方通路利用率",
                          "active_bw 列名带 (GB/s) 后缀, 官网单位 GB/s, 不做 MB/s 换算",
                          "conflict 同时保留原始列名与规范短名 (cube_wait/vec_wait/mte*_wait 等, planner 用规范名)"],
    }
    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python pipeline_parse_board.py <board_prof目录> <out.json>")
        sys.exit(1)
    write_json(parse(sys.argv[1]), sys.argv[2])
    print(f"[board] {sys.argv[1]} -> {sys.argv[2]}")
