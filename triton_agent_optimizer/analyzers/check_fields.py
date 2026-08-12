#!/usr/bin/env python3
"""字段覆盖检验 — 每个没提取到的字段, 精准告诉你"去哪个工具/哪个文件/找哪个具体列名"。

三类判定:
  OK        字段有数据
  列名不匹配   normalized 是 None, 但期望文件里有该列且有值 → parser 键和实际列名不一致 (BUG/列名变)
  合法缺      normalized 是 None, 且期望文件里该列不存在或值为空 → 版本/产品无此数据 (正常)

用法: python check_fields.py <board.json> <task.json> [diagnosis.json]
退出码: 0 = 无列名不匹配; 1 = 有列名不匹配 (需修 parser)
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else {}


# 期望字段 → (工具, 文件, 具体列名, 匹配键[, 排除键])
#   工具: "msprof op" = OPPROF 下 CSV; "msprof" = 通用 msprof 的 mindstudio_profiler_output
#   具体列名 = 本该从哪个列提取 (去真机核对这个)
#   匹配键    = 子串匹配实际列名用
BOARD_EXPECT = {
    "bandwidth": [
        ("main_mem_read_gb_s",  "msprof op", "Memory",    "aic_main_mem_read_bw",  ("main", "mem", "read", "bw")),
        ("main_mem_write_gb_s", "msprof op", "Memory",    "aic_main_mem_write_bw", ("main", "mem", "write", "bw")),
        ("l1_read_gb_s",        "msprof op", "Memory",    "aic_l1_read_bw",        ("l1", "read", "bw")),
        ("l1_write_gb_s",       "msprof op", "Memory",    "aic_l1_write_bw",       ("l1", "write", "bw")),
        ("l2_read_gb_s",        "msprof op", "Memory",    "(无此列, A2系合法缺)",   ("l2", "read", "bw")),
        ("l2_write_gb_s",       "msprof op", "Memory",    "(无此列, A2系合法缺)",   ("l2", "write", "bw")),
        ("gm_to_ub_gb_s",       "msprof op", "Memory",    "aiv_gm_to_ub_bw",       ("gm", "to_ub", "bw")),
        ("ub_to_gm_gb_s",       "msprof op", "Memory",    "aiv_ub_to_gm_bw",       ("ub", "to_gm", "bw")),
        ("ub_vector_read_gb_s", "msprof op", "MemoryUB",  "aiv_ub_read_bw_vector", ("vector", "read", "bw")),
        ("ub_vector_write_gb_s", "msprof op", "MemoryUB", "aiv_ub_write_bw_vector", ("vector", "write", "bw")),
        ("ub_scalar_read_gb_s", "msprof op", "MemoryUB",  "aiv_ub_read_bw_scalar", ("scalar", "read", "bw")),
        ("ub_scalar_write_gb_s", "msprof op", "MemoryUB", "aiv_ub_write_bw_scalar", ("scalar", "write", "bw")),
        ("ub_mte_read_gb_s",    "msprof op", "MemoryUB",  "ub_read_bw_mte(仅推理,910B3无)", ("mte", "read", "bw")),
        ("ub_mte_write_gb_s",   "msprof op", "MemoryUB",  "ub_write_bw_mte(仅推理,910B3无)", ("mte", "write", "bw")),
        ("l0a_read_gb_s",       "msprof op", "MemoryL0",  "aic_l0a_read_bw",       ("l0a", "read", "bw")),
        ("l0a_write_gb_s",      "msprof op", "MemoryL0",  "aic_l0a_write_bw",      ("l0a", "write", "bw")),
        ("l0b_read_gb_s",       "msprof op", "MemoryL0",  "aic_l0b_read_bw",       ("l0b", "read", "bw")),
        ("l0b_write_gb_s",      "msprof op", "MemoryL0",  "aic_l0b_write_bw",      ("l0b", "write", "bw")),
        ("l0c_read_gb_s",       "msprof op", "MemoryL0",  "l0c_read_bw_cube",      ("l0c", "read", "bw")),
        ("l0c_write_gb_s",      "msprof op", "MemoryL0",  "l0c_write_bw_cube",     ("l0c", "write", "bw")),
    ],
    "engine_utilization": [
        ("cube",    "msprof op", "PipeUtilization",       "aic_cube_ratio(或aic_mac_ratio)", ("aic", "cube", "ratio")),
        ("vec",     "msprof op", "PipeUtilization",       "aiv_vec_ratio",                    ("aiv", "vec", "ratio")),
        ("mte1",    "msprof op", "PipeUtilization",       "aic_mte1_ratio",                   ("mte1", "ratio")),
        ("mte2",    "msprof op", "PipeUtilization",       "aic_mte2_ratio/aiv_mte2_ratio",    ("mte2", "ratio")),
        ("mte3",    "msprof op", "PipeUtilization",       "aic_mte3_ratio/aiv_mte3_ratio",    ("mte3", "ratio")),
        ("scalar",  "msprof op", "PipeUtilization",       "aic_scalar_ratio/aiv_scalar_ratio", ("scalar", "ratio")),
        ("fixpipe", "msprof op", "PipeUtilization",       "aic_fixpipe_ratio(或aic_fixp_ratio)", ("fixp", "ratio")),
    ],
    "compute": [
        ("cube_fops",        "msprof op", "ArithmeticUtilization", "aic_cube_fops",       ("cube", "fops")),
        ("cube_ratio",       "msprof op", "ArithmeticUtilization", "aic_cube_ratio",      ("cube", "ratio")),
        ("cube_fp16_ratio",  "msprof op", "ArithmeticUtilization", "aic_cube_fp16_ratio", ("cube", "fp16", "ratio")),
        ("cube_int8_ratio",  "msprof op", "ArithmeticUtilization", "aic_cube_int8_ratio", ("cube", "int8", "ratio")),
        ("cube_instr_number", "msprof op", "ArithmeticUtilization", "aic_cube_total_instr_number", ("cube", "total", "instr")),
        ("cube_fp_instr_number", "msprof op", "ArithmeticUtilization", "aic_cube_fp_instr_number", ("cube", "fp", "instr")),
        ("vector_fops",      "msprof op", "ArithmeticUtilization", "aiv_vec_fops",        ("vec", "fops")),
        ("vec_ratio",        "msprof op", "ArithmeticUtilization", "aiv_vec_ratio",       ("aiv", "vec", "ratio")),
        ("vec_fp32_ratio",   "msprof op", "ArithmeticUtilization", "aiv_vec_fp32_ratio",  ("vec", "fp32", "ratio")),
        ("vec_fp16_ratio",   "msprof op", "ArithmeticUtilization", "aiv_vec_fp16_ratio",  ("vec", "fp16", "ratio")),
        ("aic_total_cycles", "msprof op", "ArithmeticUtilization", "aic_total_cycles",    ("aic", "total", "cycle")),
        ("aiv_total_cycles", "msprof op", "ArithmeticUtilization", "aiv_total_cycles",    ("aiv", "total", "cycle")),
    ],
    "conflict": [
        ("bank_cflt_ratio",      "msprof op", "ResourceConflictRatio", "aiv_vec_bank_cflt_ratio",      ("vec", "bank", "cflt"), ("bankgroup",)),
        ("bankgroup_cflt_ratio", "msprof op", "ResourceConflictRatio", "aiv_vec_bankgroup_cflt_ratio", ("vec", "bankgroup", "cflt")),
        ("total_cflt_ratio",     "msprof op", "ResourceConflictRatio", "aiv_vec_total_cflt_ratio",     ("vec", "total", "cflt")),
        ("resc_cflt_ratio",      "msprof op", "ResourceConflictRatio", "aiv_vec_resc_cflt_ratio",      ("vec", "resc", "cflt")),
        ("mte_cflt_ratio",       "msprof op", "ResourceConflictRatio", "aiv_vec_mte_cflt_ratio",       ("vec", "mte", "cflt")),
        ("cube_wait_ratio",      "msprof op", "ResourceConflictRatio", "aic_cube_wait_ratio",          ("cube", "wait", "ratio")),
        ("vec_wait_ratio",       "msprof op", "ResourceConflictRatio", "aiv_vec_wait_ratio",           ("vec", "wait", "ratio")),
        ("mte1_wait_ratio",      "msprof op", "ResourceConflictRatio", "aic/aiv_mte1_wait_ratio",      ("mte1", "wait", "ratio")),
        ("mte2_wait_ratio",      "msprof op", "ResourceConflictRatio", "aic/aiv_mte2_wait_ratio",      ("mte2", "wait", "ratio")),
        ("mte3_wait_ratio",      "msprof op", "ResourceConflictRatio", "aic/aiv_mte3_wait_ratio",      ("mte3", "wait", "ratio")),
    ],
}

# ★新 (P2): 官方实测字段 — 列名带 (KB)/(%)/(GB/s) 后缀, 用规范化列名精确匹配 (子串会误匹配 L1_to_GM vs GM_to_L1)
BOARD_EXPECT_NORM = {
    "traffic_kb": [   # (normalized键, 官方列名)
        ("main_mem_read_kb", "read_main_memory_datas(KB)"),
        ("main_mem_write_kb", "write_main_memory_datas(KB)"),
        ("gm_to_l1_kb", "GM_to_L1_datas(KB)"),
        ("l1_to_gm_kb", "L1_to_GM_datas(KB)(estimate)"),
        ("l0c_to_l1_kb", "L0C_to_L1_datas(KB)"),
        ("l0c_to_gm_kb", "L0C_to_GM_datas(KB)"),
        ("gm_to_ub_kb", "GM_to_UB_datas(KB)"),
        ("ub_to_gm_kb", "UB_to_GM_datas(KB)"),
    ],
    "bw_usage_rate": [
        ("gm_to_l1", "GM_to_L1_bw_usage_rate(%)"),
        ("l1_to_gm", "L1_to_GM_bw_usage_rate(%)(estimate)"),
        ("l0c_to_l1", "L0C_to_L1_bw_usage_rate(%)"),
        ("l0c_to_gm", "L0C_to_GM_bw_usage_rate(%)"),
        ("gm_to_ub", "GM_to_UB_bw_usage_rate(%)"),
        ("ub_to_gm", "UB_to_GM_bw_usage_rate(%)"),
    ],
    "active_bw_gb_s": [
        ("mte2_aiv_gb_s", "aiv_mte2_active_bw(GB/s)"),
        ("mte3_aic_gb_s", "aic_mte3_active_bw(GB/s)"),
        ("mte3_aiv_gb_s", "aiv_mte3_active_bw(GB/s)"),
        ("fixpipe_aic_gb_s", "aic_fixpipe_active_bw(GB/s)"),
    ],
    "icache_miss_rate": [
        ("cube", "aic_icache_miss_rate"),
        ("vec", "aiv_icache_miss_rate"),
    ],
}


def _norm_col(c):
    """列名规范化: 去括号块(单位/estimate后缀) + 去空格 + 小写 (与 pipeline_parse_board 同实现)."""
    import re
    return re.sub(r"\([^)]*\)", "", str(c)).replace(" ", "").lower()


def _match(name, keys, exclude=()):
    cl = name.lower()
    return all(k.lower() in cl for k in keys) and not any(x.lower() in cl for x in exclude)


def _is_na(v):
    """值是否为"无数据" (含裸 NA): 空串 / NA / N/A / nan / none / - / null"""
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in ("", "n/a", "na", "nan", "none", "-", "null")


def raw_cols_exist(raw, csv_name, keys, exclude=()):
    """期望文件里是否有匹配列 **且 rows[0] 值非 NA** (列在但值 NA → 合法缺, 不误报 BUG)"""
    csvdata = raw.get(csv_name, {})
    row0 = (csvdata.get("rows") or [{}])[0]
    for col in csvdata.get("columns", []):
        if _match(col, keys, exclude):
            if not _is_na(row0.get(col)):
                return True
    return False


def norm_key_exists(d, *keys, exclude=()):
    for k, v in (d or {}).items():
        if _match(k, keys, exclude) and v is not None:
            return True
    return False


def check_board(bd):
    raw = bd.get("raw", {})
    norm = bd.get("normalized", {})
    issues = []

    def _check(section, norm_dict, prefix):
        for entry in BOARD_EXPECT[section]:
            field, tool, src, col = entry[0], entry[1], entry[2], entry[3]
            keys = entry[4]
            exclude = entry[5] if len(entry) > 5 else ()
            if norm_dict.get(field) is None:
                issues.append((f"{prefix}.{field}", tool, src, col,
                               raw_cols_exist(raw, src, keys, exclude)))

    _check("bandwidth", norm.get("bandwidth_gb_s", {}), "bandwidth")
    _check("engine_utilization", norm.get("engine_utilization", {}), "engine")
    _check("compute", norm.get("compute", {}), "compute")
    for entry in BOARD_EXPECT["conflict"]:
        field, tool, src, col = entry[0], entry[1], entry[2], entry[3]
        keys = entry[4]
        exclude = entry[5] if len(entry) > 5 else ()
        if not norm_key_exists(norm.get("conflict", {}), *keys, exclude=exclude):
            issues.append((f"conflict.{field}", tool, src, col,
                           raw_cols_exist(raw, src, keys, exclude)))

    # ★P2: 官方实测字段 (traffic/bw_usage_rate/active_bw/icache) — 规范化列名精确核对
    for section, norm_dict, prefix in (
            ("traffic_kb", norm.get("traffic_kb", {}), "traffic_kb"),
            ("bw_usage_rate", norm.get("bw_usage_rate", {}), "bw_usage_rate"),
            ("active_bw_gb_s", norm.get("active_bw_gb_s", {}), "active_bw"),
            ("icache_miss_rate", norm.get("icache_miss_rate", {}) or {}, "icache")):
        mem_cols = [_norm_col(c) for c in raw.get("Memory", {}).get("columns", [])]
        pu_cols = [_norm_col(c) for c in raw.get("PipeUtilization", {}).get("columns", [])]
        for field, col in BOARD_EXPECT_NORM[section]:
            if norm_dict.get(field) is None:
                cols = mem_cols if section in ("traffic_kb", "bw_usage_rate") else pu_cols
                has = _norm_col(col) in cols
                issues.append((f"{prefix}.{field}", "msprof op",
                               "Memory" if section in ("traffic_kb", "bw_usage_rate") else "PipeUtilization",
                               col, has))

    if norm.get("l2_hit_rate") is None:
        issues.append(("l2_hit_rate", "msprof op", "L2Cache", "aic_total_hit_rate(%)",
                       raw_cols_exist(raw, "L2Cache", ("hit_rate",))))
    return issues


def check_task(tk):
    raw = tk.get("raw", {})
    norm = tk.get("normalized", {})
    issues = []
    opsum_cols = raw.get("op_summary", {}).get("columns", [])
    col_hit = lambda *ks: any(all(x.lower() in c.lower() for x in ks) for c in opsum_cols)  # noqa: E731

    kernels = norm.get("kernels", [])
    if kernels:
        k0 = kernels[0]
        # (normalized字段, 具体列名, 匹配键)
        for f, col, keys in (("task_duration_us", "Task Duration(us)", ("task", "duration")),
                             ("block_dim", "Block Dim", ("block", "dim")),
                             ("aicore_time_us", "aicore_time(us)", ("aicore", "time")),
                             ("aiv_time_us", "aiv_time(us)", ("aiv", "time")),
                             ("total_cycles", "Total Cycles", ("total", "cycle")),
                             ("input_shapes", "Input Shape(s)", ("input", "shape"))):
            if k0.get(f) is None:
                issues.append((f"kernel.{f}", "通用 msprof", "op_summary_*.csv", col, col_hit(*keys)))

    if not norm.get("multi_kernel"):
        issues.append(("multi_kernel", "通用 msprof", "op_statistic_*.csv", "OP Type", False))
    if not norm.get("api_overhead"):
        issues.append(("api_overhead", "通用 msprof", "api_statistic_*.csv", "API Name", False))
    if norm.get("l2_hit_rate") is None:
        issues.append(("l2_hit_rate", "通用 msprof", "l2_cache_*.csv", "Hit Rate", True))
    return issues


def check_diag(dg):
    issues = []
    if dg.get("summary") and dg.get("kernels"):
        s = dg["summary"]
        if s.get("num_kernels") is None:
            issues.append(("summary.num_kernels", "通用 msprof", "op_summary_*.csv", "Op Name 去重", False))
        for i, k in enumerate(dg["kernels"]):
            if not k.get("kernel_name"):
                issues.append((f"kernels[{i}].kernel_name", "通用 msprof", "op_summary_*.csv", "Op Name", False))
            deep = k.get("deep")
            if deep:
                if not deep.get("roofline", {}).get("bottleneck_type"):
                    issues.append((f"kernels[{i}].deep.roofline.bottleneck_type",
                                   "msprof op", "Memory.csv+ArithmeticUtilization.csv",
                                   "aic_main_mem_read_bw + aic_cube_fops(计算)", False))
                if not deep.get("bandwidth_gb_s"):
                    issues.append((f"kernels[{i}].deep.bandwidth_gb_s", "msprof op", "Memory.csv",
                                   "aic_main_mem_read_bw", False))
        return issues
    if not dg.get("roofline", {}).get("bottleneck_type"):
        issues.append(("roofline.bottleneck_type", "msprof op", "Memory.csv+ArithmeticUtilization.csv",
                       "aic_main_mem_read_bw + aic_cube_fops(计算)", False))
    if not dg.get("transfer_paths"):
        issues.append(("transfer_paths", "msprof op", "Memory.csv", "aic_*_bw", False))
    for k, v in dg.get("bottlenecks", {}).items():
        if not v.get("hint"):
            issues.append((f"bottlenecks.{k}.hint", "规则", "—", "hint", False))
    return issues


def report(bd, tk, dg):
    print("═" * 60)
    print("字段覆盖检验 — 每个缺字段: 去 [工具] 的 [文件] 找 [具体列名]")
    print("  ⚠ = 文件里有该列且非空, 但没取到 → parser 键/实际列名不符, 把真实列名告诉我")
    print("  · = 文件里该列不存在或为空 → 合法缺 (版本无此列 或 该引擎无数据)")
    print("═" * 60)
    total_bug = total_absent = 0
    for label, issues in [("board.json", check_board(bd)), ("task.json", check_task(tk)),
                          ("diagnosis.json", check_diag(dg))]:
        print(f"\n── {label} ──")
        if not issues:
            print("  ✅ 所有期望字段都有数据")
        for field, tool, src, col, has_raw in issues:
            if has_raw:
                print(f"  ⚠ {field:26s} 去 [{tool}] 的 [{src}] 找 [{col}] → raw有但没取到 → **列名不匹配**")
                total_bug += 1
            else:
                print(f"  · {field:26s} 去 [{tool}] 的 [{src}] 找 [{col}] → 该列不存在/为空 → **合法缺**")
                total_absent += 1
    print(f"\n{'=' * 60}")
    print(f"结论: 列名不匹配 {total_bug} 个 (需把真实列名告诉我修 parser) | 合法缺 {total_absent} 个 (正常)")
    return 1 if total_bug else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    bd = load(args[0]) if len(args) > 0 else {}
    tk = load(args[1]) if len(args) > 1 else {}
    dg = load(args[2]) if len(args) > 2 else {}
    sys.exit(report(bd, tk, dg))
