#!/usr/bin/env python3
"""字段覆盖检验 — 检查"理论上该有"的字段是否真有数据。

区分三种情况:
  OK        字段有数据
  字段名不匹配   normalized 是 None/空, 但原始 CSV 里有匹配列 → parser 匹配漏了 (BUG/列名变)
  源无此字段    normalized 是 None/空, 原始 CSV 也没有 → 合法缺 (版本/产品无此数据)

用法: python check_fields.py <board.json> <task.json> [diagnosis.json]
退出码: 0 = 无 BUG (可能有合法缺); 1 = 有"字段名不匹配"需修 parser
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


# 期望字段 → (源 CSV, 应含子串[, 排除子串])  子串在原始列名里找 (列名按官网核实)
BOARD_EXPECT = {
    "bandwidth": [
        ("main_mem_read_gb_s",  "Memory",   ("main", "mem", "read", "bw")),
        ("main_mem_write_gb_s", "Memory",   ("main", "mem", "write", "bw")),
        ("l1_read_gb_s",        "Memory",   ("l1", "read", "bw")),
        ("l1_write_gb_s",       "Memory",   ("l1", "write", "bw")),
        ("l2_read_gb_s",        "Memory",   ("l2", "read", "bw")),
        ("l2_write_gb_s",       "Memory",   ("l2", "write", "bw")),
        ("gm_to_ub_gb_s",       "Memory",   ("gm", "to_ub", "bw")),    # aiv_gm_to_ub_bw (MTE2)
        ("ub_to_gm_gb_s",       "Memory",   ("ub", "to_gm", "bw")),    # aiv_ub_to_gm_bw (MTE3)
        ("ub_vector_read_gb_s", "MemoryUB", ("vector", "read", "bw")),  # aiv_ub_read_bw_vector
        ("ub_vector_write_gb_s", "MemoryUB", ("vector", "write", "bw")),# aiv_ub_write_bw_vector
        ("ub_scalar_read_gb_s", "MemoryUB", ("scalar", "read", "bw")),  # aiv_ub_read_bw_scalar
        ("ub_scalar_write_gb_s", "MemoryUB", ("scalar", "write", "bw")),# aiv_ub_write_bw_scalar
        ("ub_mte_read_gb_s",    "MemoryUB", ("mte", "read", "bw")),    # 仅推理产品, 910B3 预期合法缺
        ("ub_mte_write_gb_s",   "MemoryUB", ("mte", "write", "bw")),
        ("l0a_read_gb_s",       "MemoryL0", ("l0a", "read", "bw")),    # aic_l0a_read_bw
        ("l0a_write_gb_s",      "MemoryL0", ("l0a", "write", "bw")),
        ("l0b_read_gb_s",       "MemoryL0", ("l0b", "read", "bw")),    # aic_l0b_read_bw
        ("l0b_write_gb_s",      "MemoryL0", ("l0b", "write", "bw")),
        ("l0c_read_gb_s",       "MemoryL0", ("l0c", "read", "bw")),    # l0c_read_bw_cube
        ("l0c_write_gb_s",      "MemoryL0", ("l0c", "write", "bw")),   # l0c_write_bw_cube
    ],
    "engine_utilization": [
        ("cube",    "PipeUtilization",       ("aic", "cube", "ratio")),    # aic_cube_ratio
        ("vec",     "PipeUtilization",       ("aiv", "vec", "ratio")),     # aiv_vec_ratio
        ("mte1",    "PipeUtilization",       ("mte1", "ratio")),
        ("mte2",    "PipeUtilization",       ("mte2", "ratio")),
        ("mte3",    "PipeUtilization",       ("mte3", "ratio")),
        ("scalar",  "PipeUtilization",       ("scalar", "ratio")),
        ("fixpipe", "PipeUtilization",       ("fixpipe", "ratio")),        # aic_fixpipe_ratio
    ],
    "compute": [
        ("cube_fops",        "ArithmeticUtilization", ("cube", "fops")),     # aic_cube_fops
        ("cube_ratio",       "ArithmeticUtilization", ("cube", "ratio")),
        ("cube_fp16_ratio",  "ArithmeticUtilization", ("cube", "fp16", "ratio")),
        ("cube_int8_ratio",  "ArithmeticUtilization", ("cube", "int8", "ratio")),
        ("vector_fops",      "ArithmeticUtilization", ("vec", "fops")),     # aiv_vec_fops
        ("vec_ratio",        "ArithmeticUtilization", ("aiv", "vec", "ratio")),
        ("aic_total_cycles", "ArithmeticUtilization", ("aic", "total", "cycle")),
        ("aiv_total_cycles", "ArithmeticUtilization", ("aiv", "total", "cycle")),
    ],
    "conflict": [   # ResourceConflictRatio.csv 列名 (A2 系前缀 aiv_vec_)
        ("bank_cflt_ratio",      "ResourceConflictRatio", ("vec", "bank", "cflt"), ("bankgroup",)),
        ("bankgroup_cflt_ratio", "ResourceConflictRatio", ("vec", "bankgroup", "cflt")),
        ("total_cflt_ratio",     "ResourceConflictRatio", ("vec", "total", "cflt")),
        ("resc_cflt_ratio",      "ResourceConflictRatio", ("vec", "resc", "cflt")),
        ("mte_cflt_ratio",       "ResourceConflictRatio", ("vec", "mte", "cflt")),
    ],
}


def _match(name, keys, exclude=()):
    """列名/键名是否含所有 keys 子串且不含 exclude 子串"""
    cl = name.lower()
    return all(k.lower() in cl for k in keys) and not any(x.lower() in cl for x in exclude)


def raw_cols_exist(raw, csv_name, keys, exclude=()):
    """raw 里该 CSV 是否有匹配列 **且 rows[0] 值非空**。
    列在但值为空 (如向量算子的 cube 占比) → 该引擎/通路无数据 → 判合法缺, 不误报 BUG。"""
    csvdata = raw.get(csv_name, {})
    row0 = (csvdata.get("rows") or [{}])[0]
    for col in csvdata.get("columns", []):
        if _match(col, keys, exclude):
            v = row0.get(col)
            if v is not None and str(v).strip().lower() not in ("", "n/a", "nan", "none", "-"):
                return True
    return False


def norm_key_exists(d, *keys, exclude=()):
    """normalized dict 里是否存在匹配 key (值非 None)"""
    for k, v in (d or {}).items():
        if _match(k, keys, exclude) and v is not None:
            return True
    return False


def check_board(bd):
    raw = bd.get("raw", {})
    norm = bd.get("normalized", {})
    issues = []

    def _check(section, norm_dict, prefix):
        """统一检查: (field, src, keys[, exclude]) → 判定, 带上期望列名"""
        for entry in BOARD_EXPECT[section]:
            field, src, keys = entry[0], entry[1], entry[2]
            exclude = entry[3] if len(entry) > 3 else ()
            if norm_dict.get(field) is None:
                issues.append((f"{prefix}.{field}", src,
                               raw_cols_exist(raw, src, keys, exclude), keys))

    _check("bandwidth", norm.get("bandwidth_gb_s", {}), "bandwidth")
    _check("engine_utilization", norm.get("engine_utilization", {}), "engine")
    _check("compute", norm.get("compute", {}), "compute")
    # conflict: normalized key = 原始列名原样, 用子串匹配
    for entry in BOARD_EXPECT["conflict"]:
        field, src, keys = entry[0], entry[1], entry[2]
        exclude = entry[3] if len(entry) > 3 else ()
        if not norm_key_exists(norm.get("conflict", {}), *keys, exclude=exclude):
            issues.append((f"conflict.{field}", src, raw_cols_exist(raw, src, keys, exclude), keys))

    if norm.get("l2_hit_rate") is None:
        issues.append(("l2_hit_rate", "L2Cache", raw_cols_exist(raw, "L2Cache", ("hit_rate",)),
                       ("hit_rate",)))
    return issues


def check_task(tk):
    raw = tk.get("raw", {})
    norm = tk.get("normalized", {})
    issues = []
    # op_summary 原始列
    opsum_cols = raw.get("op_summary", {}).get("columns", [])
    col_hit = lambda *ks: any(all(x.lower() in c.lower() for x in ks) for c in opsum_cols)  # noqa: E731

    kernels = norm.get("kernels", [])
    if kernels:
        k0 = kernels[0]
        for f, keys in (("task_duration_us", ("task", "duration")), ("block_dim", ("block", "dim")),
                        ("aicore_time_us", ("aicore", "time")), ("aiv_time_us", ("aiv", "time")),
                        ("total_cycles", ("total", "cycle")), ("input_shapes", ("input", "shape"))):
            if k0.get(f) is None:
                issues.append((f"kernel.{f}", "op_summary", col_hit(*keys), keys))

    if not norm.get("multi_kernel"):
        issues.append(("multi_kernel", "op_statistic", False, ("OP Type",)))
    if not norm.get("api_overhead"):
        issues.append(("api_overhead", "api_statistic", False, ("API Name",)))
    if norm.get("l2_hit_rate") is None:
        issues.append(("l2_hit_rate", "l2_cache", True, ("Hit Rate",)))
    return issues


def check_diag(dg):
    issues = []
    # 新 schema (v4): summary + kernels[].task/deep
    if dg.get("summary") and dg.get("kernels"):
        s = dg["summary"]
        if s.get("num_kernels") is None:
            issues.append(("summary.num_kernels", "整合", False, ("num_kernels",)))
        for i, k in enumerate(dg["kernels"]):
            if not k.get("kernel_name"):
                issues.append((f"kernels[{i}].kernel_name", "整合", False, ("Op Name",)))
            deep = k.get("deep")
            if deep:
                if not deep.get("roofline", {}).get("bottleneck_type"):
                    issues.append((f"kernels[{i}].deep.roofline.bottleneck_type", "整合计算",
                                   False, ("main_mem_*_bw + cube_fops",)))
                if not deep.get("bandwidth_gb_s"):
                    issues.append((f"kernels[{i}].deep.bandwidth_gb_s", "Memory.csv",
                                   False, ("aic_main_mem_read_bw",)))
        return issues
    # 旧 schema 兜底
    if not dg.get("roofline", {}).get("bottleneck_type"):
        issues.append(("roofline.bottleneck_type", "整合计算", False, ("main_mem_*_bw",)))
    if not dg.get("transfer_paths"):
        issues.append(("transfer_paths", "Memory.csv", False, ("aic_*_bw",)))
    for k, v in dg.get("bottlenecks", {}).items():
        if not v.get("hint"):
            issues.append((f"bottlenecks.{k}.hint", "规则", False, ("hint",)))
    return issues


def report(bd, tk, dg):
    print("═" * 60)
    print("字段覆盖检验 (OK=有数据 | 列名不匹配=raw有但没提取到 | 源无=合法缺)")
    print("═" * 60)
    total_bug = total_absent = 0
    for label, issues in [("board.json", check_board(bd)), ("task.json", check_task(tk)),
                          ("diagnosis.json", check_diag(dg))]:
        print(f"\n── {label} ──")
        if not issues:
            print("  ✅ 所有期望字段都有数据")
        for field, src, has_raw, keys in issues:
            exp = " / ".join(f"含'{k}'" for k in keys) if keys else ""
            if has_raw:
                print(f"  ⚠ {field:28s} 期望[{src} {exp}] → raw 有但没取到 → **字段名不匹配 (BUG/列名变)**")
                total_bug += 1
            else:
                print(f"  · {field:28s} 期望[{src} {exp}] → 该列不存在/为空 → **合法缺** (去真机核对)")
                total_absent += 1
    print(f"\n{'=' * 60}")
    print(f"结论: 字段名不匹配 {total_bug} 个 (需修 parser 列名) | 合法缺 {total_absent} 个 (正常)")
    if total_bug:
        print("  → 建议: 看 raw 里实际列名, 更新 pipeline_parse_board.py/task.py 的子串键")
    return 1 if total_bug else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    bd = load(args[0]) if len(args) > 0 else {}
    tk = load(args[1]) if len(args) > 1 else {}
    dg = load(args[2]) if len(args) > 2 else {}
    sys.exit(report(bd, tk, dg))
