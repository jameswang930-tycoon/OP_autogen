#!/usr/bin/env python3
"""910B3 理论峰值计算 + 理论/实测对照 — 纯本地可跑 (无 NPU/无 triton 依赖).

数值来源 (2026-08 联网核实, 多来源交叉确认):
  ── HBM 带宽 ──
  * HBM2e 标准 (JEDEC JESD235C): 单 stack 带宽 = 数据率 × 位宽 / 8
      3.2 Gbps × 1024 bit / 8 = 409.6 GB/s/stack
  * 910B3 = 4 stack → 理论 = 4 × 409.6 = **1638.4 GB/s**
  * 佐证: npu-smi 实测 HBM2e 64GB @1600MHz(三星); ascend-dmi 实测 ~1.54 TB/s (≈94%)
  * ⚠ 之前代码硬编码 1800 GB/s 是错的 (HBM2e 无此规格)
  ── Cube 算力 ──
  * 每 AI Core cube 每周期 16×16×16 = 4096 MAC = **8192 FLOP** (fp16)
  * 20 AI Core × 8192 × 1.8 GHz(标称) = **294.9 TFLOPS** (本机标称频率推导)
  * 官方标称 **313 TFLOPS** (≈1.91 GHz boost)
  * fp32 = fp16 / 4 (cube 用半 lane + 双字节) → 73.7 (推导) / 78.3 (官方)
  ── Vec 算力 ──
  * 全片 INT8 128 TOPS → fp16 ≈ 64 TFLOPS (2:1 推断, 非官方直接陈述)
  ── 片上 L2/UB 带宽 ──
  * 无公开官方规格 → 理论列留空, 只取实测

用法:
  python3 bench_theory.py              # 打印理论峰值 + 对照表 (有 hardware_peak.json 则算效率)
  python3 bench_theory.py --json       # 额外写 hardware_theory.json
  (run_bench.py 结束时会自动调 comparison() 写进 results.json / hardware_peak.json)
"""
import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OUT_DIR = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════
#  910B3 硬件参数表 (联网核实 2026-08)
# ═══════════════════════════════════════════════════════════════════════
HARDWARE_910B3 = {
    # ── HBM2e (JEDEC JESD235C) ──
    "hbm_type": "HBM2e",
    "hbm_stacks": 4,                # 4 stack 总位宽 4096bit
    "hbm_bus_bits": 1024,           # 每 stack 位宽
    "hbm_data_rate_gbps": 3.2,      # 针脚数据率 (DDR, 有效 3.2Gbps)
    "hbm_capacity_gb": 64,          # npu-smi 确认 64GB (65536 MB)
    # ── Cube (Da Vinci 16×16×16) ──
    "ai_cores": 20,                 # 910B3 AI Core 数
    "cube_mac_per_cycle": 4096,     # 16×16×16 = 4096 MAC/cycle/核
    "clock_ghz": 1.8,               # 标称频率 (npu-smi: nominal 1800MHz)
    "official_fp16_tflops": 313.0,  # 官方标称 FP16 峰值 (≈1.91GHz boost)
    # ── Vec ──
    "vec_int8_tops": 128.0,         # 全片向量 INT8 (fp16≈/2 推断)
    # ── 已知外部实测 (bench 之前即可对照) ──
    "ref_measure": {
        "ascend_dmi_hbm_gb_s": 1540.0,   # 工具实测 HBM 带宽 ≈94%
        "vec_ub_gb_s": 404.0,            # 我们之前 vec bench 实测
    },
}

# 每个数值的来源标注 (写进输出, 可追溯)
SOURCES = {
    "hbm_stacks": "JEDEC JESD235C: 4 stack × 1024bit @3.2Gbps → 1638.4 GB/s (多来源: 天极/JEDEC/Rambus)",
    "hbm_capacity_gb": "npu-smi info 实测 65536 MB; 多来源 64GB HBM2e",
    "hbm_freq": "npu-smi 实测 1600 MHz (三星 HBM2e)",
    "ai_cores": "本会话既有配置 (20 AI Core + 40 Vec Core)",
    "cube_mac_per_cycle": "16×16×16=4096 MAC=8192 FLOP/cycle (CSDN/昇腾架构详解确认)",
    "clock_ghz": "npu-smi info -t common: 标称 1800 MHz",
    "official_fp16_tflops": "官方/多来源 313 TFLOPS (jishuzhan/ucache/cndba/mitsea 一致)",
    "vec_int8_tops": "技术站: 全片向量 INT8 128 TOPS → fp16≈64 TFLOPS 属推断",
    "ref_measure": "ascend-dmi 实测 1.54TB/s (多来源一致); vec 404 = 我们之前实测",
}


def theoretical_peaks() -> dict:
    """从 HARDWARE_910B3 参数静态计算理论峰值 — 公式在此, 不硬编码结果."""
    h = HARDWARE_910B3
    hbm_gb_s = (h["hbm_stacks"] * h["hbm_bus_bits"] * h["hbm_data_rate_gbps"] * 1e9) / 8 / 1e9
    # stacks × bits × rate(Gbps→bps) / 8 = GB/s (decimal)
    flops_cycle = h["cube_mac_per_cycle"] * 2                      # MAC → FLOP
    fp16_nominal = (h["ai_cores"] * flops_cycle * h["clock_ghz"] * 1e9) / 1e12  # TFLOPS
    return {
        "hbm_bw_gb_s": round(hbm_gb_s, 1),                          # 1638.4
        # cube 算力 (标称频率推导 vs 官方标称)
        "cube_fp16_tflops_nominal": round(fp16_nominal, 1),         # 294.9
        "cube_fp16_tflops_official": round(h["official_fp16_tflops"], 1),  # 313
        "cube_fp32_tflops_nominal": round(fp16_nominal / 4, 1),     # 73.7
        "cube_fp32_tflops_official": round(h["official_fp16_tflops"] / 4, 1),  # 78.3
        # vec (推断)
        "vec_fp16_tflops_est": round(h["vec_int8_tops"] / 2, 1),    # 64
        # roofline 转折点 (算术强度, FLOP/byte) — fp16 用 fp16 峰值, fp32 用 fp32 峰值
        "ridge_fp16_flop_byte": round(fp16_nominal * 1e12 / (hbm_gb_s * 1e9), 1),  # ~180
        "ridge_fp32_flop_byte": round(fp16_nominal / 4 * 1e12 / (hbm_gb_s * 1e9), 1),  # ~45
        "hardware": {
            "hbm": f"{h['hbm_stacks']}×{h['hbm_bus_bits']}bit {h['hbm_type']} "
                   f"@{h['hbm_data_rate_gbps']}Gbps {h['hbm_capacity_gb']}GB",
            "cube": f"{h['ai_cores']} AI Core × {h['cube_mac_per_cycle']}MAC/cyc "
                    f"@ {h['clock_ghz']}GHz (16×16×16)",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  理论/实测对照
# ═══════════════════════════════════════════════════════════════════════

def _eff(measured, theory):
    if theory and measured:
        return measured / theory
    return None


def comparison(measured_peak: dict) -> dict:
    """measured_peak = run_bench 的 named_peak (或 hardware_peak.json['peak']).
    返回 {bandwidth:[rows], compute:[rows]} 对照表, 每行含 theory/measured/eff."""
    T = theoretical_peaks()
    rows_bw = []
    rows_c = []
    hbm = T["hbm_bw_gb_s"]

    def add(rows, metric, theory, measured, unit, note=""):
        rows.append({
            "metric": metric, "theory": theory, "measured": measured,
            "eff": _eff(measured, theory), "unit": unit, "note": note,
        })

    # ── 带宽通路 ──
    add(rows_bw, "gm_read (kernel)", hbm, measured_peak.get("gm_read_gb_s"), "GB/s")
    add(rows_bw, "gm_write (kernel)", hbm, measured_peak.get("gm_write_gb_s"), "GB/s")
    add(rows_bw, "gm_copy (kernel)", hbm, measured_peak.get("gm_copy_gb_s"), "GB/s")
    add(rows_bw, "gm_aggregate (max)", hbm, measured_peak.get("gm_bw_gb_s"),
        "GB/s", note="GM 读/写/拷贝 的 max = roofline 峰值")
    add(rows_bw, "l2_read", None, measured_peak.get("l2_read_gb_s"),
        "GB/s", note="片上缓存, 无官方理论值, 取实测")
    add(rows_bw, "vec/UB 通路", None, measured_peak.get("vec_bw_gb_s"),
        "GB/s", note="UB 数据通路, 无官方理论值, 取实测")
    add(rows_bw, "gm_to_ub (MTE)", None, measured_peak.get("gm_to_ub_gb_s"),
        "GB/s", note="msprof op per-path, 取实测")
    add(rows_bw, "l0a_feed", None, measured_peak.get("l0a_feed_gb_s"),
        "GB/s", note="msprof op per-path, 取实测")
    add(rows_bw, "l0b_feed", None, measured_peak.get("l0b_feed_gb_s"),
        "GB/s", note="msprof op per-path, 取实测")

    # ── 算力 (两列理论: 标称1.8GHz推导 / 官方标称) ──
    for metric, meas_key in [("cube_fp16", "cube_fp16_tflops"),
                             ("cube_fp32", "cube_fp32_tflops")]:
        rows_c.append({
            "metric": metric,
            "theory_nominal": T[f"{metric}_tflops_nominal"],
            "theory_official": T[f"{metric}_tflops_official"],
            "measured": measured_peak.get(meas_key),
            "eff_nominal": _eff(measured_peak.get(meas_key), T[f"{metric}_tflops_nominal"]),
            "eff_official": _eff(measured_peak.get(meas_key), T[f"{metric}_tflops_official"]),
            "unit": "TFLOPS",
        })
    add(rows_c, "vec_fp16 (推断)", T["vec_fp16_tflops_est"], None,
        "TFLOPS", note="由全片 INT8 128 TOPS ÷2 推断; vec bench 实测的是带宽非算力")

    return {"bandwidth": rows_bw, "compute": rows_c, "ridge_fp16_flop_byte": T["ridge_fp16_flop_byte"],
            "ridge_fp32_flop_byte": T["ridge_fp32_flop_byte"]}


def _load_measured() -> dict:
    p = OUT_DIR / "hardware_peak.json"
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")).get("peak", {})
    except Exception:
        pass
    return {}


def _fmt(v, w=9):
    if v is None:
        return " " * w + "—"
    return f"{v:>{w}.1f}"


def print_table(meas: dict, title="理论 vs 实测"):
    T = theoretical_peaks()
    comp = comparison(meas)
    hw = T["hardware"]
    print(f"\n══ {title} ══")
    print(f"  硬件: {hw['hbm']}")
    print(f"        {hw['cube']}")
    print(f"  来源: HBM2e JEDEC JESD235C 公式 + 官方标称 313 TFLOPS (联网核实 2026-08)")
    print(f"\n  ── 带宽通路 (GB/s) ──")
    print(f"  {'通路':<22s} {'理论':>9s} {'实测':>9s} {'效率':>7s}  说明")
    for r in comp["bandwidth"]:
        eff = f"{r['eff']*100:5.1f}%" if r["eff"] else "  —"
        print(f"  {r['metric']:<22s} {_fmt(r['theory'])} {_fmt(r['measured'])} {eff:>7s}  {r['note']}")
    print(f"\n  ── 算力 (TFLOPS) ──")
    print(f"  {'单元':<12s} {'理论(标称1.8G)':>14s} {'理论(官方)':>12s} {'实测':>9s} {'效率(标称)':>11s}")
    for r in comp["compute"]:
        eff_v = r.get("eff_nominal") or r.get("eff")
        eff = f"{eff_v*100:5.1f}%" if eff_v else "   —"
        print(f"  {r['metric']:<12s} {_fmt(r.get('theory_nominal') or r.get('theory'), 14)} "
              f"{_fmt(r.get('theory_official'), 12)} {_fmt(r.get('measured'))} {eff:>11s}  {r.get('note', '')}")
    print(f"\n  roofline 转折点 (算术强度): fp16 ≈ {T['ridge_fp16_flop_byte']} FLOP/byte, "
          f"fp32 ≈ {T['ridge_fp32_flop_byte']} FLOP/byte")
    print(f"  (kernel 实测强度 > 转折点 → compute-bound; < → memory-bound)")


def main():
    p = argparse.ArgumentParser(description="910B3 理论峰值 + 对照")
    p.add_argument("--json", action="store_true", help="额外写 hardware_theory.json")
    args = p.parse_args()

    T = theoretical_peaks()
    meas = _load_measured()
    comp = comparison(meas) if meas else comparison({})

    print("910B3 理论峰值 (公式推导):")
    for k in ["hbm_bw_gb_s", "cube_fp16_tflops_nominal", "cube_fp16_tflops_official",
              "cube_fp32_tflops_nominal", "cube_fp32_tflops_official", "vec_fp16_tflops_est"]:
        print(f"  {k:28s} {T[k]}")
    print("  来源:")
    for k in ["hbm_stacks", "hbm_capacity_gb", "cube_mac_per_cycle", "clock_ghz",
              "official_fp16_tflops", "vec_int8_tops"]:
        print(f"    • {k}: {SOURCES[k]}")

    if meas:
        print_table(meas, "理论 vs 实测 (本次 bench)")
    else:
        print_table({"gm_read_gb_s": None, "gm_write_gb_s": None, "gm_copy_gb_s": None,
                     "gm_bw_gb_s": HARDWARE_910B3["ref_measure"]["ascend_dmi_hbm_gb_s"],
                     "l2_read_gb_s": None, "vec_bw_gb_s": HARDWARE_910B3["ref_measure"]["vec_ub_gb_s"],
                     "gm_to_ub_gb_s": None, "l0a_feed_gb_s": None, "l0b_feed_gb_s": None,
                     "cube_fp16_tflops": None, "cube_fp32_tflops": None},
                    "理论 vs 参考实测 (bench 未跑, 用已知外部值)")
        print("  (跑 run_bench.py 后会自动更新 hardware_peak.json → 这里显示真机实测)")

    if args.json:
        out = OUT_DIR / "hardware_theory.json"
        out.write_text(json.dumps({"theory": T, "sources": SOURCES,
                                   "comparison": comp}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"\n→ {out}")


if __name__ == "__main__":
    main()
