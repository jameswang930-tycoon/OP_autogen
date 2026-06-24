#!/usr/bin/env python3
"""
统一分析: 读普通 msprof 模式的 op_summary.csv, 对全部用例用硬件 pipe 净时间
算出准确的带宽/算力 (排除其他 pipe 干扰), 输出 CSV.

各用例用的 pipe 时间:
  HIT/MISS (read):  aiv_mte2_time  -> pure read BW = bytes / mte2_time
  write:            aiv_mte3_time  -> pure write BW = bytes / mte3_time
  copy (1R+1W):     max(mte2,mte3) -> bottleneck-side BW
  FMA/ADD (vec):    aiv_vec_time   -> pure compute = ops / vec_time

byte/op counts inferred from kernel name (mirrors bench config).
usage: python analyze_all_msprof.py [op_summary.csv | prof_data dir] [out.csv]
"""
import sys, os, glob, csv, re
from collections import defaultdict

ELEM = 2
N_CORES = 20
N_VEC = 40
FREQ_GHZ = 1.8
DATA_MB = 256
NITERS = 256
T_VEC_TFLOPS = 128 * FREQ_GHZ * N_VEC / 1e3   # 9.216
AGG_DDR = N_CORES * 32 * FREQ_GHZ              # 1152
AGG_L2  = N_VEC   * 110 * FREQ_GHZ             # 7920
AGG_W   = N_CORES * 64 * FREQ_GHZ             # 2304 (ub_to_l2 write)

# HIT config (mirror bench)
HIT_BLK_ELEMS = 49152 * 3 * 10
HIT_OUTER = 256

def align_data(mb, grid, max_tile):
    unit = grid * max_tile * 3
    target = int(mb*1024*1024/ELEM)
    return max(unit, (target//unit)*unit)

def fnum(v):
    v = (v or "").strip()
    if v in ("","NA","N/A"): return None
    try: return float(v)
    except ValueError: return None

def find_summary(arg):
    if arg and os.path.isfile(arg): return arg
    base = arg if arg and os.path.isdir(arg) else "prof_data"
    cands = glob.glob(os.path.join(base,"**","*op_summary*.csv"), recursive=True)
    return max(cands, key=os.path.getmtime) if cands else None

def classify(name):
    """return (case, grid, tile, kind)"""
    g = re.search(r"_g(\d+)", name); t = re.search(r"_T(\d+)", name)
    grid = int(g.group(1)) if g else None
    tile = int(t.group(1)) if t else None
    if "read_hit_scalar" in name: return ("HIT_scalar", grid, tile, "read")
    if "read_hit_tile"   in name: return ("HIT_vector", grid, tile, "read")
    m = re.search(r"MISS(\d+)", name)
    if m:                         return (f"MISS_{m.group(1)}MB", grid, tile, "read")
    if "write_kernel"    in name: return ("write", grid, tile, "write")
    if "copy_kernel"     in name: return ("copy", grid, tile, "copy")
    if "compute_add"     in name: return ("Vec_ADD", grid, tile, "vec_add")
    if "compute_kernel"  in name: return ("Vec_FMA", grid, tile, "vec_fma")
    return (None, grid, tile, None)

def bytes_or_ops(case, kind, grid, tile):
    """return (amount, unit) ; amount=bytes for transfer, ops for vec"""
    if kind == "read" and case.startswith("HIT"):
        return grid * HIT_BLK_ELEMS * HIT_OUTER * ELEM, "bytes"
    if kind == "read":  # MISS
        m = re.search(r"MISS_(\d+)MB", case)
        mb = int(m.group(1))
        unit = grid * 16384 * 3
        N = max(unit, int(mb*1024*1024/ELEM)//unit*unit)
        return N * ELEM, "bytes"
    if kind == "write":
        N = align_data(DATA_MB, grid, 32768)
        return N * ELEM, "bytes"
    if kind == "copy":
        N = align_data(DATA_MB, grid, 32768)
        return 2 * N * ELEM, "bytes"   # 1R+1W
    if kind in ("vec_fma","vec_add"):
        ops_per = 2 if kind == "vec_fma" else 1
        N = align_data(DATA_MB, grid, 32768)
        return N * NITERS * ops_per, "ops"
    return None, None

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "msprof_accurate.csv"
    src = find_summary(arg)
    if not src:
        print("no op_summary.csv found"); return
    print(f"source: {src}")

    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))

    # aggregate per kernel name (sum times across calls, then per-call)
    agg = defaultdict(lambda: {"dur":0.0,"mte2":0.0,"mte3":0.0,"vec":0.0,"n":0})
    for r in rows:
        name = (r.get("Op Name","") or "").strip()
        case, grid, tile, kind = classify(name)
        if case is None:
            continue
        a = agg[name]
        a["dur"]  += fnum(r.get("Task Duration(us)")) or 0
        a["mte2"] += fnum(r.get("aiv_mte2_time(us)")) or 0
        a["mte3"] += fnum(r.get("aiv_mte3_time(us)")) or 0
        a["vec"]  += fnum(r.get("aiv_vec_time(us)")) or 0
        a["n"]    += 1

    out_rows = []
    for name, a in sorted(agg.items()):
        case, grid, tile, kind = classify(name)
        n = a["n"]
        if not n: continue
        dur = a["dur"]/n; mte2 = a["mte2"]/n; mte3 = a["mte3"]/n; vec = a["vec"]/n
        amount, unit = bytes_or_ops(case, kind, grid, tile)
        if amount is None: continue

        # pick the pipe time that defines this case's true metric
        if kind == "read":
            pipe_t, pipe = mte2, "MTE2"
        elif kind == "write":
            pipe_t, pipe = mte3, "MTE3"
        elif kind == "copy":
            pipe_t = max(mte2, mte3); pipe = "MTE2" if mte2>=mte3 else "MTE3"
        else:  # vec
            pipe_t, pipe = vec, "Vec"

        if unit == "bytes":
            # accurate BW from pipe net time
            acc_metric = amount / (pipe_t*1e-6) / 1e9 if pipe_t else 0   # GB/s
            e2e_metric = amount / (dur*1e-6) / 1e9 if dur else 0
            mkind = "GB/s"
        else:
            acc_metric = amount / (pipe_t*1e-6) / 1e12 if pipe_t else 0  # TFLOPS
            e2e_metric = amount / (dur*1e-6) / 1e12 if dur else 0
            mkind = "TFLOPS"

        out_rows.append({
            "case": case, "grid": grid, "tile": tile, "tile_kb": tile*ELEM/1024 if tile else "",
            "metric_kind": mkind,
            "msprof_accurate": round(acc_metric,3),   # the trustworthy value
            "perf_e2e": round(e2e_metric,3),          # perf_counter-equivalent
            "pipe_used": pipe,
            "pipe_time_us": round(pipe_t,3),
            "total_us": round(dur,3),
            "pipe_ratio": round(pipe_t/dur,3) if dur else "",  # how much this pipe dominates
            "calls": n,
        })

    # sort: case, then tile
    order = {"HIT_vector":0,"HIT_scalar":1,"MISS_256MB":2,"MISS_512MB":3,
             "MISS_1024MB":4,"write":5,"copy":6,"Vec_FMA":7,"Vec_ADD":8}
    out_rows.sort(key=lambda r:(order.get(r["case"],99), r["grid"] or 0, r["tile"] or 0))

    with open(out_csv,"w",newline="",encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    # console summary
    print(f"\n  {'case':<13s} {'grid':>4s} {'TILE':>6s} {'metric':>7s} "
          f"{'msprof准确':>10s} {'perf端到端':>10s} {'pipe':>5s} {'pipe占比':>8s}")
    print("  " + "-"*78)
    for r in out_rows:
        print(f"  {r['case']:<13s} {str(r['grid']):>4s} {str(r['tile']):>6s} "
              f"{r['metric_kind']:>7s} {r['msprof_accurate']:>10.1f} {r['perf_e2e']:>10.1f} "
              f"{r['pipe_used']:>5s} {str(r['pipe_ratio']):>8s}")

    print(f"""
  -> {out_csv} ({len(out_rows)} rows)
  ============================================================
  msprof_accurate = amount / pipe_net_time   (硬件pipe净时间, 准确值)
                    read->MTE2, write->MTE3, copy->max(MTE2,MTE3), vec->Vec
  perf_e2e        = amount / total_time      (perf_counter口径, 含其他pipe)
  pipe_ratio      = pipe_time / total_time   (接近1=该pipe主导, perf可信;
                                              远小于1=被其他pipe拖累, perf失真)
  ============================================================""")

if __name__ == "__main__":
    main()
