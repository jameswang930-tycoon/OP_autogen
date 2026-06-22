#!/usr/bin/env python3
"""
Plot bench_result.csv (from bench_matmul.py) - one PNG per case.
复用 vecadd/plot_bench.py 逻辑, 标签通用化 (core / tile)。

X axis: tile size in KB (linear).  Knee marked.
Y axis: TFLOPS (solid) + FLOP/cyc/core (dashed).
grid = N_CUBE (cube core 数) 画为单条线。

Usage:
  python bench_matmul.py             # produces bench_result.csv
  python plot_bench.py bench_result.csv [outdir]
"""
import sys, os, csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mark_knee(xs, ys):
    """knee = smallest x reaching 95% of peak"""
    if not ys:
        return None
    peak = max(ys)
    for x, y in zip(xs, ys):
        if y >= peak * 0.95:
            return x
    return None


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_bench.py bench_result.csv [outdir]")
        return
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "plots"
    os.makedirs(outdir, exist_ok=True)

    with open(src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"read {src}: {len(rows)} rows")

    # group: case -> grid -> [(tile_kb, metric, secondary)]
    cases = defaultdict(lambda: defaultdict(list))
    kind_of = {}
    for r in rows:
        case = r["case"]
        grid = int(r["grid"])
        kb = float(r["tile_kb"])
        val = float(r["metric_value"])
        sec = float(r["secondary"]) if r["secondary"] not in ("", None) else None
        cases[case][grid].append((kb, val, sec))
        kind_of[case] = r["metric_kind"]

    colors = {20: "#1f77b4", 40: "#d62728"}
    saved = []

    for case in sorted(cases):
        kind = kind_of[case]
        is_tf = (kind == "TFLOPS")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax2 = ax.twinx() if is_tf else None

        for grid in sorted(cases[case]):
            pts = sorted(cases[case][grid])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            c = colors.get(grid, "#2ca02c")
            ax.plot(xs, ys, "o-", color=c, label=f"grid={grid}",
                    markersize=5, linewidth=1.9)

            # knee
            knee = mark_knee(xs, ys)
            if knee is not None:
                ky = dict(zip(xs, ys))[knee]
                ax.axvline(knee, color=c, ls="--", alpha=0.4)
                unit = "TFLOPS"
                ax.annotate(f"knee\n{knee:.0f}KB\n{ky:.0f} {unit}", (knee, ky),
                            textcoords="offset points", xytext=(8, -38),
                            fontsize=8, color=c)
            # peak
            py = max(ys); px = xs[ys.index(py)]
            ax.annotate(f"peak {py:.0f} TFLOPS\n@ {px:.0f}KB", (px, py),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=9, color=c, fontweight="bold")

            # secondary axis: FLOP/cyc/core (dashed)
            if is_tf and ax2 is not None:
                secs = [p[2] for p in pts if p[2] is not None]
                if secs:
                    ax2.plot(xs, secs, "s--", color=c, alpha=0.45, markersize=3)

        ax.set_xlabel("tile size (BM*BN*BK * 2B / 1024, KB)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)

        if is_tf:
            ax.set_ylabel("TFLOPS (solid)", fontsize=11)
            ax2.set_ylabel("FLOP/cyc/core (dashed)", fontsize=11)
            ax.set_title(f"{case}  -  TFLOPS (solid) + FLOP/cyc/core (dashed)",
                         fontsize=12, fontweight="bold")
        else:
            ax.set_ylabel("aggregate bandwidth GB/s", fontsize=11)
            ax.set_title(f"{case}  -  TILE -> bandwidth",
                         fontsize=12, fontweight="bold")

        path = os.path.join(outdir, f"{case}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        saved.append(path)
        print(f"  {case}: {sum(len(v) for v in cases[case].values())} pts, "
              f"grids={sorted(cases[case].keys())}")

    print(f"\nDone. {len(saved)} PNGs -> {outdir}/")


if __name__ == "__main__":
    main()
