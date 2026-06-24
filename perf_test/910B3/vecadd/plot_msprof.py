#!/usr/bin/env python3
"""
Plot msprof_accurate.csv (from analyze_all_msprof.py) - one PNG per case.
Uses the msprof hardware-accurate value (pipe net time), NOT perf_counter.
X: KB per transfer (linear).  Y: GB/s (transfer) or TFLOPS (vec).
grid lines separated, knee + peak annotated.

usage: python plot_msprof.py msprof_accurate.csv [outdir]
"""
import sys, os, csv
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mark_knee(xs, ys):
    if not ys: return None
    peak = max(ys)
    for x, y in zip(xs, ys):
        if y >= peak * 0.95:
            return x
    return None


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_msprof.py msprof_accurate.csv [outdir]")
        return
    src = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else "plots_msprof"
    os.makedirs(outdir, exist_ok=True)

    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"read {src}: {len(rows)} rows")

    # group: case -> grid -> [(kb, accurate_value)]
    cases = defaultdict(lambda: defaultdict(list))
    kind_of = {}
    for r in rows:
        case = r["case"]
        grid = int(r["grid"]) if r["grid"] not in ("", None) else 0
        kb = float(r["tile_kb"]) if r["tile_kb"] not in ("", None) else 0
        val = float(r["msprof_accurate"])
        cases[case][grid].append((kb, val))
        kind_of[case] = r["metric_kind"]

    colors = {20: "#1f77b4", 40: "#d62728", 0: "#2ca02c"}
    saved = []

    for case in sorted(cases):
        kind = kind_of[case]
        is_vec = (kind == "TFLOPS")
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for grid in sorted(cases[case]):
            pts = sorted(cases[case][grid])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            c = colors.get(grid, "#888")
            lbl = f"grid={grid}" if grid else "grid=40"
            ax.plot(xs, ys, "o-", color=c, label=lbl, markersize=5, linewidth=1.9)

            knee = mark_knee(xs, ys)
            if knee is not None:
                ky = dict(zip(xs, ys))[knee]
                ax.axvline(knee, color=c, ls="--", alpha=0.4)
                unit = "TFLOPS" if is_vec else "GB/s"
                ax.annotate(f"knee\n{knee:.0f}KB\n{ky:.0f} {unit}", (knee, ky),
                            textcoords="offset points", xytext=(8, -38),
                            fontsize=8, color=c)
            py = max(ys); px = xs[ys.index(py)]
            unit = "TFLOPS" if is_vec else "GB/s"
            # if peak is near the right edge, place label to the left
            xr = (px - min(xs)) / (max(xs) - min(xs) + 1e-9)
            dx = -70 if xr > 0.7 else 5
            ax.annotate(f"peak {py:.0f} {unit}\n@ {px:.0f}KB", (px, py),
                        textcoords="offset points", xytext=(dx, 10),
                        fontsize=9, color=c, fontweight="bold")

        ax.set_xlabel("KB per transfer (TILE x 2B / 1024)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=10)
        if is_vec:
            ax.set_ylabel("TFLOPS (msprof Vec-time accurate)", fontsize=11)
            ax.set_title(f"{case}  -  msprof accurate compute (pure Vec time)",
                         fontsize=12, fontweight="bold")
        else:
            ax.set_ylabel("aggregate bandwidth GB/s (msprof accurate)", fontsize=11)
            ax.set_title(f"{case}  -  msprof accurate BW (pipe net time)",
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
