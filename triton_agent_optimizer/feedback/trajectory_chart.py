#!/usr/bin/env python3
"""
优化轨迹图 — 6 阶段加速比曲线 (v4 兼容)。

═══ 图表设计 ═══
  单 Y 轴: Speedup / TFLOPS (同一条线)
  X 轴:    Round (按实际轮数, 非均分)
  背景:    6 个浅色阶段色带 (宽度 = 各阶段轮次占比)
  阶段标签: 顶部彩色小标签
  点:      绿色 ● KEEP  |  红色 × REVERT  |  灰色 ■ Baseline
  标注:    超过 8% 的跳跃自动标注策略名
  虚线:    PyTorch baseline (灰色水平)
  标题:    总轮次 | TFLOPS 变化 | 加速比 | vs PyTorch

═══ 怎么运行 ═══
  前置: 先跑 main.py 生成 outputs/<op>/optimization_trajectory.json
        (可选) 先跑 bench_910b3/bench_pytorch.py 生成 pytorch_tflops.json → 图上出 PyTorch 虚线
  运行:
    python3 feedback/trajectory_chart.py outputs/matmul
    # 输出: outputs/matmul/final_output/trajectory_chart.png
  依赖: matplotlib (pip install matplotlib)

  v4 说明: 读 state.initial_tflops / state.pytorch_tflops (scheduler 设基准时算好);
            hist 存 speedup (每轮 vs 初始), cumulative = running best。
"""

from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# 6 阶段颜色
TIER_BG   = ["#e8f5e9","#fff3e0","#e3f2fd","#fce4ec","#f3e5f5","#e0f2f1"]
TIER_FG   = ["#2e7d32","#e65100","#1565c0","#c62828","#6a1b9a","#00695c"]
TIER_NAME = ["Algorithm","Fusion","Tiling","Memory","Compute","Architecture"]


def _load_pytorch_bench(kernel_dir: Path, state: dict) -> Optional[dict]:
    """PyTorch 基准 (tflops + time_us): 优先 state (scheduler 存好的), 缺则按算子自动读 bench json.
    ★不再回退到误导的默认 — 没有真实基准就返回 None, 图上不画虚线.
    ★时间 time_us 是直接可比口径 (同算子同形状); tflops 仅兜底展示.
    按算子选基准: attention → pytorch_attention; 多matmul/MLP → pytorch_mlp; 单 matmul → pytorch."""
    st_tf = state.get("pytorch_tflops")
    st_tu = state.get("pytorch_time_us")
    if st_tf or st_tu:
        return {"tflops": st_tf, "time_us": st_tu}
    bench_dir = _PROJECT_DIR / "bench_910b3"
    op = kernel_dir.name.lower()
    # ★显式算子映射优先 (bench_910b3/bench_config.PT_BENCH_MAP); 旧启发式兜底
    try:
        from bench_910b3.bench_config import PT_BENCH_MAP
        f = PT_BENCH_MAP.get(op)
        if f and (bench_dir / f).exists():
            try:
                return json.loads((bench_dir / f).read_text(encoding="utf-8"))
            except Exception:
                pass
    except Exception:
        pass
    cands = (["pytorch_attention_tflops.json", "pytorch_mlp_tflops.json", "pytorch_tflops.json"]
             if "attention" in op
             else ["pytorch_mlp_tflops.json", "pytorch_tflops.json"])
    for f in cands:
        p = bench_dir / f
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def generate(kernel_dir: Path, output_path: Optional[Path] = None) -> Path:
    traj_file = kernel_dir / "optimization_trajectory.json"
    if not traj_file.exists():
        raise FileNotFoundError(f"Not found: {traj_file}")
    traj = json.loads(traj_file.read_text(encoding="utf-8"))
    history = traj.get("history", [])
    state = traj.get("state", {})

    if len(history) < 1:
        raise ValueError("Need at least 1 round")

    # ── 数据 (v4: hist 存 speedup = 每轮 vs 初始基准; cumulative = 历史最优) ──
    rounds = [r.get("round", i + 1) for i, r in enumerate(history)]
    decisions = [r.get("decision", "?") for r in history]
    strategies = [r.get("change") or r.get("strategy", "") for r in history]
    reasons = [r.get("error", "") for r in history]
    speeds = [r.get("speedup", 1.0) for r in history]           # 每轮 vs baseline
    cum_speeds = list(np.maximum.accumulate(np.array(speeds)))  # running best
    # 基准点: 第 0 轮前 speedup=1.0 (初始)
    rounds = [0] + rounds
    cum_speeds = [1.0] + cum_speeds
    speeds = [1.0] + speeds
    decisions = ["BASELINE"] + decisions
    strategies = ["Baseline"] + strategies

    # TFLOPS: 从 state.baseline_ns 算 (需 M/N/K, 存在 state 里) 或默认
    initial_tflops = state.get("initial_tflops") or 6.4
    # ★PyTorch 基准: 优先 state (scheduler 存的), 缺则自动按算子读 bench json; None = 无真实数据 → 不画误导虚线
    # ★对比用时间 (time_us, 同算子同形状直接可比); tflops 仅兜底展示
    pt_bench = _load_pytorch_bench(kernel_dir, state) or {}
    pytorch_tflops = pt_bench.get("tflops")
    pytorch_time_us = pt_bench.get("time_us")
    # ★F3: hist 若存了每轮真实 tflops (kernel 结构变化后 FLOPs 变) 就逐轮用, 否则 initial×speedup 兜底
    hist_tflops = [r.get("tflops") for r in history]
    per_round_tf = [
        (h if isinstance(h, (int, float)) else initial_tflops * cs)
        for h, cs in zip(hist_tflops, cum_speeds[1:])
    ]
    tflops_arr = np.array([initial_tflops] + per_round_tf)

    # ── Tier ranges ──
    tier_ranges = []
    cur_tier, t_start = 0, 0
    for i, r in enumerate(history):
        t = r.get("tier",0)
        if t != cur_tier:
            if cur_tier > 0: tier_ranges.append((cur_tier, t_start, i-1))
            cur_tier, t_start = t, i
    if cur_tier > 0: tier_ranges.append((cur_tier, t_start, len(history)-1))

    # ── Plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    # 尝试中文字体
    try:
        import matplotlib.font_manager as fm
        for f in fm.fontManager.ttflist:
            if "SimHei" in f.name or "Microsoft YaHei" in f.name or "Noto Sans CJK" in f.name:
                plt.rcParams["font.family"] = f.name; break
    except: pass

    plt.rcParams.update({"font.size":11, "axes.titlesize":13, "axes.labelsize":12, "figure.dpi":150})
    fig, ax = plt.subplots(figsize=(22, 10))

    # ═══ 阶段背景 ═══
    for tier, start, end in tier_ranges:
        if start < end:
            ax.axvspan(start - 0.4, end + 0.4, alpha=0.55,
                       color=TIER_BG[(tier-1)%6], zorder=0)

    # ═══ PyTorch baseline (只有真实基准数据才画, 缺则不画误导虚线) ═══
    # ★口径说明: torch 侧含 aclnn 框架 kernel (eager 全图), triton 侧只统计目标 kernel (非 aclnn).
    #   ★对比用时间: pytorch_speedup = 我们 baseline_ns / pytorch_ns (同算子同形状, 时间直接可比).
    #     y 轴是"相对我们 baseline 的加速比" → 虚线位置直接 = "PyTorch 用时是 baseline 的几分之几".
    #     若 pytorch 快于我们的 baseline → 虚线 >1; 慢 → <1.
    if pt_bench:
        _pt_ns = (pytorch_time_us or 0) * 1000.0
        _base_ns = state.get("baseline_ns")
        if _pt_ns and _base_ns:
            pytorch_speedup = _base_ns / _pt_ns            # ★时间口径 (首选)
        elif pytorch_tflops and initial_tflops:
            pytorch_speedup = pytorch_tflops / initial_tflops   # 兜底 (无时间数据时)
        else:
            pytorch_speedup = None
        if pytorch_speedup:
            ax.axhline(y=pytorch_speedup, color="gray", linestyle="--", linewidth=2.5,
                       alpha=0.8, zorder=1)
            _lbl_t = f"{pytorch_time_us:.0f}us" if pytorch_time_us else f"{pytorch_tflops:.1f} TFLOPS"
            ax.text(len(rounds)-1, pytorch_speedup + 0.03,
                    f"PyTorch eager* ({_lbl_t})\n*口径: torch 含框架 kernel, triton 不含 → 仅参考",
                    fontsize=9, color="gray", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # ═══ Running Best ═══
    running_best = np.maximum.accumulate(np.array(cum_speeds))
    ax.plot(rounds, running_best, color="#1565c0", linewidth=3, alpha=0.85, zorder=2)
    ax.fill_between(rounds, 1.0, running_best, alpha=0.06, color="#1565c0")

    # ═══ Points ═══
    r_arr, s_arr = np.array(rounds), np.array(cum_speeds)
    bl = np.array([d=="BASELINE" for d in decisions])
    kp = np.array([d=="KEEP" for d in decisions])
    rv = np.array([d=="REVERT" for d in decisions])

    if bl.any(): ax.scatter(r_arr[bl], s_arr[bl], c="gray", s=120, marker="s",
                             zorder=5, label="Baseline")
    if kp.any(): ax.scatter(r_arr[kp], s_arr[kp], c="#2ecc71", s=70,
                             edgecolors="white", linewidth=0.5, zorder=5,
                             label=f"KEEP ({kp.sum()})")
    if rv.any(): ax.scatter(r_arr[rv], s_arr[rv], c="#e74c3c", s=90, marker="X",
                             linewidth=1.5, zorder=5, label=f"REVERT ({rv.sum()})")

    # ═══ Annotations ═══
    for i in range(1, len(rounds)):
        if decisions[i]=="KEEP" and speeds[i] > 1.07:
            ax.annotate(strategies[i][:30],
                xy=(rounds[i], cum_speeds[i]),
                xytext=(rounds[i]+0.4, cum_speeds[i]+0.07),
                fontsize=7.5, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1, alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="#1565c0", lw=0.7),
                zorder=10)
        if decisions[i]=="REVERT" and speeds[i] < 0.97:
            ax.annotate(reasons[i][:35],
                xy=(rounds[i], cum_speeds[i]),
                xytext=(rounds[i]+0.5, cum_speeds[i]-0.06),
                fontsize=7, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.7, alpha=0.5),
                zorder=9)

    # ═══ Phase labels (top) ═══
    ylim = ax.get_ylim()
    for tier, start, end in tier_ranges:
        if start < end:
            mid = (start+end)/2
            ax.text(mid, ylim[1]*0.975, f"T{tier}: {TIER_NAME[tier-1]}",
                    fontsize=9.5, fontweight="bold", color=TIER_FG[(tier-1)%6],
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.35", fc=TIER_BG[(tier-1)%6],
                              alpha=0.85, ec=TIER_FG[(tier-1)%6], lw=1.2),
                    zorder=20)

    # ═══ Right Y: TFLOPS ═══
    ax2 = ax.twinx()
    ymin, ymax = ax.get_ylim()
    ax2.set_ylim(ymin * initial_tflops, ymax * initial_tflops)
    ax2.set_ylabel("Throughput (TFLOPS)", fontsize=12, color="#7b1fa2")
    ax2.tick_params(axis="y", labelcolor="#7b1fa2")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ═══ Title ═══
    total_rounds = len(rounds) - 1
    final_s = running_best[-1]
    final_t = tflops_arr[-1]
    # ★vs PyTorch 用时间对比 (直接可比): 我们最优用时 vs pytorch 用时
    _vs_pt = ""
    if pt_bench:
        _cur_ns = state.get("baseline_ns")
        _final_us = (_cur_ns / final_s / 1000.0) if (_cur_ns and final_s) else None
        if pytorch_time_us and _final_us:
            _vs_pt = (f"  |  vs PyTorch({pytorch_time_us:.0f}us): "
                      f"我们最优 {_final_us:.0f}us = {_final_us/pytorch_time_us*100:.0f}%")
        elif pytorch_tflops:
            _vs_pt = (f"  |  vs PyTorch({pytorch_tflops:.1f} TFLOPS): "
                      f"{final_t/pytorch_tflops*100:.0f}%")
    title = (
        f"Optimization Trajectory: {kernel_dir.name}     "
        f"Rounds: {total_rounds}  |  "
        f"TFLOPS: {initial_tflops:.1f} -> {final_t:.2f}  |  "
        f"Speedup: {final_s:.2f}x{_vs_pt}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=22)
    ax.set_ylabel("Cumulative Speedup (x)", fontsize=13, color="#1565c0")
    ax.set_xlabel("Optimization Round", fontsize=13)
    ax.set_xlim(-0.8, len(rounds)-0.2)
    ax.set_ylim(0.88, max(running_best)*1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))
    ax.tick_params(axis="y", labelcolor="#1565c0")
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out = output_path or (kernel_dir / "final_output" / "trajectory_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"[chart] {out} ({total_rounds} rounds, "
          f"{initial_tflops:.1f}->{final_t:.2f} TFLOPS, {final_s:.2f}x)")
    return out


def _self_test():
    kd = _PROJECT_DIR / "outputs" / "vector_add_fp16_N65536"
    if not (kd / "optimization_trajectory.json").exists():
        print("[chart] SKIP: no trajectory")
        return
    try: generate(kd); print("[chart] OK")
    except Exception as e: print(f"[chart] ERROR: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1: generate(Path(sys.argv[1]))
    else: _self_test()
