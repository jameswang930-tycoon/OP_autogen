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
    bench_dir = _PROJECT_DIR / "bench_910b3" / "outputs"   # ★产物统一在 bench_910b3/outputs/
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
    # ★v4.6: 快测门 REVERT/未测轮 hist 诚实记 speedup=null (未跑 msprof 不编数) —
    #   .get 默认值只在 key 缺失时生效, 值为 null 时拿到 None → np 比较直接 TypeError.
    #   画图回退 prev_speedup (= 当前已接受水平, 与 scheduler 失败轮同口径, 防假掉点);
    #   连 prev_speedup 都没有的脏行 → NaN (点不画, fmax 累积忽略).
    def _speed(r):
        v = r.get("speedup")
        if not isinstance(v, (int, float)):
            v = r.get("prev_speedup")
        return float(v) if isinstance(v, (int, float)) else float("nan")
    speeds = [_speed(r) for r in history]                       # 每轮 vs baseline
    cum_speeds = list(np.fmax.accumulate(np.array(speeds, dtype=float)))  # running best (fmax 忽略 NaN)
    # 基准点: 第 0 轮前 speedup=1.0 (初始)
    rounds = [0] + rounds
    cum_speeds = [1.0] + cum_speeds
    speeds = [1.0] + speeds
    decisions = ["BASELINE"] + decisions
    strategies = ["Baseline"] + strategies

    # TFLOPS: state.initial_tflops (scheduler 算的) — 非 cube 算子可能没有 → None, 不画误导轴
    #   (★修复: 原来 `or 6.4` 给访存型算子硬画假数, 已移除 — 访存型算子该看 GB/s, 不是 TFLOPS)
    initial_tflops = state.get("initial_tflops")
    # ★PyTorch 基准: 优先 state (scheduler 存的), 缺则自动按算子读 bench json; None = 无真实数据 → 不画误导虚线
    # ★对比用时间 (time_us, 同算子同形状直接可比); tflops 仅兜底展示
    pt_bench = _load_pytorch_bench(kernel_dir, state) or {}
    # (pytorch_tflops 只用于 _load_pytorch_bench 内部/state 读取; 本函数只用时间口径对比)
    pytorch_time_us = pt_bench.get("time_us")
    # ★F3: hist 若存了每轮真实 tflops (kernel 结构变化后 FLOPs 变) 就逐轮用, 否则 initial×speedup 兜底
    hist_tflops = [r.get("tflops") for r in history]
    tflops_arr = None
    if initial_tflops:
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
    # ★CJK 字体普遍无 bold/italic 变体 → matplotlib 找 bold 失败刷警告 (无害但烦人), 抑制
    try:
        import logging
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    except Exception:
        pass

    # 中文字体: ★按字形覆盖检测 (扫描每个字体的 cmap 是否真含中文码点), 不是按名字猜 —
    #   名字匹配不可靠 (服务器字体名千奇百怪, 装了也叫不到); 选第一个覆盖中文的.
    #   font.family 直接设具体字体名 (不走 sans-serif 列表 → 避免 bold/italic 时 fallback).
    #   ★Last Resort 是 matplotlib 自带"最后手段"占位字体: 对所有码点都有占位 glyph,
    #     get_char_index 一定非 0 → 会误选, 渲染成方块带字 — 必须排除.
    #     (名字 "Last Resort High-Efficiency" / 文件 lastResort.ttf, 两处都要查)
    def _cjk_covers(fe):
        if "last resort" in fe.name.lower() or "lastresort" in fe.fname.lower():
            return False
        try:
            f = ft2font.FT2Font(fe.fname)
            # '你' U+4F60 + 常用 CJK 区边界 U+9FA5: 都覆盖才认
            return f.get_char_index(0x4F60) != 0 and f.get_char_index(0x9FA5) != 0
        except Exception:
            return False

    try:
        import matplotlib.font_manager as fm
        from matplotlib import ft2font

        _cjk_fonts = [f for f in fm.fontManager.ttflist if _cjk_covers(f)]
        if not _cjk_fonts:
            # ★字体可能是装 matplotlib 缓存之后才装的 → 强制重扫系统字体再测
            try:
                fm._load_fontmanager(try_read_cache=False)
                _cjk_fonts = [f for f in fm.fontManager.ttflist if _cjk_covers(f)]
            except Exception:
                pass
        if _cjk_fonts:
            # 黑体(sans)优先, 排序稳定
            _pref = ["Noto Sans CJK", "WenQuanYi", "SimHei", "Microsoft YaHei",
                     "Source Han", "Droid Sans Fallback"]
            _cjk_fonts.sort(key=lambda f: next(
                (i for i, k in enumerate(_pref) if k in f.name), 99))
            plt.rcParams["font.family"] = _cjk_fonts[0].name
            print(f"[chart] 中文字体: {_cjk_fonts[0].name}")
        else:
            plt.rcParams["font.family"] = "DejaVu Sans"
            print("[chart] ⚠ 未找到真实中文字体 (中文会变方块)! 服务器装:\n"
                  "      apt-get install fonts-noto-cjk\n"
                  "      装完若仍乱码: rm -rf ~/.cache/matplotlib (字体缓存)")
    except Exception as e:
        print(f"[chart] 中文字体检测异常: {e}")
    plt.rcParams["axes.unicode_minus"] = False

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
        # ★统一 Event 设备端口径: bench_pytorch 现用 Event 测 time_us → 我们的 baseline_e2e_event_ns (Event)
        #   优先 Event-vs-Event (同口径); 缺 Event 基线才退化用 msprof baseline_e2e_ns.
        _base_ns = state.get("baseline_e2e_event_ns") or state.get("baseline_e2e_ns") or state.get("baseline_ns")
        if _pt_ns and _base_ns:
            pytorch_speedup = _base_ns / _pt_ns
        else:
            pytorch_speedup = None
        if pytorch_speedup:
            ax.axhline(y=pytorch_speedup, color="gray", linestyle="--", linewidth=2.5,
                       alpha=0.8, zorder=1)
            ax.text(rounds[0]+0.5, pytorch_speedup + 0.03,
                    f"PyTorch ({pytorch_time_us:.0f}us, Event)\n口径: 双端 Event 设备侧",
                    fontsize=9, color="gray", ha="left", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # ═══ Industrial baseline (工业级天花板: torch.compile/TorchAir 或 CANN-FA) — 第二条虚线 ═══
    #   口径与 PyTorch 线同: 我们端到端基线 vs 工业级端到端 (都是 msprof Σ全部).
    #   ★这才是"看优化效果"的对比 — 我们最终 kernel vs 工业级实现.
    industrial_time_us = state.get("industrial_time_us")
    if industrial_time_us:
        _ind_ns = industrial_time_us * 1000.0
        # ★industrial 现用 Event 测 → 优先 Event 基线 (同口径); 缺则 msprof
        _base_ns = state.get("baseline_e2e_event_ns") or state.get("baseline_e2e_ns") or state.get("baseline_ns")
        if _ind_ns and _base_ns:
            ind_speedup = _base_ns / _ind_ns
            ax.axhline(y=ind_speedup, color="#d32f2f", linestyle="--", linewidth=2.0,
                       alpha=0.8, zorder=1)
            ax.text(len(rounds)-1, ind_speedup + 0.03,
                    f"Industrial (torch.compile/CANN-FA) ({industrial_time_us:.0f}us, Event)",
                    fontsize=9, color="#d32f2f", ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

    # ═══ Running Best ═══
    # ★fmax: 未测轮 (NaN) 不毒化后续累积 (np.maximum 会传播 NaN)
    running_best = np.fmax.accumulate(np.array(cum_speeds))
    ax.plot(rounds, running_best, color="#1565c0", linewidth=3, alpha=0.85, zorder=2)
    ax.fill_between(rounds, 1.0, running_best, alpha=0.06, color="#1565c0")

    # ═══ Points ═══
    # ★bug 修复 (2026-08-12): 原来点全画在 cumulative best 上 → REVERT/FAIL 轮掉速不可见
    #   (× 挂在最高点, 像"回退轮依然保持最优", 误导). 现在**点画实际 speedup**:
    #   KEEP 贴合蓝线 (KEEP 即新 best), REVERT/FAIL 真实掉下来, 一眼看出尝试失败.
    r_arr, s_arr = np.array(rounds), np.array(speeds)
    bl = np.array([d == "BASELINE" for d in decisions])
    kp = np.array([d == "KEEP" for d in decisions])
    rv = np.array([d == "REVERT" for d in decisions])
    fl = np.array([d == "FAIL" for d in decisions])

    if bl.any(): ax.scatter(r_arr[bl], s_arr[bl], c="gray", s=120, marker="s",
                             zorder=5, label="Baseline")
    if kp.any(): ax.scatter(r_arr[kp], s_arr[kp], c="#2ecc71", s=70,
                             edgecolors="white", linewidth=0.5, zorder=5,
                             label=f"KEEP ({kp.sum()})")
    if rv.any(): ax.scatter(r_arr[rv], s_arr[rv], c="#e74c3c", s=90, marker="X",
                             linewidth=1.5, zorder=5, label=f"REVERT ({rv.sum()})")
    if fl.any(): ax.scatter(r_arr[fl], s_arr[fl], c="#f39c12", s=90, marker="v",
                             edgecolors="white", linewidth=0.5, zorder=5,
                             label=f"FAIL ({fl.sum()})")

    # ═══ Annotations (防重叠: KEEP 只标每 tier 内 speedup 最高的轮 — 完整记录在 rounds.csv;
    #   文本截短; _free_y 垂直错开) ═══
    _used_spots = []
    def _free_y(x, y, step=0.04):
        while any(abs(x - u) < 1.3 and abs(y - v) < 0.065 for u, v in _used_spots):
            y += step
        _used_spots.append((x, y))
        return y

    # 每 tier 只标 speedup 最大的 KEEP 轮 (i 是 rounds 索引, history 索引 = i-1)
    _best_keep = set()
    for _t, _s, _e in tier_ranges:
        _cand = [i for i in range(_s, _e + 1)
                 if i + 1 < len(decisions) and decisions[i + 1] == "KEEP"]
        if _cand:
            _best_keep.add(max(_cand, key=lambda i: speeds[i + 1]) + 1)

    for i in range(1, len(rounds)):
        if decisions[i]=="KEEP" and i in _best_keep and speeds[i] > 1.07:
            ax.annotate(strategies[i][:14],
                xy=(rounds[i], speeds[i]),
                xytext=(rounds[i]+0.4, _free_y(rounds[i], speeds[i]+0.07)),
                fontsize=6.8, color="#1565c0",
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=0.8, alpha=0.6),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.85, ec="#1565c0", lw=0.6),
                zorder=10)
        if decisions[i]=="REVERT" and speeds[i] < 0.97:
            ax.annotate(reasons[i][:16],
                xy=(rounds[i], speeds[i]),
                xytext=(rounds[i]+0.5, _free_y(rounds[i], speeds[i]-0.06)),
                fontsize=6.6, color="#c62828",
                arrowprops=dict(arrowstyle="->", color="#c62828", lw=0.7, alpha=0.5),
                bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.85, ec="#c62828", lw=0.6),
                zorder=9)
        if decisions[i]=="FAIL" and np.isfinite(speeds[i]):
            ax.annotate("×失败",
                xy=(rounds[i], speeds[i]),
                xytext=(rounds[i]+0.3, _free_y(rounds[i], speeds[i]-0.05)),
                fontsize=6.5, color="#b9770e",
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=0.6, alpha=0.6),
                zorder=9)

    # ═══ Phase labels (top) — ★短段也标 (只标 T{n} 编号防重叠), 不丢任何 tier ═══
    ylim = ax.get_ylim()
    for tier, start, end in tier_ranges:
        if start < end:
            mid = (start+end)/2
            if (end - start) >= 4:
                _pl, _py, _ps = f"T{tier}: {TIER_NAME[tier-1]}", ylim[1]*0.975, 9.5
            else:
                _pl, _py, _ps = f"T{tier}", ylim[1]*0.96, 8.0   # ★短段: 只画编号, 塞得下不重叠
            ax.text(mid, _py, _pl,
                    fontsize=_ps, fontweight="bold", color=TIER_FG[(tier-1)%6],
                    ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc=TIER_BG[(tier-1)%6],
                              alpha=0.85, ec=TIER_FG[(tier-1)%6], lw=1.0),
                    zorder=20)

    # ═══ Right Y: TFLOPS (只有有意义的 initial_tflops 才画 — 访存型算子无 → 不画误导轴) ═══
    if tflops_arr is not None:
        ax2 = ax.twinx()
        ymin, ymax = ax.get_ylim()
        ax2.set_ylim(ymin * initial_tflops, ymax * initial_tflops)
        ax2.set_ylabel("Throughput (TFLOPS)", fontsize=12, color="#7b1fa2")
        ax2.tick_params(axis="y", labelcolor="#7b1fa2")
        ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ═══ Title (含绝对耗时: 相对加速比看不出 baseline 2s→1s 与 2ms→1ms 的差别) ═══
    total_rounds = len(rounds) - 1
    final_s = running_best[-1]
    final_t = tflops_arr[-1] if tflops_arr is not None else None
    _tf_s = f"{initial_tflops:.1f} -> {final_t:.2f}" if final_t is not None else "—"
    # ★绝对耗时 (Event 优先: 工业级口径; 缺则 msprof 端到端; 都没有不显示)
    _base_ns0 = state.get("baseline_e2e_event_ns") or state.get("baseline_e2e_ns") or state.get("baseline_ns")
    _abs_s = ""
    if _base_ns0 and final_s:
        _final_us = _base_ns0 / final_s / 1000.0
        _src = "Event" if (state.get("baseline_e2e_event_ns")) else "msprof"
        _abs_s = f"  |  耗时: {_base_ns0/1000:.0f}us → {_final_us:.0f}us ({_src})"
    # ★vs PyTorch 用端到端口径对比 (直接可比, 两端都 msprof Σ全部): 我们最优用时 vs pytorch 用时
    #   (★修复: 原 tflops 兜底对比已移除 — 跨算子类型用 TFLOPS 比失真)
    _vs_pt = ""
    if pt_bench or state.get("industrial_time_us"):
        _cur_ns = state.get("baseline_e2e_ns") or state.get("baseline_ns")
        _final_us = (_cur_ns / final_s / 1000.0) if (_cur_ns and final_s) else None
        if pytorch_time_us and _final_us:
            _vs_pt = (f"  |  vs PyTorch({pytorch_time_us:.0f}us): "
                      f"我们最优 {_final_us:.0f}us = {_final_us/pytorch_time_us*100:.0f}%")
        _ind_us = state.get("industrial_time_us")
        if _ind_us and _final_us:
            _vs_pt += (f"  |  vs Industrial({_ind_us:.0f}us): "
                       f"{_final_us/_ind_us*100:.0f}%")
    # ★title 拆两行 (一行塞不下会顶到图外/挤压): 主行 = 轨迹信息, 次行 = 对比口径
    _title_1 = (f"Optimization Trajectory: {kernel_dir.name}   "
                f"Rounds: {total_rounds}  |  TFLOPS: {_tf_s}  |  "
                f"Speedup: {final_s:.2f}x{_abs_s}")
    _title_2 = _vs_pt.strip()
    title = _title_1 + (f"\n{_title_2}" if _title_2 else "")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.set_ylabel("Cumulative Speedup (x)", fontsize=13, color="#1565c0")
    ax.set_xlabel("Optimization Round", fontsize=13)
    ax.set_xlim(-0.8, len(rounds)-0.2)
    ax.set_ylim(0.88, max(running_best)*1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2fx"))
    ax.tick_params(axis="y", labelcolor="#1565c0")
    # ★轮次多时 x 刻度稀疏 (防 round 数字挤在一起): 最多 ~16 个刻度
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=16))
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ★两行 title 下 tight_layout 常失败 ("top/bottom margins cannot be made large enough")
    #   → 显式留顶部空间 (title 两行) + 底部 x 轴
    try:
        fig.subplots_adjust(top=0.86, bottom=0.09, left=0.055, right=0.98)
    except Exception:
        pass
    out = output_path or (kernel_dir / "final_output" / "trajectory_chart.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ★每轮明细导出 CSV (图同数据, 机器/Excel 可读; 含 FAIL 轮与 error)
    try:
        import csv as _csv
        csv_path = out.parent / "rounds.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as _f:
            _w = _csv.writer(_f)
            _w.writerow(["round", "tier", "decision", "result", "strategy", "change",
                         "speedup", "kernel_speedup", "e2e_event_us", "error"])
            for _h in history:
                _evt = _h.get("e2e_event_ns")
                _w.writerow([
                    _h.get("round"), _h.get("tier"), _h.get("decision"), _h.get("result"),
                    (_h.get("strategy") or ""), (_h.get("change") or ""),
                    _h.get("speedup"), _h.get("kernel_speedup"),
                    (round(_evt / 1000.0, 1) if _evt else ""),
                    (_h.get("error") or ""),
                ])
    except Exception:
        pass

    print(f"[chart] {out} ({total_rounds} rounds, "
          f"{initial_tflops if initial_tflops is not None else 'N/A'}->"
          f"{final_t if final_t is not None else 'N/A'} TFLOPS, {final_s:.2f}x)"
          f"{_abs_s}")
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
