#!/usr/bin/env python3
"""重新测量 outputs/ 下所有算子的 best_kernel.py 端到端耗时 — ★工业级方法 (do_bench 同口径).

方法 (与 verify Event / bench_common.measure_event / triton testing.do_bench 一致):
  - 注入 KERNEL_EVENT_TIME 分支 (agents.verifier._inject_event_timing)
  - 多窗口 median: KERNEL_EVENT_REPS 个独立 Event 窗口 (每窗口包 KERNEL_LOOP 次) → median
  - ★破 L2 (工业级口径): 每窗口前重放输入分配 (新地址, _keep 持有防 caching allocator 复用)
    — Ascend 无清 L2 API, 重建等效 do_bench 的 clear_cache / 工业级基准的 n_buf 输入轮换
  - 同时测热 L2 版本 (rebuild_inputs=False) → 输出"虚高倍数" (冷/热)

★--l2 模式 (2026-08-13): 对每个算子额外用 msprof op 实测 L2Cache.csv 命中率 (热/冷各一次),
  用数据定论破 L2 是否生效:
    - 热高冷低 + 耗时无差异 → L2 命中不省 MTE2 时间 (硬件行为, 虚高 1.0 是真实结果)
    - 两模式命中率都低 → "热 L2 虚高"假设在 910B3/triton-ascend 上不成立, 口径本来就一致
    - 热高冷低 + 耗时差异大 → 重建未生效 (bug, 需排查注入)

终端输出表格 + 每算子 outputs/<op>/final_output/remeasure_best.json。

用法 (在仓库根 triton_agent_optimizer/ 下):
  python3 feedback/remeasure_best.py                        # 全部算子的 best_kernel.py
  python3 feedback/remeasure_best.py --op matmul            # 只测一个算子
  python3 feedback/remeasure_best.py --l2 --op vector_add   # 加测 L2 命中率 (msprof op, 较慢)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))   # ★仓库相对路径导入 (feedback/ → 仓库根)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from agents.verifier import _inject_event_timing  # noqa: E402


def _inject(kernel_path: Path, rebuild: bool) -> tuple:
    """注入 Event 计时文件 → (evt_path, error_or_None)."""
    try:
        src = kernel_path.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"读文件失败: {e}"
    injected = _inject_event_timing(src, rebuild_inputs=rebuild)
    if not injected:
        return None, "无标准 KERNEL_LOOP 循环 (无法注入 Event 计时)"
    tag = "cold" if rebuild else "hot"
    evt = kernel_path.parent / f"event_remeasure_{tag}.py"
    evt.write_text(injected, encoding="utf-8")
    return evt, None


def measure(kernel_path: Path, rebuild: bool, loop: int = 30, reps: int = 5) -> tuple:
    """对 kernel 跑一次 Event 计时 (注入 KERNEL_EVENT_TIME 分支).
    rebuild=True → 破 L2 (工业级口径); False → 热 L2 (量化虚高).
    返回 (median_us, error_or_None)."""
    evt, err = _inject(kernel_path, rebuild)
    if evt is None:
        return None, err
    env = dict(os.environ, KERNEL_EVENT_TIME="1", KERNEL_LOOP=str(loop),
               KERNEL_EVENT_REPS=str(reps), MATMUL_VERIFY="")
    try:
        r = subprocess.run(["python3", str(evt)], capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace",
                           timeout=1800, env=env)
    except Exception as e:
        return None, f"运行失败: {str(e)[:120]}"
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"EVENT_E2E_US:([\d.]+)", out)
    if not m:
        return None, (out.strip()[-200:] or "无 EVENT_E2E_US 输出")
    return float(m.group(1)), None


def _kernel_name(src: str) -> str:
    """第一个 @triton.jit 函数名 (msprof op 的 kernel-name 用)."""
    m = re.search(r"@triton\.jit\s*\ndef\s+(\w+)\s*\(", src)
    return m.group(1) if m else ""


def measure_l2_hit(kernel_path: Path, rebuild: bool, loop: int = 30, reps: int = 5,
                   work_root: Path | None = None) -> tuple:
    """★实测 L2 命中率 (msprof op 的 L2Cache.csv aic_total_hit_rate).
    返回 (hit_rate_0to1_or_None, error_or_None)."""
    evt, err = _inject(kernel_path, rebuild)
    if evt is None:
        return None, err
    src = kernel_path.read_text(encoding="utf-8")
    kname = _kernel_name(src)
    if not kname:
        return None, "无 @triton.jit kernel 名"
    tag = "cold" if rebuild else "hot"
    work = (work_root or evt.parent) / f"l2_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, KERNEL_EVENT_TIME="1", KERNEL_LOOP=str(loop),
               KERNEL_EVENT_REPS=str(reps), MATMUL_VERIFY="")
    cmd = ["msprof", "op", f"--kernel-name={kname}", f"--output={work}",
           "--warm-up=5", "python3", str(evt)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="backslashreplace",
                           timeout=3600, env=env)
    except Exception as e:
        return None, f"msprof op 运行失败: {str(e)[:120]}"
    # 找 L2Cache.csv (OPPROF_*/L2Cache.csv 或直接)
    cands = sorted(work.rglob("L2Cache.csv"))
    if not cands:
        return None, (f"无 L2Cache.csv: {(r.stdout or '')[-150:] + (r.stderr or '')[-150:]}")
    try:
        with open(cands[0], encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        return None, f"L2Cache.csv 读取失败: {e}"
    if not rows:
        return None, "L2Cache.csv 空"
    for col in ("aic_total_hit_rate(%)", "aiv_total_hit_rate(%)", "aic_total_hit_rate",
                "total_hit_rate(%)"):
        v = rows[0].get(col)
        if v is not None and v.strip():
            try:
                val = float(v)
                return (round(val / 100.0, 4) if val > 1 else round(val, 4)), None
            except ValueError:
                continue
    return None, f"L2Cache.csv 无命中率列: {list(rows[0].keys())[:10]}"


def main():
    p = argparse.ArgumentParser(description="重测所有 best_kernel.py 端到端 (工业级口径, 破 L2)")
    p.add_argument("--op", type=str, default=None, help="只测指定算子 (缺省=全部)")
    p.add_argument("--loop", type=int, default=30, help="每 Event 窗口包 KERNEL_LOOP 次 (默认 30)")
    p.add_argument("--reps", type=int, default=5, help="独立 Event 窗口数, 取 median (默认 5)")
    p.add_argument("--skip-existing", action="store_true",
                   help="已有 remeasure_best.json 的算子跳过")
    p.add_argument("--l2", action="store_true",
                   help="★加测 L2 命中率 (msprof op 热/冷各一次, 慢 ~2-4min/算子; 用数据定论破 L2 是否生效)")
    args = p.parse_args()

    out_root = _PROJECT_DIR / "outputs"
    if not out_root.exists():
        print(f"❌ 无 outputs/ 目录: {out_root} (先跑过 main.py 优化)")
        return 1

    # 收集算子: outputs/*/best_kernel.py (best_kernel 缺失的跳过)
    cands = []
    for kd in sorted(out_root.iterdir()):
        if not kd.is_dir():
            continue
        bk = kd / "best_kernel.py"
        if args.op and kd.name != args.op:
            continue
        if args.op and kd.name == args.op and not bk.exists():
            print(f"⚠ {kd.name}: 无 best_kernel.py")
            return 1
        if bk.exists():
            cands.append((kd.name, bk))
    if not cands:
        print("⚠ 没有找到任何 best_kernel.py (outputs/ 下每个算子目录需有 best_kernel.py)")
        return 1

    print(f"═══ 重测 {len(cands)} 个算子的 best_kernel.py 端到端 (Event, 多窗口 median x{args.reps}, "
          f"LOOP x{args.loop}){' + L2 命中率' if args.l2 else ''} ═══")
    print(f"  冷L2 = 每窗口重建输入 (工业级口径, 与 do_bench 同效) | 热L2 = 同批输入 (旧 verify 口径)")
    print("")

    rows = []
    for name, bk in cands:
        rs = _PROJECT_DIR / "outputs" / name / "final_output" / "remeasure_best.json"
        if args.skip_existing and rs.exists():
            print(f"  ⏭ {name:20s} 已有 remeasure_best.json (--skip-existing)")
            try:
                rows.append(json.loads(rs.read_text(encoding="utf-8")))
            except Exception:
                pass
            continue
        print(f"  ⏳ {name} ...", flush=True)
        cold, err_c = measure(bk, rebuild=True, loop=args.loop, reps=args.reps)
        hot, err_h = measure(bk, rebuild=False, loop=args.loop, reps=args.reps)
        if cold is None or hot is None:
            print(f"  ⚠ {name}: 冷={err_c or cold} | 热={err_h or hot}")
            continue
        inflate = round(cold / hot, 2) if hot else None   # 冷/热 = 热L2 快了多少倍 (虚高倍数)
        row = {"op": name, "kernel": str(bk),
               "cold_l2_us_industrial": round(cold, 2),   # ★工业级口径 (破 L2)
               "hot_l2_us_old_verify": round(hot, 2),     # 旧 verify 口径 (热 L2)
               "l2_inflate_x": inflate,                   # >1 = 热L2 虚高 (假快) 倍数
               "loop": args.loop, "reps": args.reps,
               "method": "Event 注入多窗口 median, 每窗口重建输入破 L2 (do_bench 同款)",
               "measured_at": datetime.now().isoformat(timespec="seconds")}
        if args.l2:
            l2c, err_l2c = measure_l2_hit(bk, rebuild=True, loop=args.loop, reps=args.reps)
            l2h, err_l2h = measure_l2_hit(bk, rebuild=False, loop=args.loop, reps=args.reps)
            row["l2_hit_rate_cold"] = l2c
            row["l2_hit_rate_hot"] = l2h
            print(f"    L2命中率: 冷={l2c if l2c is not None else err_l2c} "
                  f"热={l2h if l2h is not None else err_l2h}")
        rows.append(row)
        rs.parent.mkdir(parents=True, exist_ok=True)
        rs.write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"    → 冷L2(工业级) {cold:8.1f}us | 热L2(旧) {hot:8.1f}us | 虚高 {inflate}x "
              f"→ {rs}")

    # ── 终端汇总表 ──
    print("")
    print("═" * 96)
    if args.l2:
        print(f"  {'算子':<20}{'冷L2(us)':>10}{'热L2(us)':>10}{'虚高':>7}"
              f"{'冷命中率':>10}{'热命中率':>10}")
        print("═" * 96)
        for r in rows:
            _i = f"{r['l2_inflate_x']}x" if r.get("l2_inflate_x") else "—"
            _lc = f"{r['l2_hit_rate_cold']:.2f}" if r.get("l2_hit_rate_cold") is not None else "—"
            _lh = f"{r['l2_hit_rate_hot']:.2f}" if r.get("l2_hit_rate_hot") is not None else "—"
            print(f"  {r['op']:<20}{r['cold_l2_us_industrial']:>10.1f}"
                  f"{r['hot_l2_us_old_verify']:>10.1f}{_i:>7}{_lc:>10}{_lh:>10}")
        print("═" * 96)
        print("  ★定论: 热命中≈冷命中 → L2 复用本就不显著, 虚高1.0是真实结果 (口径一致);")
        print("         热高冷低+耗时无差 → L2 命中不省 MTE2 时间; 热高冷低+耗时差大 → 重建未生效")
    else:
        print(f"  {'算子':<20}{'冷L2 工业级(us)':>16}{'热L2 旧口径(us)':>16}{'虚高倍数':>10}")
        print("═" * 96)
        for r in rows:
            _i = f"{r['l2_inflate_x']}x" if r.get("l2_inflate_x") else "—"
            print(f"  {r['op']:<20}{r['cold_l2_us_industrial']:>16.1f}"
                  f"{r['hot_l2_us_old_verify']:>16.1f}{_i:>10}")
    if rows:
        _vals = [r["l2_inflate_x"] for r in rows if r.get("l2_inflate_x")]
        if _vals:
            _avg = sum(_vals) / len(_vals)
            _mx = max(_vals); _mn = min(_vals)
            print("═" * 96)
            print(f"  虚高倍数 (热L2假快): 平均 {_avg:.2f}x | 最小 {_mn:.2f}x | 最大 {_mx:.2f}x")
            print(f"  ★>1.5x 的算子: {[r['op'] for r in rows if (r.get('l2_inflate_x') or 1) > 1.5]} "
                  f"(旧 verify Event 偏乐观, 与工业级对比时用冷L2列)")
    print("")
    print(f"  单算子明细 → outputs/<op>/final_output/remeasure_best.json")
    print(f"  (--l2 时 json 含 l2_hit_rate_cold/hot 命中率, 用数据定论破 L2 是否生效)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
