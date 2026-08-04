#!/usr/bin/env python3
"""simulator 完整翻译 — trace.json + instr_exe.csv → 人话。不删任何内容。

trace.json  : 每个事件 → 时间[起..止] + 通道 + 名称 + 时长 + 核/线程 (含 start/end/并行)
instr_exe.csv: 每条指令 → 名称 + 管道(中文) + 调用次数 + cycles + 耗时(running_time 优先,
              0 时用 cycles÷1.9GHz) + detail 里的搬运字节数

════════════════════════════════════════════════════════════════════
★ 输入/输出路径 (由你改动下面两个变量即可, 不用传命令行参数):
    INPUT  = 输入 sim_prof 目录 (含 OPPROF_*/simulator/trace.json + instr_exe.csv)
    OUTPUT = 输出翻译文本的路径 (None = 直接打印到屏幕)
  例:
    INPUT  = "input/matmul/e2e_run/03_sim/sim_prof"
    OUTPUT = "input/matmul/e2e_run/06_diagnosis/sim_translation.txt"
  也可命令行: python translate_trace.py <sim_prof目录> [--instr|--trace|--all]
════════════════════════════════════════════════════════════════════
"""
import csv
import json
import re
import sys
from pathlib import Path

# ── ★ 由你改动: 输入 sim_prof 目录 + 输出文件 ──
INPUT = "input/matmul/e2e_run/03_sim/sim_prof"   # ← 改成你的 sim_prof 目录
OUTPUT = None                                     # ← 改输出路径; None=打印屏幕

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PIPE_CN = {
    "VECTOR": "向量计算", "SCALAR": "标量计算", "CUBE": "矩阵乘(Cube)",
    "MTE1": "搬运L1→L0A/B", "MTE2": "搬运GM/L2→L1/UB", "MTE3": "搬运UB→GM/L2",
    "FIXP": "FIXPIPE转换", "FLOWCTRL": "控制流", "CACHEMISS": "ICACHE未命中",
    "ALL": "全核", "USEMASK": "自定义打点",
}
AIC_FREQ_MHZ = 1900.0


def _find_sim(base):
    for opprof in sorted(Path(base).glob("OPPROF_*")):
        sim = opprof / "simulator"
        if sim.is_dir():
            return sim
    return None


def _dur_ns(rt, cyc):
    try:
        rt_us = float(rt)
        if rt_us > 0:
            return rt_us * 1000.0
    except ValueError:
        pass
    try:
        return int(cyc) * 1000.0 / AIC_FREQ_MHZ
    except ValueError:
        return 0.0


def translate_trace(trace_path):
    data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    xs = [e for e in events if e.get("ph") == "X"]
    print(f"── trace.json: {len(events)} 事件 ({len(xs)} 个完整段) ──")
    cats = {}
    for e in xs:
        cats[e.get("cat", "?")] = cats.get(e.get("cat", "?"), 0) + 1
    print(f"   通道分布: {cats}")
    print()
    for e in xs:
        name = e.get("name", "?")
        ts = float(e.get("ts", 0))
        dur = float(e.get("dur", 0))
        cat = e.get("cat", "?")
        print(f"   [{ts:>8.1f}..{ts + dur:>8.1f}] {str(cat):12s} {name}  "
              f"时长={dur:.1f}ns pid={e.get('pid')} tid={e.get('tid')}")


def translate_instr(csv_path, limit=None):
    print(f"── {Path(csv_path).name} (指令级, 已按指令名聚合) ──")
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8", newline="")))
    print(f"   总 {len(rows)} 条指令")
    print()
    shown = rows if limit is None else rows[:limit]
    for r in shown:
        instr = r.get("instr", "")
        pipe = r.get("pipe", "")
        cc = r.get("call_count", "")
        cyc = r.get("cycles", "")
        rt = r.get("running_time(us)", "")
        detail = r.get("detail", "")
        dur = _dur_ns(rt, cyc)
        size = 0
        m = re.search(r"XD:X[23]=(\w+)", detail or "")
        if m:
            try:
                size = int(m.group(1), 16)
            except ValueError:
                pass
        pipe_cn = PIPE_CN.get(pipe, pipe or "?")
        size_s = f"{size}B" if size else "-"
        print(f"   {str(instr)[:24]:24s} 管道={str(pipe_cn)[:16]:16s} 次数={str(cc)[:5]:5s} "
              f"cycles={str(cyc)[:6]:6s} 耗时={dur:8.1f}ns 搬运={size_s}")
    if limit is not None and len(rows) > limit:
        print(f"   ... (仅显示前 {limit}, 共 {len(rows)})")


if __name__ == "__main__":
    import io
    args = sys.argv[1:]
    repo = Path(__file__).resolve().parent.parent   # 仓库根 (analyzers/..)
    # 输入: 命令行参数 > 脚本顶部 INPUT 配置
    if args:
        base = args[0]
    elif INPUT:
        base = str(repo / INPUT) if not Path(INPUT).is_absolute() else INPUT
    else:
        print("用法: python translate_trace.py <sim_prof目录> [--instr|--trace|--all]  或改脚本顶部 INPUT")
        sys.exit(1)
    mode = next((a for a in args if a.startswith("--")), "--all")

    # 输出重定向到 buffer, 最后写文件或打印
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf

    sim = _find_sim(base)
    if sim is None:
        sys.stdout = old
        sys.exit(f"❌ 找不到 {base}/OPPROF_*/simulator/   (改脚本顶部 INPUT 或传命令行参数)")

    if mode in ("--trace", "--all"):
        t = sim / "trace.json"
        if t.exists():
            translate_trace(t)
        else:
            print("  ⚠ 无 trace.json")
        print()

    if mode in ("--instr", "--all"):
        csvs = sorted(sim.glob("core*/*_instr_exe.csv"))
        if csvs:
            # 翻译第一个核的 (全核内容相似, 数量一致); 提示其余核
            translate_instr(csvs[0])
            if len(csvs) > 1:
                print(f"\n  (其余 {len(csvs) - 1} 个核的 instr_exe 结构相同, 数值不同)")
        else:
            print("  ⚠ 无 *_instr_exe.csv")

    sys.stdout = old
    text = buf.getvalue()
    if OUTPUT:
        out = Path(OUTPUT) if Path(OUTPUT).is_absolute() else repo / OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[translate_trace] 已写入: {out}")
    else:
        print(text, end="")
