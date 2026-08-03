#!/usr/bin/env python3
"""真实数据报告 — 参照 costModel/cost_emulator/simulator.py 的输出格式。

输入: e2e_run/05_merged/merged.json (由 run_all.sh 三源合并产出, 真实数据)
输出: 与 simulator.py 相同样式的报告
  - 默认:  Gantt 流水图 + op 表 + 时间/引擎/带宽利用率 + 并行度 + 关键路径
  - --llm: 紧凑结构化文本 (EXECUTION SUMMARY / PER-OP / ... 供 LLM)
  - --nx:  node-link JSON 图 (nodes=ops, links=依赖, graph=汇总)

与 simulator.py 的区别: simulator 用 SATURATION 模型算 duration, 本脚本用
merged.json 里的真实 duration (来自 simulator instr_exe per-call) + 真实结构
(dst/src/size/依赖) + 真机端到端 (total_ns/cores)。同步 op (set_flag/wait_flag/
pipe_barrier) 作为独立引擎 (Sync) 计入调度。

用法:
  python real_report.py [merged.json] [--llm|--nx]
  默认找 input/matmul/e2e_run/05_merged/merged.json
"""
import json
import sys
from pathlib import Path

# 统一 UTF-8 输出 (服务器 Linux 正常; Windows GBK 控制台也不崩溃)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ── 引擎表 (对照 simulator.py, 新增 Sync/FixPipe) ────────────────────────────
ENGINE_FOR = {
    'gm_to_ub': 0, 'ub_to_gm': 1,
    'vadd': 2, 'vmul': 2, 'vsub': 2, 'vdiv': 2, 'vmax': 2, 'vmin': 2,
    'vexp': 2, 'vlog': 2, 'vabs': 2, 'vrelu': 2, 'vsqrt': 2, 'vtanh': 2,
    'vneg': 2, 'vrsqrt': 2, 'vln': 2,
    'gm_to_l1': 3, 'l1_to_l0': 4,
    'matmul': 5, 'matrixmul': 5, 'mix_matmul': 5, 'mmadL1': 5, 'batchMmadL1': 5,
    'l0_to_gm': 6,
    'set_flag': 7, 'wait_flag': 7, 'pipe_barrier': 7, 'sync_block': 7,
    'fixpipe': 8,
}
ENG_NAME = {0: 'GM→UB', 1: 'UB→GM', 2: 'VecUnit', 3: 'GM→L1', 4: 'L1→L0',
            5: 'CubeUnit', 6: 'L0→GM', 7: 'Sync', 8: 'FixPipe'}
N_ENG = len(ENG_NAME)

# 真机峰值参考 (来自 board.json memory_bandwidth_gb_s, 按通路); 其余标注未校准
PEAK_GB_S = {0: None, 1: None, 3: None, 4: None, 5: None, 6: None, 7: 0.0, 8: None}


class Op:
    __slots__ = ('idx', 'name', 'dst', 'src', 'src2', 'engine', 'size_kb',
                 'duration', 'start', 'end', 'cycles', 'pipe', 'core_id',
                 'effective_bw', 'bw_util', 'regime', 'line')

    def __init__(self, idx, name, dst, src, engine, size_kb=0.0, duration=0.0,
                 src2='', cycles=0, pipe='', core_id=''):
        self.idx, self.name, self.dst, self.src, self.src2 = idx, name, dst, src, src2
        self.engine, self.size_kb, self.duration = engine, size_kb, duration
        self.start = self.end = 0.0
        self.cycles, self.pipe, self.core_id = cycles, pipe, core_id
        self.effective_bw, self.bw_util, self.regime = 0.0, 0.0, ''
        self.line = 0

    def sig(self) -> str:
        if self.engine == 7 and not self.dst and not self.src:
            return f"{self.name}[pipe={self.pipe or '?'}]"   # 同步 op 无 buffer
        if self.src2:
            return f"{self.name}({self.dst}, {self.src}, {self.src2})"
        if self.name == 'vadd' and not self.src2:
            return f"{self.name}({self.dst}, {self.src}, 0)"
        return f"{self.name}({self.dst}, {self.src})"


def _fmt_size(kb: float) -> str:
    return f"{kb / 1024:.3g} MB" if kb >= 1024 else f"{kb:.3g} KB"


def _fmt_ns(ns: float) -> str:
    return f"{ns / 1000:.3g} µs" if ns >= 1000 else f"{ns:.3g} ns"


# ── 从 merged.json 构建 op 流 + 依赖 ─────────────────────────────────────────
def build_ops(merged: dict) -> tuple[list[Op], dict, dict]:
    ops = []
    for i, m in enumerate(merged["per_op_statistics"]):
        t = m.get("op_type") or "unknown"
        eng = ENGINE_FOR.get(t, 0)
        size = m.get("size_kb") or 0.0
        dur = m.get("duration_ns")
        dur = float(dur) if isinstance(dur, (int, float)) else 0.0
        op = Op(i, t, m.get("dst") or "", m.get("src") or "", eng,
                size_kb=size, duration=dur, src2=m.get("src2") or "",
                cycles=m.get("cycles") or 0,
                pipe=m.get("pipeline_channel") or "",
                core_id=m.get("core_id") or "")
        # 有效带宽 (真实): GB/s = size_bytes / duration_s / 1e9 = size_kb*1024/duration_ns
        if size > 0 and dur > 0:
            op.effective_bw = size * 1024.0 / dur
            peak = PEAK_GB_S.get(eng)
            op.bw_util = (op.effective_bw / peak) if peak else 0.0
            op.regime = 'saturated' if op.bw_util >= 0.95 else ('floor' if op.bw_util <= 0.5 else 'ramp')
        elif eng == 7:
            op.regime = 'sync'
        ops.append(op)

    n = len(ops)
    deps = {i: set() for i in range(n)}
    for i, m in enumerate(merged["per_op_statistics"]):
        for d in (m.get("dependencies") or []):
            j = d.get("from_op_id")
            if j is not None and 0 <= j < i:
                deps[i].add(j)
    return ops, deps, merged.get("execution_summary", {})


# ── ASAP 调度 (真实 duration) ────────────────────────────────────────────────
def schedule(ops, deps):
    done, e_free = {}, [0.0] * N_ENG
    for i, op in enumerate(ops):
        earliest = e_free[op.engine]
        for j in deps[i]:
            earliest = max(earliest, done[j])
        op.start, op.end = earliest, earliest + op.duration
        done[i], e_free[op.engine] = op.end, op.end


def hazard_detail(pred: Op, succ: Op) -> str:
    if succ.dst and succ.dst == pred.dst:
        return f"WAW on '{pred.dst}'"
    if (succ.src and succ.src == pred.dst) or (succ.src2 and succ.src2 == pred.dst):
        return f"RAW on '{pred.dst}'"
    if pred.src and succ.dst == pred.src:
        return f"WAR on '{pred.src}'"
    return "engine serialization"


# ── Gantt (默认模式) ─────────────────────────────────────────────────────────
def render(ops, deps):
    H = max((op.end for op in ops), default=0.0)
    GANTT_W = 90
    scale = GANTT_W / H if H > 0 else 1.0
    PAD = '  '

    def row(eng_ops):
        r = ['·'] * GANTT_W
        for op in eng_ops:
            c0 = int(round(op.start * scale))
            c1 = max(c0 + 1, int(round(op.end * scale)))
            for t in range(c0, min(c1, GANTT_W)):
                r[t] = '█'
        return ''.join(r)

    print()
    print(PAD + '┌─ Pipeline Execution Graph (REAL data from merged.json) ' + '─' * 10 + '┐')
    print()
    print(PAD + f'Time axis: {GANTT_W} cols ≈ {_fmt_ns(H)} makespan (scheduled from REAL per-op durations)')
    for eng in range(N_ENG):
        print(f"{PAD}  {ENG_NAME[eng]:10} │ {row([o for o in ops if o.engine == eng])}")
    print()

    print(PAD + f"{'Op':<4} {'Instruction':<34} {'Engine':<9} {'Size':<8} {'ns[..]':<16} {'BW%':<6} waits-for")
    print(PAD + '─' * 110)
    for op in ops:
        ws = ', '.join(f"op{j}({hazard_detail(ops[j], op)})" for j in sorted(deps[op.idx])) or '—'
        print(PAD + f"{op.idx:<4} {op.sig():<34} {ENG_NAME[op.engine]:<9} "
                    f"{_fmt_size(op.size_kb):<8} [{op.start:.1f}..{op.end:.1f}] "
                    f"{op.bw_util:5.0%} {ws}")
    print()

    print(PAD + 'Time breakdown (op duration ÷ total, sorted):')
    for op in sorted(ops, key=lambda o: o.duration, reverse=True):
        pct = op.duration / H if H else 0.0
        bar = ('█' * int(round(pct * 20))).ljust(20)
        print(PAD + f"  op{op.idx:<2} {op.sig():<34} [{bar}] {pct:6.1%}  ({op.duration:.1f}/{H:.1f} ns)")
    print()

    print(PAD + 'Engine utilization:')
    for eng in range(N_ENG):
        busy = sum(op.end - op.start for op in ops if op.engine == eng)
        pct = int(100 * busy / H) if H else 0
        bar = '█' * (pct // 5) + '░' * (20 - pct // 5)
        print(PAD + f"  {ENG_NAME[eng]:10} [{bar}] {pct:3}%  ({busy:.1f}/{H:.1f} ns)")
    print()

    print(PAD + 'Bandwidth utilization (effective ÷ peak, REAL size÷time):')
    for op in ops:
        peak = PEAK_GB_S.get(op.engine)
        if op.regime == 'sync':
            print(PAD + f"  op{op.idx:<2} {ENG_NAME[op.engine]:8} [sync]  (no data transfer)")
        elif peak:
            pct = int(round(100 * op.bw_util))
            bar = ('█' * (pct // 5)).ljust(20)
            print(PAD + f"  op{op.idx:<2} {ENG_NAME[op.engine]:8} [{bar}] {pct:3}%  "
                        f"({op.effective_bw:.4g}/{peak:.0f} GB/s, {op.regime})")
        else:
            print(PAD + f"  op{op.idx:<2} {ENG_NAME[op.engine]:8}  effective={op.effective_bw:.4g} GB/s  "
                        f"(peak 未校准, 见 board.json memory_bandwidth)")
    print()

    pairs = [(i, j) for i in range(len(ops)) for j in range(i + 1, len(ops))
             if ops[i].start < ops[j].end and ops[j].start < ops[i].end]
    print(PAD + f'Parallel overlap ({len(pairs)} pair(s)):')
    for i, j in pairs[:15]:
        s, e = max(ops[i].start, ops[j].start), min(ops[i].end, ops[j].end)
        print(PAD + f"  op{i} ∥ op{j}  (overlap {s:.1f}..{e:.1f} ns)")
    if not pairs:
        print(PAD + 'Execution is fully sequential — no parallel overlap.')
    print()
    print(PAD + '└' + '─' * 40 + '┘')
    print()


# ── LLM 模式 ─────────────────────────────────────────────────────────────────
def render_llm(ops, deps, summary):
    H = max((op.end for op in ops), default=0.0)
    print("=== EXECUTION SUMMARY ===")
    print(f"total_ns: {H:.2f}   (real board end-to-end: {summary.get('total_ns')} ns)")
    print(f"num_ops: {len(ops)}")
    print(f"num_cores: {summary.get('num_cores')}")
    pairs = [(i, j) for i in range(len(ops)) for j in range(i + 1, len(ops))
             if ops[i].start < ops[j].end and ops[j].start < ops[i].end]
    print(f"execution_mode: {'parallel' if pairs else 'sequential'}")
    print()

    print("=== TIME BREAKDOWN ===")
    for op in sorted(ops, key=lambda o: o.duration, reverse=True):
        r = op.duration / H if H else 0.0
        print(f"op{op.idx}: {op.sig()}  duration_ns={op.duration:.2f}  time_ratio={r:.2%}  "
              f"cycles={op.cycles}  pipe={op.pipe}")
    print()

    print("=== PER-OP STATISTICS ===")
    for op in ops:
        print(f"op{op.idx}: {op.sig()}")
        print(f"  engine: {ENG_NAME[op.engine]}  size: {_fmt_size(op.size_kb)}  "
              f"region/core: {op.core_id}")
        print(f"  ns=[{op.start:.2f}..{op.end:.2f}]  duration_ns={op.end - op.start:.2f}  "
              f"time_ratio={(op.end - op.start) / H:.2%}")
        print(f"  effective_bw={op.effective_bw:.4g} GB/s  regime={op.regime}")
        if deps[op.idx]:
            for j in sorted(deps[op.idx]):
                print(f"  blocked_by: op{j} — {hazard_detail(ops[j], op)}")
        else:
            print(f"  blocked_by: none")
        print()

    print("=== ENGINE UTILIZATION ===")
    for eng in range(N_ENG):
        busy = sum(op.end - op.start for op in ops if op.engine == eng)
        pct = busy / H if H else 0.0
        print(f"{ENG_NAME[eng]}: busy={busy:.2f}/{H:.2f} ns  utilization={pct:.2%}")
    print()

    print("=== BANDWIDTH UTILIZATION ===")
    for op in ops:
        if op.regime == 'sync':
            print(f"op{op.idx} ({ENG_NAME[op.engine]}): sync — no data transfer")
        else:
            peak = PEAK_GB_S.get(op.engine)
            ps = f"{peak:.0f}" if peak else "未校准"
            print(f"op{op.idx} ({ENG_NAME[op.engine]}): effective={op.effective_bw:.4g} GB/s  "
                  f"peak={ps} GB/s  util={op.bw_util:.2%}  regime={op.regime}")
    print()

    print("=== PARALLELISM ===")
    print(f"parallel_pairs: {len(pairs)}")
    for i, j in pairs[:20]:
        s, e = max(ops[i].start, ops[j].start), min(ops[i].end, ops[j].end)
        print(f"  op{i} || op{j}: overlap_ns={e - s:.2f}")
    print()

    # critical path (topo DP over hazard edges + same-engine serialization)
    preds = {i: set(deps[i]) for i in range(len(ops))}
    last = {}
    for op in ops:
        if op.engine in last:
            preds[op.idx].add(last[op.engine])
        last[op.engine] = op.idx
    dist = [0.0] * len(ops); back = [None] * len(ops)
    for i in range(len(ops)):
        best, b = 0.0, None
        for p in preds[i]:
            if dist[p] > best:
                best, b = dist[p], p
        dist[i], back[i] = best + ops[i].duration, b
    sink = max(range(len(ops)), key=lambda i: dist[i])
    path = []
    cur = sink
    while cur is not None:
        path.append(cur); cur = back[cur]
    path.reverse()
    print("=== CRITICAL PATH ===")
    print(f"length_ns: {dist[sink]:.2f}  (= {dist[sink] / H:.0%} of scheduled makespan)")
    print(f"path: {' -> '.join(f'op{i}' for i in path)}")
    for k in range(1, len(path)):
        print(f"  op{path[k-1]} -> op{path[k]}: {hazard_detail(ops[path[k-1]], ops[path[k]])}")
    print()


# ── NX 模式 (无 networkx, 纯 dict) ────────────────────────────────────────────
def render_nx(ops, deps):
    H = max((op.end for op in ops), default=0.0)
    nodes = []
    for op in ops:
        nodes.append({
            "id": f"op{op.idx}", "idx": op.idx, "instruction": op.sig(),
            "op_name": op.name, "dst": op.dst, "src": op.src,
            "engine": op.engine, "engine_name": ENG_NAME[op.engine],
            "size_kb": op.size_kb, "duration_ns": round(op.end - op.start, 4),
            "start_ns": round(op.start, 4), "end_ns": round(op.end, 4),
            "cycles": op.cycles, "pipe": op.pipe,
            "effective_bw_gb_s": round(op.effective_bw, 4),
            "bw_utilization": round(op.bw_util, 4), "regime": op.regime,
        })
    links = []
    for i, op in enumerate(ops):
        for j in sorted(deps[i]):
            links.append({
                "source": f"op{j}", "target": f"op{i}",
                "reason": hazard_detail(ops[j], op),
                "delay_ns": round(op.start - ops[j].end, 4),
            })
    eng_util = {}
    for eng in range(N_ENG):
        busy = sum(op.end - op.start for op in ops if op.engine == eng)
        eng_util[ENG_NAME[eng]] = round(busy / H, 4) if H else 0.0
    bottleneck = max(ops, key=lambda o: o.duration)
    print(json.dumps({
        "directed": True,
        "graph": {
            "total_ns": round(H, 4), "num_ops": len(ops),
            "engine_utilization": eng_util,
            "bottleneck": {"op": f"op{bottleneck.idx}",
                           "instruction": bottleneck.sig(),
                           "duration_ns": round(bottleneck.duration, 4)},
        },
        "nodes": nodes, "links": links,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    llm = '--llm' in args
    nx = '--nx' in args
    path = next((a for a in args if not a.startswith('--')), None)
    if not path:
        path = Path(__file__).resolve().parent / "e2e_run/05_merged/merged.json"
    p = Path(path)
    if not p.exists():
        sys.exit(f"❌ 找不到 {p}\n  先跑: bash analyzers/run_server_flow.sh 生成 merged.json")
    merged = json.loads(p.read_text(encoding="utf-8"))
    ops, deps, summary = build_ops(merged)
    if not ops:
        sys.exit("❌ merged.json 无 per_op_statistics")
    schedule(ops, deps)
    if nx:
        render_nx(ops, deps)
    elif llm:
        render_llm(ops, deps, summary)
    else:
        render(ops, deps)
        print(f"  注: 调度 makespan={max(o.end for o in ops):.2f} ns 为真实 per-op 时长之和的 ASAP 排布;")
        print(f"      真机端到端 total_ns={summary.get('total_ns')} ns (board.json, 含真实流水重叠)")
