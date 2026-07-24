#!/usr/bin/env python3
"""
msprof op simulator 分析器 —— 唯一工作模式: 运行 msprof 采集 + 解析 trace.json。
输出到 outputs/<kernel>/roundN/msprof/pipeline_report.json

═══════════════════════════════════════════════════════════════════════════════
  完整流程 (在 910B3 服务器上运行)
═══════════════════════════════════════════════════════════════════════════════

  1. 编译: Ascend 编译器编译 Triton kernel → binary (--run-mode=sim -g -DASCENDC_TRACE_ON)
  2. 采集: msprof op simulator ./binary
  3. 产出: OPPROF_{timestamp}_XXX/simulator/trace.json (Chrome Trace Event Format)
  4. 解析: 本脚本解析 trace.json, 提取流水线通道 (VECTOR/MTE2/MTE3/Cube/...)
  5. 保存:
     outputs/<kernel>/round0/msprof/
       ├── OPPROF_{timestamp}_XXX/          # msprof 原始中间产物
       └── pipeline_report.json             # ★ 解析最终产物 (29字段, 16[OK]+13❌)

  通道 → 7-engine 映射:
    MTE2   → GM→UB (Vector入口) 或 GM→L1 (Matrix入口, 自动修正)
    MTE3   → UB→GM
    MTE1   → L1→L0
    VECTOR → VecUnit
    Cube   → CubeUnit
    FIXP   → L0→GM

═══════════════════════════════════════════════════════════════════════════════
  pipeline_report.json 字段 (29字段, 对齐 HIVMIR report 结构)
═══════════════════════════════════════════════════════════════════════════════

  [OK] msprof 直接提供 (16):
    op_id, op_type, engine, pipeline_channel, core_id, trace_event_name,
    duration_ns, start_ns, end_ns, time_ratio,
    total_ns, num_ops, execution_mode, num_cores,
    engine_utilization, parallel_pairs, critical_path(basic)

  ❌ HIVMIR 补充 (13):
    instruction, dst, src, src2, size_kb, variable_name, memory_region,
    effective_bw_gb_s, peak_bw_gb_s, bw_utilization, regime,
    wait_before_start_ns, blocked_by
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union


# ═══════════════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════════════

TBD = "待补充"

PIPELINE_MAP: Dict[str, Tuple[str, str, bool]] = {
    "VECTOR":   ("VecUnit",   "vadd",        True),
    "Cube":     ("CubeUnit",  "matrixmul",   True),
    "CUBE":     ("CubeUnit",  "matrixmul",   True),
    "MTE1":     ("L1→L0",    "l1_to_l0",    True),
    "MTE2":     ("GM→UB",    "gm_to_ub",    True),
    "MTE3":     ("UB→GM",    "ub_to_gm",    True),
    "FIXP":     ("L0→GM",    "l0_to_gm",    True),
    "SCALAR":   ("Scalar",   "scalar",       False),
    "FLOWCTRL": ("FlowCtrl", "sync",         False),
    "CACHEMISS":("CacheMiss","cache_miss",   False),
}

# pipeline_report.json 中 msprof 提供的字段
MSPROF_FIELDS = [
    "op_id", "op_type", "engine", "pipeline_channel", "core_id",
    "trace_event_name", "duration_ns", "start_ns", "end_ns", "time_ratio",
    "total_ns", "num_ops", "execution_mode", "num_cores",
    "engine_utilization", "parallel_pairs", "critical_path",
]

# pipeline_report.json 中 HIVMIR 待补充的字段
HIVMIR_NEEDED_FIELDS = [
    "instruction", "dst", "src", "src2",
    "size_kb", "variable_name", "memory_region",
    "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization", "regime",
    "wait_before_start_ns", "blocked_by",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineOp:
    """流水线中的一个操作 (op)。29 字段对齐 HIVMIR report。"""
    # ── msprof [OK] ──
    op_id: int
    op_type: str
    engine: str = ""
    pipeline_channel: str = ""
    core_id: str = ""
    trace_event_name: str = ""
    duration_ns: float = 0.0
    start_ns: float = 0.0
    end_ns: float = 0.0
    time_ratio: float = 0.0
    # ── HIVMIR ❌ ──
    instruction: str = TBD
    dst: str = TBD
    src: str = TBD
    src2: str = ""
    size_kb: str = TBD
    variable_name: str = TBD
    memory_region: str = TBD
    effective_bw_gb_s: str = TBD
    peak_bw_gb_s: str = TBD
    bw_utilization: str = TBD
    regime: str = TBD
    wait_before_start_ns: str = TBD
    blocked_by: List[str] = field(default_factory=list)


@dataclass
class PipelineReport:
    """msprof 完整流水线报告。29 字段全结构, msprof 填 16, 其余标 TBD。"""
    # SECTION 1
    total_ns: float = 0.0
    num_ops: int = 0
    execution_mode: str = "unknown"
    num_cores: int = 0
    # SECTION 2
    time_breakdown: List[dict] = field(default_factory=list)
    # SECTION 3
    ops: List[PipelineOp] = field(default_factory=list)
    # SECTION 4
    engine_utilization: Dict[str, float] = field(default_factory=dict)
    # SECTION 5
    bandwidth_utilization: str = TBD
    # SECTION 6
    parallel_pairs: List[dict] = field(default_factory=list)
    # SECTION 7
    critical_path: List[int] = field(default_factory=list)
    critical_path_length_ns: float = 0.0
    critical_path_fraction: str = TBD
    critical_path_edges: List[dict] = field(default_factory=list)
    # 元数据
    source: str = "msprof_op_simulator"
    trace_json_path: str = ""
    profiler_output_dir: str = ""
    generated_at: str = ""
    msprof_fields: List[str] = field(default_factory=lambda: list(MSPROF_FIELDS))
    hivmir_fields: List[str] = field(default_factory=lambda: list(HIVMIR_NEEDED_FIELDS))


# ═══════════════════════════════════════════════════════════════════════════════
#  trace.json 解析器
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TraceEvent:
    ph: str; cat: str; name: str
    pid: int; tid: int
    ts_us: float; dur_us: float = 0.0
    eid: int = 0; args: dict = field(default_factory=dict)


class TraceJsonParser:
    """解析 msprof op simulator 产出的 trace.json (Chrome Trace Event Format)。"""

    def __init__(self):
        self._process_names: Dict[int, str] = {}
        self._thread_names: Dict[Tuple[int, int], str] = {}

    def parse(self, trace_json_path: Path) -> PipelineReport:
        with open(trace_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        events_list = raw if isinstance(raw, list) else raw.get("traceEvents", raw.get("events", []))

        events = self._parse_events(events_list)
        hw_events = [e for e in events if self._layer_of(e) == "AscendHardware"]

        ops = self._extract_pipeline_ops(hw_events)
        ops = self._resolve_mte2(ops)
        ops.sort(key=lambda o: o.start_ns)
        for i, op in enumerate(ops):
            op.op_id = i

        total_ns = max((op.end_ns for op in ops), default=0.0)
        for op in ops:
            op.time_ratio = op.duration_ns / total_ns if total_ns > 0 else 0.0

        ops = self._analyze_dependencies(ops, events)
        engine_util = self._compute_engine_utilization(ops, total_ns)
        parallel_pairs = self._detect_parallelism(ops)
        cp_ops, cp_len = self._compute_critical_path(ops)
        core_ids = set(op.core_id for op in ops if op.core_id)

        return PipelineReport(
            total_ns=total_ns,
            num_ops=len(ops),
            execution_mode="parallel" if parallel_pairs else "sequential",
            num_cores=len(core_ids),
            ops=ops,
            time_breakdown=[
                {"op_id": op.op_id, "op_type": op.op_type, "engine": op.engine,
                 "duration_ns": round(op.duration_ns, 2),
                 "time_ratio": round(op.time_ratio, 4)}
                for op in sorted(ops, key=lambda o: o.time_ratio, reverse=True)
            ],
            engine_utilization=engine_util,
            parallel_pairs=parallel_pairs,
            critical_path=cp_ops,
            critical_path_length_ns=cp_len,
            critical_path_fraction=f"{cp_len/total_ns:.0%}" if total_ns > 0 else TBD,
            trace_json_path=str(trace_json_path),
            generated_at=datetime.now().isoformat(),
        )

    # ── event 解析 ──

    def _parse_events(self, events_list: List[dict]) -> List[_TraceEvent]:
        parsed: List[_TraceEvent] = []
        for ev in events_list:
            ph = ev.get("ph", "")
            if ph == "M":
                if ev.get("name") == "process_name":
                    self._process_names[int(ev.get("pid", 0))] = ev.get("args", {}).get("name", "")
                elif ev.get("name") == "thread_name":
                    self._thread_names[(int(ev.get("pid", 0)), int(ev.get("tid", 0)))] = ev.get("args", {}).get("name", "")
                continue
            ts = float(str(ev.get("ts", 0)))
            dur = float(ev.get("dur", 0)) if "dur" in ev else 0.0
            eid = int(ev.get("id", 0)) if "id" in ev else 0
            pid = int(ev.get("pid", 0)) if isinstance(ev.get("pid"), (int, str)) else 0
            tid = int(ev.get("tid", 0)) if isinstance(ev.get("tid"), (int, str)) else 0
            parsed.append(_TraceEvent(ph=ph, cat=str(ev.get("cat", "")), name=str(ev.get("name", "")),
                                       pid=pid, tid=tid, ts_us=ts, dur_us=dur, eid=eid, args=ev.get("args", {})))
        return parsed

    def _layer_of(self, ev: _TraceEvent) -> str:
        pname = self._process_names.get(ev.pid, "")
        if "Python" in pname: return "Python"
        if "CANN" in pname: return "CANN"
        if "Ascend" in pname or "Hardware" in pname: return "AscendHardware"
        return "AscendHardware" if ev.pid > 700000000 else "CANN"

    # ── 流水线提取 ──

    def _extract_pipeline_ops(self, hw_events: List[_TraceEvent]) -> List[PipelineOp]:
        complete = sorted([e for e in hw_events if e.ph == "X" and e.dur_us > 0], key=lambda e: e.ts_us)
        ops: List[PipelineOp] = []
        for op_id, ev in enumerate(complete):
            ch = self._identify_channel(ev)
            if ch is None: continue
            info = PIPELINE_MAP.get(ch)
            if info is None or not info[2]: continue
            eng, otype, _ = info
            core = self._thread_names.get((ev.pid, ev.tid), f"tid_{ev.tid}")
            ops.append(PipelineOp(
                op_id=op_id, op_type=otype, engine=eng,
                pipeline_channel=ch, core_id=core, trace_event_name=ev.name,
                start_ns=ev.ts_us * 1000.0, end_ns=(ev.ts_us + ev.dur_us) * 1000.0,
                duration_ns=ev.dur_us * 1000.0,
            ))
        return ops

    def _identify_channel(self, ev: _TraceEvent) -> Optional[str]:
        cat_u = ev.cat.upper()
        for ch in PIPELINE_MAP:
            if ch.upper() == cat_u: return ch
        name_u = ev.name.upper()
        for ch in PIPELINE_MAP:
            if ch.upper() in name_u or name_u in ch.upper(): return ch
        if isinstance(ev.args, dict):
            pipe = str(ev.args.get("pipeline", ev.args.get("channel", ""))).upper()
            for ch in PIPELINE_MAP:
                if ch.upper() in pipe: return ch
        return None

    # ── MTE2 修正 ──

    def _resolve_mte2(self, ops: List[PipelineOp]) -> List[PipelineOp]:
        for i, op in enumerate(ops):
            if op.pipeline_channel != "MTE2": continue
            lookahead = [o for o in ops[i+1:i+6] if o.core_id == op.core_id]
            ch_ahead = [o.pipeline_channel for o in lookahead]
            if any(c in ("Cube", "CUBE") for c in ch_ahead) and "MTE1" in ch_ahead:
                op.engine = "GM→L1"; op.op_type = "gm_to_l1"
        return ops

    # ── 依赖 ──

    def _analyze_dependencies(self, ops: List[PipelineOp], events: List[_TraceEvent]) -> List[PipelineOp]:
        flow_map: Dict[int, Tuple[float, float]] = {}
        for ev in events:
            if ev.ph == "s" and ev.eid > 0:
                flow_map.setdefault(ev.eid, [0.0, 0.0])
                flow_map[ev.eid] = (ev.ts_us, flow_map[ev.eid][1])
            elif ev.ph == "f" and ev.eid > 0:
                flow_map.setdefault(ev.eid, [0.0, 0.0])
                flow_map[ev.eid] = (flow_map[ev.eid][0], ev.ts_us)
        for fid, (src_ts, dst_ts) in flow_map.items():
            if src_ts == 0 or dst_ts == 0: continue
            src_op = self._find_nearest(ops, src_ts * 1000.0)
            dst_op = self._find_nearest(ops, dst_ts * 1000.0)
            if src_op and dst_op and src_op.op_id != dst_op.op_id:
                entry = f"op{src_op.op_id}(flow_{fid}_待HIVMIR确认类型)"
                if entry not in dst_op.blocked_by:
                    dst_op.blocked_by.append(entry)
        for i in range(1, len(ops)):
            for j in range(i - 1, -1, -1):
                if ops[i].engine == ops[j].engine:
                    gap = ops[i].start_ns - ops[j].end_ns
                    if 0 <= gap < ops[i].duration_ns * 5:
                        entry = f"op{ops[j].op_id}(engine_serial)"
                        if entry not in ops[i].blocked_by:
                            ops[i].blocked_by.append(entry)
                    break
        return ops

    def _find_nearest(self, ops: List[PipelineOp], ts_ns: float) -> Optional[PipelineOp]:
        if not ops: return None
        closest = min(ops, key=lambda o: abs(o.start_ns - ts_ns))
        return closest if abs(closest.start_ns - ts_ns) < closest.duration_ns * 10 else None

    # ── 引擎利用率 ──

    def _compute_engine_utilization(self, ops: List[PipelineOp], total_ns: float) -> Dict[str, float]:
        if total_ns <= 0: return {}
        u: Dict[str, float] = {}
        for op in ops:
            u[op.engine] = u.get(op.engine, 0.0) + op.duration_ns
        return {e: round(v / total_ns, 4) for e, v in u.items()}

    # ── 并行 ──

    def _detect_parallelism(self, ops: List[PipelineOp]) -> List[dict]:
        pairs: List[dict] = []
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                a, b = ops[i], ops[j]
                if a.start_ns < b.end_ns and b.start_ns < a.end_ns:
                    ov = min(a.end_ns, b.end_ns) - max(a.start_ns, b.start_ns)
                    if ov > 0: pairs.append({"op_a": a.op_id, "op_b": b.op_id, "overlap_ns": round(ov, 2)})
        return pairs

    # ── 关键路径 ──

    def _compute_critical_path(self, ops: List[PipelineOp]) -> Tuple[List[int], float]:
        if not ops: return [], 0.0
        s = sorted(ops, key=lambda o: o.end_ns)
        path, cur, length = [], 0.0, 0.0
        for op in s:
            if op.start_ns >= cur:
                path.append(op.op_id); cur = op.end_ns; length += op.duration_ns
        return path, length


# ═══════════════════════════════════════════════════════════════════════════════
#  主类: MsprofAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class MsprofAnalyzer:
    """msprof op simulator 分析器。

    唯一工作模式: 运行 msprof → 解析 trace.json → 写入 outputs/<kernel>/roundN/msprof/

    Usage:
        analyzer = MsprofAnalyzer()

        # round0 (基准分析)
        report = analyzer.analyze(binary_path=Path("./binary"), kernel_name="vector_add_fp16_N65536")

        # roundN (在某个 Tier 下)
        report = analyzer.analyze(binary_path=Path("./binary"), kernel_name="vector_add_fp16_N65536",
                                   tier="01_block_size_launch", round_number=3)

        # 直接解析已有 trace.json
        report = analyzer.analyze_trace(Path("OPPROF_xxx/simulator/trace.json"),
                                         kernel_name="vector_add_fp16_N65536", round_number=0)
    """

    def __init__(self, msprof_bin: Optional[str] = None, timeout_seconds: int = 120):
        self.msprof_bin = msprof_bin or shutil.which("msprof")
        self.timeout = timeout_seconds
        self._parser = TraceJsonParser()

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口: 完整采集 + 解析
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze(
        self,
        binary_path: Path,
        kernel_name: str,
        tier: str = "",
        round_number: int = 0,
    ) -> PipelineReport:
        """运行 msprof op simulator → 解析 → 保存到 outputs/<kernel>/roundN/msprof/。

        Args:
            binary_path: 已编译的算子二进制文件
            kernel_name: Triton kernel 名 (用于 outputs/ 目录)
            tier: Tier 文件夹名 (如 "01_block_size_launch"), round0 时为空
            round_number: 轮次号 (0 = 基准分析)
        """
        if not self.msprof_bin:
            raise RuntimeError(
                "msprof not found. Run on Ascend 910B3 server with CANN installed.\n"
                "Or use analyze_trace() with a pre-generated trace.json."
            )
        binary_path = Path(binary_path)
        if not binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        # 1. 运行 msprof
        print(f"[msprof] Running: msprof op simulator {binary_path}")
        t0 = time.time()
        result = subprocess.run(
            [self.msprof_bin, "op", "simulator", str(binary_path)],
            capture_output=True, timeout=self.timeout,
            encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"msprof failed (code {result.returncode}):\n{result.stderr[:1000]}")
        print(f"[msprof] Done in {time.time() - t0:.1f}s")

        # 2. 找 OPPROF 目录
        opprof_dirs = sorted(Path.cwd().glob("OPPROF_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not opprof_dirs:
            raise RuntimeError("No OPPROF_* directory created")
        prof_dir = opprof_dirs[0]
        print(f"[msprof] Found: {prof_dir}")

        # 3. 找 trace.json
        trace_paths = list(prof_dir.glob("simulator/trace.json"))
        if not trace_paths:
            trace_paths = list(prof_dir.glob("simulator/core*/trace.json"))
        if not trace_paths:
            raise FileNotFoundError(f"No trace.json in {prof_dir}/simulator/")

        # 4. 解析
        if len(trace_paths) == 1:
            report = self._parser.parse(trace_paths[0])
        else:
            reports = [self._parser.parse(p) for p in trace_paths]
            report = self._merge_reports(reports)
        report.profiler_output_dir = str(prof_dir)

        # 5. 保存到 outputs/
        output_path = self._save(report, prof_dir, kernel_name, tier, round_number)
        print(f"[msprof] Report saved → {output_path}")
        print(f"[msprof] {report.num_ops} ops, {report.total_ns:.1f}ns, "
              f"{'parallel' if report.parallel_pairs else 'sequential'}")
        return report

    def analyze_trace(
        self,
        trace_json_path: Path,
        kernel_name: str,
        tier: str = "",
        round_number: int = 0,
    ) -> PipelineReport:
        """直接解析已有 trace.json (不需要 msprof 命令)。"""
        trace_json_path = Path(trace_json_path)
        if not trace_json_path.exists():
            raise FileNotFoundError(f"trace.json not found: {trace_json_path}")

        report = self._parser.parse(trace_json_path)

        # 找 OPPROF 目录 (trace.json 的上上级)
        prof_dir = trace_json_path.parent.parent
        report.profiler_output_dir = str(prof_dir)

        output_path = self._save(report, prof_dir, kernel_name, tier, round_number)
        print(f"[msprof] Report saved → {output_path}")
        return report

    # ═══════════════════════════════════════════════════════════════════════════
    #  内部: 保存
    # ═══════════════════════════════════════════════════════════════════════════

    def _save(
        self, report: PipelineReport, prof_dir: Path,
        kernel_name: str, tier: str, round_number: int,
    ) -> Path:
        """保存 pipeline_report.json + 拷贝 OPPROF 中间产物。"""
        root = _find_outputs_root()
        if tier:
            round_dir = root / kernel_name / tier / f"round{round_number}"
        else:
            round_dir = root / kernel_name / f"round{round_number}"
        msprof_dir = round_dir / "msprof"
        msprof_dir.mkdir(parents=True, exist_ok=True)

        # pipeline_report.json
        json_path = msprof_dir / "pipeline_report.json"
        json_path.write_text(json.dumps(self._report_to_dict(report), indent=2, ensure_ascii=False), encoding="utf-8")

        # 拷贝 OPPROF 中间产物
        dest_opprof = msprof_dir / prof_dir.name
        if not dest_opprof.exists():
            shutil.copytree(prof_dir, dest_opprof, dirs_exist_ok=True)

        return json_path

    def _report_to_dict(self, report: PipelineReport) -> dict:
        return {
            "meta": {
                "source": report.source,
                "generated_at": report.generated_at,
                "trace_json_path": report.trace_json_path,
                "profiler_output_dir": report.profiler_output_dir,
                "msprof_fields_provided": report.msprof_fields,
                "hivmir_fields_pending": report.hivmir_fields,
                "note": (
                    "msprof 提供 timing/engine/channel (16字段) [OK], "
                    "HIVMIR 补充 buffer名/size/依赖 (13字段) ❌"
                ),
            },
            "execution_summary": {
                "total_ns": report.total_ns,
                "num_ops": report.num_ops,
                "execution_mode": report.execution_mode,
                "num_cores": report.num_cores,
            },
            "time_breakdown": report.time_breakdown,
            "per_op_statistics": [
                {
                    "op_id": op.op_id,
                    "op_type": op.op_type,
                    "engine": op.engine,
                    "instruction": op.instruction,
                    "dst": op.dst, "src": op.src, "src2": op.src2,
                    "size_kb": op.size_kb,
                    "variable_name": op.variable_name,
                    "memory_region": op.memory_region,
                    "duration_ns": round(op.duration_ns, 2),
                    "start_ns": round(op.start_ns, 2),
                    "end_ns": round(op.end_ns, 2),
                    "time_ratio": round(op.time_ratio, 4),
                    "effective_bw_gb_s": op.effective_bw_gb_s,
                    "peak_bw_gb_s": op.peak_bw_gb_s,
                    "bw_utilization": op.bw_utilization,
                    "regime": op.regime,
                    "wait_before_start_ns": op.wait_before_start_ns,
                    "blocked_by": op.blocked_by,
                    "pipeline_channel": op.pipeline_channel,
                    "core_id": op.core_id,
                    "trace_event_name": op.trace_event_name,
                }
                for op in report.ops
            ],
            "engine_utilization": report.engine_utilization,
            "bandwidth_utilization": report.bandwidth_utilization,
            "parallelism": {
                "parallel_pairs": report.parallel_pairs,
                "total_pairs": len(report.parallel_pairs),
            },
            "critical_path": {
                "path": report.critical_path,
                "length_ns": report.critical_path_length_ns,
                "fraction": report.critical_path_fraction,
                "edges": report.critical_path_edges,
            },
        }

    def _merge_reports(self, reports: List[PipelineReport]) -> PipelineReport:
        if len(reports) == 1: return reports[0]
        all_ops = [op for r in reports for op in r.ops]
        all_ops.sort(key=lambda o: o.start_ns)
        for i, op in enumerate(all_ops):
            op.op_id = i
        total_ns = max((op.end_ns for op in all_ops), default=0.0)
        for op in all_ops:
            op.time_ratio = op.duration_ns / total_ns if total_ns > 0 else 0.0
        return PipelineReport(
            total_ns=total_ns, num_ops=len(all_ops),
            execution_mode="parallel", num_cores=len(reports),
            ops=all_ops,
            time_breakdown=[
                {"op_id": op.op_id, "op_type": op.op_type, "engine": op.engine,
                 "duration_ns": round(op.duration_ns, 2), "time_ratio": round(op.time_ratio, 4)}
                for op in sorted(all_ops, key=lambda o: o.time_ratio, reverse=True)
            ],
            generated_at=datetime.now().isoformat(),
        )


def _find_outputs_root() -> Path:
    """找到 outputs/ 根目录。"""
    # 从当前文件向上找 triton_agent_optimizer/outputs/
    return Path(__file__).resolve().parent.parent / "outputs"


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    import tempfile
    print("=" * 60)
    print("MsprofAnalyzer Self-Test")
    print("=" * 60)

    mock_trace = [
        {"ph": "M", "name": "process_name", "pid": 1, "tid": 0, "cat": "__metadata",
         "args": {"name": "Ascend Hardware"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 100,
         "cat": "__metadata", "args": {"name": "core0.veccore0"}},
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 200,
         "cat": "__metadata", "args": {"name": "core0.veccore1"}},
        # Vector pipeline (tid=100)
        {"ph": "X", "cat": "MTE2", "name": "gm_to_ub", "pid": 1, "tid": 100,
         "ts": 0.0, "dur": 1.622, "args": {}},
        {"ph": "X", "cat": "VECTOR", "name": "vadd", "pid": 1, "tid": 100,
         "ts": 1.622, "dur": 0.324, "args": {}},
        {"ph": "X", "cat": "MTE3", "name": "ub_to_gm", "pid": 1, "tid": 100,
         "ts": 1.946, "dur": 1.710, "args": {}},
        # Matrix pipeline (tid=200) — MTE2 should resolve to GM→L1
        {"ph": "X", "cat": "MTE2", "name": "gm_to_l1", "pid": 1, "tid": 200,
         "ts": 5.0, "dur": 7.172, "args": {}},
        {"ph": "X", "cat": "MTE1", "name": "l1_to_l0", "pid": 1, "tid": 200,
         "ts": 12.172, "dur": 1.379, "args": {}},
        {"ph": "X", "cat": "Cube", "name": "matrixmul", "pid": 1, "tid": 200,
         "ts": 13.551, "dur": 1.748, "args": {}},
        {"ph": "X", "cat": "FIXP", "name": "l0_to_gm", "pid": 1, "tid": 200,
         "ts": 15.299, "dur": 7.172, "args": {}},
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(mock_trace, f, indent=2)
        tmp_path = Path(f.name)

    try:
        parser = TraceJsonParser()
        report = parser.parse(tmp_path)

        print(f"\nEXECUTION SUMMARY:")
        print(f"  total_ns: {report.total_ns:.1f}  ops: {report.num_ops}  mode: {report.execution_mode}  cores: {report.num_cores}")

        print(f"\nOPS:")
        for op in report.ops:
            filled = sum(1 for v in [op.engine, op.duration_ns] if v)
            pending = sum(1 for v in [op.instruction, op.dst, op.size_kb] if v == TBD)
            msprof_n = 2
            hivmir_n = sum(1 for v in [op.instruction, op.dst, op.src, op.src2, op.size_kb, op.variable_name, op.memory_region, op.effective_bw_gb_s, op.peak_bw_gb_s, op.bw_utilization, op.regime, op.wait_before_start_ns] if v == TBD) + (1 if op.blocked_by == [] or op.blocked_by == [TBD] else 0)
            print(f"  op{op.op_id}: {op.op_type:12s} engine={op.engine:8s} ch={op.pipeline_channel:8s} "
                  f"dur={op.duration_ns:.0f}ns [{op.start_ns:.0f}..{op.end_ns:.0f}] "
                  f"ratio={op.time_ratio:.2%} "
                  f"(msprof OK={msprof_n}, HIVMIR TBD={hivmir_n})")

        gm_ub = [o for o in report.ops if o.op_type == "gm_to_ub"]
        gm_l1 = [o for o in report.ops if o.op_type == "gm_to_l1"]
        print(f"\nMTE2 resolution: gm_to_ub={len(gm_ub)} gm_to_l1={len(gm_l1)}")
        assert len(gm_ub) >= 1, "Vector pipeline: MTE2→GM→UB"
        assert len(gm_l1) >= 1, "Matrix pipeline: MTE2→GM→L1"

        print(f"\nENGINE UTILIZATION: {json.dumps(report.engine_utilization, indent=2)}")
        print(f"PARALLEL PAIRS: {len(report.parallel_pairs)}")
        print(f"CRITICAL PATH: {report.critical_path} ({report.critical_path_length_ns:.1f}ns)")

        # 验证 JSON 输出 (不写 outputs)
        analyzer = MsprofAnalyzer(msprof_bin="mock")
        d = analyzer._report_to_dict(report)
        assert "meta" in d and "per_op_statistics" in d
        assert d["meta"]["msprof_fields_provided"] == list(MSPROF_FIELDS)
        assert d["meta"]["hivmir_fields_pending"] == list(HIVMIR_NEEDED_FIELDS)
        assert len(d["per_op_statistics"]) == report.num_ops
        for opd in d["per_op_statistics"]:
            assert opd["instruction"] == TBD
            assert opd["size_kb"] == TBD
            assert isinstance(opd["duration_ns"], (int, float)) and opd["duration_ns"] > 0

        print(f"\nJSON format: {len(json.dumps(d))} chars, all assertions passed [OK]")

    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
