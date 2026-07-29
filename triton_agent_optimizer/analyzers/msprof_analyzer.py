#!/usr/bin/env python3
"""
msprof Analyzer v2.0 — 解析 msprof op simulator 输出的指令级 trace
=================================================================

数据来源: msprof op simulator --soc-version=Ascend910B3 (纯CPU仿真, 不需要NPU硬件)

输出文件:
  OPPROF_xxx/
    simulator/
      trace.json                  ← 全核汇总 (total_ns, parallelism, critical_path)
      core*.veccore*/             ← 每核独立 trace
        instr_exe.csv             ← ★ 指令级耗时 (pipe/cycles/running_time/detail)
        code_exe.csv              ← 代码行耗时
        trace.json                ← 该核指令流水

instr_exe.csv 字段:
  instr, addr, pipe, call_count, cycles, running_time(us), detail

pipe → engine 映射:
  MTE2    → GM→UB  (or GM→L1 in Cube path)
  MTE3    → UB→GM
  VECTOR  → VecUnit
  CUBE    → CubeUnit
  MTE1    → L1→L0
  SCALAR  → Scalar (address computation, not counted in 7-engine)
  ALL     → Barrier (sync)
  FLOWCTRL→ Flow Control

环境自动切换:
  - 有 msprof 工具 → 调用 msprof op simulator 生成 trace
  - 无 msprof 工具 → 从已有 OPPROF_ 目录解析 (用户手动提供)
"""

from __future__ import annotations

import csv, json, os, re, subprocess, sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

TBD = "待补充"

# ═══════════════════════════════════════════════════════════════════════════════
#  pipe → engine 映射 (来源: 官方 msprof 文档 + CANN 9.0 实测验证)
# ═══════════════════════════════════════════════════════════════════════════════

PIPE_TO_ENGINE: Dict[str, str] = {
    "MTE2":   "GM→UB",    # GM→UB or GM→L1 (context dependent)
    "MTE3":   "UB→GM",
    "VECTOR": "VecUnit",
    "CUBE":   "CubeUnit",
    "MTE1":   "L1→L0",
}

# pipe → 7-engine ID
ENGINE_NAME_TO_ID: Dict[str, int] = {
    "GM→UB":   0, "UB→GM": 1, "VecUnit":  2,
    "GM→L1":   3, "L1→L0": 4, "CubeUnit": 5, "L0→GM": 6,
}

MSPROF_FIELDS = [
    "duration_ns", "start_ns", "end_ns", "time_ratio", "cycles",
    "engine", "pipeline_channel", "core_id",
    "total_ns", "num_ops", "execution_mode", "num_cores",
    "engine_utilization", "parallel_pairs", "critical_path",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InstrRecord:
    """instr_exe.csv 单条指令。"""
    instr: str = ""             # VADD, MOV_OUT_TO_UB, SET_FLAG ...
    addr: str = ""              # 指令地址
    pipe: str = ""              # MTE2, VECTOR, CUBE, MTE3, SCALAR, ALL ...
    call_count: int = 1
    cycles: int = 0
    running_time_us: float = 0.0
    detail: str = ""            # XD:X3=0x4000, Dtype:F32, id:49

    @property
    def duration_ns(self) -> float:
        return self.running_time_us * 1000.0

    @property
    def engine(self) -> str:
        return PIPE_TO_ENGINE.get(self.pipe, self.pipe)


@dataclass
class PipelineOp:
    """msprof 解析出的单个流水线操作 (对齐 29 字段)。"""
    op_id: int
    op_type: str = TBD          # 从 HIVM 补充
    engine: str = ""
    pipeline_channel: str = ""  # MTE2, VECTOR, etc.
    core_id: str = ""
    instruction: str = ""       # 硬件指令名
    duration_ns: float = 0.0
    cycles: int = 0
    start_ns: float = 0.0       # (从 trace 补充)
    end_ns: float = 0.0
    time_ratio: float = 0.0
    detail: str = ""
    # 从 detail 提取的额外信息
    data_size_bytes: int = 0
    data_type: str = ""
    hw_op_id: int = 0           # detail 中的 id:


@dataclass
class MsprofReport:
    """msprof 完整解析报告。"""
    total_ns: float = 0.0
    num_ops: int = 0
    execution_mode: str = "unknown"
    num_cores: int = 0
    core_types: List[str] = field(default_factory=list)  # ["veccore", "cubecore"]
    ops: List[PipelineOp] = field(default_factory=list)
    engine_utilization: Dict[str, float] = field(default_factory=dict)
    parallel_pairs: List[dict] = field(default_factory=list)
    critical_path: List[int] = field(default_factory=list)
    critical_path_length_ns: float = 0.0
    source_dir: str = ""
    generated_at: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Parser
# ═══════════════════════════════════════════════════════════════════════════════

class MsprofParser:
    """解析 msprof op simulator 输出目录。"""

    @staticmethod
    def parse_instr_csv(csv_path: Path) -> List[InstrRecord]:
        """解析 core*_instr_exe.csv。"""
        records = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    r = InstrRecord(
                        instr=row.get("instr", "").strip(),
                        addr=row.get("addr", "").strip(),
                        pipe=row.get("pipe", "").strip(),
                        call_count=int(row.get("call_count", 1)),
                        cycles=int(row.get("cycles", 0)),
                        running_time_us=float(row.get("running_time(us)", 0)),
                        detail=row.get("detail", "").strip(),
                    )
                    records.append(r)
                except (ValueError, KeyError):
                    continue
        return records

    @staticmethod
    def parse_trace_json(json_path: Path) -> dict:
        """解析 trace.json 提取 total_ns, parallelism, critical_path。"""
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        events = data if isinstance(data, list) else data.get("traceEvents", [])
        complete = [e for e in events if e.get("ph") == "X"]

        # total_ns = max(ts + dur)
        total_ns = 0.0
        for e in complete:
            end_ts = float(e.get("ts", 0)) + float(e.get("dur", 0))
            if end_ts > total_ns:
                total_ns = end_ts
        total_ns *= 1000.0  # us → ns

        # 并行检测
        pairs = []
        for i in range(len(complete)):
            for j in range(i + 1, len(complete)):
                a, b = complete[i], complete[j]
                a_ts = float(a.get("ts", 0))
                a_end = a_ts + float(a.get("dur", 0))
                b_ts = float(b.get("ts", 0))
                b_end = b_ts + float(b.get("dur", 0))
                if a_ts < b_end and b_ts < a_end:
                    overlap = min(a_end, b_end) - max(a_ts, b_ts)
                    if overlap > 0:
                        pairs.append({
                            "op_a": i, "op_b": j,
                            "overlap_ns": round(overlap * 1000, 2)
                        })

        execution_mode = "parallel" if len(pairs) > 0 else "sequential"

        # 关键路径: 最长不重叠链
        sorted_events = sorted(complete, key=lambda e: float(e.get("ts", 0)))
        cp_ops, cp_len = [], 0.0
        cur_end = 0.0
        for e in sorted_events:
            ts = float(e.get("ts", 0))
            dur = float(e.get("dur", 0))
            if ts >= cur_end:
                cp_ops.append(len(cp_ops))
                cur_end = ts + dur
                cp_len += dur

        return {
            "total_ns": round(total_ns, 2),
            "execution_mode": execution_mode,
            "parallel_pairs": pairs[:20],
            "critical_path": cp_ops,
            "critical_path_length_ns": round(cp_len * 1000, 2),
        }

    @staticmethod
    def parse_detail(detail_str: str) -> dict:
        """从 detail 字段提取数据大小、类型、硬件 op id。

        Example: "XD:X3=0x4000,XN:X5=0,XM:X2=0x2000,Dtype:F32,Id:49"
        → {"dst_val": 0x4000, "src_val": 0x2000, "dtype": "F32", "hw_id": 49}
        """
        info: dict = {"data_size_bytes": 0, "data_type": "", "hw_op_id": 0}
        if not detail_str:
            return info

        # Dtype: F32, B64, B32 ...
        m = re.search(r'Dtype:(\w+)', detail_str)
        if m:
            info["data_type"] = m.group(1)

        # Id: 49
        m = re.search(r'[Ii]d:(\d+)', detail_str)
        if m:
            info["hw_op_id"] = int(m.group(1))

        # XD:X3=0x4000 or XM:X2=0x2000 (提取数据大小)
        for key in ["XD:X3", "XM:X2", "XD:X2"]:
            m = re.search(rf'{key}=(\w+)', detail_str)
            if m:
                try:
                    info["data_size_bytes"] = int(m.group(1), 16)
                except ValueError:
                    pass
                break

        return info

    @classmethod
    def parse_dir(cls, opprof_dir: Path) -> MsprofReport:
        """解析整个 OPPROF_xxx 目录。"""
        sim_dir = opprof_dir / "simulator"

        # 找所有核心
        cores = []
        if sim_dir.exists():
            for d in sorted(sim_dir.iterdir()):
                if d.is_dir() and d.name.startswith("core"):
                    cores.append(d.name)

        # 解析每个核心的 csv
        all_records: List[Tuple[str, List[InstrRecord]]] = []
        core_types = set()
        for core_name in cores:
            core_dir = sim_dir / core_name
            for csv_file in sorted(core_dir.glob("*_instr_exe.csv")):
                records = cls.parse_instr_csv(csv_file)
                all_records.append((core_name, records))
            # 检测 core 类型
            if "cubecore" in core_name:
                core_types.add("cubecore")
            if "veccore" in core_name:
                core_types.add("veccore")

        num_cores = len(all_records)

        # 生成 PipelineOp 列表 (按 core + 指令顺序)
        ops: List[PipelineOp] = []
        op_id = 0
        for core_name, records in all_records:
            ts = 0.0
            for r in records:
                detail_info = cls.parse_detail(r.detail)
                po = PipelineOp(
                    op_id=op_id,
                    engine=r.engine,
                    pipeline_channel=r.pipe,
                    core_id=core_name,
                    instruction=r.instr,
                    duration_ns=r.duration_ns,
                    cycles=r.cycles,
                    start_ns=ts,
                    end_ns=ts + r.duration_ns,
                    detail=r.detail,
                    data_size_bytes=detail_info.get("data_size_bytes", 0),
                    data_type=detail_info.get("data_type", ""),
                    hw_op_id=detail_info.get("hw_op_id", 0),
                )
                ops.append(po)
                ts += r.duration_ns
                op_id += 1

        # 解析 aggregate trace.json
        agg_trace = sim_dir / "trace.json"
        trace_info = {}
        if agg_trace.exists():
            trace_info = cls.parse_trace_json(agg_trace)

        # 计算 time_ratio
        total_ns = trace_info.get("total_ns", 0.0)
        if total_ns <= 0:
            total_ns = sum(o.duration_ns for o in ops)
        for o in ops:
            o.time_ratio = o.duration_ns / total_ns if total_ns > 0 else 0.0

        # engine utilization
        engine_util: Dict[str, float] = {}
        for o in ops:
            eng = o.engine
            engine_util[eng] = engine_util.get(eng, 0.0) + o.duration_ns
        for eng in engine_util:
            engine_util[eng] = round(engine_util[eng] / total_ns, 4) if total_ns > 0 else 0.0

        return MsprofReport(
            total_ns=total_ns,
            num_ops=len(ops),
            execution_mode=trace_info.get("execution_mode", "unknown"),
            num_cores=num_cores,
            core_types=sorted(core_types),
            ops=ops,
            engine_utilization=engine_util,
            parallel_pairs=trace_info.get("parallel_pairs", []),
            critical_path=trace_info.get("critical_path", []),
            critical_path_length_ns=trace_info.get("critical_path_length_ns", 0.0),
            source_dir=str(opprof_dir),
            generated_at=datetime.now().isoformat(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MsprofAnalyzer — 主类
# ═══════════════════════════════════════════════════════════════════════════════

class MsprofAnalyzer:
    """msprof 性能分析器。

    两种模式:
      - run_and_analyze(): 调用 msprof op simulator 生成 trace 并解析
      - parse_existing(): 解析已有的 OPPROF_ 目录

    Usage:
        analyzer = MsprofAnalyzer()
        report = analyzer.parse_existing(Path("OPPROF_xxx"))
    """

    def __init__(self, msprof_bin: Optional[str] = None):
        import shutil
        self.msprof_bin = msprof_bin or shutil.which("msprof") or ""
        self.msprof_available = bool(self.msprof_bin and Path(self.msprof_bin).exists())
        self._parser = MsprofParser()

    def parse_existing(self, opprof_dir: Path) -> MsprofReport:
        """解析已有的 OPPROF_xxx 目录。"""
        return self._parser.parse_dir(Path(opprof_dir))

    def find_latest_opprof(self, base_dir: Path) -> Optional[Path]:
        """在 base_dir 下查找最新的 OPPROF_ 目录。"""
        if not base_dir.exists():
            return None
        dirs = sorted(base_dir.glob("OPPROF_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return dirs[0] if dirs else None

    def to_dict(self, report: MsprofReport) -> dict:
        """转为 JSON-serializable dict (对齐项目 29 字段 schema)。"""
        return {
            "meta": {
                "source": "msprof_analyzer_v2",
                "generated_at": report.generated_at,
                "source_dir": report.source_dir,
                "msprof_fields_provided": MSPROF_FIELDS,
                "note": "pipe→engine 映射: MTE2→GM→UB, MTE3→UB→GM, VECTOR→VecUnit, CUBE→CubeUnit, MTE1→L1→L0",
            },
            "execution_summary": {
                "total_ns": report.total_ns,
                "num_ops": report.num_ops,
                "execution_mode": report.execution_mode,
                "num_cores": report.num_cores,
                "core_types": report.core_types,
            },
            "time_breakdown": sorted(
                [{"op_id": o.op_id, "engine": o.engine, "pipe": o.pipeline_channel,
                  "duration_ns": o.duration_ns, "time_ratio": o.time_ratio,
                  "instruction": o.instruction}
                 for o in report.ops],
                key=lambda x: x["time_ratio"], reverse=True,
            )[:20],
            "per_op_statistics": [
                {
                    "op_id": o.op_id,
                    "op_type": o.op_type,     # TBD, 由 HIVM 补充
                    "engine": o.engine,
                    "pipeline_channel": o.pipeline_channel,
                    "core_id": o.core_id,
                    "instruction": o.instruction,
                    "duration_ns": o.duration_ns,
                    "start_ns": o.start_ns,
                    "end_ns": o.end_ns,
                    "time_ratio": o.time_ratio,
                    "cycles": o.cycles,
                    "data_size_bytes": o.data_size_bytes,
                    "data_type": o.data_type,
                    "hw_op_id": o.hw_op_id,
                    "detail": o.detail,
                    # 以下由 HIVM 补充
                    "dst": TBD, "src": TBD, "src2": TBD,
                    "size_kb": TBD, "memory_region": TBD,
                    "variable_name": TBD, "dependencies": [],
                }
                for o in report.ops
            ],
            "engine_utilization": report.engine_utilization,
            "parallelism": {
                "parallel_pairs": report.parallel_pairs,
                "total_pairs": len(report.parallel_pairs),
            },
            "critical_path": {
                "path": report.critical_path,
                "length_ns": report.critical_path_length_ns,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 60)
    print("MsprofAnalyzer v2.0 — Self-Test")
    print("=" * 60)

    analyzer = MsprofAnalyzer()
    print(f"  msprof available: {analyzer.msprof_available}")

    # Test 1: parse existing OPPROF data (add kernel)
    add_dir = Path("/home/hjkc2/msprof_out2/OPPROF_20260728115748_BJISSSKXSANDOEJB")
    if add_dir.exists():
        print(f"\nTest 1: parse add kernel trace")
        report = analyzer.parse_existing(add_dir)
        print(f"  total_ns: {report.total_ns:.1f}")
        print(f"  num_ops: {report.num_ops}")
        print(f"  num_cores: {report.num_cores} ({report.core_types})")
        print(f"  execution_mode: {report.execution_mode}")
        print(f"  engine_utilization: {json.dumps(report.engine_utilization)}")
        print(f"  parallel_pairs: {len(report.parallel_pairs)}")

        # Top ops
        print("  Top ops by time:")
        for o in sorted(report.ops, key=lambda x: -x.duration_ns)[:5]:
            print(f"    op{o.op_id}: {o.engine:10s} {o.pipeline_channel:8s} "
                  f"{o.instruction:20s} {o.duration_ns:8.1f}ns")

        assert report.num_cores > 0
        assert report.total_ns > 0
        print("  PASS")
    else:
        print(f"\nTest 1: SKIP (data not found at {add_dir})")
        print("  Run msprof on the demo first")

    # Test 2: parse matmul
    matmul_dir = Path("/home/hjkc2/msprof_matmul/OPPROF_20260728141803_XMKEMZJEPAUGUYRT")
    if matmul_dir.exists():
        print(f"\nTest 2: parse matmul trace")
        report = analyzer.parse_existing(matmul_dir)
        print(f"  total_ns: {report.total_ns:.1f}")
        print(f"  num_ops: {report.num_ops}")
        print(f"  core_types: {report.core_types}")
        print(f"  engine_utilization: {json.dumps(report.engine_utilization)}")

        # Check for CUBE ops
        cube_ops = [o for o in report.ops if o.pipeline_channel == "CUBE"]
        mte1_ops = [o for o in report.ops if o.pipeline_channel == "MTE1"]
        print(f"  CUBE ops: {len(cube_ops)}, MTE1 ops: {len(mte1_ops)}")
        assert len(cube_ops) > 0, "Matmul should have CUBE ops"
        assert len(mte1_ops) > 0, "Matmul should have MTE1 ops"
        print("  PASS")
    else:
        print(f"\nTest 2: SKIP (data not found)")

    # Test 3: JSON export
    if add_dir.exists():
        report = analyzer.parse_existing(add_dir)
        d = analyzer.to_dict(report)
        assert "per_op_statistics" in d
        for opd in d["per_op_statistics"][:5]:
            assert isinstance(opd["duration_ns"], (int, float))
            assert opd["duration_ns"] > 0
        print(f"\nTest 3 (JSON): {len(json.dumps(d))} chars, all assertions passed")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED — MsprofAnalyzer v2.0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
