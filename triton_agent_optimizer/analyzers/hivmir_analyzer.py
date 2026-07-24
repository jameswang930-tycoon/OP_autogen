#!/usr/bin/env python3
"""
HIVMIR 分析器 —— 解析编译器中间产物 HIVMIR，提取算子语义信息。
输出到 outputs/<kernel>/roundN/hivmir/hivmir_report.json

═══════════════════════════════════════════════════════════════════════════════
  完整流程
═══════════════════════════════════════════════════════════════════════════════

  1. Triton kernel (.py) → Ascend 编译器编译 → HIVMIR 中间产物 (.mlir 文本)
  2. 本脚本解析 HIVMIR:
     ✅ op_type, instruction, dst, src, src2 — 操作语义
     ✅ size_kb — 精确数据大小 (memref<N>KB)
     ✅ memory_region — GM/UB/L1/L0 (buffer 前缀)
     ✅ variable_name — 变量名 (含地址偏移)
     ✅ dependencies — RAW/WAR/WAW (def-use chain 分析)
     ✅ buffers — producer/consumer 汇总
     ❌ 时序 (start_ns/end_ns/duration_ns) — msprof 提供
     ❌ 带宽/regime — msprof + SATURATION_PARAMS 计算
  3. 输出: outputs/<kernel>/roundN/hivmir/
       ├── compiler_output/              # HIVMIR 编译器原始输出
       └── hivmir_report.json            # ★ 解析最终产物 (29字段, 9✅+16❌)

═══════════════════════════════════════════════════════════════════════════════
  HIVMIR 文本格式 (编译器 trace 输出)
═══════════════════════════════════════════════════════════════════════════════

  格式示例 (Vector add, 128KB tile):
    hivm.alloc %ub_1 : memref<128KB>
    hivm.alloc %ub_2 : memref<128KB>
    hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
    hivm.vadd %ub_2, %ub_1, 3.0
    hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>

  支持格式:
    1. hivm.OP %dst, %src : memref<SIZE>     — 编译器 trace
    2. OP(dst, src, ...)                     — 纯文本 / DSL
    3. hivm.hir.load/store/vadd/matmul ...   — MLIR 全格式

  依赖分析规则:
    RAW: op 的 src 被之前的 op 写过 → RAW
    WAR: op 的 dst 被之前的 op 读过 → WAR
    WAW: op 的 dst 被之前的 op 写过 → WAW

═══════════════════════════════════════════════════════════════════════════════
  hivmir_report.json 字段 (29字段, 对齐 msprof pipeline_report.json)
═══════════════════════════════════════════════════════════════════════════════

  ✅ HIVMIR 提供 (9):
    op_id, op_type, instruction, dst, src, src2,
    size_kb, variable_name, dependencies

  ❌ msprof 补充 (16):
    engine, pipeline_channel, core_id, trace_event_name,
    duration_ns, start_ns, end_ns, time_ratio,
    effective_bw_gb_s, peak_bw_gb_s, bw_utilization, regime,
    wait_before_start_ns, total_ns, execution_mode, num_cores,
    engine_utilization, parallel_pairs, critical_path
"""

from __future__ import annotations

import json
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════════════════════════

TBD = "待补充"

OP_TO_ENGINE: Dict[str, str] = {
    "gm_to_ub":  "GM→UB",   "ub_to_gm":  "UB→GM",
    "gm_to_l1":  "GM→L1",   "l1_to_l0":  "L1→L0",   "l0_to_gm":  "L0→GM",
    "vadd":      "VecUnit", "vsub":      "VecUnit",  "vmul":      "VecUnit",
    "vdiv":      "VecUnit", "vmax":      "VecUnit",  "vmin":      "VecUnit",
    "vexp":      "VecUnit", "vlog":      "VecUnit",
    "matrixmul": "CubeUnit",
}

BUFFER_REGION: Dict[str, str] = {"gm": "GM", "ub": "UB", "l1": "L1", "l0": "L0"}

HIVMIR_FIELDS = [
    "op_id", "op_type", "instruction", "dst", "src", "src2",
    "size_kb", "memory_region_dst", "memory_region_src",
    "variable_name", "address_offset", "line_number",
    "dependencies", "buffers_info",
]

MSPROF_NEEDED_FIELDS = [
    "engine", "pipeline_channel", "core_id", "trace_event_name",
    "duration_ns", "start_ns", "end_ns", "time_ratio",
    "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization", "regime",
    "wait_before_start_ns",
    "total_ns", "execution_mode", "num_cores",
    "engine_utilization", "parallel_pairs", "critical_path",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构 (29字段对齐 msprof PipelineOp)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HIVMIROp:
    """HIVMIR 解析出的单个操作。29 字段对齐 msprof PipelineOp。"""
    # ── HIVMIR ✅ ──
    op_id: int
    op_type: str
    instruction: str = ""
    dst: str = ""
    src: str = ""
    src2: str = ""
    size_kb: float = 0.0
    memory_region_dst: str = ""
    memory_region_src: str = ""
    variable_name: str = ""
    address_offset: str = ""
    line_number: int = 0
    scalar: float = 0.0
    dependencies: List[dict] = field(default_factory=list)  # [{from_op_id, type}]
    # ── msprof ❌ ──
    engine: str = TBD
    pipeline_channel: str = TBD
    core_id: str = TBD
    trace_event_name: str = TBD
    duration_ns: str = TBD
    start_ns: str = TBD
    end_ns: str = TBD
    time_ratio: str = TBD
    effective_bw_gb_s: str = TBD
    peak_bw_gb_s: str = TBD
    bw_utilization: str = TBD
    regime: str = TBD
    wait_before_start_ns: str = TBD
    blocked_by: str = TBD


@dataclass
class BufferInfo:
    """buffer 信息。"""
    name: str
    region: str
    size_kb: float
    producers: List[int] = field(default_factory=list)
    consumers: List[int] = field(default_factory=list)


@dataclass
class HIVMIRReport:
    """HIVMIR 完整解析报告。7-section 对齐 msprof PipelineReport。"""
    # SECTION 1
    total_ns: str = TBD
    num_ops: int = 0
    execution_mode: str = TBD
    num_cores: str = TBD
    # SECTION 2
    time_breakdown: str = TBD
    # SECTION 3
    ops: List[HIVMIROp] = field(default_factory=list)
    # SECTION 4
    engine_utilization: str = TBD
    # SECTION 5
    bandwidth_utilization: str = TBD
    # SECTION 6
    parallelism: str = TBD
    # SECTION 7
    critical_path: str = TBD
    # 元数据
    source: str = "hivmir_compiler_trace"
    hivmir_file_path: str = ""
    generated_at: str = ""
    hivmir_fields: List[str] = field(default_factory=lambda: list(HIVMIR_FIELDS))
    msprof_fields: List[str] = field(default_factory=lambda: list(MSPROF_NEEDED_FIELDS))
    # HIVMIR 独有
    buffers: Dict[str, BufferInfo] = field(default_factory=dict)
    raw_dependencies: List[dict] = field(default_factory=list)
    war_dependencies: List[dict] = field(default_factory=list)
    waw_dependencies: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
#  HIVMIR 解析器
# ═══════════════════════════════════════════════════════════════════════════════

class HIVMIRParser:
    """解析 HIVMIR 文本 (编译器 trace 输出)。"""

    def __init__(self):
        self.ops: List[HIVMIROp] = []
        self.buffers: Dict[str, BufferInfo] = {}
        self._last_write: Dict[str, int] = {}
        self._last_read:  Dict[str, int] = {}

    def parse(self, hivmir_text: str) -> HIVMIRReport:
        self.ops = []; self.buffers = {}; self._last_write = {}; self._last_read = {}
        lines = hivmir_text.strip().split("\n")
        op_id = 0

        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("/*"): continue
            if any(line.startswith(kw) for kw in ("func.", "return", "module", "}")): continue

            alloc_info = self._try_parse_alloc(line)
            if alloc_info:
                name, size_kb, region = alloc_info
                self.buffers[name] = BufferInfo(name=name, region=region, size_kb=size_kb)
                continue

            op = self._try_parse_op(line, line_no, op_id)
            if op:
                self._update_buffer_usage(op, op_id)
                self.ops.append(op)
                op_id += 1

        self._analyze_dependencies()
        return self._build_report()

    # ── alloc ──
    def _try_parse_alloc(self, line: str) -> Optional[Tuple[str, float, str]]:
        m = re.search(r'(?:hivm\.)?alloc\s+%(\w+)\s*,?\s*:?\s*memref<(\d+\.?\d*)\s*(KB|MB|GB|B)?>', line)
        if m: return (m.group(1), self._to_kb(float(m.group(2)), m.group(3) or "KB"), self._get_region(m.group(1)))
        m = re.search(r'alloc\(\s*(\w+)\s*,\s*(\d+\.?\d*)\s*(KB|MB|GB|B)?\s*\)', line)
        if m: return (m.group(1), self._to_kb(float(m.group(2)), m.group(3) or "KB"), self._get_region(m.group(1)))
        m = re.search(r'%\w+\s*=\s*(?:memref\.)?alloc\b.*?memref<(\d+)x(?:\w+)>', line)
        if m:
            size_kb = (int(m.group(1)) * 2) / 1024.0
            region = "UB"; name_m = re.search(r'(%\w+)', line)
            return (name_m.group(1) if name_m else "unknown", size_kb, region)
        return None

    # ── op ──
    def _try_parse_op(self, line: str, line_no: int, op_id: int) -> Optional[HIVMIROp]:
        # 格式 A: hivm.OP %dst, %src : memref<SIZE>
        m = re.match(r'(?:hivm\.)?(\w+)\s+%(\w+)\s*,\s*%(\w+)(?:\s*,\s*(%?\w+))?(?:\s*:\s*memref<(\d+\.?\d*)\s*(KB|MB|GB|B)?>)?', line)
        if m:
            return self._build_op(m.group(1), m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), line, line_no, op_id)
        # 格式 B: OP(dst, src)
        m = re.match(r'(\w+)\(\s*(\w+)\s*,\s*(\w+)(?:\s*,\s*([^)]+))?\s*\)', line)
        if m:
            return self._build_op(m.group(1), m.group(2), m.group(3), m.group(4), None, None, line, line_no, op_id)
        # 格式 C: MLIR hivm.hir.load/store/vadd/matmul
        m = re.match(r'(?:hivm\.hir\.)?(load|store|vadd|vmul|vsub|matmul|batchMmadL1)\s+', line)
        if m:
            tmap = {"load": "gm_to_ub", "store": "ub_to_gm", "vadd": "vadd", "vmul": "vmul", "vsub": "vsub", "matmul": "matrixmul", "batchMmadL1": "matrixmul"}
            otype = tmap.get(m.group(1), m.group(1))
            ins = re.search(r'ins\(([^)]+)\)', line); outs = re.search(r'outs\(([^)]+)\)', line)
            dst = "unknown"; src = "unknown"; src2 = ""
            if outs: dst_m = re.search(r'(%\w+)', outs.group(1)); dst = dst_m.group(1) if dst_m else "unknown"
            if ins:
                parts = re.findall(r'(%\w+)', ins.group(1))
                if len(parts) >= 1: src = parts[0]
                if len(parts) >= 2: src2 = parts[1]
            sizes = re.findall(r'memref<(\d+)x\w+>', line)
            size_kb = (int(sizes[0]) * 2) / 1024.0 if sizes else 64.0
            return HIVMIROp(op_id=op_id, op_type=otype, instruction=f"{otype}({dst}, {src}{', '+src2 if src2 else ''})",
                            dst=dst, src=src, src2=src2, size_kb=size_kb,
                            memory_region_dst=self._get_region(dst), memory_region_src=self._get_region(src),
                            variable_name=dst, line_number=line_no)
        return None

    def _build_op(self, op_type, dst, src, arg3, size_str, size_unit, raw_line, line_no, op_id):
        scalar = 0.0; src2 = ""
        if arg3:
            arg3 = arg3.strip().lstrip("%")
            try: scalar = float(arg3)
            except ValueError: src2 = arg3
        size_kb = 64.0
        if size_str: size_kb = self._to_kb(float(size_str), size_unit or "KB")
        else:
            buf = self.buffers.get(dst) or self.buffers.get(src)
            if buf: size_kb = buf.size_kb
        addr_offset = ""; var_name = dst
        om = re.search(r'(\w+)\s*\+\s*(\w+)\s*\*\s*(\d+\.?\d*)\s*(KB|MB|B)?', raw_line)
        if om:
            addr_offset = f"+ {om.group(2)}*{om.group(3)}{om.group(4) or 'KB'}"
            var_name = f"{om.group(1)}{addr_offset}"
        if src2: instr = f"{op_type}({dst}, {src}, {src2})"
        elif op_type == "vadd" and scalar: instr = f"{op_type}({dst}, {src}, {scalar})"
        else: instr = f"{op_type}({dst}, {src})"
        return HIVMIROp(op_id=op_id, op_type=op_type, instruction=instr,
                        dst=dst, src=src, src2=src2, scalar=scalar, size_kb=size_kb,
                        memory_region_dst=self._get_region(dst), memory_region_src=self._get_region(src),
                        variable_name=var_name, address_offset=addr_offset, line_number=line_no,
                        engine=OP_TO_ENGINE.get(op_type, TBD))

    # ── 依赖 ──
    def _update_buffer_usage(self, op: HIVMIROp, op_id: int):
        for buf_name, is_dst in [(op.dst, True), (op.src, False), (op.src2, False)]:
            if not buf_name or buf_name == "unknown": continue
            if buf_name not in self.buffers:
                self.buffers[buf_name] = BufferInfo(name=buf_name, region=self._get_region(buf_name), size_kb=op.size_kb)
            if is_dst: self.buffers[buf_name].producers.append(op_id)
            else: self.buffers[buf_name].consumers.append(op_id)

    def _analyze_dependencies(self):
        for op in self.ops:
            for buf_name in [op.src, op.src2]:
                if not buf_name or buf_name == "unknown": continue
                if buf_name in self._last_write and self._last_write[buf_name] != op.op_id:
                    op.dependencies.append({"from_op_id": self._last_write[buf_name], "type": "RAW"})
            if op.dst and op.dst != "unknown":
                if op.dst in self._last_read and self._last_read[op.dst] != op.op_id:
                    op.dependencies.append({"from_op_id": self._last_read[op.dst], "type": "WAR"})
                if op.dst in self._last_write and self._last_write[op.dst] != op.op_id:
                    op.dependencies.append({"from_op_id": self._last_write[op.dst], "type": "WAW"})
            if op.dst and op.dst != "unknown": self._last_write[op.dst] = op.op_id
            for buf_name in [op.src, op.src2]:
                if buf_name and buf_name != "unknown": self._last_read[buf_name] = op.op_id

    # ── 报告 ──
    def _build_report(self) -> HIVMIRReport:
        r = HIVMIRReport(num_ops=len(self.ops), ops=self.ops, buffers=self.buffers, generated_at=datetime.now().isoformat())
        for op in self.ops:
            for d in op.dependencies:
                entry = {"from_op": d["from_op_id"], "to_op": op.op_id, "type": d["type"],
                         "from_instruction": self.ops[d["from_op_id"]].instruction if d["from_op_id"] < len(self.ops) else "?",
                         "to_instruction": op.instruction,
                         "buffer": self._shared_buf(self.ops[d["from_op_id"]] if d["from_op_id"] < len(self.ops) else None, op)}
                {"RAW": r.raw_dependencies, "WAR": r.war_dependencies, "WAW": r.waw_dependencies}[d["type"]].append(entry)
        return r

    def _shared_buf(self, a, b) -> str:
        if a is None: return "?"
        if b.src == a.dst: return b.src
        if b.src2 == a.dst: return b.src2
        if b.dst == a.dst or b.dst == a.src: return b.dst
        return "?"

    @staticmethod
    def _get_region(name: str) -> str:
        if not name or name == "unknown": return "Unknown"
        p = name.lstrip("%").split("_")[0].lower()
        return "GM" if p.startswith("arg") else BUFFER_REGION.get(p, "Unknown")

    @staticmethod
    def _to_kb(size: float, unit: str) -> float:
        u = unit.upper()
        if u == "GB": return size * 1024 * 1024
        if u == "MB": return size * 1024
        if u == "B":  return size / 1024
        return size


# ═══════════════════════════════════════════════════════════════════════════════
#  DSL → HIVMIR mock 转换 (本地测试用)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_mock_hivmir_from_dsl(dsl_program: str) -> str:
    stmts = re.findall(r'\w+\([^)]+\)', dsl_program)
    lines = []
    for stmt in stmts:
        m = re.match(r'(\w+)\((.+)\)', stmt)
        if not m: continue
        name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
        if name == "alloc": lines.append(f"hivm.alloc %{args[0]} : memref<{args[1]}>")
        elif name == "vadd": lines.append(f"hivm.vadd %{args[0]}, %{args[1]}, {args[2] if len(args)>2 else '1.0'}")
        elif name == "matrixmul": lines.append(f"hivm.matrixmul %{args[0]}, %{args[1]}, %{args[2]} : memref<64KB>")
        else: lines.append(f"hivm.{name} %{args[0]}, %{args[1]} : memref<64KB>")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  主类: HIVMIRAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class HIVMIRAnalyzer:
    """HIVMIR 分析器。

    唯一工作模式: 解析 HIVMIR 文本 → 写入 outputs/<kernel>/roundN/hivmir/

    Usage:
        analyzer = HIVMIRAnalyzer()

        # round0
        report = analyzer.analyze(hivmir_text, kernel_name="vector_add_fp16_N65536")

        # roundN in tier
        report = analyzer.analyze(hivmir_text, kernel_name="vector_add_fp16_N65536",
                                   tier="01_block_size_launch", round_number=3)
    """

    def __init__(self):
        self._parser = HIVMIRParser()

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口
    # ═══════════════════════════════════════════════════════════════════════════

    def analyze(
        self,
        hivmir_text: str,
        kernel_name: str,
        tier: str = "",
        round_number: int = 0,
        source_path: str = "",
    ) -> HIVMIRReport:
        """解析 HIVMIR 文本 → 保存到 outputs/<kernel>/roundN/hivmir/。

        Args:
            hivmir_text: HIVMIR 文本内容
            kernel_name: Triton kernel 名
            tier: Tier 文件夹名, round0 时为空
            round_number: 轮次号 (0 = 基准分析)
            source_path: HIVMIR 来源文件路径 (记录在报告中)
        """
        report = self._parser.parse(hivmir_text)
        report.hivmir_file_path = source_path
        report.generated_at = datetime.now().isoformat()

        # 保存
        root = _find_outputs_root()
        if tier:
            round_dir = root / kernel_name / tier / f"round{round_number}"
        else:
            round_dir = root / kernel_name / f"round{round_number}"
        hivmir_dir = round_dir / "hivmir"
        hivmir_dir.mkdir(parents=True, exist_ok=True)

        # hivmir_report.json
        json_path = hivmir_dir / "hivmir_report.json"
        json_path.write_text(
            json.dumps(self._report_to_dict(report), indent=2, ensure_ascii=False),
            encoding="utf-8")

        # compiler_output/
        comp_dir = hivmir_dir / "compiler_output"
        comp_dir.mkdir(exist_ok=True)
        (comp_dir / "hivmir_output.mlir").write_text(hivmir_text, encoding="utf-8")

        print(f"[hivmir] Report saved → {json_path}")
        print(f"[hivmir] {report.num_ops} ops, {len(report.buffers)} buffers, "
              f"RAW={len(report.raw_dependencies)}, WAR={len(report.war_dependencies)}, "
              f"WAW={len(report.waw_dependencies)}")
        return report

    def analyze_file(
        self, hivmir_path: Path, kernel_name: str,
        tier: str = "", round_number: int = 0,
    ) -> HIVMIRReport:
        """解析 HIVMIR 文件。"""
        hivmir_path = Path(hivmir_path)
        if not hivmir_path.exists():
            raise FileNotFoundError(f"HIVMIR file not found: {hivmir_path}")
        text = hivmir_path.read_text(encoding="utf-8")
        return self.analyze(text, kernel_name, tier, round_number, str(hivmir_path))

    # ═══════════════════════════════════════════════════════════════════════════
    #  输出格式化
    # ═══════════════════════════════════════════════════════════════════════════

    def _report_to_dict(self, report: HIVMIRReport) -> dict:
        return {
            "meta": {
                "source": report.source,
                "generated_at": report.generated_at,
                "hivmir_file_path": report.hivmir_file_path,
                "hivmir_fields_provided": report.hivmir_fields,
                "msprof_fields_pending": report.msprof_fields,
                "note": (
                    "HIVMIR 提供 buffer名/size/依赖/指令 (9字段) ✅, "
                    "msprof 补充 timing/bandwidth/regime (16字段) ❌"
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
                    "scalar": op.scalar if op.scalar else 0.0,
                    "size_kb": op.size_kb,
                    "memory_region_dst": op.memory_region_dst,
                    "memory_region_src": op.memory_region_src,
                    "variable_name": op.variable_name,
                    "address_offset": op.address_offset,
                    "line_number": op.line_number,
                    "duration_ns": op.duration_ns,
                    "start_ns": op.start_ns,
                    "end_ns": op.end_ns,
                    "time_ratio": op.time_ratio,
                    "effective_bw_gb_s": op.effective_bw_gb_s,
                    "peak_bw_gb_s": op.peak_bw_gb_s,
                    "bw_utilization": op.bw_utilization,
                    "regime": op.regime,
                    "wait_before_start_ns": op.wait_before_start_ns,
                    "blocked_by": op.blocked_by,
                    "pipeline_channel": op.pipeline_channel,
                    "core_id": op.core_id,
                    "trace_event_name": op.trace_event_name,
                    "dependencies": op.dependencies,
                }
                for op in report.ops
            ],
            "engine_utilization": report.engine_utilization,
            "bandwidth_utilization": report.bandwidth_utilization,
            "parallelism": report.parallelism,
            "critical_path": report.critical_path,
            "buffers": {
                name: {
                    "region": buf.region, "size_kb": buf.size_kb,
                    "producers": buf.producers, "consumers": buf.consumers,
                }
                for name, buf in report.buffers.items()
            },
            "dependencies_summary": {
                "total": len(report.raw_dependencies) + len(report.war_dependencies) + len(report.waw_dependencies),
                "raw": report.raw_dependencies,
                "war": report.war_dependencies,
                "waw": report.waw_dependencies,
            },
        }


def _find_outputs_root() -> Path:
    return Path(__file__).resolve().parent.parent / "outputs"


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 60)
    print("HIVMIRAnalyzer Self-Test")
    print("=" * 60)

    parser = HIVMIRParser()

    # Test 1: Vector add
    text1 = """
hivm.alloc %gm_1 : memref<128KB>
hivm.alloc %ub_1 : memref<128KB>
hivm.alloc %ub_2 : memref<128KB>
hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
hivm.vadd %ub_2, %ub_1, 2.0
hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>
"""
    r1 = parser.parse(text1)
    print(f"\nTest 1 — Vector add: {r1.num_ops} ops, {len(r1.buffers)} buffers")
    print(f"  RAW={len(r1.raw_dependencies)} WAR={len(r1.war_dependencies)} WAW={len(r1.waw_dependencies)}")
    for op in r1.ops:
        deps = ", ".join(f"op{d['from_op_id']}({d['type']})" for d in op.dependencies)
        filled = sum(1 for v in [op.instruction, op.dst, op.size_kb] if v and v != TBD)
        pending = sum(1 for v in [op.duration_ns, op.start_ns, op.regime] if v == TBD)
        print(f"  op{op.op_id}: {op.instruction:35s} size={op.size_kb}KB "
              f"region={op.memory_region_dst} deps=[{deps}] "
              f"(HIVMIR OK={filled}, msprof TBD={pending})")
    assert r1.num_ops == 3 and len(r1.raw_dependencies) == 2

    # Test 2: Matrix pipeline
    text2 = """
hivm.gm_to_l1 %l1_a1, %gm_a1 : memref<256KB>
hivm.l1_to_l0 %l0_a1, %l1_a1 : memref<128KB>
hivm.matrixmul %l0_c1, %l0_a1, %l0_b1
hivm.l0_to_gm %gm_c1, %l0_c1 : memref<256KB>
"""
    r2 = parser.parse(text2)
    print(f"\nTest 2 — Matrix pipeline: {r2.num_ops} ops")
    for op in r2.ops:
        print(f"  op{op.op_id}: {op.instruction:40s} size={op.size_kb}KB")
    assert r2.num_ops == 4

    # Test 3: WAR + WAW
    text3 = """
hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
hivm.vadd %ub_2, %ub_1, 2.0
hivm.gm_to_ub %ub_1, %gm_2 : memref<128KB>
"""
    r3 = parser.parse(text3)
    print(f"\nTest 3 — WAR+WAW: RAW={len(r3.raw_dependencies)} WAR={len(r3.war_dependencies)} WAW={len(r3.waw_dependencies)}")
    for w in r3.war_dependencies + r3.waw_dependencies:
        print(f"  {w['type']}: op{w['from_op']}->op{w['to_op']} buffer={w['buffer']}")
    assert len(r3.war_dependencies) >= 1 and len(r3.waw_dependencies) >= 1

    # Test 4: JSON 格式验证 (不写 outputs, 只验证序列化)
    analyzer = HIVMIRAnalyzer()
    d = analyzer._report_to_dict(r1)
    assert d["meta"]["hivmir_fields_provided"] == list(HIVMIR_FIELDS)
    assert d["meta"]["msprof_fields_pending"] == list(MSPROF_NEEDED_FIELDS)
    for opd in d["per_op_statistics"]:
        assert opd["instruction"] != TBD  # HIVMIR has this
        assert opd["size_kb"] != TBD       # HIVMIR has this
        assert opd["duration_ns"] == TBD   # msprof needed
    assert "buffers" in d and "dependencies_summary" in d
    print(f"\nJSON format: {len(json.dumps(d))} chars, all assertions passed [OK]")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
