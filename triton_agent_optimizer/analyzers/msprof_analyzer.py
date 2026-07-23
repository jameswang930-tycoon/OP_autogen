#!/usr/bin/env python3
"""
Simulator 包装器 — 提供 Python API 调用 cost_emulator/simulator.py。

两种输出模式:
  --llm 模式 (主要, AI 消费):  每轮必跑, 返回结构化 SimulatorResult
  Gantt  模式 (次要, 人读):    按需生成, 返回原始 ASCII 文本

解析器复用 fusion_pipeline/complete_data_merge.py 的 SimulatorOutputParser
的解析逻辑, 但作为独立模块存在, 不依赖 fusion_pipeline 的 matplotlib/pandas。

使用:
    from analyzers.msprof_analyzer import MsprofAnalyzer

    analyzer = MsprofAnalyzer(simulator_path=paths.simulator_path,
                               python_exe=paths.python_executable)

    # AI 消费 (每轮必跑)
    result = analyzer.run_llm("alloc(gm_1, 128KB) gm_to_ub(ub_1, gm_1)")
    print(f"total_ns={result.total_ns}, bottleneck={result.time_breakdown[0]}")

    # 人读 (按需生成)
    gantt_text = analyzer.run_human("alloc(gm_1, 128KB) gm_to_ub(ub_1, gm_1)")
    Path("debug_gantt.txt").write_text(gantt_text)
"""

from __future__ import annotations

import os
import re
import sys
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockedByInfo:
    """一个 op 被另一个 op 阻塞的详细信息。"""
    blocker_op_id: int
    hazard_type: str          # 'RAW' | 'WAR' | 'WAW'
    buffer_name: str          # 被阻塞的 buffer 名
    description: str          # 人类可读描述
    is_avoidable: bool = False  # WAR 且无 RAW/WAW 重叠 → 可通过独立 buffer 避免


@dataclass
class SimulatorOp:
    """单个操作 (op) 的完整性能信息。

    字段来源: simulator --llm PER-OP STATISTICS section
    """
    op_id: int
    op_type: str              # gm_to_ub / ub_to_gm / vadd / gm_to_l1 / l1_to_l0 / matrixmul / l0_to_gm
    instruction: str          # 原始指令文本
    engine: str = ""          # GM→UB / UB→GM / VecUnit / ...
    dst: str = ""
    src: str = ""
    src2: str = ""            # 第二源 (仅 matrixmul)

    size_kb: float = 0.0
    duration_ns: float = 0.0
    start_ns: float = 0.0
    end_ns: float = 0.0

    effective_bw: float = 0.0   # GB/s
    peak_bw: float = 0.0        # GB/s
    bw_utilization: float = 0.0 # 0.0 ~ 1.0
    regime: str = "unknown"     # floor / ramp / saturated / flat

    wait_before_start_ns: float = 0.0
    blocked_by: List[BlockedByInfo] = field(default_factory=list)
    time_ratio: float = 0.0     # 0.0 ~ 1.0

    @property
    def is_bottleneck_candidate(self) -> bool:
        """是否可能是瓶颈候选 (时间占比 > 10%)。"""
        return self.time_ratio > 0.10

    @property
    def is_saturated(self) -> bool:
        """带宽是否已饱和。"""
        return self.regime == "saturated"

    @property
    def is_below_peak(self) -> bool:
        """带宽利用率是否在 peak 以下 (floor 或 ramp)。"""
        return self.regime in ("floor", "ramp")


@dataclass
class ParallelPair:
    """一对并行执行的 op。"""
    op_a: int
    op_b: int
    overlap_start_ns: float
    overlap_end_ns: float
    overlap_ns: float


@dataclass
class CriticalPathEdge:
    """关键路径上的一条边。"""
    from_op: int
    to_op: int
    reason: str  # e.g. "RAW on 'ub_1'" / "engine serialization (GM→UB reused)"


@dataclass
class SimulatorResult:
    """simulator --llm --critical-path 完整输出。

    7 个 section 的结构化表示:
      EXECUTION SUMMARY / TIME BREAKDOWN / PER-OP STATISTICS /
      ENGINE UTILIZATION / BANDWIDTH UTILIZATION / PARALLELISM /
      CRITICAL PATH
    """
    total_ns: float = 0.0
    num_ops: int = 0
    execution_mode: str = "sequential"   # 'parallel' | 'sequential'

    ops: List[SimulatorOp] = field(default_factory=list)
    engine_utilization: Dict[str, float] = field(default_factory=dict)
    parallel_pairs: List[ParallelPair] = field(default_factory=list)
    sequential_root_causes: List[str] = field(default_factory=list)

    critical_path: List[int] = field(default_factory=list)
    critical_path_length_ns: float = 0.0
    critical_path_fraction: float = 0.0
    critical_path_edges: List[CriticalPathEdge] = field(default_factory=list)

    raw_output: str = ""  # 原始输出文本 (调试用)

    # ── 便捷查询 ──────────────────────────────────────────────────────────────

    def get_op(self, op_id: int) -> Optional[SimulatorOp]:
        """按 id 获取 op。"""
        for op in self.ops:
            if op.op_id == op_id:
                return op
        return None

    def get_bottleneck_op(self) -> Optional[SimulatorOp]:
        """获取瓶颈 op (time_ratio 最大的, 且在被关键路径上)。"""
        if not self.ops:
            return None
        cp_set = set(self.critical_path)
        cp_ops = [op for op in self.ops if op.op_id in cp_set]
        if not cp_ops:
            return max(self.ops, key=lambda o: o.time_ratio)
        return max(cp_ops, key=lambda o: o.time_ratio)

    def get_ops_on_critical_path(self) -> List[SimulatorOp]:
        """获取关键路径上的所有 op。"""
        return [op for op in self.ops if op.op_id in set(self.critical_path)]

    def get_time_breakdown(self) -> List[SimulatorOp]:
        """按 time_ratio 从大到小排序的 op 列表。"""
        return sorted(self.ops, key=lambda o: o.time_ratio, reverse=True)

    def get_engine_bottleneck(self) -> Tuple[str, float]:
        """获取利用率最高的引擎。"""
        if not self.engine_utilization:
            return ("unknown", 0.0)
        max_eng = max(self.engine_utilization, key=self.engine_utilization.get)
        return (max_eng, self.engine_utilization[max_eng])


# ═══════════════════════════════════════════════════════════════════════════════
#  解析器
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatorOutputParser:
    """解析 simulator --llm --critical-path 的文本输出。

    处理 7 个 section:
      EXECUTION SUMMARY → TIME BREAKDOWN → PER-OP STATISTICS →
      ENGINE UTILIZATION → BANDWIDTH UTILIZATION → PARALLELISM →
      CRITICAL PATH

    正则表达式设计为容忍编码问题 (如 → 被 GBK 编码破坏为 → 等)。
    """

    def __init__(self):
        self.result = SimulatorResult()

    def parse(self, text: str) -> SimulatorResult:
        """主入口: 解析完整 simulator 输出文本。"""
        self.result = SimulatorResult(raw_output=text)

        self._parse_execution_summary(text)
        self._parse_per_op_statistics(text)
        self._parse_engine_utilization(text)
        self._parse_parallelism(text)
        self._parse_critical_path(text)

        return self.result

    # ── EXECUTION SUMMARY ──────────────────────────────────────────────────

    def _parse_execution_summary(self, text: str):
        m = re.search(r'total_ns:\s*([\d.]+)', text)
        if m:
            self.result.total_ns = float(m.group(1))

        m = re.search(r'num_ops:\s*(\d+)', text)
        if m:
            self.result.num_ops = int(m.group(1))

        m = re.search(r'execution_mode:\s*(\w+)', text)
        if m:
            self.result.execution_mode = m.group(1)

    # ── PER-OP STATISTICS ──────────────────────────────────────────────────

    def _parse_per_op_statistics(self, text: str):
        """解析 PER-OP STATISTICS section。

        通过定位 opN: header 来分割每个 op 块, 然后逐行解析。
        """
        # 找到 PER-OP STATISTICS section 的起点
        sec_start = text.find("=== PER-OP STATISTICS ===")
        if sec_start == -1:
            return
        # 找到下一个 section 的起点 (ENGINE UTILIZATION)
        sec_end = text.find("=== ENGINE UTILIZATION ===", sec_start)
        section = text[sec_start:sec_end] if sec_end != -1 else text[sec_start:]

        # 用 "op\d+:" 作为 op 块的分隔符
        op_blocks = re.split(r'\n(?=op\d+:\s)', section)
        for block in op_blocks:
            op = self._parse_single_op_block(block)
            if op:
                self.result.ops.append(op)

    def _parse_single_op_block(self, block: str) -> Optional[SimulatorOp]:
        """解析单个 op 的文本块。"""
        # ---- header: opN: op_type(args) ----
        header_m = re.match(
            r'op(\d+):\s+(\w+)\(([^)]+)\)', block.strip()
        )
        if not header_m:
            return None

        op_id = int(header_m.group(1))
        op_type = header_m.group(2)
        operands = [p.strip() for p in header_m.group(3).split(",")]
        dst = operands[0] if len(operands) > 0 else ""
        src = operands[1] if len(operands) > 1 else ""
        src2 = operands[2] if len(operands) > 2 else ""

        op = SimulatorOp(
            op_id=op_id,
            op_type=op_type,
            instruction=f"{op_type}({header_m.group(3)})",
            dst=dst,
            src=src,
            src2=src2,
        )

        # ---- engine ----
        m = re.search(r'engine:\s*(\S+(?:\s*\S+)*)', block)
        if m:
            op.engine = m.group(1).strip()

        # ---- size ----
        m = re.search(r'size:\s*([\d.]+)\s*KB', block)
        if m:
            op.size_kb = float(m.group(1))

        # ---- cycles_ns ----
        m = re.search(r'cycles_ns:\s*\[([\d.]+)\.\.([\d.]+)\]', block)
        if m:
            op.start_ns = float(m.group(1))
            op.end_ns = float(m.group(2))

        # ---- duration_ns (in block, not just header) ----
        m = re.search(r'duration_ns=([\d.]+)', block)
        if m:
            op.duration_ns = float(m.group(1))

        # ---- time_ratio ----
        m = re.search(r'time_ratio=([\d.]+)%', block)
        if m:
            op.time_ratio = float(m.group(1)) / 100.0

        # ---- bandwidth ----
        m = re.search(
            r'effective=([\d.]+)\s*GB/s\s+peak=([\d.]+)\s*GB/s\s+utilization=([\d.]+)%',
            block
        )
        if m:
            op.effective_bw = float(m.group(1))
            op.peak_bw = float(m.group(2))
            op.bw_utilization = float(m.group(3)) / 100.0

        # ---- regime ----
        m = re.search(r'regime=(\w+)', block)
        if m:
            op.regime = m.group(1)

        # ---- wait_ns_before_start ----
        m = re.search(r'wait_ns_before_start:\s*([\d.]+)', block)
        if m:
            op.wait_before_start_ns = float(m.group(1))

        # ---- blocked_by (可能有多行) ----
        self._parse_blocked_by(block, op)

        return op

    def _parse_blocked_by(self, block: str, op: SimulatorOp):
        """解析 blocked_by 行 (可能有多个)。"""
        # 匹配: blocked_by: opN via RAW on 'buf' ...
        pattern = re.compile(
            r'blocked_by:\s*op(\d+)\s+via\s+(\w+)\s+on\s+\'(\w+)\''
            r'(?:\s*[-–—]\s*(.+))?'
        )
        for m in pattern.finditer(block):
            blocker_id = int(m.group(1))
            htype = m.group(2)
            buf = m.group(3)
            desc = m.group(4).strip() if m.group(4) else f"{buf} {htype} dependency"

            # WAR 且无 RAW/WAW → avoidable
            avoidable = (htype == "WAR")

            op.blocked_by.append(BlockedByInfo(
                blocker_op_id=blocker_id,
                hazard_type=htype,
                buffer_name=buf,
                description=desc,
                is_avoidable=avoidable,
            ))

    # ── ENGINE UTILIZATION ─────────────────────────────────────────────────

    def _parse_engine_utilization(self, text: str):
        sec_start = text.find("=== ENGINE UTILIZATION ===")
        if sec_start == -1:
            return
        sec_end = text.find("=== BANDWIDTH UTILIZATION ===", sec_start)
        section = text[sec_start:sec_end] if sec_end != -1 else text[sec_start:]

        # 匹配: EngineName: busy=X/Y ns  utilization=Z%
        pattern = re.compile(
            r'([\w→]+):\s*busy=([\d.]+)/([\d.]+)\s*ns\s+utilization=([\d.]+)%'
        )
        for m in pattern.finditer(section):
            eng_name = m.group(1).strip()
            util = float(m.group(4)) / 100.0
            self.result.engine_utilization[eng_name] = util

    # ── PARALLELISM ────────────────────────────────────────────────────────

    def _parse_parallelism(self, text: str):
        sec_start = text.find("=== PARALLELISM ===")
        if sec_start == -1:
            return
        sec_end = text.find("=== CRITICAL PATH ===", sec_start)
        section = text[sec_start:sec_end] if sec_end != -1 else text[sec_start:]

        # 并行对: opA || opB: overlap=[X..Y]  overlap_ns=Z
        pair_pattern = re.compile(
            r'op(\d+)\s*\|\|\s*op(\d+):\s*overlap=\[([\d.]+)\.\.([\d.]+)\]\s+overlap_ns=([\d.]+)'
        )
        for m in pair_pattern.finditer(section):
            self.result.parallel_pairs.append(ParallelPair(
                op_a=int(m.group(1)),
                op_b=int(m.group(2)),
                overlap_start_ns=float(m.group(3)),
                overlap_end_ns=float(m.group(4)),
                overlap_ns=float(m.group(5)),
            ))

        # 串行根因: opA->opB: RAW on 'buf' ...
        cause_pattern = re.compile(
            r'op(\d+)->op(\d+):\s+(\w+)\s+on\s+\'(\w+)\''
        )
        for m in cause_pattern.finditer(section):
            self.result.sequential_root_causes.append(
                f"op{m.group(1)}->op{m.group(2)}: {m.group(3)} on '{m.group(4)}'"
            )

    # ── CRITICAL PATH ──────────────────────────────────────────────────────

    def _parse_critical_path(self, text: str):
        sec_start = text.find("=== CRITICAL PATH ===")
        if sec_start == -1:
            return
        section = text[sec_start:]

        m = re.search(r'length_ns:\s*([\d.]+)', section)
        if m:
            self.result.critical_path_length_ns = float(m.group(1))

        m = re.search(r'fraction_of_makespan:\s*([\d.]+)%', section)
        if m:
            self.result.critical_path_fraction = float(m.group(1)) / 100.0

        # path: opA -> opB -> opC
        path_m = re.search(r'path:\s*(op\d+(?:\s*->\s*op\d+)*)', section)
        if path_m:
            ids = re.findall(r'op(\d+)', path_m.group(1))
            self.result.critical_path = [int(i) for i in ids]

        # edges: opA -> opB: reason
        edge_pattern = re.compile(
            r'op(\d+)\s*->\s*op(\d+):\s*(.+?)(?=\n\s*op\d+|\n\s*per_op|\n\n|\Z)',
            re.DOTALL
        )
        # 更精确的边缘匹配: 在 "edges:" 和 "per_op:" 之间
        edges_start = section.find("edges:")
        per_op_start = section.find("per_op:", edges_start) if edges_start != -1 else -1
        if edges_start != -1:
            edges_section = section[edges_start:per_op_start] if per_op_start != -1 else section[edges_start:]
            for m in re.finditer(
                r'op(\d+)\s*->\s*op(\d+):\s*(.+)', edges_section
            ):
                self.result.critical_path_edges.append(CriticalPathEdge(
                    from_op=int(m.group(1)),
                    to_op=int(m.group(2)),
                    reason=m.group(3).strip(),
                ))


# ═══════════════════════════════════════════════════════════════════════════════
#  主类: MsprofAnalyzer
# ═══════════════════════════════════════════════════════════════════════════════

class MsprofAnalyzer:
    """Simulator 包装器。

    通过 subprocess 调用 cost_emulator/simulator.py。
    不 import simulator 模块 (避免依赖 networkx 等可选包)。

    Usage:
        analyzer = MsprofAnalyzer(simulator_path, python_exe)
        result = analyzer.run_llm("alloc(gm_1, 128KB) gm_to_ub(ub_1, gm_1)")
        gantt  = analyzer.run_human("alloc(gm_1, 128KB) gm_to_ub(ub_1, gm_1)")
    """

    def __init__(
        self,
        simulator_path: Optional[Path] = None,
        python_exe: Optional[str] = None,
        timeout_seconds: int = 30,
    ):
        """
        Args:
            simulator_path: simulator.py 的路径。默认从 config 读取。
            python_exe: Python 可执行文件。默认从 config 读取。
            timeout_seconds: subprocess 超时 (秒)。
        """
        if simulator_path is None:
            from config import paths
            simulator_path = paths.simulator_path
        if python_exe is None:
            from config import paths
            python_exe = paths.python_executable

        self.simulator_path = Path(simulator_path)
        self.python_exe = python_exe
        self.timeout = timeout_seconds

        self._parser = SimulatorOutputParser()

        # 检测是否需要设置 PYTHONIOENCODING (Windows GBK 问题)
        self._needs_utf8_env = (sys.platform == "win32")

    # ═══════════════════════════════════════════════════════════════════════
    #  主要模式: --llm (AI 消费, 每轮必跑)
    # ═══════════════════════════════════════════════════════════════════════

    def run_llm(self, dsl_program: str) -> SimulatorResult:
        """运行 simulator --llm --critical-path, 返回结构化结果。

        这是 Agent 每轮必跑的分析数据源。

        Args:
            dsl_program: DSL 程序文本, e.g. "alloc(gm_1, 128KB) gm_to_ub(ub_1, gm_1)"

        Returns:
            SimulatorResult with all 7 sections parsed.

        Raises:
            subprocess.TimeoutExpired: simulator 超时。
            RuntimeError: simulator 非零退出码。
        """
        raw = self._invoke(["--llm", "--critical-path", dsl_program])
        return self._parser.parse(raw)

    def run_llm_file(self, dsl_file: Path) -> SimulatorResult:
        """从文件读取 DSL 程序, 运行 simulator --llm --critical-path。"""
        dsl_program = dsl_file.read_text(encoding="utf-8").strip()
        return self.run_llm(dsl_program)

    def parse_llm_output(self, raw_output: str) -> SimulatorResult:
        """解析已有的 simulator --llm 输出文本 (不重新运行 simulator)。

        用途: 从日志文件恢复之前的分析结果。
        """
        return self._parser.parse(raw_output)

    # ═══════════════════════════════════════════════════════════════════════
    #  次要模式: Gantt (人读, 按需生成)
    # ═══════════════════════════════════════════════════════════════════════

    def run_human(self, dsl_program: str) -> str:
        """运行 simulator --critical-path (默认模式), 返回 ASCII Gantt 流水图。

        包含: Pipeline Execution Graph / 操作表格 / 时间占比柱状图 /
              引擎利用率 / 带宽利用率 / 并行分析 / 关键路径

        注意: Windows 需要 PYTHONIOENCODING=utf-8 (µ 字符的 GBK 编码问题)。

        Args:
            dsl_program: DSL 程序文本。

        Returns:
            原始 ASCII Gantt 文本 (不解析, 直接保存或打印)。
        """
        return self._invoke(["--critical-path", dsl_program])

    def run_human_to_file(self, dsl_program: str, output_path: Path) -> Path:
        """运行 simulator Gantt 模式, 输出保存到文件。"""
        text = self.run_human(dsl_program)
        output_path.write_text(text, encoding="utf-8")
        return output_path

    # ═══════════════════════════════════════════════════════════════════════
    #  验证模式: --verify (内存容量检查)
    # ═══════════════════════════════════════════════════════════════════════

    def run_verify(self, dsl_program: str) -> Tuple[bool, str]:
        """运行 simulator --verify, 检查内存容量是否正确。

        Returns:
            (passed, report_text)
        """
        raw = self._invoke(["--verify", dsl_program])
        passed = "PASS" in raw and "FAIL" not in raw
        return passed, raw

    # ═══════════════════════════════════════════════════════════════════════
    #  内部: subprocess 调用
    # ═══════════════════════════════════════════════════════════════════════

    def _invoke(self, args: List[str]) -> str:
        """调用 simulator.py 并返回 stdout。

        Windows 环境自动设置 PYTHONIOENCODING=utf-8 以处理 µ 字符。
        """
        cmd = [self.python_exe, str(self.simulator_path)] + args

        env = os.environ.copy()
        if self._needs_utf8_env:
            env["PYTHONIOENCODING"] = "utf-8"

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=self.timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Simulator timed out after {self.timeout}s.\n"
                f"DSL program may be too complex (large loop expansion?)."
            )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(
                f"Simulator exited with code {result.returncode}.\n"
                f"stderr: {stderr[:500]}\n"
                f"cmd: {' '.join(cmd)[:200]}"
            )

        return result.stdout

    # ═══════════════════════════════════════════════════════════════════════
    #  DSL 生成 (当前为 stub, 后续实现)
    # ═══════════════════════════════════════════════════════════════════════

    def generate_dsl_from_kernel(self, kernel_code: str) -> str:
        """从 Triton kernel 提取 DSL 程序。

        TODO: 这是当前最大的 gap —— 目前没有自动化的 kernel→DSL 转换器。
        可能的实现路径:
          1. 复用 /triton-plan skill 的逻辑 (NL/PyTorch → DSL)
          2. 通过编译器提取 HIVMIR 后转换为 DSL
          3. 由 LLM 理解 kernel 代码后生成 DSL

        当前返回占位符, 方便测试其他组件。
        """
        raise NotImplementedError(
            "Kernel → DSL conversion is not yet implemented.\n"
            "Workaround: manually provide DSL program via run_llm(dsl_string).\n"
            "See /triton-plan skill for partial NL→DSL conversion logic."
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """用 example_output 中的文件验证解析器。"""
    from pathlib import Path

    example_dir = Path(__file__).resolve().parent.parent / "example_output"
    test_files = [
        "01_vector_add_saturated.txt",
        "04_matrix_pipeline_parallel.txt",
    ]

    parser = SimulatorOutputParser()

    for fname in test_files:
        fpath = example_dir / fname
        if not fpath.exists():
            print(f"[SKIP] {fname} not found")
            continue

        # 尝试 UTF-8, 回退到 GBK (Windows 默认编码)
        try:
            text = fpath.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = fpath.read_text(encoding="gbk")
        result = parser.parse(text)

        print(f"\n{'='*60}")
        print(f"Test: {fname}")
        print(f"  total_ns:      {result.total_ns:.2f}")
        print(f"  num_ops:       {result.num_ops}")
        print(f"  execution_mode: {result.execution_mode}")
        print(f"  ops parsed:    {len(result.ops)}")
        print(f"  engines:       {list(result.engine_utilization.keys())}")
        print(f"  parallel_pairs: {len(result.parallel_pairs)}")
        print(f"  critical_path:  {len(result.critical_path)} ops "
              f"({result.critical_path_length_ns:.1f} ns, "
              f"{result.critical_path_fraction:.0%})")

        bottleneck = result.get_bottleneck_op()
        if bottleneck:
            print(f"  bottleneck:    op{bottleneck.op_id} ({bottleneck.op_type}) "
                  f"time_ratio={bottleneck.time_ratio:.2%} "
                  f"regime={bottleneck.regime} "
                  f"blocked_by={len(bottleneck.blocked_by)} ops")

        # 检查解析完整性
        assert result.num_ops == len(result.ops), \
            f"num_ops mismatch: {result.num_ops} != {len(result.ops)}"
        assert result.total_ns > 0, "total_ns should be > 0"

        for op in result.ops:
            assert op.engine, f"op{op.op_id} has no engine"
            assert op.size_kb > 0, f"op{op.op_id} size_kb=0"
            assert op.duration_ns > 0, f"op{op.op_id} duration_ns=0"
            assert op.regime != "unknown", f"op{op.op_id} regime unknown"

    print(f"\n{'='*60}")
    print("All parser tests passed.")


if __name__ == "__main__":
    _self_test()
