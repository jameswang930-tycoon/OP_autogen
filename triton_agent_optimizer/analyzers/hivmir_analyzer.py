#!/usr/bin/env python3
"""
HIVMIR Analyzer v2.0 — 解析真实 AscendNPU-IR MLIR HIVM Dialect
=================================================================

通过 bishengir-compile / bishengir-opt 获取 HIVM IR 并解析。

真实 HIVM dialect 语法 (CANN 8.5.1 / ascendnpu-ir 1.1.0 验证; 2026-08-18 由 CANN 9.0 统一修订):
  - alloc:   %buf = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
  - load:    hivm.hir.load ins(%gm_buf : type) outs(%ub_buf : type)
  - store:   hivm.hir.store ins(%ub_buf : type) outs(%gm_buf : type)
  - vadd:    hivm.hir.vadd ins(%a, %b : t, t) outs(%c : t)
  - vmul/vsub/vdiv/vmax/vmin/vexp: same as vadd
  - matmul:  hivm.hir.matmul ins(%A, %B : t, t) outs(%C : t) {a_transpose, block_sizes=[M,K,N]}
  - address: #hivm.address_space<gm|ub|l1>
  - function: hacc.entry, hacc.function_kind<DEVICE>

数据来源: bishengir-compile + bishengir-opt --print-ir-after-all
不需要 NPU 硬件，纯 CPU 编译。
"""

from __future__ import annotations

import json, re, sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

TBD = "待补充"

# HIVM op → engine 映射
OP_TO_ENGINE: Dict[str, str] = {
    "gm_to_ub":  "GM→UB",   "ub_to_gm":  "UB→GM",
    "gm_to_l1":  "GM→L1",   "l1_to_l0":  "L1→L0",   "l0_to_gm":  "L0→GM",
    "vadd":  "VecUnit", "vsub":  "VecUnit", "vmul":  "VecUnit",
    "vdiv":  "VecUnit", "vmax":  "VecUnit", "vmin":  "VecUnit",
    "vexp":  "VecUnit", "vlog":  "VecUnit", "vabs":  "VecUnit",
    "vrelu": "VecUnit", "vsqrt": "VecUnit", "vtanh": "VecUnit",
    "vbrc": "VecUnit", "vcvt": "VecUnit", "vmov": "VecUnit",
    "vsel": "VecUnit", "vcmp": "VecUnit", "vdul": "VecUnit",
    "vdup": "VecUnit", "vbitsel": "VecUnit", "vconv": "VecUnit",
    "matmul":    "CubeUnit", "matrixmul":  "CubeUnit",
    "mix_matmul":"CubeUnit", "mmadL1":     "CubeUnit",
    "batchMmadL1":"CubeUnit",
}

# memref shape×dtype → size_kb 计算
DTYPE_SIZES = {
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
}

# #hivm.address_space<值> → 友好区域名 (triton-ascend al.ascend_address_space 对接 hivm::AddressSpace)
ADDRESS_SPACE_MAP = {
    "ub": "UB", "cbuf": "L1", "ca": "L0A", "cb": "L0B", "cc": "L0C", "gm": "GM",
}

# 同步 op (前缀 hivm.hir., 真实数据已确认; 非计算引擎, 计入 op_id 与 simulator SET_FLAG/WAIT_FLAG/BAR 对齐)
SYNC_OPS = {"set_flag", "wait_flag", "pipe_barrier", "sync_block"}

# IR lowering 后的函数名 → op 类型
LOWERED_FUNC_MAP = {
    "load_gm_to_ubuf": "gm_to_ub",
    "load_gm_to_ubuf_1d": "gm_to_ub",
    "store_ubuf_to_gm": "ub_to_gm",
    "store_ubuf_to_gm_1d": "ub_to_gm",
    "vadd_1d": "vadd", "vmul_1d": "vmul",
}

# 提供的字段
HIVMIR_FIELDS = [
    "op_id", "op_type", "instruction", "dst", "src", "src2",
    "size_kb", "memory_region", "variable_name", "dependencies",
]

# msprof 待补充的字段
MSPROF_NEEDED = [
    "engine", "pipeline_channel", "core_id",
    "duration_ns", "start_ns", "end_ns", "time_ratio",
    "effective_bw_gb_s", "peak_bw_gb_s", "bw_utilization", "regime",
    "wait_before_start_ns",
    "total_ns", "execution_mode", "num_cores",
    "engine_utilization", "parallel_pairs", "critical_path",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HIVMIROp:
    """单个 HIVM 操作。"""
    op_id: int
    op_type: str = ""            # gm_to_ub, vadd, matmul, ...
    engine: str = TBD            # 从 op_type 派生
    instruction: str = ""        # 操作文本
    dst: str = ""                # 目标 SSA 值 (%buf_z)
    src: str = ""                # 源 SSA 值 (%buf_x)
    src2: str = ""               # 第二源 (%buf_y)
    size_kb: float = 0.0         # 数据大小
    memory_region: str = ""      # GM / UB / L1
    variable_name: str = ""      # SSA 变量名
    dependencies: List[dict] = field(default_factory=list)
    # 额外
    dtype: str = ""              # f16, f32, ...
    attrs: dict = field(default_factory=dict)  # {a_transpose, block_sizes, ...}


@dataclass
class BufferInfo:
    """Buffer 信息。"""
    name: str
    region: str = "unknown"
    size_kb: float = 0.0
    producers: List[int] = field(default_factory=list)
    consumers: List[int] = field(default_factory=list)


@dataclass
class HIVMIRReport:
    """HIVM 完整解析报告。"""
    num_ops: int = 0
    ops: List[HIVMIROp] = field(default_factory=list)
    buffers: Dict[str, BufferInfo] = field(default_factory=dict)
    raw_deps: List[dict] = field(default_factory=list)   # RAW
    war_deps: List[dict] = field(default_factory=list)   # WAR
    waw_deps: List[dict] = field(default_factory=list)   # WAW
    kernel_name: str = ""
    source_path: str = ""
    generated_at: str = ""
    # 兼容旧接口 (msprof 补充)
    total_ns: str = TBD
    execution_mode: str = TBD
    num_cores: str = TBD


# ═══════════════════════════════════════════════════════════════════════════════
#  MLIR HIVM Dialect Parser
# ═══════════════════════════════════════════════════════════════════════════════

class HIVMIRParser:
    """解析真实 HIVM dialect MLIR 文本。

    支持两种格式:
      格式 A (输入 HIVM IR): hivm.hir.load/vadd/store/matmul + memref.alloc + #hivm.address_space
      格式 B (lowered IR):   call @load_gm_to_ubuf_1d_half + memref.cast
    """

    def __init__(self):
        self.ops: List[HIVMIROp] = []
        self.buffers: Dict[str, BufferInfo] = {}
        self._last_write: Dict[str, int] = {}
        self._last_read: Dict[str, int] = {}
        self._alloc_sizes: Dict[str, float] = {}  # SSA名 → size_kb

    def parse(self, mlir_text: str) -> HIVMIRReport:
        self.ops = []
        self.buffers = {}
        self._last_write = {}
        self._last_read = {}
        self._alloc_sizes = {}

        # 尝试格式 A
        hir_ops = re.findall(r'hivm\.hir\.\w+', mlir_text)
        has_hivm = len(hir_ops) > 0

        if has_hivm:
            self._parse_format_a(mlir_text)
        else:
            self._parse_format_b(mlir_text)

        self._analyze_dependencies()
        return self._build_report()

    # ── 格式 A: 输入 HIVM IR ──────────────────────────────────────────────

    def _parse_format_a(self, text: str):
        """解析 hivm.hir.* + memref.alloc + #hivm.address_space<>"""
        lines = text.strip().split("\n")
        op_id = 0
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("//"):
                i += 1
                continue

            # memref.alloc: %buf = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
            alloc = self._try_parse_alloc(line)
            if alloc:
                ssa_name, size_kb, region, dtype = alloc
                self._alloc_sizes[ssa_name] = size_kb
                self.buffers[ssa_name] = BufferInfo(
                    name=ssa_name, region=region, size_kb=size_kb)
                i += 1
                continue

            # sync op (非 ins/outs 语法, 计入 op_id 使与 simulator SET_FLAG/WAIT_FLAG/BAR 对齐)
            sync = re.match(r'hivm\.hir\.(set_flag|wait_flag|pipe_barrier|sync_block)\b', line)
            if sync:
                op_name = sync.group(1)
                op = HIVMIROp(
                    op_id=op_id, op_type=op_name, engine="Sync",
                    instruction=line,
                    dst="", src="", src2="",
                    size_kb=0.0, memory_region="",
                    variable_name=f"sync_{op_id}",
                    dtype="", attrs={},
                )
                self.ops.append(op)
                op_id += 1
                i += 1
                continue

            # hivm.hir.* ops (可能跨多行)
            if line.startswith("hivm.hir."):
                # 检查是否已经是一个完整的单行 op (结尾可能是 ) 或 } 属性)
                if "ins(" in line and "outs(" in line:
                    full_op = line
                    op = self._try_parse_hir_op(full_op, op_id)
                    if op:
                        self._update_buffer_usage(op, op_id)
                        self.ops.append(op)
                        op_id += 1
                    i += 1
                    continue

                # 跨多行 op: 收集直到完整
                op_lines = [line]
                j = i + 1
                while j < len(lines):
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    op_lines.append(next_line)
                    full = " ".join(op_lines)
                    if "outs(" in full and (full.rstrip().endswith(")")
                                            or full.rstrip().endswith("}")):
                        break
                    j += 1

                full_op = " ".join(op_lines)
                op = self._try_parse_hir_op(full_op, op_id)
                if op:
                    self._update_buffer_usage(op, op_id)
                    self.ops.append(op)
                    op_id += 1
                i = j + 1
                continue

            i += 1

    def _try_parse_alloc(self, line: str) -> Optional[Tuple[str, float, str, str]]:
        """%buf = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
        支持 1D/2D/3D (memref<256x256xf32,...>); 动态维 (memref<?x?xf32,...>) → size_kb=0 未知;
        address_space 值映射为友好区域名 (cbuf→L1, cc→L0C, gm→GM, ...)。"""
        m = re.match(
            r'(%\w+)\s*=\s*memref\.alloc\(\)\s*:\s*'
            r'memref<([0-9?]+(?:x[0-9?]+)*)\s*x\s*(\w+)\s*,\s*'
            r'(?:strided<[^>]*>\s*,\s*)?#hivm\.address_space<(\w+)>', line)
        if not m:
            return None
        ssa = m.group(1)
        dims_str = m.group(2)
        dtype = m.group(3)
        region_raw = m.group(4)
        region = ADDRESS_SPACE_MAP.get(region_raw, region_raw)
        if "?" in dims_str:
            size_kb = 0.0            # 动态维: 编译期尺寸未知
        else:
            size_kb = 1.0
            for d in dims_str.split("x"):
                size_kb *= int(d)
            size_kb = size_kb * DTYPE_SIZES.get(dtype, 2) / 1024.0
        return (ssa, size_kb, region, dtype)

    def _try_parse_hir_op(self, line: str, op_id: int) -> Optional[HIVMIROp]:
        """hivm.hir.OP ins(...) outs(...) [{attrs}]"""
        m = re.match(
            r'hivm\.hir\.(\w+)\s+ins\((.+?)\)\s*outs\((.+?)\)\s*(\{.*\})?\s*$', line)
        if not m:
            return None

        op_name = m.group(1)        # load / store / vadd / matmul / ...
        ins_str = m.group(2).strip()
        outs_str = m.group(3).strip()
        attrs_str = m.group(4) or ""

        # 解析 outs → dst
        dst = self._parse_first_operand(outs_str)
        # 解析 ins → src, src2
        srcs = self._parse_operands(ins_str)
        src = srcs[0] if len(srcs) > 0 else ""
        src2 = srcs[1] if len(srcs) > 1 else ""

        # op type 映射
        if op_name == "load":
            op_type = "gm_to_ub"
        elif op_name == "store":
            op_type = "ub_to_gm"
        elif op_name in ("matmul", "mix_matmul", "batchMmadL1", "mmadL1"):
            op_type = op_name
        else:
            op_type = op_name   # vadd, vmul, vsub, etc.

        engine = OP_TO_ENGINE.get(op_type, TBD)

        # size_kb: 优先 alloc_sizes, 否则从 op 的 ins/outs 类型签名提取
        # (真实 HIVM 尺寸在 tensor<MxNxdtype>/memref<MxNxdtype> 里, 见 _size_from_type)
        size_kb = self._alloc_sizes.get(dst, self._alloc_sizes.get(src, 0.0))
        if not size_kb:
            size_kb = self._size_from_type(outs_str) or self._size_from_type(ins_str)

        # memory_region: 从 buffers 查询 (alloc 时存储)
        mem_region = ""
        if dst in self.buffers:
            mem_region = self.buffers[dst].region
        if not mem_region and src in self.buffers:
            mem_region = self.buffers[src].region
        if not mem_region:
            mem_region = self._get_region(dst) or self._get_region(src)

        # 解析 attrs
        attrs = {}
        if attrs_str:
            attrs = self._parse_attrs(attrs_str)

        return HIVMIROp(
            op_id=op_id, op_type=op_type, engine=engine,
            instruction=f"hivm.hir.{op_name}({dst}, {src}{', '+src2 if src2 else ''})",
            dst=dst, src=src, src2=src2,
            size_kb=size_kb, memory_region=mem_region,
            variable_name=dst if dst else src,
            dtype="f16", attrs=attrs,
        )

    @staticmethod
    def _size_from_type(type_str: str) -> float:
        """从 op 的 ins/outs 类型串提取 size_kb。

        真实 HIVM (AscendNPU-IR 文档确认) 尺寸在 op 类型签名里:
          tensor<16x32xf32> / memref<256x256xf16, #hivm.address_space<ub>>
        含 ? 动态维 → 返回 0 (未知)。
        """
        m = re.search(r'(?:tensor|memref)<([0-9x?]+)x(\w+)', type_str or "")
        if not m:
            return 0.0
        dims_str, dtype = m.group(1), m.group(2)
        if "?" in dims_str:
            return 0.0
        total = 1
        for d in dims_str.split("x"):
            total *= int(d)
        return total * DTYPE_SIZES.get(dtype, 2) / 1024.0

    def _parse_first_operand(self, s: str) -> str:
        """从 '(%buf : type)' 或 '%buf, ... : type, type' 提取第一个操作数"""
        s = s.strip().lstrip("(").rstrip(")")
        # 找到第一个 %name
        m = re.match(r'(%\w+)', s)
        return m.group(1) if m else ""

    def _parse_operands(self, s: str) -> List[str]:
        """从 ins 区域提取所有 %name 操作数"""
        s = s.strip().lstrip("(").rstrip(")")
        # 移除类型标注部分 (everything after the last memref)
        # 找到所有 %name
        return re.findall(r'(%\w+)', s)

    def _parse_attrs(self, s: str) -> dict:
        """解析 {a_transpose, block_sizes=[16,16,16]}"""
        attrs = {}
        for key in ["a_transpose", "b_transpose"]:
            if key in s:
                attrs[key] = True
        m = re.search(r'block_sizes\s*=\s*\[([^\]]+)\]', s)
        if m:
            attrs["block_sizes"] = [int(x.strip()) for x in m.group(1).split(",")]
        return attrs

    # ── 格式 B: lowered IR ────────────────────────────────────────────────

    def _parse_format_b(self, text: str):
        """解析 call @load_gm_to_ubuf_* 等 lowered 函数调用"""
        op_id = 0

        # 先解析 alloc (lowered IR 中也有 memref.alloc)
        for line in text.split("\n"):
            line = line.strip()
            alloc = self._try_parse_alloc(line)
            if alloc:
                ssa_name, size_kb, region, dtype = alloc
                self._alloc_sizes[ssa_name] = size_kb
                self.buffers[ssa_name] = BufferInfo(
                    name=ssa_name, region=region, size_kb=size_kb)

        # 解析 call @op_name + memref.cast
        for line in text.split("\n"):
            line = line.strip()
            op = self._try_parse_lowered_call(line, op_id)
            if op:
                self._update_buffer_usage(op, op_id)
                self.ops.append(op)
                op_id += 1

    def _try_parse_lowered_call(self, line: str, op_id: int) -> Optional[HIVMIROp]:
        """call @load_gm_to_ubuf_1d_half(%cast, %cast_2, %c0_i32, ...)"""
        m = re.match(r'call\s+@(\w+)\((.+?)\)\s*:', line)
        if not m:
            return None

        func_name = m.group(1)
        args_str = m.group(2)

        op_type = "unknown"
        for key, val in LOWERED_FUNC_MAP.items():
            if key in func_name:
                op_type = val
                break

        if op_type == "unknown":
            return None

        # 提取第一个 memref (dst) 和第二个 memref (src)
        args = [a.strip() for a in args_str.split(",")]
        # 提取 %name 从 args
        names = []
        for a in args:
            nm = re.search(r'(%\w+)', a)
            if nm:
                names.append(nm.group(1))

        src = names[0] if len(names) > 0 else ""
        dst = names[1] if len(names) > 1 else ""

        engine = OP_TO_ENGINE.get(op_type, TBD)
        size_kb = self._alloc_sizes.get(dst, self._alloc_sizes.get(src, 0.0))
        mem_region = self._get_region(dst) or self._get_region(src)

        return HIVMIROp(
            op_id=op_id, op_type=op_type, engine=engine,
            instruction=f"call @{func_name}({src}, {dst})",
            dst=dst, src=src,
            size_kb=size_kb, memory_region=mem_region,
            variable_name=dst if dst else src,
        )

    # ── 依赖分析 ─────────────────────────────────────────────────────────

    def _update_buffer_usage(self, op: HIVMIROp, op_id: int):
        for buf_name, is_dst in [(op.dst, True), (op.src, False), (op.src2, False)]:
            if not buf_name or buf_name == "unknown":
                continue
            if buf_name not in self.buffers:
                self.buffers[buf_name] = BufferInfo(name=buf_name)
            if is_dst:
                self.buffers[buf_name].producers.append(op_id)
            else:
                self.buffers[buf_name].consumers.append(op_id)

    def _analyze_dependencies(self):
        for op in self.ops:
            # RAW: src 被之前的 op 写过
            for buf_name in [op.src, op.src2]:
                if buf_name and buf_name in self._last_write:
                    if self._last_write[buf_name] != op.op_id:
                        op.dependencies.append(
                            {"from_op_id": self._last_write[buf_name], "type": "RAW",
                             "buffer": buf_name})
            # WAR: dst 被之前的 op 读过
            if op.dst and op.dst in self._last_read:
                if self._last_read[op.dst] != op.op_id:
                    op.dependencies.append(
                        {"from_op_id": self._last_read[op.dst], "type": "WAR",
                         "buffer": op.dst})
            # WAW: dst 被之前的 op 写过
            if op.dst and op.dst in self._last_write:
                if self._last_write[op.dst] != op.op_id:
                    op.dependencies.append(
                        {"from_op_id": self._last_write[op.dst], "type": "WAW",
                         "buffer": op.dst})

            # 更新
            if op.dst:
                self._last_write[op.dst] = op.op_id
            for buf_name in [op.src, op.src2]:
                if buf_name:
                    self._last_read[buf_name] = op.op_id

    # ── Build Report ──────────────────────────────────────────────────────

    def _build_report(self) -> HIVMIRReport:
        report = HIVMIRReport(
            num_ops=len(self.ops),
            ops=self.ops,
            buffers=self.buffers,
            generated_at=datetime.now().isoformat(),
        )
        for op in self.ops:
            for d in op.dependencies:
                entry = {
                    "from_op": d["from_op_id"], "to_op": op.op_id,
                    "type": d["type"],
                    "from_inst": (self.ops[d["from_op_id"]].instruction
                                  if d["from_op_id"] < len(self.ops) else "?"),
                    "to_inst": op.instruction,
                    "buffer": d.get("buffer", "?"),
                }
                if d["type"] == "RAW":
                    report.raw_deps.append(entry)
                elif d["type"] == "WAR":
                    report.war_deps.append(entry)
                else:
                    report.waw_deps.append(entry)
        return report

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_region(name: str) -> str:
        if not name:
            return "unknown"
        # 从 address_space 推断(在alloc时已存储), 或从变量名前缀推断
        if name.startswith("%gm"):
            return "GM"
        if name.startswith("%ub"):
            return "UB"
        if name.startswith("%l1"):
            return "L1"
        if name.startswith("%l0"):
            return "L0"
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  HIVMIRAnalyzer — 主类
# ═══════════════════════════════════════════════════════════════════════════════

class HIVMIRAnalyzer:
    """HIVM 分析器。

    输入: HIVM MLIR 文本 (从编译器或手写 MLIR 文件)
    输出: HIVMIRReport (结构化 op 列表 + 依赖)

    Usage:
        analyzer = HIVMIRAnalyzer()
        report = analyzer.analyze(mlir_text)
        json_out = analyzer.to_dict(report)
    """

    def __init__(self):
        self._parser = HIVMIRParser()

    def analyze(self, mlir_text: str, source_path: str = "") -> HIVMIRReport:
        report = self._parser.parse(mlir_text)
        report.source_path = source_path
        return report

    def analyze_file(self, mlir_path: Path) -> HIVMIRReport:
        mlir_path = Path(mlir_path)
        text = mlir_path.read_text(encoding="utf-8")
        return self.analyze(text, str(mlir_path))

    def to_dict(self, report: HIVMIRReport) -> dict:
        """转为 JSON-serializable dict（对齐项目 29 字段 schema）。"""
        return {
            "meta": {
                "source": "hivmir_analyzer_v2",
                "generated_at": report.generated_at,
                "source_path": report.source_path,
                "hivmir_fields_provided": HIVMIR_FIELDS,
                "msprof_fields_pending": MSPROF_NEEDED,
            },
            "execution_summary": {
                "total_ns": report.total_ns,
                "num_ops": report.num_ops,
                "execution_mode": report.execution_mode,
                "num_cores": report.num_cores,
            },
            "per_op_statistics": [
                {
                    "op_id": op.op_id,
                    "op_type": op.op_type,
                    "engine": op.engine,
                    "instruction": op.instruction,
                    "dst": op.dst, "src": op.src, "src2": op.src2,
                    "size_kb": op.size_kb,
                    "memory_region": op.memory_region,
                    "variable_name": op.variable_name,
                    "dtype": op.dtype,
                    "attrs": op.attrs,
                    "dependencies": op.dependencies,
                    # msprof 待补字段
                    "duration_ns": TBD, "start_ns": TBD, "end_ns": TBD,
                    "time_ratio": TBD,
                    "effective_bw_gb_s": TBD, "peak_bw_gb_s": TBD,
                    "bw_utilization": TBD, "regime": TBD,
                    "wait_before_start_ns": TBD,
                    "pipeline_channel": TBD, "core_id": TBD,
                }
                for op in report.ops
            ],
            "buffers": {
                name: {"region": b.region, "size_kb": b.size_kb,
                       "producers": b.producers, "consumers": b.consumers}
                for name, b in report.buffers.items()
            },
            "dependencies_summary": {
                "total": len(report.raw_deps) + len(report.war_deps) + len(report.waw_deps),
                "raw": report.raw_deps, "war": report.war_deps, "waw": report.waw_deps,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 60)
    print("HIVMIRAnalyzer v2.0 — Self-Test")
    print("=" * 60)

    analyzer = HIVMIRAnalyzer()

    # Test 1: Input HIVM IR (格式 A)
    mlir_a = """
func.func @add_kernel(%x: memref<1024xf16, #hivm.address_space<gm>>,
                       %y: memref<1024xf16, #hivm.address_space<gm>>,
                       %z: memref<1024xf16, #hivm.address_space<gm>>)
    attributes {hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {

    %buf_x = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_y = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_z = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>

    hivm.hir.load ins(%x : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_x : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%y : memref<1024xf16, #hivm.address_space<gm>>)
                 outs(%buf_y : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%buf_x, %buf_y : memref<1024xf16, #hivm.address_space<ub>>,
                                       memref<1024xf16, #hivm.address_space<ub>>)
                 outs(%buf_z : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%buf_z : memref<1024xf16, #hivm.address_space<ub>>)
                  outs(%z : memref<1024xf16, #hivm.address_space<gm>>)
    return
}
"""
    r1 = analyzer.analyze(mlir_a)
    print(f"\nTest 1 (格式A): {r1.num_ops} ops, {len(r1.buffers)} buffers")
    for op in r1.ops:
        deps = ", ".join(f"op{d['from_op_id']}({d['type']})" for d in op.dependencies)
        print(f"  op{op.op_id}: {op.op_type:10s} dst={op.dst:10s} src={op.src:10s} "
              f"size={op.size_kb:.1f}KB region={op.memory_region} deps=[{deps}]")
    assert r1.num_ops == 4, f"Expected 4 ops (2 loads + 1 vadd + 1 store), got {r1.num_ops}"
    assert r1.ops[0].op_type == "gm_to_ub"
    assert r1.ops[1].op_type == "gm_to_ub"
    assert r1.ops[2].op_type == "vadd"
    assert r1.ops[3].op_type == "ub_to_gm"
    # 验证依赖: load→vadd(RAW), vadd→store(RAW)
    assert len(r1.raw_deps) >= 2, f"Expected >=2 RAW deps, got {len(r1.raw_deps)}"
    print("  PASS")

    # Test 2: Lowered IR (格式 B)
    mlir_b = """
    %alloc = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %alloc_0 = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %alloc_1 = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %cast = memref.cast %arg0 : memref<1024xf16, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_2 = memref.cast %alloc : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_half(%cast, %cast_2, %c0_i32, %cst, %c0) : (...)
    %cast_3 = memref.cast %arg1 : memref<1024xf16, #hivm.address_space<gm>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<gm>>
    %cast_4 = memref.cast %alloc_0 : memref<1024xf16, #hivm.address_space<ub>> to memref<?xf16, strided<[?], offset: ?>, #hivm.address_space<ub>>
    call @load_gm_to_ubuf_1d_half(%cast_3, %cast_4, %c0_i32, %cst, %c0) : (...)
    call @vadd_1d_half(%cast_2, %cast_4, %cast_7, %cast_8) : (...)
    call @store_ubuf_to_gm_1d_half(%cast_9, %cast_10, %c0_i32) : (...)
"""
    r2 = analyzer.analyze(mlir_b)
    print(f"\nTest 2 (格式B): {r2.num_ops} ops")
    for op in r2.ops:
        print(f"  op{op.op_id}: {op.op_type:10s} size={op.size_kb:.1f}KB")
    assert r2.num_ops == 4, f"Expected 4 ops, got {r2.num_ops}"
    print("  PASS")

    # Test 3: JSON 导出
    d = analyzer.to_dict(r1)
    assert d["meta"]["hivmir_fields_provided"] == HIVMIR_FIELDS
    for opd in d["per_op_statistics"]:
        assert opd["instruction"] != TBD
        assert isinstance(opd["size_kb"], (int, float))
    print(f"\nTest 3 (JSON): {len(json.dumps(d))} chars, all assertions passed")

    # Test 4: 复杂融合算子
    mlir_c = """
    %buf_x = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_y = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_w = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_t = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    %buf_z = memref.alloc() : memref<1024xf16, #hivm.address_space<ub>>
    hivm.hir.load ins(%x : memref<1024xf16, #hivm.address_space<gm>>) outs(%buf_x : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%y : memref<1024xf16, #hivm.address_space<gm>>) outs(%buf_y : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.load ins(%w : memref<1024xf16, #hivm.address_space<gm>>) outs(%buf_w : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.vadd ins(%buf_x, %buf_y : memref<1024xf16, #hivm.address_space<ub>>, memref<1024xf16, #hivm.address_space<ub>>) outs(%buf_t : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.vmul ins(%buf_t, %buf_w : memref<1024xf16, #hivm.address_space<ub>>, memref<1024xf16, #hivm.address_space<ub>>) outs(%buf_z : memref<1024xf16, #hivm.address_space<ub>>)
    hivm.hir.store ins(%buf_z : memref<1024xf16, #hivm.address_space<ub>>) outs(%z : memref<1024xf16, #hivm.address_space<gm>>)
"""
    r3 = analyzer.analyze(mlir_c)
    print(f"\nTest 4 (fused add-mul): {r3.num_ops} ops")
    assert r3.num_ops == 6, f"Expected 6 ops, got {r3.num_ops}"
    expected = ["gm_to_ub", "gm_to_ub", "gm_to_ub", "vadd", "vmul", "ub_to_gm"]
    actual_types = [op.op_type for op in r3.ops]
    assert actual_types == expected, f"Expected {expected}, got {actual_types}"
    # 验证依赖
    raw_chain = [d for d in r3.raw_deps if d["type"] == "RAW"]
    assert len(raw_chain) > 0, "Should have RAW dependencies"
    print(f"  RAW deps: {len(r3.raw_deps)}, WAR: {len(r3.war_deps)}, WAW: {len(r3.waw_deps)}")
    print("  PASS")

    # Test 5: 真实 matmul 格式 (2026-08-03 服务器 hivm_try.txt 确认):
    #   cube op = hivm.hir.mmadL1, sync = hivm.hir.set_flag/wait_flag/pipe_barrier,
    #   2D alloc (memref<128x128xf16, cbuf>), address_space={cbuf,cc,gm}, 动态维 <?x?>
    mlir_d = """
func.func @matmul_kernel(%A: memref<256x256xf16, #hivm.address_space<gm>>,
                          %B: memref<256x256xf16, #hivm.address_space<gm>>,
                          %C: memref<256x256xf32, #hivm.address_space<gm>>) {
    %l1_a = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
    %l1_b = memref.alloc() : memref<128x128xf16, #hivm.address_space<cbuf>>
    %l0c  = memref.alloc() : memref<128x128xf32, #hivm.address_space<cc>>
    %dyn  = memref.alloc() : memref<?x?xf16, #hivm.address_space<cbuf>>
    hivm.hir.load ins(%A : memref<256x256xf16, #hivm.address_space<gm>>) outs(%l1_a : memref<128x128xf16, #hivm.address_space<cbuf>>)
    hivm.hir.load ins(%B : memref<256x256xf16, #hivm.address_space<gm>>) outs(%l1_b : memref<128x128xf16, #hivm.address_space<cbuf>>)
    hivm.hir.set_flag [set_pipe = #hivm.pipe<mte2>, wait_pipe = #hivm.pipe<cube>, flag_id = 0 : i32]
    hivm.hir.mmadL1 ins(%l1_a, %l1_b, %c0_i1, %m, %k, %n : memref<128x128xf16, #hivm.address_space<cbuf>>, memref<128x128xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%l0c : memref<128x128xf32, #hivm.address_space<cc>>) {lhs_m = 128 : i32, rhs_n = 128 : i32, l0b_k = 128 : i32}
    hivm.hir.wait_flag [set_pipe = #hivm.pipe<cube>, wait_pipe = #hivm.pipe<mte3>, flag_id = 0 : i32]
    hivm.hir.store ins(%l0c : memref<128x128xf32, #hivm.address_space<cc>>) outs(%C : memref<256x256xf32, #hivm.address_space<gm>>)
    hivm.hir.pipe_barrier [pipe = #hivm.pipe<mte3>]
    return
}
"""
    r5 = analyzer.analyze(mlir_d)
    print(f"\nTest 5 (真实 matmul 格式): {r5.num_ops} ops, {len(r5.buffers)} buffers")
    for op in r5.ops:
        print(f"  op{op.op_id}: {op.op_type:12s} engine={op.engine:8s} size={op.size_kb:.1f}KB region={op.memory_region}")
    # 预期 7 ops: 2 load + 1 set_flag + 1 mmadL1 + 1 wait_flag + 1 store + 1 pipe_barrier
    assert r5.num_ops == 7, f"Expected 7 ops, got {r5.num_ops}"
    types = [op.op_type for op in r5.ops]
    assert types == ["gm_to_ub", "gm_to_ub", "set_flag", "mmadL1",
                     "wait_flag", "ub_to_gm", "pipe_barrier"], f"got {types}"
    mmad = [op for op in r5.ops if op.op_type == "mmadL1"][0]
    assert mmad.engine == "CubeUnit", f"mmadL1 engine={mmad.engine}"
    assert mmad.size_kb == 64.0, f"mmadL1 size={mmad.size_kb}"  # dst %l0c = 128x128xf32 = 64KB
    # 2D alloc + 区域映射
    assert r5.buffers["%l1_a"].region == "L1", r5.buffers["%l1_a"].region
    assert r5.buffers["%l0c"].region == "L0C", r5.buffers["%l0c"].region
    assert r5.buffers["%l1_a"].size_kb == 32.0, r5.buffers["%l1_a"].size_kb
    # 动态维 alloc → size_kb 0 (未知), 但仍解析出区域
    assert r5.buffers["%dyn"].size_kb == 0.0 and r5.buffers["%dyn"].region == "L1"
    # load op 的区域来自 dst alloc (L1)
    assert r5.ops[0].memory_region == "L1", r5.ops[0].memory_region
    print("  PASS")

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED — HIVMIRAnalyzer v2.0")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
