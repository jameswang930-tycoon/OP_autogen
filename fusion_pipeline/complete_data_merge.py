#!/usr/bin/env python3
"""
完整的数据合并脚本：整合 HIVMIR 和 msprof op simulator

功能：
1. 解析 simulator --llm --critical-path 输出（获取时序信息）
2. 解析 HIVMIR 文本（获取变量名、依赖关系、数据大小）
3. 合并两个数据源
4. 生成完整报告和可视化

输出字段：
- Op: 操作序号
- 操作类型: gm_to_ub, vadd 等
- 引擎: GM→UB, VecUnit 等
- SIZE: 搬运数据大小（KB）
- Times: 执行时间（ns）
- BW util: 带宽利用率
- Regime: floor/ramp/saturated/flat
- waitFor: 依赖的操作
- 依赖类型: RAW/WAR/WAW
- 时间占比: 百分比
"""

import re
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd

# ============================================================================
#  Part 1: Simulator 输出解析器
# ============================================================================

@dataclass
class SimulatorOp:
    """Simulator 操作信息"""
    op_id: int
    op_type: str  # gm_to_ub, vadd, etc.
    engine: str
    dst: str
    src: str
    src2: str = ""
    size_kb: float = 0.0
    duration_ns: float = 0.0
    start_ns: float = 0.0
    end_ns: float = 0.0
    effective_bw: float = 0.0
    peak_bw: float = 0.0
    bw_utilization: float = 0.0
    regime: str = "unknown"
    wait_before_start: float = 0.0
    blocked_by: str = ""
    time_ratio: float = 0.0


class SimulatorOutputParser:
    """解析 simulator --llm --critical-path 输出"""

    def __init__(self):
        self.ops: List[SimulatorOp] = []
        self.total_ns: float = 0.0
        self.execution_mode: str = ""
        self.engine_utilization: Dict[str, Tuple[float, float]] = {}
        self.critical_path: List[int] = []

    def parse(self, simulator_output: str) -> List[SimulatorOp]:
        """解析完整的 simulator 输出"""
        self.ops = []

        # 解析 EXECUTION SUMMARY
        self._parse_execution_summary(simulator_output)

        # 解析 PER-OP STATISTICS
        self._parse_per_op_statistics(simulator_output)

        # 解析 ENGINE UTILIZATION
        self._parse_engine_utilization(simulator_output)

        # 解析 CRITICAL PATH
        self._parse_critical_path(simulator_output)

        return self.ops

    def _parse_execution_summary(self, text: str):
        """解析 EXECUTION SUMMARY 部分"""
        match = re.search(r'total_ns:\s*([\d.]+)', text)
        if match:
            self.total_ns = float(match.group(1))

        match = re.search(r'execution_mode:\s*(\w+)', text)
        if match:
            self.execution_mode = match.group(1)

    def _parse_per_op_statistics(self, text: str):
        """解析 PER-OP STATISTICS 部分"""
        # 分割每个 operation
        op_pattern = re.compile(r'(op\d+):\s+(\w+)\(([^)]+)\)')
        op_matches = op_pattern.finditer(text)

        for match in op_matches:
            op_id_str = match.group(1)
            op_type = match.group(2)
            operands = match.group(3)

            op_id = int(op_id_str.replace('op', ''))

            # 解析操作数
            parts = [p.strip() for p in operands.split(',')]
            dst = parts[0] if len(parts) > 0 else ""
            src = parts[1] if len(parts) > 1 else ""
            src2 = parts[2] if len(parts) > 2 else ""

            # 解析 size
            size_match = re.search(r'size:\s*([\d.]+)\s*KB', text[match.end():match.end()+500])
            size_kb = float(size_match.group(1)) if size_match else 64.0

            # 解析时间
            cycles_match = re.search(r'cycles_ns:\s*\[([\d.]+)\.\.([\d.]+)\]', text[match.end():match.end()+500])
            start_ns = float(cycles_match.group(1)) if cycles_match else 0.0
            end_ns = float(cycles_match.group(2)) if cycles_match else 0.0

            duration_match = re.search(r'duration_ns=([\d.]+)', text[match.end():match.end()+500])
            duration_ns = float(duration_match.group(1)) if duration_match else 0.0

            # 解析引擎
            engine_match = re.search(r'engine:\s*([^\n]+)', text[match.end():match.end()+500])
            engine = engine_match.group(1).strip() if engine_match else ""

            # 解析带宽
            bw_match = re.search(r'effective=([\d.]+)\s*GB/s\s+peak=([\d.]+)\s*GB/s\s+utilization=([\d.]+)%',
                               text[match.end():match.end()+500])
            effective_bw = float(bw_match.group(1)) if bw_match else 0.0
            peak_bw = float(bw_match.group(2)) if bw_match else 0.0
            bw_utilization = float(bw_match.group(3)) / 100.0 if bw_match else 0.0

            # 解析 regime
            regime_match = re.search(r'regime=(\w+)', text[match.end():match.end()+500])
            regime = regime_match.group(1) if regime_match else "unknown"

            # 解析 wait time
            wait_match = re.search(r'wait_ns_before_start:\s*([\d.]+)', text[match.end():match.end()+500])
            wait_before_start = float(wait_match.group(1)) if wait_match else 0.0

            # 解析 blocked_by
            blocked_match = re.search(r'blocked_by:\s*([^\n]+)', text[match.end():match.end()+500])
            blocked_by = blocked_match.group(1).strip() if blocked_match else "none"

            # 计算时间占比
            time_ratio = (duration_ns / self.total_ns * 100) if self.total_ns > 0 else 0.0

            op = SimulatorOp(
                op_id=op_id,
                op_type=op_type,
                engine=engine,
                dst=dst,
                src=src,
                src2=src2,
                size_kb=size_kb,
                duration_ns=duration_ns,
                start_ns=start_ns,
                end_ns=end_ns,
                effective_bw=effective_bw,
                peak_bw=peak_bw,
                bw_utilization=bw_utilization,
                regime=regime,
                wait_before_start=wait_before_start,
                blocked_by=blocked_by,
                time_ratio=time_ratio,
            )

            self.ops.append(op)

    def _parse_engine_utilization(self, text: str):
        """解析 ENGINE UTILIZATION 部分"""
        pattern = re.compile(r'(\w+→?\w*):\s*busy=([\d.]+)/([\d.]+)\s*ns\s+utilization=([\d.]+)%')
        for match in pattern.finditer(text):
            engine_name = match.group(1)
            busy_ns = float(match.group(2))
            total_ns = float(match.group(3))
            self.engine_utilization[engine_name] = (busy_ns, total_ns)

    def _parse_critical_path(self, text: str):
        """解析 CRITICAL PATH 部分"""
        path_match = re.search(r'path:\s*(op\d+(?:\s*->\s*op\d+)*)', text)
        if path_match:
            path_str = path_match.group(1)
            op_ids = re.findall(r'op(\d+)', path_str)
            self.critical_path = [int(op_id) for op_id in op_ids]


# ============================================================================
#  Part 2: HIVMIR 解析器（增强版）
# ============================================================================

@dataclass
class HIVMIROp:
    """HIVMIR 操作信息"""
    op_id: int
    op_type: str
    engine: str
    dst: str
    src: str
    src2: str = ""
    size_kb: float = 64.0
    size_bytes: int = 65536
    variable_name: str = ""
    address_offset: str = ""
    dependencies: List[Tuple[int, str]] = field(default_factory=list)
    line_number: int = 0
    memory_region: str = ""


class HIVMIRParser:
    """解析 HIVMIR 文本（增强版，基于 AscendNPU-IR 文档）"""

    # 操作类型到引擎的映射
    OP_TO_ENGINE = {
        'gm_to_ub': 'GM→UB',
        'ub_to_gm': 'UB→GM',
        'gm_to_l1': 'GM→L1',
        'l1_to_l0': 'L1→L0',
        'l0_to_gm': 'L0→GM',
        'vadd': 'VecUnit',
        'vsub': 'VecUnit',
        'vmul': 'VecUnit',
        'matrixmul': 'CubeUnit',
    }

    # 缓冲区前缀到内存区域的映射
    BUFFER_REGION = {
        'gm': 'Global Memory',
        'ub': 'Unified Buffer',
        'l1': 'L1 SRAM',
        'l0': 'L0 Register',
    }

    def __init__(self):
        self.ops: List[HIVMIROp] = []
        self.buffers: Dict[str, Tuple[str, float]] = {}  # name -> (region, size_kb)

    def parse(self, hivmir_text: str) -> List[HIVMIROp]:
        """解析 HIVMIR 文本"""
        self.ops = []
        self.buffers = {}

        lines = hivmir_text.strip().split('\n')

        op_id = 0
        for line_no, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过注释和空行
            if not line or line.startswith('//') or line.startswith('/*'):
                continue

            # 解析 alloc
            if 'alloc' in line or 'hivm.alloc' in line:
                self._parse_alloc(line)

            # 解析操作
            op = self._parse_operation(line)
            if op:
                op.op_id = op_id
                op.line_number = line_no
                self.ops.append(op)
                op_id += 1

        # 分析依赖关系
        self._analyze_dependencies()

        return self.ops

    def _parse_alloc(self, line: str):
        """解析内存分配"""
        # 格式: hivm.alloc %buf_name : memref<64KB>
        match = re.search(r'%(\w+)\s*,?\s*:\s*memref<(\d+)(KB|MB|GB)?>', line)
        if match:
            buf_name = match.group(1)
            size = float(match.group(2)) if match.group(2) else 64.0
            unit = match.group(3) if match.group(3) else 'KB'

            # 转换为 KB
            if unit == 'MB':
                size *= 1024
            elif unit == 'GB':
                size *= 1024 * 1024

            # 确定内存区域
            region = self._get_buffer_region(buf_name)
            self.buffers[buf_name] = (region, size)

    def _parse_operation(self, line: str) -> Optional[HIVMIROp]:
        """解析单个操作"""
        # 匹配操作模式
        # 格式: hivm.op_type %dst, %src : memref<size>
        op_pattern = re.compile(r'hivm\.(\w+)\s+%(\w+)\s*,\s*%(\w+)(?:\s*,\s*%(\w+))?\s*(?::\s*memref<(\d+)(KB|MB|GB)?>)?')

        match = op_pattern.search(line)
        if not match:
            return None

        op_type = match.group(1)
        dst = match.group(2)
        src = match.group(3)
        src2 = match.group(4) or ""

        # 解析大小
        size_kb = 64.0
        if match.group(5):
            size_kb = float(match.group(5))
            unit = match.group(6) if match.group(6) else 'KB'
            if unit == 'MB':
                size_kb *= 1024
            elif unit == 'GB':
                size_kb *= 1024 * 1024

        # 获取引擎
        engine = self.OP_TO_ENGINE.get(op_type, 'Unknown')

        # 获取内存区域
        memory_region = self._get_buffer_region(dst)

        # 构建变量名（包含地址偏移）
        variable_name = dst
        address_offset = ""

        # 检查是否有地址偏移（如 gm_1 + m*1KB）
        offset_match = re.search(r'\+\s*(\w+)\*(\d+)(KB|MB|GB)?', line)
        if offset_match:
            address_offset = f"+ {offset_match.group(1)}*{offset_match.group(2)}{offset_match.group(3) or 'KB'}"
            variable_name = f"{dst}{address_offset}"

        return HIVMIROp(
            op_id=0,  # 稍后赋值
            op_type=op_type,
            engine=engine,
            dst=dst,
            src=src,
            src2=src2,
            size_kb=size_kb,
            size_bytes=int(size_kb * 1024),
            variable_name=variable_name,
            address_offset=address_offset,
            memory_region=memory_region,
        )

    def _get_buffer_region(self, buffer_name: str) -> str:
        """根据缓冲区名获取内存区域"""
        prefix = buffer_name.split('_')[0].lower()
        return self.BUFFER_REGION.get(prefix, 'Unknown')

    def _analyze_dependencies(self):
        """分析 RAW/WAR/WAW 依赖"""
        # 记录每个 buffer 的最后写入操作
        last_write: Dict[str, int] = {}
        # 记录每个 buffer 的最后读取操作
        last_read: Dict[str, int] = {}

        for op in self.ops:
            # RAW: 读之前有写
            if op.src and op.src in last_write:
                writer_id = last_write[op.src]
                op.dependencies.append((writer_id, 'RAW'))

            if op.src2 and op.src2 in last_write:
                writer_id = last_write[op.src2]
                op.dependencies.append((writer_id, 'RAW'))

            # WAR: 写之前有读
            if op.dst and op.dst in last_read:
                reader_id = last_read[op.dst]
                op.dependencies.append((reader_id, 'WAR'))

            # WAW: 写之前有写
            if op.dst and op.dst in last_write:
                writer_id = last_write[op.dst]
                op.dependencies.append((writer_id, 'WAW'))

            # 更新记录
            if op.dst:
                last_write[op.dst] = op.op_id

            if op.src:
                last_read[op.src] = op.op_id

            if op.src2:
                last_read[op.src2] = op.op_id


# ============================================================================
#  Part 3: 数据合并
# ============================================================================

@dataclass
class CombinedOp:
    """合并后的完整操作信息"""
    op_id: int
    op_type: str
    engine: str
    size_kb: float
    size_bytes: int
    variable_name: str
    address_offset: str
    duration_ns: float
    start_ns: float
    end_ns: float
    effective_bw_gb_s: float
    peak_bw_gb_s: float
    bw_utilization: float
    regime: str
    wait_before_start_ns: float
    blocked_by: str
    dependencies: List[Tuple[int, str]]
    time_ratio: float
    memory_region: str
    line_number: int


class DataMerger:
    """合并 simulator 和 HIVMIR 数据"""

    def merge(self,
              sim_ops: List[SimulatorOp],
              hivmir_ops: List[HIVMIROp]) -> List[CombinedOp]:
        """合并两个数据源"""
        combined_ops = []

        # 建立 op_type -> list 映射
        # 假设两者的操作顺序一致
        max_ops = max(len(sim_ops), len(hivmir_ops))

        for i in range(max_ops):
            sim_op = sim_ops[i] if i < len(sim_ops) else None
            hivmir_op = hivmir_ops[i] if i < len(hivmir_ops) else None

            if sim_op and hivmir_op:
                # 两者都有，合并数据
                combined = CombinedOp(
                    op_id=i,
                    op_type=sim_op.op_type,
                    engine=sim_op.engine,
                    size_kb=hivmir_op.size_kb,  # 使用 HIVMIR 的精确大小
                    size_bytes=hivmir_op.size_bytes,
                    variable_name=hivmir_op.variable_name,  # 使用 HIVMIR 的变量名
                    address_offset=hivmir_op.address_offset,
                    duration_ns=sim_op.duration_ns,
                    start_ns=sim_op.start_ns,
                    end_ns=sim_op.end_ns,
                    effective_bw_gb_s=sim_op.effective_bw,
                    peak_bw_gb_s=sim_op.peak_bw,
                    bw_utilization=sim_op.bw_utilization,
                    regime=sim_op.regime,
                    wait_before_start_ns=sim_op.wait_before_start,
                    blocked_by=sim_op.blocked_by,
                    dependencies=hivmir_op.dependencies,  # 使用 HIVMIR 的依赖分析
                    time_ratio=sim_op.time_ratio,
                    memory_region=hivmir_op.memory_region,
                    line_number=hivmir_op.line_number,
                )
            elif sim_op:
                # 只有 simulator 数据
                combined = CombinedOp(
                    op_id=i,
                    op_type=sim_op.op_type,
                    engine=sim_op.engine,
                    size_kb=sim_op.size_kb,
                    size_bytes=int(sim_op.size_kb * 1024),
                    variable_name=f"{sim_op.dst}",
                    address_offset="",
                    duration_ns=sim_op.duration_ns,
                    start_ns=sim_op.start_ns,
                    end_ns=sim_op.end_ns,
                    effective_bw_gb_s=sim_op.effective_bw,
                    peak_bw_gb_s=sim_op.peak_bw,
                    bw_utilization=sim_op.bw_utilization,
                    regime=sim_op.regime,
                    wait_before_start_ns=sim_op.wait_before_start,
                    blocked_by=sim_op.blocked_by,
                    dependencies=[],
                    time_ratio=sim_op.time_ratio,
                    memory_region="Unknown",
                    line_number=0,
                )
            else:
                # 只有 HIVMIR 数据（缺少时序）
                combined = CombinedOp(
                    op_id=i,
                    op_type=hivmir_op.op_type,
                    engine=hivmir_op.engine,
                    size_kb=hivmir_op.size_kb,
                    size_bytes=hivmir_op.size_bytes,
                    variable_name=hivmir_op.variable_name,
                    address_offset=hivmir_op.address_offset,
                    duration_ns=0.0,  # 缺少时序
                    start_ns=0.0,
                    end_ns=0.0,
                    effective_bw_gb_s=0.0,
                    peak_bw_gb_s=0.0,
                    bw_utilization=0.0,
                    regime="unknown",
                    wait_before_start_ns=0.0,
                    blocked_by="",
                    dependencies=hivmir_op.dependencies,
                    time_ratio=0.0,
                    memory_region=hivmir_op.memory_region,
                    line_number=hivmir_op.line_number,
                )

            combined_ops.append(combined)

        return combined_ops


# ============================================================================
#  Part 4: 报告生成（完整版）
# ============================================================================

class CompleteReportGenerator:
    """生成完整的分析报告"""

    @staticmethod
    def generate_text_report(ops: List[CombinedOp],
                            total_ns: float,
                            output_file: str = None) -> str:
        """生成文本报告"""
        lines = []

        lines.append("=" * 120)
        lines.append("算子融合分析报告 - 完整操作流水（数据来源：HIVMIR + msprof op simulator）")
        lines.append("=" * 120)
        lines.append("")
        lines.append(f"总执行时间: {total_ns:.2f} ns")
        lines.append(f"操作数量: {len(ops)}")
        lines.append("")

        # 详细操作表格
        header = (f"{'Op':>4s}  {'操作类型':>12s}  {'引擎':>10s}  "
                  f"{'SIZE(KB)':>10s}  {'变量名':>20s}  "
                  f"{'Times(ns)':>12s}  {'BW util':>8s}  {'Regime':>10s}  "
                  f"{'waitFor':>15s}  {'依赖类型':>10s}  {'时间占比':>8s}")
        lines.append(header)
        lines.append("-" * 120)

        for op in ops:
            # 依赖信息
            dep_str = ", ".join([f"op{d[0]}" for d in op.dependencies])
            if not dep_str:
                dep_str = "-"

            # 依赖类型
            dep_types = ", ".join([d[1] for d in op.dependencies])
            if not dep_types:
                dep_types = "-"

            line = (f"{op.op_id:>4d}  {op.op_type:>12s}  {op.engine:>10s}  "
                    f"{op.size_kb:>10.1f}  {op.variable_name:>20s}  "
                    f"{op.duration_ns:>12.1f}  {op.bw_utilization:>7.1%}  {op.regime:>10s}  "
                    f"{dep_str:>15s}  {dep_types:>10s}  {op.time_ratio:>7.2f}%")
            lines.append(line)

        lines.append("")
        lines.append("=" * 120)
        lines.append("")

        # 时间占比统计（从大到小排序）
        lines.append("时间占比统计（从大到小排序）：")
        lines.append("-" * 120)

        sorted_ops = sorted(ops, key=lambda x: x.time_ratio, reverse=True)

        for i, op in enumerate(sorted_ops, 1):
            dep_info = ", ".join([f"op{d[0]}({d[1]})" for d in op.dependencies])
            if not dep_info:
                dep_info = "无依赖"

            lines.append(f"{i:>2d}. Op{op.op_id:>3d} ({op.op_type:>12s}): "
                        f"{op.time_ratio:>6.2f}%  "
                        f"时长={op.duration_ns:>10.1f}ns  "
                        f"大小={op.size_kb:>8.1f}KB  "
                        f"引擎={op.engine:>10s}  "
                        f"变量={op.variable_name:<20s}  "
                        f"依赖={dep_info}")

        lines.append("")
        lines.append("=" * 120)
        lines.append("")

        # 引擎利用率统计
        lines.append("引擎利用率统计：")
        lines.append("-" * 120)

        engine_stats = {}
        for op in ops:
            if op.engine not in engine_stats:
                engine_stats[op.engine] = {'busy_ns': 0.0, 'ops': []}
            engine_stats[op.engine]['busy_ns'] += op.duration_ns
            engine_stats[op.engine]['ops'].append(op.op_id)

        for engine, stats in sorted(engine_stats.items()):
            util = stats['busy_ns'] / total_ns * 100 if total_ns > 0 else 0
            ops_str = ", ".join([f"op{op_id}" for op_id in stats['ops']])
            lines.append(f"{engine:>12s}: busy={stats['busy_ns']:>10.1f}ns  "
                        f"utilization={util:>6.2f}%  ops=[{ops_str}]")

        lines.append("")
        lines.append("=" * 120)

        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"\n✓ 报告已保存到: {output_file}")

        return report_text

    @staticmethod
    def generate_visualization(ops: List[CombinedOp],
                              output_file: str = None,
                              title: str = "操作时间占比分析"):
        """生成可视化图表"""
        # 按时间占比排序
        sorted_ops = sorted(ops, key=lambda x: x.time_ratio, reverse=True)

        # 准备数据
        labels = [f"Op{op.op_id}\n{op.op_type}\n{op.size_kb:.0f}KB" for op in sorted_ops]
        values = [op.time_ratio for op in sorted_ops]
        colors = plt.cm.viridis([v / max(values) if max(values) > 0 else 0 for v in values])

        # 创建图表
        fig = plt.figure(figsize=(18, 12))

        # 子图1: 柱状图（时间占比）
        ax1 = fig.add_subplot(2, 2, 1)
        bars = ax1.bar(range(len(labels)), values, color=colors)
        ax1.set_xlabel('操作序号', fontsize=12)
        ax1.set_ylabel('时间占比 (%)', fontsize=12)
        ax1.set_title(f'{title} - 时间占比', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # 在柱子上添加数值
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

        # 子图2: 饼图（Top 10）
        ax2 = fig.add_subplot(2, 2, 2)
        top_n = min(10, len(sorted_ops))
        top_labels = [f"Op{op.op_id} ({op.op_type})\n{op.time_ratio:.1f}%" for op in sorted_ops[:top_n]]
        top_values = [op.time_ratio for op in sorted_ops[:top_n]]

        if len(sorted_ops) > top_n:
            other_time = sum(op.time_ratio for op in sorted_ops[top_n:])
            top_labels.append(f"其他 ({len(sorted_ops) - top_n}个)\n{other_time:.1f}%")
            top_values.append(other_time)

        colors_pie = plt.cm.Set3.colors[:len(top_values)]
        wedges, texts, autotexts = ax2.pie(top_values, labels=top_labels, autopct='',
                                           startangle=90, colors=colors_pie)
        ax2.set_title(f'{title} - 分布（Top 10）', fontsize=14, fontweight='bold')

        # 子图3: 引擎分布
        ax3 = fig.add_subplot(2, 2, 3)
        engine_stats = {}
        for op in sorted_ops:
            if op.engine not in engine_stats:
                engine_stats[op.engine] = 0
            engine_stats[op.engine] += op.time_ratio

        engines = list(engine_stats.keys())
        engine_values = list(engine_stats.values())
        colors_engine = plt.cm.Pastel1.colors[:len(engines)]

        ax3.barh(engines, engine_values, color=colors_engine)
        ax3.set_xlabel('时间占比 (%)', fontsize=12)
        ax3.set_ylabel('引擎', fontsize=12)
        ax3.set_title('引擎时间分布', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)

        for i, (eng, val) in enumerate(zip(engines, engine_values)):
            ax3.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10)

        # 子图4: 操作时序图
        ax4 = fig.add_subplot(2, 2, 4)
        for i, op in enumerate(sorted_ops[:top_n]):
            ax4.barh(i, op.duration_ns, left=op.start_ns,
                    height=0.6, label=f"Op{op.op_id}",
                    color=plt.cm.tab10(i % 10))

        ax4.set_xlabel('时间 (ns)', fontsize=12)
        ax4.set_ylabel('操作序号', fontsize=12)
        ax4.set_title('操作时序图（Top 10）', fontsize=14, fontweight='bold')
        ax4.set_yticks(range(top_n))
        ax4.set_yticklabels([f"Op{op.op_id}" for op in sorted_ops[:top_n]])
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存到: {output_file}")

        return fig


# ============================================================================
#  Part 5: 主流程
# ============================================================================

def run_simulator_and_parse(dsl_program: str) -> Tuple[List[SimulatorOp], str, float]:
    """运行 simulator 并解析输出"""
    import subprocess

    # 获取 simulator 路径
    simulator_path = Path(__file__).parent.parent / "costModel" / "cost_emulator" / "simulator.py"

    # 运行 simulator
    cmd = ["python", str(simulator_path), "--llm", "--critical-path", dsl_program]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"运行 simulator 失败: {e}")
        print(f"错误输出: {e.stderr}")
        return [], "", 0.0

    # 解析输出
    parser = SimulatorOutputParser()
    ops = parser.parse(output)

    return ops, output, parser.total_ns


def parse_hivmir_file(hivmir_file: str) -> List[HIVMIROp]:
    """解析 HIVMIR 文件"""
    with open(hivmir_file, 'r', encoding='utf-8') as f:
        hivmir_text = f.read()

    parser = HIVMIRParser()
    return parser.parse(hivmir_text)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='合并 HIVMIR 和 simulator 数据')
    parser.add_argument('--dsl', type=str, help='DSL 程序字符串')
    parser.add_argument('--hivmir', type=str, help='HIVMIR 文件路径')
    parser.add_argument('--output-dir', type=str, default='./fusion_analysis_output', help='输出目录')

    args = parser.parse_args()

    print("\n" + "=" * 120)
    print("完整数据合并流程：HIVMIR + msprof op simulator")
    print("=" * 120)

    # Step 1: 运行 simulator
    print("\n[Step 1] 运行 msprof op simulator...")
    if args.dsl:
        dsl_program = args.dsl
    else:
        # 示例 DSL
        dsl_program = "alloc(gm_1, 128KB) alloc(ub_1, 128KB) alloc(ub_2, 128KB) gm_to_ub(ub_1, gm_1) vadd(ub_2, ub_1, 2.0) ub_to_gm(gm_2, ub_2)"

    sim_ops, sim_output, total_ns = run_simulator_and_parse(dsl_program)
    print(f"  ✓ 解析了 {len(sim_ops)} 个 simulator 操作")

    # Step 2: 解析 HIVMIR
    print("\n[Step 2] 解析 HIVMIR...")
    if args.hivmir:
        hivmir_ops = parse_hivmir_file(args.hivmir)
    else:
        # 生成示例 HIVMIR
        sample_hivmir = """
hivm.alloc %gm_1 : memref<128KB>
hivm.alloc %ub_1 : memref<128KB>
hivm.alloc %ub_2 : memref<128KB>
hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
hivm.vadd %ub_2, %ub_1, 2.0
hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>
"""
        parser = HIVMIRParser()
        hivmir_ops = parser.parse(sample_hivmir)

    print(f"  ✓ 解析了 {len(hivmir_ops)} 个 HIVMIR 操作")

    # Step 3: 合并数据
    print("\n[Step 3] 合并数据...")
    merger = DataMerger()
    combined_ops = merger.merge(sim_ops, hivmir_ops)
    print(f"  ✓ 合并了 {len(combined_ops)} 个操作")

    # Step 4: 生成报告
    print("\n[Step 4] 生成报告和可视化...")
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 文本报告
    report = CompleteReportGenerator.generate_text_report(
        combined_ops, total_ns,
        output_file=str(output_path / "complete_fusion_report.txt")
    )

    # 可视化
    fig = CompleteReportGenerator.generate_visualization(
        combined_ops,
        output_file=str(output_path / "complete_fusion_analysis.png")
    )

    print("\n" + "=" * 120)
    print("流程完成！")
    print("=" * 120)
    print(f"\n输出目录: {output_path}")
    print(f"  - complete_fusion_report.txt (详细报告)")
    print(f"  - complete_fusion_analysis.png (可视化图表)")

    return report


if __name__ == "__main__":
    main()