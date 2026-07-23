#!/usr/bin/env python3
"""
完整的算子融合分析流程：
  1. 解析 HIVMIR（从编译器中间产物）
  2. 运行 msprof op simulator（时序模拟）
  3. 合并两个数据源
  4. 生成详细分析报告
  5. 可视化时间占比

输入:
  - Triton kernel (.py 文件)
  - PyTorch 基准代码
  - Shape 信息

输出:
  - 每个操作的详细信息（操作类型、引擎、大小、时间、带宽利用率、依赖）
  - 时间占比图（从大到小排序）
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd

# 添加 costModel 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "costModel" / "cost_emulator"))

from simulator import (
    ENGINE_FOR, ENG_NAME, MEMORY_CAPACITY_KB,
    bandwidth_profile, peak_bandwidth_gb_s,
    Op, emulate_program, build_deps, schedule
)


# ===============================================================================
#  Part 1: HIVMIR 解析器
# ===============================================================================

@dataclass
class HIVMIROperation:
    """HIVM IR 操作的详细信息"""
    op_id: int
    op_type: str  # gm_to_ub, vadd, matrixmul, etc.
    engine: str   # GM→UB, VecUnit, etc.
    dst: str      # 目标缓冲区
    src: str      # 源缓冲区
    src2: str = ""  # 第二个源（用于 matrixmul）
    size_kb: float = 0.0
    size_bytes: int = 0
    variable_name: str = ""
    dependencies: List[Tuple[int, str]] = field(default_factory=list)  # (op_id, dep_type)
    line_number: int = 0


class HIVMIRParser:
    """解析 HIVM IR 文本，提取详细操作信息"""

    # HIVM IR 操作模式（基于 AscendNPU-IR 推断）
    OP_PATTERNS = {
        r'hivm\.gm_to_ub\s*%\w+,\s*%\w+\s*:\s*memref<(\d+)(KB|MB|GB)?>': 'gm_to_ub',
        r'hivm\.ub_to_gm\s*%\w+,\s*%\w+\s*:\s*memref<(\d+)(KB|MB|GB)?>': 'ub_to_gm',
        r'hivm\.gm_to_l1\s*%\w+,\s*%\w+\s*:\s*memref<(\d+)(KB|MB|GB)?>': 'gm_to_l1',
        r'hivm\.l1_to_l0\s*%\w+,\s*%\w+\s*:\s*memref<(\d+)(KB|MB|GB)?>': 'l1_to_l0',
        r'hivm\.l0_to_gm\s*%\w+,\s*%\w+\s*:\s*memref<(\d+)(KB|MB|GB)?>': 'l0_to_gm',
        r'hivm\.vadd\s*%\w+,\s*%\w+,\s*[\d.]+': 'vadd',
        r'hivm\.matrixmul\s*%\w+,\s*%\w+,\s*%\w+': 'matrixmul',
    }

    # 引擎映射
    ENGINE_MAP = {
        'gm_to_ub': 'GM→UB',
        'ub_to_gm': 'UB→GM',
        'gm_to_l1': 'GM→L1',
        'l1_to_l0': 'L1→L0',
        'l0_to_gm': 'L0→GM',
        'vadd': 'VecUnit',
        'matrixmul': 'CubeUnit',
    }

    def __init__(self):
        self.operations: List[HIVMIROperation] = []
        self.buffer_map: Dict[str, str] = {}  # buffer_name -> region

    def parse_ir_text(self, ir_text: str) -> List[HIVMIROperation]:
        """解析 HIVM IR 文本"""
        self.operations = []
        lines = ir_text.strip().split('\n')

        op_id = 0
        for line_no, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            # 匹配操作
            op_info = self._parse_operation(line)
            if op_info:
                op_info.op_id = op_id
                op_info.line_number = line_no
                self.operations.append(op_info)
                op_id += 1

        # 分析依赖关系
        self._analyze_dependencies()

        return self.operations

    def _parse_operation(self, line: str) -> Optional[HIVMIROperation]:
        """解析单个 HIVM 操作"""
        for pattern, op_type in self.OP_PATTERNS.items():
            match = re.search(pattern, line)
            if match:
                # 提取操作数
                operands = re.findall(r'%(\w+)', line)
                dst = operands[0] if len(operands) > 0 else ""
                src = operands[1] if len(operands) > 1 else ""
                src2 = operands[2] if len(operands) > 2 else ""

                # 提取大小
                size_kb = self._extract_size(line)

                # 提取变量名（从 memref 类型）
                var_name = self._extract_variable_name(line)

                return HIVMIROperation(
                    op_id=0,  # 稍后赋值
                    op_type=op_type,
                    engine=self.ENGINE_MAP.get(op_type, 'Unknown'),
                    dst=dst,
                    src=src,
                    src2=src2,
                    size_kb=size_kb,
                    size_bytes=int(size_kb * 1024),
                    variable_name=var_name,
                )

        return None

    def _extract_size(self, line: str) -> float:
        """从 memref 中提取大小（KB）"""
        match = re.search(r'memref<(\d+)(KB|MB|GB)?', line)
        if match:
            size = float(match.group(1))
            unit = match.group(2) or 'KB'

            # 转换为 KB
            if unit == 'MB':
                size *= 1024
            elif unit == 'GB':
                size *= 1024 * 1024

            return size
        return 64.0  # 默认大小

    def _extract_variable_name(self, line: str) -> str:
        """提取变量名"""
        # 简单实现：返回第一个操作数
        match = re.search(r'%(\w+)', line)
        return match.group(1) if match else ""

    def _analyze_dependencies(self):
        """分析操作之间的依赖关系（RAW/WAW/WAR）"""
        # 最后写入的 buffer -> op_id
        last_write: Dict[str, int] = {}

        for op in self.operations:
            # RAW: 读之前有写
            if op.src and op.src in last_write:
                writer_id = last_write[op.src]
                op.dependencies.append((writer_id, 'RAW'))

            if op.src2 and op.src2 in last_write:
                writer_id = last_write[op.src2]
                op.dependencies.append((writer_id, 'RAW'))

            # WAR: 写之前有读（简化处理）
            # WAW: 写之前有写
            if op.dst:
                if op.dst in last_write:
                    writer_id = last_write[op.dst]
                    op.dependencies.append((writer_id, 'WAW'))

                last_write[op.dst] = op.op_id


# ===============================================================================
#  Part 2: msprof op simulator 结果提取
# ===============================================================================

@dataclass
class SimulatorResult:
    """模拟器结果"""
    op_id: int
    op_type: str
    engine: str
    size_kb: float
    duration_ns: float
    start_ns: float
    end_ns: float
    effective_bw_gb_s: float
    bw_utilization: float
    regime: str  # floor/ramp/saturated/flat


class SimulatorAnalyzer:
    """运行 simulator 并提取结果"""

    def __init__(self, dsl_program: str):
        self.dsl_program = dsl_program
        self.operations: List[SimulatorResult] = []

    def run_simulation(self) -> List[SimulatorResult]:
        """运行模拟器"""
        # 解析 DSL 程序
        ops = emulate_program(self.dsl_program)
        if not ops:
            print("Warning: DSL parsing failed, using empty ops list")
            return []

        # 分配大小和时长
        assign_sizes(ops)

        # 构建依赖并调度
        build_deps(ops)
        schedule(ops)

        # 提取结果
        self.operations = []
        for i, op in enumerate(ops):
            result = SimulatorResult(
                op_id=i,
                op_type=op.name,
                engine=ENG_NAME.get(op.engine, f"Engine{op.engine}"),
                size_kb=op.size_kb,
                duration_ns=op.duration,
                start_ns=op.start,
                end_ns=op.end,
                effective_bw_gb_s=op.effective_bw,
                bw_utilization=op.bw_utilization,
                regime=op.regime,
            )
            self.operations.append(result)

        return self.operations


def assign_sizes(ops: List[Op]):
    """为操作分配大小（简化版，从 simulator.py 复制）"""
    for op in ops:
        if op.size_kb == 0.0:
            op.size_kb = 64.0  # 默认大小

        # 计算时长
        duration, bw, util, regime = bandwidth_profile(op.engine, op.size_kb)
        op.duration = duration
        op.effective_bw = bw
        op.bw_utilization = util
        op.regime = regime


# ===============================================================================
#  Part 3: 数据合并与映射
# ===============================================================================

@dataclass
class CombinedOperation:
    """合并后的操作信息"""
    op_id: int
    op_type: str
    engine: str
    size_kb: float
    size_bytes: int
    variable_name: str
    duration_ns: float
    start_ns: float
    end_ns: float
    effective_bw_gb_s: float
    bw_utilization: float
    regime: str
    dependencies: List[Tuple[int, str]]
    time_ratio: float  # 占比百分比


class OperationMerger:
    """合并 HIVMIR 和 simulator 的数据"""

    def __init__(self):
        self.operations: List[CombinedOperation] = []

    def merge(self,
              hivmir_ops: List[HIVMIROperation],
              sim_ops: List[SimulatorResult]) -> List[CombinedOperation]:
        """合并两个数据源"""
        self.operations = []

        # 建立映射：HIVMIR 的 op_type -> simulator 的 op_type
        op_map = {i: op for i, op in enumerate(sim_ops)}

        total_time = max(op.end_ns for op in sim_ops) if sim_ops else 1.0

        for hivmir_op in hivmir_ops:
            # 找到对应的 simulator 结果
            sim_op = op_map.get(hivmir_op.op_id)

            if sim_op:
                combined = CombinedOperation(
                    op_id=hivmir_op.op_id,
                    op_type=hivmir_op.op_type,
                    engine=hivmir_op.engine,
                    size_kb=hivmir_op.size_kb,
                    size_bytes=hivmir_op.size_bytes,
                    variable_name=hivmir_op.variable_name,
                    duration_ns=sim_op.duration_ns,
                    start_ns=sim_op.start_ns,
                    end_ns=sim_op.end_ns,
                    effective_bw_gb_s=sim_op.effective_bw_gb_s,
                    bw_utilization=sim_op.bw_utilization,
                    regime=sim_op.regime,
                    dependencies=hivmir_op.dependencies,
                    time_ratio=sim_op.duration_ns / total_time * 100 if total_time > 0 else 0,
                )
                self.operations.append(combined)

        return self.operations


# ===============================================================================
#  Part 4: 报告生成
# ===============================================================================

class ReportGenerator:
    """生成详细分析报告"""

    @staticmethod
    def generate_text_report(operations: List[CombinedOperation], output_file: str = None):
        """生成文本报告"""
        lines = []
        lines.append("=" * 100)
        lines.append("算子融合分析报告 - 详细操作流水")
        lines.append("=" * 100)
        lines.append("")

        # 表头
        header = (f"{'Op':>4s}  {'操作类型':>12s}  {'引擎':>10s}  "
                  f"{'大小':>10s}  {'时长(ns)':>12s}  "
                  f"{'带宽利用率':>10s}  {'Regime':>10s}  {'依赖':>20s}")
        lines.append(header)
        lines.append("-" * 100)

        # 每个操作的详细信息
        for op in operations:
            dep_str = ", ".join([f"op{d[0]}({d[1]})" for d in op.dependencies])
            if not dep_str:
                dep_str = "-"

            line = (f"{op.op_id:>4d}  {op.op_type:>12s}  {op.engine:>10s}  "
                    f"{op.size_kb:>8.1f}KB  {op.duration_ns:>12.1f}  "
                    f"{op.bw_utilization:>9.1%}  {op.regime:>10s}  {dep_str:>20s}")
            lines.append(line)

        lines.append("")
        lines.append("=" * 100)
        lines.append("")

        # 时间占比统计
        lines.append("时间占比统计（从大到小排序）：")
        lines.append("-" * 100)

        sorted_ops = sorted(operations, key=lambda x: x.time_ratio, reverse=True)
        for i, op in enumerate(sorted_ops, 1):
            lines.append(f"{i:>2d}. Op{op.op_id:>3d} ({op.op_type:>12s}): "
                        f"{op.time_ratio:>6.2f}%  "
                        f"时长={op.duration_ns:>10.1f}ns  "
                        f"引擎={op.engine:>10s}")

        report_text = "\n".join(lines)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"报告已保存到: {output_file}")

        return report_text

    @staticmethod
    def generate_visualization(operations: List[CombinedOperation], output_file: str = None):
        """生成时间占比可视化图"""
        # 按时间占比排序
        sorted_ops = sorted(operations, key=lambda x: x.time_ratio, reverse=True)

        # 准备数据
        labels = [f"Op{op.op_id}\n{op.op_type}" for op in sorted_ops]
        values = [op.time_ratio for op in sorted_ops]
        colors = plt.cm.viridis([v / 100 for v in values])

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：柱状图
        ax1.bar(range(len(labels)), values, color=colors)
        ax1.set_xlabel('操作序号', fontsize=12)
        ax1.set_ylabel('时间占比 (%)', fontsize=12)
        ax1.set_title('各操作时间占比（从大到小）', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # 右图：饼图（Top 10）
        top_n = min(10, len(sorted_ops))
        top_labels = [f"Op{op.op_id} ({op.op_type})" for op in sorted_ops[:top_n]]
        top_values = [op.time_ratio for op in sorted_ops[:top_n]]

        if len(sorted_ops) > top_n:
            other_time = sum(op.time_ratio for op in sorted_ops[top_n:])
            top_labels.append(f"其他 ({len(sorted_ops) - top_n}个)")
            top_values.append(other_time)

        ax2.pie(top_values, labels=top_labels, autopct='%1.1f%%',
                startangle=90, colors=plt.cm.Set3.colors[:len(top_values)])
        ax2.set_title('时间占比分布（Top 10）', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {output_file}")

        return fig


# ===============================================================================
#  Part 5: 完整流程
# ===============================================================================

class FusionAnalysisPipeline:
    """完整的融合分析流程"""

    def __init__(self, triton_file: str, shapes: dict, dtype: str = 'fp16'):
        self.triton_file = Path(triton_file)
        self.shapes = shapes
        self.dtype = dtype

        self.hivmir_ops: List[HIVMIROperation] = []
        self.sim_ops: List[SimulatorResult] = []
        self.combined_ops: List[CombinedOperation] = []

    def step1_generate_hivmir(self) -> str:
        """
        Step 1: 从 Triton 代码生成 HIVMIR

        实际实现需要：
        1. 调用华为编译器链
        2. 使用 --mlir-print-ir-after-all 插桩
        3. 解析 HIVM dialect

        这里提供示例 HIVMIR
        """
        print("\n[Step 1] 生成 HIVMIR...")

        # 示例：基于 vadd kernel 的 HIVMIR（需要替换为实际编译输出）
        sample_hivmir = """
// HIVM IR for vadd kernel
hivm.alloc %gm_a, %gm_b, %gm_c : memref<256KB>
hivm.alloc %ub_a, %ub_b, %ub_c : memref<64KB>

// Tile 0: load inputs
hivm.gm_to_ub %ub_a, %gm_a : memref<64KB>
hivm.gm_to_ub %ub_b, %gm_b : memref<64KB>

// Compute
hivm.vadd %ub_c, %ub_a, %ub_b, 1.0

// Store output
hivm.ub_to_gm %gm_c, %ub_c : memref<64KB>

// Tile 1: repeat
hivm.gm_to_ub %ub_a, %gm_a : memref<64KB>
hivm.gm_to_ub %ub_b, %gm_b : memref<64KB>
hivm.vadd %ub_c, %ub_a, %ub_b, 1.0
hivm.ub_to_gm %gm_c, %ub_c : memref<64KB>
"""

        # 解析 HIVMIR
        parser = HIVMIRParser()
        self.hivmir_ops = parser.parse_ir_text(sample_hivmir)

        print(f"  ✓ 已解析 {len(self.hivmir_ops)} 个 HIVMIR 操作")

        return sample_hivmir

    def step2_run_simulator(self) -> str:
        """
        Step 2: 运行 msprof op simulator

        使用 costModel/cost_emulator/simulator.py
        """
        print("\n[Step 2] 运行 msprof op simulator...")

        # 构建 DSL 程序（基于 HIVMIR）
        dsl_parts = []
        for op in self.hivmir_ops:
            if op.op_type == 'gm_to_ub':
                dsl_parts.append(f"gm_to_ub({op.dst}, {op.src})")
            elif op.op_type == 'ub_to_gm':
                dsl_parts.append(f"ub_to_gm({op.dst}, {op.src})")
            elif op.op_type == 'vadd':
                dsl_parts.append(f"vadd({op.dst}, {op.src}, 1.0)")
            elif op.op_type == 'matrixmul':
                dsl_parts.append(f"matrixmul({op.dst}, {op.src}, {op.src2})")

        dsl_program = "alloc(gm_a, 256KB) alloc(ub_a, 64KB) " + " ".join(dsl_parts)

        # 运行模拟器
        analyzer = SimulatorAnalyzer(dsl_program)
        self.sim_ops = analyzer.run_simulation()

        print(f"  ✓ 模拟器已生成 {len(self.sim_ops)} 个操作结果")

        return dsl_program

    def step3_merge_data(self) -> List[CombinedOperation]:
        """Step 3: 合并两个数据源"""
        print("\n[Step 3] 合并 HIVMIR 和 simulator 数据...")

        merger = OperationMerger()
        self.combined_ops = merger.merge(self.hivmir_ops, self.sim_ops)

        print(f"  ✓ 已合并 {len(self.combined_ops)} 个操作")

        return self.combined_ops

    def step4_generate_reports(self, output_dir: str = "./fusion_analysis_output"):
        """Step 4: 生成报告和可视化"""
        print("\n[Step 4] 生成分析报告...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 文本报告
        text_report = ReportGenerator.generate_text_report(
            self.combined_ops,
            output_file=str(output_path / "fusion_analysis_report.txt")
        )

        # 可视化
        fig = ReportGenerator.generate_visualization(
            self.combined_ops,
            output_file=str(output_path / "fusion_analysis_plot.png")
        )

        print(f"\n报告已生成在目录: {output_path}")

        return text_report

    def run(self):
        """运行完整流程"""
        print("\n" + "=" * 100)
        print("开始算子融合分析流程")
        print("=" * 100)

        self.step1_generate_hivmir()
        self.step2_run_simulator()
        self.step3_merge_data()
        report = self.step4_generate_reports()

        print("\n" + "=" * 100)
        print("流程完成！")
        print("=" * 100)

        return report


# ===============================================================================
#  主函数
# ===============================================================================

def main():
    """主函数：演示完整流程"""

    # 示例配置（需要替换为实际输入）
    config = {
        'triton_file': 'example_kernel.py',  # Triton kernel 文件路径
        'shapes': {'N': 32768},               # Shape 信息
        'dtype': 'fp16',                      # 数据类型
    }

    # 创建流程实例
    pipeline = FusionAnalysisPipeline(
        triton_file=config['triton_file'],
        shapes=config['shapes'],
        dtype=config['dtype']
    )

    # 运行完整流程
    report = pipeline.run()

    # 打印报告
    print("\n" + report)


if __name__ == "__main__":
    main()