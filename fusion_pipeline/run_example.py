#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例运行脚本 - 演示完整流程

这个脚本演示如何：
1. 运行 simulator 获取时序信息
2. 解析 HIVMIR 获取详细信息
3. 合并数据生成完整报告

使用方法:
  python run_example.py
"""

import sys
import io
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fusion_pipeline.complete_data_merge import (
    run_simulator_and_parse,
    HIVMIRParser,
    DataMerger,
    CompleteReportGenerator
)


def main():
    """运行完整示例"""

    print("\n" + "=" * 100)
    print("算子融合分析示例")
    print("=" * 100)

    # 示例 DSL 程序（向量加法）
    dsl_program = """
    alloc(gm_x, 256KB)
    alloc(gm_y, 256KB)
    alloc(gm_out, 256KB)
    alloc(ub_x, 64KB)
    alloc(ub_y, 64KB)
    alloc(ub_out, 64KB)

    gm_to_ub(ub_x, gm_x)
    gm_to_ub(ub_y, gm_y)
    vadd(ub_out, ub_x, ub_y, 1.0)
    ub_to_gm(gm_out, ub_out)

    gm_to_ub(ub_x, gm_x)
    gm_to_ub(ub_y, gm_y)
    vadd(ub_out, ub_x, ub_y, 1.0)
    ub_to_gm(gm_out, ub_out)

    gm_to_ub(ub_x, gm_x)
    gm_to_ub(ub_y, gm_y)
    vadd(ub_out, ub_x, ub_y, 1.0)
    ub_to_gm(gm_out, ub_out)

    gm_to_ub(ub_x, gm_x)
    gm_to_ub(ub_y, gm_y)
    vadd(ub_out, ub_x, ub_y, 1.0)
    ub_to_gm(gm_out, ub_out)
    """

    # 示例 HIVMIR
    hivmir_text = """
// HIVM IR for vadd kernel (tiled execution)
hivm.alloc %gm_x, %gm_y, %gm_out : memref<256KB>
hivm.alloc %ub_x, %ub_y, %ub_out : memref<64KB>

// Tile 0
hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

// Tile 1
hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

// Tile 2
hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

// Tile 3
hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>
"""

    # Step 1: 运行 simulator
    print("\n[Step 1] 运行 msprof op simulator...")
    sim_ops, sim_output, total_ns = run_simulator_and_parse(dsl_program)
    print(f"  ✓ 解析了 {len(sim_ops)} 个操作")
    print(f"  ✓ 总执行时间: {total_ns:.2f} ns")

    # Step 2: 解析 HIVMIR
    print("\n[Step 2] 解析 HIVMIR...")
    hivmir_parser = HIVMIRParser()
    hivmir_ops = hivmir_parser.parse(hivmir_text)
    print(f"  ✓ 解析了 {len(hivmir_ops)} 个操作")

    # Step 3: 合并数据
    print("\n[Step 3] 合并数据...")
    merger = DataMerger()
    combined_ops = merger.merge(sim_ops, hivmir_ops)
    print(f"  ✓ 合并了 {len(combined_ops)} 个操作")

    # Step 4: 生成报告
    print("\n[Step 4] 生成报告...")
    output_dir = Path("./example_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 文本报告
    report = CompleteReportGenerator.generate_text_report(
        combined_ops,
        total_ns,
        output_file=str(output_dir / "example_report.txt")
    )

    # 可视化
    fig = CompleteReportGenerator.generate_visualization(
        combined_ops,
        output_file=str(output_dir / "example_analysis.png"),
        title="向量加法算子分析"
    )

    # 打印摘要
    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
    print(f"\n输出文件:")
    print(f"  - {output_dir / 'example_report.txt'}")
    print(f"  - {output_dir / 'example_analysis.png'}")

    print("\n" + "=" * 100)
    print("关键发现:")
    print("=" * 100)

    # 找出时间占比最大的操作
    max_op = max(combined_ops, key=lambda x: x.time_ratio)
    print(f"\n瓶颈操作: Op{max_op.op_id} ({max_op.op_type})")
    print(f"  - 时间占比: {max_op.time_ratio:.2f}%")
    print(f"  - 执行时间: {max_op.duration_ns:.2f} ns")
    print(f"  - 数据大小: {max_op.size_kb:.1f} KB")
    print(f"  - 引擎: {max_op.engine}")
    print(f"  - 带宽利用率: {max_op.bw_utilization:.1%}")

    # 找出依赖链
    print(f"\n依赖链分析:")
    print(f"  - 总操作数: {len(combined_ops)}")
    print(f"  - RAW 依赖: {sum(1 for op in combined_ops for dep in op.dependencies if dep[1] == 'RAW')}")
    print(f"  - WAR 依赖: {sum(1 for op in combined_ops for dep in op.dependencies if dep[1] == 'WAR')}")
    print(f"  - WAW 依赖: {sum(1 for op in combined_ops for dep in op.dependencies if dep[1] == 'WAW')}")

    print("\n" + report)


if __name__ == "__main__":
    main()