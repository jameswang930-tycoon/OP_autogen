#!/usr/bin/env python3
"""
在 910B3 服务器上运行的完整流程脚本

使用方法:
  python run_fusion_analysis.py --kernel vadd_kernel.py --shapes '{"N": 32768}' --dtype fp16

输出:
  - fusion_analysis_output/fusion_analysis_report.txt
  - fusion_analysis_output/fusion_analysis_plot.png
  - fusion_analysis_output/operations.json
"""

import argparse
import json
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fusion_pipeline.combine_hivmir_msprof import FusionAnalysisPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='算子融合分析工具')

    parser.add_argument(
        '--kernel',
        type=str,
        required=True,
        help='Triton kernel 文件路径 (.py)'
    )

    parser.add_argument(
        '--shapes',
        type=str,
        required=True,
        help='Shape 信息 (JSON 格式)'
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='fp16',
        choices=['fp16', 'fp32', 'bf16'],
        help='数据类型 (默认: fp16)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='./fusion_analysis_output',
        help='输出目录 (默认: ./fusion_analysis_output)'
    )

    parser.add_argument(
        '--hivmir-file',
        type=str,
        default=None,
        help='HIVMIR 文件路径（可选，如果不提供则生成示例）'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 解析 shapes
    try:
        shapes = json.loads(args.shapes)
    except json.JSONDecodeError:
        print(f"错误: shapes 参数不是有效的 JSON: {args.shapes}")
        sys.exit(1)

    # 检查 kernel 文件
    kernel_path = Path(args.kernel)
    if not kernel_path.exists():
        print(f"错误: kernel 文件不存在: {args.kernel}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("算子融合分析工具 - 华为昇腾 910B3")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - Kernel: {args.kernel}")
    print(f"  - Shapes: {shapes}")
    print(f"  - Dtype: {args.dtype}")
    print(f"  - Output: {args.output_dir}")

    # 创建流程实例
    pipeline = FusionAnalysisPipeline(
        triton_file=args.kernel,
        shapes=shapes,
        dtype=args.dtype
    )

    # 如果提供了 HIVMIR 文件，使用它
    if args.hivmir_file:
        hivmir_path = Path(args.hivmir_file)
        if hivmir_path.exists():
            with open(hivmir_path, 'r', encoding='utf-8') as f:
                hivmir_text = f.read()
            # 直接使用提供的 HIVMIR
            from fusion_pipeline.combine_hivmir_msprof import HIVMIRParser
            parser = HIVMIRParser()
            pipeline.hivmir_ops = parser.parse_ir_text(hivmir_text)
            print(f"\n已加载 HIVMIR 文件: {args.hivmir_file}")

    # 运行完整流程
    try:
        report = pipeline.run()
        print("\n" + report)
        print("\n✓ 分析完成！")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()