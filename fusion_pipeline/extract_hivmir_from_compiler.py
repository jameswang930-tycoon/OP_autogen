#!/usr/bin/env python3
"""
从华为编译器提取 HIVMIR

该脚本演示如何：
1. 调用华为编译器链
2. 使用 MLIR pass 插桩获取 IR
3. 解析 HIVM dialect

注意：
- 需要在昇腾 910B3 服务器上运行
- 需要安装 CANN 工具包
- 需要配置环境变量
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class HIVMIRExtractor:
    """从编译器提取 HIVMIR"""

    def __init__(self, triton_file: str, output_dir: str = "./hivmir_output"):
        self.triton_file = Path(triton_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_with_compiler(self) -> str:
        """
        使用华为编译器提取 HIVMIR

        实际流程：
        1. 编译 Triton 代码到 Ascend NPU
        2. 使用 --mlir-print-ir-after-all 插桩
        3. 提取 HIVM dialect 层的 IR

        注意：这需要在昇腾服务器上运行
        """
        print(f"\n[Step 1] 编译 Triton kernel: {self.triton_file}")

        # 示例编译命令（需要根据实际环境调整）
        # 实际命令可能是：
        # bishengir-compile --mlir-print-ir-after-all input.py -o output.om

        output_ir_file = self.output_dir / "hivmir_output.mlir"

        # 模拟编译过程（实际需要调用编译器）
        # 这里提供一个示例 HIVMIR
        sample_hivmir = self._generate_sample_hivmir()

        # 保存到文件
        with open(output_ir_file, 'w', encoding='utf-8') as f:
            f.write(sample_hivmir)

        print(f"  ✓ HIVMIR 已保存到: {output_ir_file}")

        return str(output_ir_file)

    def _generate_sample_hivmir(self) -> str:
        """生成示例 HIVMIR（用于演示）"""
        return """
// HIVM IR - Auto-generated from Triton kernel
// Kernel: vadd
// Target: Ascend 910B3

module {
  // Memory allocation
  hivm.alloc %gm_x, %gm_y, %gm_out : memref<256KB>
  hivm.alloc %ub_x, %ub_y, %ub_out : memref<64KB>

  // Tile 0: Load inputs from GM to UB
  hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
  hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>

  // Compute: vector addition
  hivm.vadd %ub_out, %ub_x, %ub_y, 1.0

  // Store output from UB to GM
  hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

  // Tile 1: Repeat for next block
  hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
  hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
  hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
  hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

  // Tile 2: Repeat for next block
  hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
  hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
  hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
  hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>

  // Tile 3: Repeat for next block
  hivm.gm_to_ub %ub_x, %gm_x : memref<64KB>
  hivm.gm_to_ub %ub_y, %gm_y : memref<64KB>
  hivm.vadd %ub_out, %ub_x, %ub_y, 1.0
  hivm.ub_to_gm %gm_out, %ub_out : memref<64KB>
}
"""

    def parse_hivmir_pass_dump(self, pass_dump_file: str) -> List[str]:
        """
        解析 pass dump 文件，提取 HIVM dialect

        Args:
            pass_dump_file: pass dump 文件路径

        Returns:
            List[str]: 每个 pass 后的 IR
        """
        with open(pass_dump_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按 pass 分割
        passes = content.split('// -----// IR Dump')

        hivm_irs = []
        for pass_ir in passes:
            if 'hivm.' in pass_ir:
                hivm_irs.append(pass_ir)

        return hivm_irs


def compile_with_instrumentation(triton_file: str, output_dir: str) -> str:
    """
    编译 Triton kernel 并插桩获取 HIVMIR

    实际实现需要：
    1. 设置编译器环境变量
    2. 调用 bishengir-compile 或类似工具
    3. 使用 MLIR debugging flags

    示例命令：
        bishengir-compile \\
            --mlir-print-ir-after-all \\
            --mlir-print-ir-after-change \\
            input.py \\
            -o output.om \\
            > ir_dump.txt
    """
    extractor = HIVMIRExtractor(triton_file, output_dir)
    return extractor.extract_with_compiler()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='提取 HIVMIR')
    parser.add_argument('triton_file', help='Triton kernel 文件路径')
    parser.add_argument('--output-dir', default='./hivmir_output', help='输出目录')

    args = parser.parse_args()

    # 检查文件
    if not Path(args.triton_file).exists():
        print(f"错误: 文件不存在: {args.triton_file}")
        sys.exit(1)

    # 提取 HIVMIR
    output_file = compile_with_instrumentation(args.triton_file, args.output_dir)
    print(f"\nHIVMIR 已提取: {output_file}")


if __name__ == "__main__":
    main()