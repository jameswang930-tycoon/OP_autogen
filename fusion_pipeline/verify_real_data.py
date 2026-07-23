#!/usr/bin/env python3
"""
验证脚本 - 在 910B3 服务器上运行，获取真实的 HIVMIR 格式

这个脚本用于：
1. 编译 Triton kernel 并捕获编译器输出
2. 分析真实的 HIVMIR 格式
3. 更新解析器以适配真实格式

使用方法（在 910B3 服务器上）:
  python verify_real_data.py --kernel vadd_kernel.py
"""

import subprocess
import sys
import re
import json
from pathlib import Path


def compile_and_capture_ir(kernel_file: str, output_dir: str = "./verify_output"):
    """
    编译 Triton kernel 并捕获所有中间 IR

    需要在 910B3 服务器上运行，使用华为编译器
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Step 1: 编译 Triton kernel 并捕获 IR")
    print("=" * 80)

    # 方法 1: 使用 bishengir-compile（如果可用）
    try:
        cmd = [
            "bishengir-compile",
            "--mlir-print-ir-after-all",
            "--mlir-print-ir-after-change",
            kernel_file,
            "-o", f"{output_path}/output.om"
        ]

        print(f"\n执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            # 保存完整输出
            with open(f"{output_path}/full_ir_dump.txt", 'w') as f:
                f.write(result.stdout)

            print(f"✓ 编译成功，IR 已保存到: {output_path}/full_ir_dump.txt")
            return result.stdout
        else:
            print(f"✗ 编译失败: {result.stderr}")
    except FileNotFoundError:
        print("bishengir-compile 未找到，尝试其他方法...")

    # 方法 2: 使用 Triton 编译（如果可用）
    try:
        cmd = [
            "python", "-c",
            f"""
import torch
import triton
import sys
sys.path.insert(0, '{kernel_file}')

# 强制打印 IR
import os
os.environ['TRITON_PRINT_IR'] = '1'

# 导入 kernel
from {Path(kernel_file).stem} import *

# 编译（触发 IR 打印）
"""
        ]

        print(f"\n执行命令: python (Triton 编译)")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            with open(f"{output_path}/triton_ir_dump.txt", 'w') as f:
                f.write(result.stdout)

            print(f"✓ Triton IR 已保存到: {output_path}/triton_ir_dump.txt")
            return result.stdout
        else:
            print(f"✗ Triton 编译失败: {result.stderr}")
    except Exception as e:
        print(f"✗ Triton 方法失败: {e}")

    # 方法 3: 直接运行并捕获输出
    try:
        cmd = ["python", kernel_file]
        print(f"\n执行命令: python {kernel_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        with open(f"{output_path}/kernel_output.txt", 'w') as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        print(f"✓ Kernel 输出已保存到: {output_path}/kernel_output.txt")
        return result.stdout
    except Exception as e:
        print(f"✗ 执行失败: {e}")

    return None


def analyze_hivmir_format(ir_text: str):
    """
    分析 HIVMIR 的真实格式，提取操作模式和字段名
    """
    print("\n" + "=" * 80)
    print("Step 2: 分析 HIVMIR 格式")
    print("=" * 80)

    if not ir_text:
        print("✗ 没有可分析的 IR 文本")
        return

    # 查找 HIVM 相关的行
    hivm_lines = []
    for line in ir_text.split('\n'):
        if 'hivm' in line.lower() or 'HIVM' in line:
            hivm_lines.append(line)

    if hivm_lines:
        print(f"\n找到 {len(hivm_lines)} 行 HIVM 相关内容:")
        print("-" * 80)
        for line in hivm_lines[:20]:  # 显示前 20 行
            print(line)
    else:
        print("\n未找到 HIVM 相关内容，显示所有操作:")
        # 尝试查找其他操作模式
        op_patterns = re.findall(r'(gm_to_\w+|ub_to_\w+|vadd|matrixmul)', ir_text)
        if op_patterns:
            print(f"找到的操作类型: {set(op_patterns)}")

        # 显示前 100 行供分析
        print("\n前 100 行 IR:")
        print("-" * 80)
        for line in ir_text.split('\n')[:100]:
            if line.strip():
                print(line)

    # 提取操作模式
    print("\n" + "=" * 80)
    print("Step 3: 提取操作模式")
    print("=" * 80)

    patterns = {
        '操作类型': r'(gm_to_\w+|ub_to_\w+|l1_to_\w+|l0_to_\w+|v\w+|matrix\w+)',
        '引擎名称': r'(GM→\w+|UB→\w+|L1→\w+|VecUnit|CubeUnit)',
        '缓冲区名称': r'%(\w+)',
        '数据大小': r'memref<(\d+)(KB|MB|GB)',
        '数据类型': r'(fp16|fp32|bf16|int\d+)',
    }

    results = {}
    for name, pattern in patterns.items():
        matches = re.findall(pattern, ir_text, re.IGNORECASE)
        if matches:
            results[name] = list(set(matches[:10]))  # 取前 10 个不重复的
            print(f"{name}: {results[name]}")

    return results


def generate_parser_update(format_info: dict, output_file: str = "parser_update.json"):
    """
    基于真实格式生成解析器更新建议
    """
    print("\n" + "=" * 80)
    print("Step 4: 生成解析器更新建议")
    print("=" * 80)

    update = {
        'status': 'needs_verification',
        'message': '请根据真实 HIVMIR 格式更新 complete_data_merge.py 中的解析器',
        'detected_patterns': format_info,
        'recommended_updates': []
    }

    print("\n建议的更新:")
    print("-" * 80)

    if format_info.get('操作类型'):
        print(f"1. 更新 OP_TO_ENGINE 映射，添加: {format_info['操作类型']}")

    if format_info.get('引擎名称'):
        print(f"2. 验证引擎名称: {format_info['引擎名称']}")

    if format_info.get('缓冲区名称'):
        print(f"3. 更新缓冲区前缀映射: {format_info['缓冲区名称'][:5]}")

    # 保存更新建议
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(update, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 更新建议已保存到: {output_file}")

    return update


def main():
    import argparse

    parser = argparse.ArgumentParser(description='验证真实 HIVMIR 格式')
    parser.add_argument('--kernel', type=str, help='Triton kernel 文件路径')
    parser.add_argument('--ir-file', type=str, help='已有的 IR 文件路径（跳过编译）')
    parser.add_argument('--output-dir', type=str, default='./verify_output', help='输出目录')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("HIVMIR 真实格式验证工具")
    print("=" * 80)
    print("\n警告: 此脚本需要在华为昇腾 910B3 服务器上运行")
    print("      需要安装 CANN 工具包和华为编译器\n")

    # 获取 IR 文本
    if args.ir_file:
        print(f"\n从文件读取 IR: {args.ir_file}")
        with open(args.ir_file, 'r', encoding='utf-8') as f:
            ir_text = f.read()
    elif args.kernel:
        ir_text = compile_and_capture_ir(args.kernel, args.output_dir)
    else:
        print("\n错误: 请提供 --kernel 或 --ir-file 参数")
        sys.exit(1)

    if ir_text:
        # 分析格式
        format_info = analyze_hivmir_format(ir_text)

        # 生成更新建议
        generate_parser_update(format_info, f"{args.output_dir}/parser_update.json")

        print("\n" + "=" * 80)
        print("验证完成")
        print("=" * 80)
        print(f"\n下一步:")
        print(f"1. 查看输出文件: {args.output_dir}/")
        print(f"2. 根据真实格式更新 complete_data_merge.py")
        print(f"3. 更新 config.py 中的字段映射")


if __name__ == "__main__":
    main()