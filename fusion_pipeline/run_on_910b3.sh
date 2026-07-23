#!/bin/bash
# 在华为昇腾 910B3 服务器上运行完整流程的脚本
# 使用方法: bash run_on_910b3.sh <triton_kernel.py> [output_dir]

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法: $0 <triton_kernel.py> [output_dir]"
    echo "示例: $0 example_kernels/vadd_kernel.py ./my_analysis"
    exit 1
fi

KERNEL_FILE=$1
OUTPUT_DIR=${2:-"./fusion_analysis_output_$(date +%Y%m%d_%H%M%S)"}

# 检查文件是否存在
if [ ! -f "$KERNEL_FILE" ]; then
    print_error "文件不存在: $KERNEL_FILE"
    exit 1
fi

print_info "配置信息:"
echo "  - Kernel 文件: $KERNEL_FILE"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 当前时间: $(date)"
echo ""

# Step 1: 提取 HIVMIR
print_info "Step 1: 编译 Triton kernel 并提取 HIVMIR..."
HIVMIR_DIR="$OUTPUT_DIR/hivmir"
mkdir -p "$HIVMIR_DIR"

python fusion_pipeline/extract_hivmir_from_compiler.py \
    "$KERNEL_FILE" \
    --output-dir "$HIVMIR_DIR"

if [ -f "$HIVMIR_DIR/hivmir_output.mlir" ]; then
    print_info "✓ HIVMIR 已生成: $HIVMIR_DIR/hivmir_output.mlir"
else
    print_warn "HIVMIR 文件未生成，使用示例数据"
fi

# Step 2: 运行性能基准测试
print_info "Step 2: 运行性能基准测试..."
PROF_DIR="$OUTPUT_DIR/prof_data"
mkdir -p "$PROF_DIR"

# 运行 kernel（如果包含 benchmark 代码）
python "$KERNEL_FILE" 2>&1 | tee "$OUTPUT_DIR/benchmark_output.log" || {
    print_warn "基准测试未执行或失败，继续..."
}

# 运行 msprof（如果可用）
if command -v msprof &> /dev/null; then
    print_info "运行 msprof 分析..."
    msprof --op-mode "$PROF_DIR" 2>&1 | tee "$OUTPUT_DIR/msprof_output.log" || {
        print_warn "msprof 分析失败，继续..."
    }
else
    print_warn "msprof 未安装，跳过性能分析"
fi

# Step 3: 合并数据并生成报告
print_info "Step 3: 合并数据并生成报告..."

if [ -f "$HIVMIR_DIR/hivmir_output.mlir" ]; then
    python fusion_pipeline/complete_data_merge.py \
        --hivmir "$HIVMIR_DIR/hivmir_output.mlir" \
        --output-dir "$OUTPUT_DIR"
else
    print_warn "使用示例 HIVMIR 运行..."
    python fusion_pipeline/run_example.py
    cp -r example_output/* "$OUTPUT_DIR/"
fi

# 检查输出
if [ -f "$OUTPUT_DIR/complete_fusion_report.txt" ]; then
    print_info "✓ 报告已生成: $OUTPUT_DIR/complete_fusion_report.txt"
    print_info "✓ 图表已生成: $OUTPUT_DIR/complete_fusion_analysis.png"
else
    print_error "报告生成失败"
    exit 1
fi

# 打印摘要
echo ""
echo "========================================================================================================"
print_info "分析完成！"
echo "========================================================================================================"
echo ""
echo "输出文件:"
echo "  - $OUTPUT_DIR/complete_fusion_report.txt (详细报告)"
echo "  - $OUTPUT_DIR/complete_fusion_analysis.png (可视化图表)"
echo "  - $OUTPUT_DIR/hivmir/hivmir_output.mlir (HIVMIR 中间产物)"
echo ""
echo "查看报告:"
echo "  cat $OUTPUT_DIR/complete_fusion_report.txt"
echo ""
echo "========================================================================================================"