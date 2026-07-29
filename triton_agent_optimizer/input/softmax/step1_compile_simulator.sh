#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → HIVM MLIR + msprof op simulator trace
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  服务器上直接执行:
#    cd triton_agent_optimizer/input/softmax
#    bash step1_compile_simulator.sh
#
#  指定 kernel (默认 softmax_kernel):
#    bash step1_compile_simulator.sh fused_gelu_kernel
#
# ═════════════════════════════════════════════════════════════════════════════════
#  参考:
#   TRITON_KERNEL_DUMP: https://github.com/triton-lang/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md
#   msprof op simulator: https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/debug_guide/profiling.html
# ═════════════════════════════════════════════════════════════════════════════════

set -e

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

echo "=============================================="
echo " Step 1: HIVM MLIR + msprof simulator trace"
echo " kernel: $KERNEL"
echo "=============================================="

# ── 1a. HIVM MLIR 提取 ──
export TRITON_KERNEL_DUMP=1
export TRITON_DUMP_DIR=./hivmir
export TRITON_ALWAYS_COMPILE=1

# ── 1b. msprof op simulator (CANN 8.5 新语法: 命令放最后) ──
msprof op simulator \
    --kernel-name="$KERNEL" \
    --soc-version=Ascend910B3 \
    --output=./msprof_sim \
    python3 run_and_profile.py

echo ""
echo "=============================================="
echo " 产物检查"
echo "=============================================="

echo ""
echo "── HIVM MLIR ──"
if ls hivmir/*.mlir 2>/dev/null; then
    for f in hivmir/*.mlir; do
        echo "  $f ($(wc -c < "$f") bytes)"
    done
else
    echo "  WARNING: 未找到 .mlir 文件 (检查 hivmir/ 目录)"
    ls -la hivmir/ 2>/dev/null || echo "  hivmir/ 不存在"
fi

echo ""
echo "── msprof simulator trace ──"
OPPROF=$(ls -dt msprof_sim/OPPROF_*/ 2>/dev/null | head -1)
if [ -n "$OPPROF" ]; then
    echo "  $OPPROF"
    echo "  instr_exe.csv:"
    find "$OPPROF/simulator" -name "*_instr_exe.csv" 2>/dev/null | while read f; do
        lines=$(wc -l < "$f")
        echo "    $f ($lines lines)"
    done
    echo "  trace.json:"
    find "$OPPROF/simulator" -name "trace.json" 2>/dev/null | while read f; do
        echo "    $f ($(wc -c < "$f") bytes)"
    done
else
    echo "  WARNING: 未找到 OPPROF 目录"
fi

echo ""
echo "=============================================="
echo " Step 1 完成"
echo "=============================================="
