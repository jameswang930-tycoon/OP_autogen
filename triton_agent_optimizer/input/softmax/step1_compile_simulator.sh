#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → HIVM MLIR + msprof simulator trace
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  服务器上执行:
#    cd triton_agent_optimizer/input/softmax
#    bash step1_compile_simulator.sh [softmax_kernel|fused_gelu_kernel]
#
# ═════════════════════════════════════════════════════════════════════════════════
#  注意事项 (常见报错):
#   1. TRITON_KERNEL_DUMP 在 Ascend 后端不工作 → 必须用 TRITON_DEBUG=1
#      来源: https://gitcode.com/Ascend/triton-ascend/blob/main/docs/en/environment_variable_and_compiler_options_reference.md
#      → TRITON_KERNEL_DUMP "used for CUDA backend. On Ascend side, prefer TRITON_DEBUG=1"
#   2. TRITON_DEBUG=1 输出到 ~/.triton/dump/<hash>/  (固定路径, 不可改)
#   3. triton-ascend 3.5.x 曾出现 kernel.npuir.mlir 不生成的问题 (pass 重命名)
#      来源: https://gitcode.com/Ascend/triton-ascend/blob/release/3.5.x/docs/zh/FAQ.md
#      修复: bishengir pass 名已从 hivm-inject-sync 更新为 hivm-graph-sync-solver
#   4. msprof op simulator 需要 LD_LIBRARY_PATH 指向 simulator lib
#      来源: https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/debug_guide/profiling.html
#      命令: export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
#   5. CANN 8.5.x + msprof op simulator 已知兼容性问题 (华为云博客实战记录)
#      如 simulator 模式失败, 备选: 分开跑 TRITON_DEBUG + msprof op (真机模式)
# ═════════════════════════════════════════════════════════════════════════════════

# 注: 不使用 set -e, 避免 cp/ls 失败时静默退出

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# ── 找 CANN 安装路径 ──
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ]; then
    CANN_HOME="$ASCEND_TOOLKIT_HOME"
elif [ -n "${ASCEND_HOME:-}" ]; then
    CANN_HOME="$ASCEND_HOME"
else
    for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann /usr/local/Ascend; do
        if [ -d "$d" ]; then CANN_HOME="$d"; break; fi
    done
fi

echo "=============================================="
echo " Step 1: HIVM MLIR + msprof simulator trace"
echo " kernel: $KERNEL"
echo " CANN:   ${CANN_HOME:-未找到}"
echo "=============================================="

# ── 1a. HIVM MLIR 提取 ──
# 清理旧 dump, 确保拿到最新产物
rm -rf ~/.triton/dump/ 2>/dev/null || true
rm -rf "$HERE/hivmir" 2>/dev/null || true

echo ""
echo "── 1a. 提取 HIVM MLIR (TRITON_DEBUG=1 → ~/.triton/dump/) ──"
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1

python3 run_and_profile.py
TRITON_RC=$?

# 拷贝 dump 产物到项目目录
DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)
mkdir -p "$HERE/hivmir"

echo "DUMP_DIR=$DUMP_DIR"
echo "dump 目录下所有文件:"
if [ -n "$DUMP_DIR" ]; then
    find "$DUMP_DIR" -type f 2>/dev/null
    echo ""
    echo "--- 拷贝 .mlir / .ttir / .ttadapter / .npuir 文件 ---"
    find "$DUMP_DIR" -maxdepth 1 -type f \( -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" \) 2>/dev/null | while read f; do
        cp "$f" "$HERE/hivmir/" 2>/dev/null && echo "  已拷贝: $(basename "$f")"
    done
    # 如果顶层没有 .mlir, 递归找
    if ! ls "$HERE/hivmir/"*.mlir 2>/dev/null && ! ls "$HERE/hivmir/"*.ttir 2>/dev/null; then
        echo "  顶层无 .mlir, 递归搜索..."
        find "$DUMP_DIR" -type f \( -name "*.mlir" -o -name "*.ttir" \) 2>/dev/null | while read f; do
            cp "$f" "$HERE/hivmir/" 2>/dev/null && echo "  已拷贝: $(basename "$f")"
        done
    fi
    echo "HIVM 产物:"
    ls -la "$HERE/hivmir/"
else
    echo "WARNING: TRITON_DEBUG=1 未生成 dump 目录"
fi

# ── 1b. msprof op simulator ──
echo ""
echo "── 1b. msprof op simulator 采集 ──"

# 设置 simulator LD_LIBRARY_PATH (必须!)
SIM_LIB=""
for d in "$CANN_HOME/tools/simulator/Ascend910B3/lib" \
         "$CANN_HOME/tools/simulator/dav_2201/lib" \
         "/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/cann/tools/simulator/Ascend910B3/lib"; do
    if [ -d "$d" ]; then SIM_LIB="$d"; break; fi
done

if [ -n "$SIM_LIB" ]; then
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    echo "SIM lib: $SIM_LIB"
else
    echo "WARNING: 未找到 simulator lib 路径"
    echo "  查找位置: \$CANN_HOME/tools/simulator/Ascend910B3/lib"
fi

rm -rf "$HERE/msprof_sim" 2>/dev/null || true

msprof op simulator \
    --kernel-name="$KERNEL" \
    --soc-version=Ascend910B3 \
    --output="$HERE/msprof_sim" \
    python3 run_and_profile.py 2>&1 || true

# ── 产物检查 ──
echo ""
echo "=============================================="
echo " 产物检查"
echo "=============================================="

echo ""
echo "── HIVM MLIR ──"
if ls "$HERE/hivmir/"*.mlir 2>/dev/null; then
    for f in "$HERE/hivmir/"*.mlir; do
        echo "  $f ($(wc -c < "$f") bytes)"
    done
else
    echo "  [FAIL] 未找到 .mlir 文件"
fi

echo ""
echo "── msprof simulator trace ──"
OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
if [ -n "$OPPROF" ]; then
    echo "  $OPPROF"
    find "$OPPROF/simulator" -name "*_instr_exe.csv" 2>/dev/null | while read f; do
        echo "    $f ($(wc -l < "$f") lines)"
    done
else
    echo "  [FAIL] 未找到 OPPROF 目录"
    echo ""
    echo "  常见原因 + 解决方案:"
    echo "  1. LD_LIBRARY_PATH 未设 → 已自动设置"
    echo "  2. CANN 8.5.x msprof 兼容性 → 尝试 msprof op (真机模式) 替代"
    echo "  3. triton kernel 编译失败 → 先确认 python3 run_and_profile.py 单独能跑"
    echo ""
    echo "  备选方案 (真机 msprof):"
    echo "    msprof op --kernel-name=$KERNEL --output=./msprof_sim python3 run_and_profile.py"
fi

echo ""
echo "=============================================="
echo " Step 1 完成"
echo "=============================================="
