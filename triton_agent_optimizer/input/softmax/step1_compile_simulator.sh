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
#  已知问题:
#   - aicpu_legacy.tar.gz warning → CANN 可选包, 忽略, 不影响功能
#   - triton-ascend 3.5.x: kernel.npuir.mlir 可能不生成 (pass 重命名 bug)
#     来源: https://gitcode.com/Ascend/triton-ascend/blob/release/3.5.x/docs/zh/FAQ.md
#     修复: 合并请求 !1656, 如未修复则只有 .ttir.mlir + .ttadapter.mlir
#   - 大量 INFO 日志 → triton JIT 编译正常输出, PIPE 到文件即可
# ═════════════════════════════════════════════════════════════════════════════════

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
mkdir -p "$LOG_DIR"

ERRORS=""
WARNS=""
OKS=""

add_err()  { ERRORS="$ERRORS\n  [ERROR] $*"; }
add_warn() { WARNS="$WARNS\n  [WARN] $*"; }
add_ok()   { OKS="$OKS\n  [OK] $*"; }

echo "=============================================="
echo " Step 1: HIVM MLIR + msprof simulator trace"
echo " kernel: $KERNEL"
echo " logs:   $LOG_DIR"
echo "=============================================="

# ── 检查 triton-ascend 版本 ──
echo ""
echo "── 环境信息 ──"
python3 -c "import triton; print('triton version:', getattr(triton, '__version__', 'unknown'))" 2>/dev/null || add_warn "无法获取 triton 版本"
python3 -c "import torch_npu; print('torch_npu:', torch_npu.__version__)" 2>/dev/null || true
echo "CANN_HOME: ${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME:-未设置}}"

# ── 清理旧产物 ──
rm -rf ~/.triton/dump/ 2>/dev/null || true
rm -rf "$HERE/hivmir" 2>/dev/null || true
rm -rf "$HERE/msprof_sim" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  1a. HIVM MLIR 提取
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 1a. 提取 HIVM MLIR (TRITON_DEBUG=1) ──"

export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1

# 抑制 INFO 日志, 只保留关键输出
python3 run_and_profile.py > "$LOG_DIR/triton_run.log" 2>&1
TRITON_RC=$?

# 从日志中提取 dump 路径
DUMP_PATH=$(grep -oP 'Dumping intermediate results to \K.*' "$LOG_DIR/triton_run.log" 2>/dev/null | head -1)
echo "Triton 退出码: $TRITON_RC"

# ── 查找 dump 文件 ──
FOUND_MLIRS=""
DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)

if [ -n "$DUMP_DIR" ]; then
    echo "dump 目录: $DUMP_DIR"
    FOUND_MLIRS=$(find "$DUMP_DIR" -type f \( -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" \) 2>/dev/null)
fi

# 也检查 triton cache 目录 (编译产物可能在里面)
CACHE_DIR=$(ls -dt ~/.triton/cache/*/ 2>/dev/null | head -1)
if [ -n "$CACHE_DIR" ]; then
    echo "cache 目录: $CACHE_DIR"
    CACHE_MLIRS=$(find "$CACHE_DIR" -type f \( -name "*.mlir" -o -name "*.o" \) 2>/dev/null)
fi

# ── 拷贝 .mlir 文件 ──
mkdir -p "$HERE/hivmir"

if [ -n "$FOUND_MLIRS" ]; then
    echo ""
    echo "找到的 MLIR 文件:"
    echo "$FOUND_MLIRS"
    echo "$FOUND_MLIRS" | while read f; do
        [ -f "$f" ] && cp "$f" "$HERE/hivmir/" && echo "  已拷贝: $(basename "$f")"
    done
    add_ok "HIVM MLIR: $(echo "$FOUND_MLIRS" | wc -l) 个文件"
else
    add_err "HIVM MLIR: 未找到任何 .mlir 文件"
    echo ""
    echo "  检查以下位置:"
    echo "    ~/.triton/dump/     → $(ls ~/.triton/dump/ 2>/dev/null | wc -l) 个子目录"
    echo "    ~/.triton/cache/    → $(ls ~/.triton/cache/ 2>/dev/null | wc -l) 个子目录"

    # 尝试从 cache 拷贝 .o 文件 (可能编译成功但没有 dump)
    if [ -n "$CACHE_DIR" ] && ls "$CACHE_DIR"/*.o 2>/dev/null; then
        cp "$CACHE_DIR"/*.o "$HERE/hivmir/" 2>/dev/null
        add_warn "只有 .o 编译产物, 无 .mlir (triton-ascend 3.5.x pass 重命名 bug)"
        echo "  备选: 拷贝了 .o 文件, 可用 bishengir-opt 反编译回 MLIR"
    fi

    # 检查 triton 运行日志是否有编译错误
    if grep -qi "error\|ERROR\|FAIL" "$LOG_DIR/triton_run.log" 2>/dev/null; then
        echo ""
        echo "  !! triton 日志中有错误:"
        grep -i "error\|ERROR\|FAIL" "$LOG_DIR/triton_run.log" | head -10
    fi
fi

echo ""
echo "hivmir/ 产物:"
ls -la "$HERE/hivmir/" 2>/dev/null || echo "  (空)"

# ═════════════════════════════════════════════════════════════════════════════════
#  1b. msprof op simulator
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 1b. msprof op simulator 采集 ──"

# 找 simulator lib
SIM_LIB=""
for d in "${ASCEND_TOOLKIT_HOME:-}/tools/simulator/Ascend910B3/lib" \
         "${ASCEND_HOME:-}/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/cann/tools/simulator/Ascend910B3/lib" \
         "/usr/local/Ascend/tools/simulator/Ascend910B3/lib"; do
    if [ -d "$d" ]; then SIM_LIB="$d"; break; fi
done

MSPROF_SIM_OK=0
if [ -z "$SIM_LIB" ]; then
    add_warn "msprof simulator: 未找到 simulator lib (可能不在此环境)"
    echo "  查找路径: \$CANN_HOME/tools/simulator/Ascend910B3/lib"
else
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    echo "SIM lib: $SIM_LIB"

    msprof op simulator \
        --kernel-name="$KERNEL" \
        --soc-version=Ascend910B3 \
        --output="$HERE/msprof_sim" \
        python3 run_and_profile.py > "$LOG_DIR/msprof_simulator.log" 2>&1 || true

    OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
    if [ -n "$OPPROF" ]; then
        INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | head -1)
        if [ -n "$INSTRS" ]; then
            add_ok "msprof simulator: $(basename "$OPPROF") ($(wc -l < "$INSTRS") 条指令)"
            MSPROF_SIM_OK=1
        else
            add_warn "msprof simulator: OPPROF 目录生成但无 instr_exe.csv"
        fi
    else
        add_warn "msprof simulator: 未生成 OPPROF 目录"
    fi
fi

# ── 备选: 真机 msprof op ──
if [ "$MSPROF_SIM_OK" = "0" ]; then
    echo ""
    echo "── 1b备选: 真机 msprof op (simulator 不可用) ──"
    MSPROF_BIN=$(command -v msprof 2>/dev/null || echo "")
    if [ -n "$MSPROF_BIN" ]; then
        rm -rf "$HERE/msprof_timing" 2>/dev/null || true
        msprof op \
            --kernel-name="$KERNEL" \
            --output="$HERE/msprof_timing" \
            python3 run_and_profile.py > "$LOG_DIR/msprof_op.log" 2>&1 || true

        OPPROF_HW=$(ls -dt "$HERE/msprof_timing"/OPPROF_*/ 2>/dev/null | head -1)
        if [ -n "$OPPROF_HW" ]; then
            add_ok "msprof 真机: $(basename "$OPPROF_HW")"
            ls "$OPPROF_HW"/*.csv 2>/dev/null | while read f; do
                echo "  $(basename "$f") ($(wc -l < "$f") lines)"
            done
        else
            add_warn "msprof 真机: 也未生成 OPPROF 目录"
        fi
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  汇总
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo " Step 1 完成 — 诊断汇总"
echo "=============================================="

echo -e "$OKS"
echo -e "$WARNS"
echo -e "$ERRORS"

echo ""
echo "── 产物清单 ──"
echo "hivmir/       ($(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files)"
find "$HERE/hivmir" -type f 2>/dev/null | while read f; do
    echo "  $(basename "$f") ($(wc -c < "$f") bytes)"
done

echo "msprof_sim/   ($(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files)"
echo "msprof_timing/ ($(find "$HERE/msprof_timing" -type f 2>/dev/null | wc -l) files)"
echo "日志:         $LOG_DIR/"

echo ""
echo "── triton 运行日志 (最后 20 行) ──"
tail -20 "$LOG_DIR/triton_run.log" 2>/dev/null || echo "  (无日志)"

echo ""
echo "下一步:"
if ls "$HERE/hivmir/"*.mlir 2>/dev/null || ls "$HERE/hivmir/"*.ttir 2>/dev/null; then
    echo "  bash step2_parse_hivm.sh"
fi
if ls "$HERE/msprof_sim"/OPPROF_*/simulator/*/*_instr_exe.csv 2>/dev/null; then
    echo "  bash step3_parse_msprof.sh"
elif ls "$HERE/msprof_timing"/OPPROF_*/*.csv 2>/dev/null; then
    echo "  (真机 msprof 数据需手动解析)"
fi
