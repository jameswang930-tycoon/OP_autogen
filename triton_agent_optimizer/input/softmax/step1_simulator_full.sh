#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → msprof op simulator → trace.json + instr_exe.csv
#  参照 matmul kernel 已验证可用的命令格式
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法: bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#  先清理:
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ sim_build/
#    rm -rf ~/.triton/cache/
#
# ═════════════════════════════════════════════════════════════════════════════════
#  ★ matmul 验证通过的 msprof op simulator 命令 (本脚本参照此格式):
#    msprof op simulator --application="python3 bench_matmul.py" \
#       --kernel-name="matmul_kernel" --soc-version=Ascend910B3 \
#       --launch-count=5 --core-id=0 --output=./sim_result
# ═════════════════════════════════════════════════════════════════════════════════

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

RED='\033[31m'; GRN='\033[32m'; YLW='\033[33m'; NC='\033[0m'
_err() { echo -e "${RED}[ERR]${NC} $*" | tee -a "$LOG_DIR/summary.log"; }
_wrn() { echo -e "${YLW}[WARN]${NC} $*"; }
_ok()  { echo -e "${GRN}[OK]${NC}  $*" | tee -a "$LOG_DIR/summary.log"; }
_inf() { echo -e "[INFO] $*"; }

# ── 环境 ──
MSPROF=$(command -v msprof 2>/dev/null || echo "")
PY=$(command -v python3 2>/dev/null || echo "")
echo "msprof: ${MSPROF:-未找到}"
echo "python3: ${PY:-未找到}"
[ -z "$MSPROF" ] && { _err "msprof 未找到"; exit 1; }
[ -z "$PY" ] && { _err "python3 未找到"; exit 1; }

# ── 清理 ──
rm -rf "$HERE/msprof_sim" "$HERE/hivmir" "$HERE/sim_build" 2>/dev/null || true
rm -rf ~/.triton/cache/ 2>/dev/null || true
mkdir -p "$HERE/hivmir"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: msprof op simulator (直接跑, 参照 matmul 验证格式)
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════ msprof op simulator ══════════"
echo "kernel: $KERNEL"

msprof op simulator \
    --application="python3 $HERE/run_and_profile.py" \
    --kernel-name="$KERNEL" \
    --soc-version=Ascend910B3 \
    --launch-count=5 \
    --core-id=0 \
    --output="$HERE/msprof_sim" \
    > "$LOG_DIR/msprof.log" 2>&1
R1=$?
echo "退出码: $R1"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)

if [ -n "$OPPROF" ]; then
    _ok "OPPROF: $(basename $OPPROF)"
    echo ""
    echo "=== 全部文件 ==="
    find "$OPPROF" -type f | while read f; do
        echo "  $f ($(wc -c < "$f") bytes)"
    done

    TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
    INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
    CSV=$(find "$OPPROF" -name "*.csv" 2>/dev/null | wc -l)

    echo ""
    _ok "★★★ trace.json: $TRACES ★★★"
    _ok "★★★ instr_exe.csv: $INSTRS ★★★"
    _ok "CSV: $CSV"
else
    _err "OPPROF 未生成"
    echo ""
    echo "=== msprof 日志 ==="
    cat "$LOG_DIR/msprof.log"
    echo ""
    echo "=== 诊断 ==="
    grep -i "error\|fail\|cannot\|UNKNOWN\|Failed\|invalid\|dumped" "$LOG_DIR/msprof.log" 2>/dev/null | head -15 || echo "(无)"
    echo ""
    _wrn "msprof op simulator 对 $KERNEL 不工作"
    _inf "对比: matmul_kernel (tl.dot, Cube) 可正常产出"
    _inf "当前 kernel 是纯 Vector + reduction, 模拟器可能不支持此类型"
    _inf "备选: 用 HIVM MLIR (语义) + msprof op 真机 PipeUtilization.csv (时序) 合并分析"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 2: 顺便提取 HIVM MLIR (语义分析用)
# ═════════════════════════════════════════════════════════════════════════════════
echo ""
echo "══════════ HIVM MLIR 提取 ══════════"

export TRITON_ALWAYS_COMPILE=1
$PY run_and_profile.py > "$LOG_DIR/hivm_run.log" 2>&1

# 找 .mlir 文件
MLIRS=$(find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" 2>/dev/null)
MLIR_CNT=$(echo "$MLIRS" | grep -c '.' 2>/dev/null || echo 0)
echo "$MLIRS" | while read f; do [ -f "$f" ] && cp "$f" "$HERE/hivmir/" 2>/dev/null; done
[ "$MLIR_CNT" -gt 0 ] && _ok "HIVM: $MLIR_CNT 个文件 → hivmir/" || _wrn "HIVM: 0 个"

echo ""
echo "══════════ 汇总 ══════════"
echo "msprof_sim: $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
echo "hivmir:     $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
find "$HERE" -name "trace.json" 2>/dev/null | while read f; do echo "★★★ $f ($(wc -c < "$f") bytes)"; done
echo "日志: $LOG_DIR"
