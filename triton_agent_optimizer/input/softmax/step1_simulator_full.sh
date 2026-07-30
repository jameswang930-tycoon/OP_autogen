#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → trace.json (cache → ttadapter.mlir → .o → 可执行文件)
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法: bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#  先清理:
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ sim_build/
#    rm -rf ~/.triton/cache/
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
_hdr() { echo ""; echo "══════════ $* ══════════"; }

# ── 环境 ──
CANN="${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME}}"
[ -z "$CANN" ] && for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do [ -d "$d" ] && { CANN="$d"; break; }; done
BISHENG=$(command -v bisheng 2>/dev/null || echo "")
BISHENGIR=$(command -v bishengir-compile 2>/dev/null || echo "")
MSPROF=$(command -v msprof 2>/dev/null || echo "")
PY=$(command -v python3 2>/dev/null || echo "")
SIM_LIB=$(dirname "$(find /usr/local/Ascend -name "libruntime_camodel.so" -type f 2>/dev/null | head -1)" 2>/dev/null || echo "")

echo "CANN:      ${CANN:-未找到}"
echo "bisheng:   ${BISHENG:-未找到}"
echo "bishengir: ${BISHENGIR:-未找到}"
echo "msprof:    ${MSPROF:-未找到}"
echo "SIM lib:   ${SIM_LIB:-未找到}"

[ -z "$PY" ] && { _err "python3 未找到"; exit 1; }
[ -z "$BISHENGIR" ] && { _err "bishengir-compile 未找到"; exit 1; }
[ -z "$BISHENG" ] && { _err "bisheng 未找到"; exit 1; }
[ -z "$MSPROF" ] && { _err "msprof 未找到"; exit 1; }
[ -z "$SIM_LIB" ] && { _err "simulator lib 未找到"; exit 1; }

# ── 清理 ──
rm -rf "$HERE/hivmir" "$HERE/sim_build" "$HERE/msprof_sim" 2>/dev/null || true
rm -rf ~/.triton/cache/ ~/.triton/dump/ 2>/dev/null || true
mkdir -p "$HERE/hivmir" "$HERE/sim_build"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: 跑 kernel → cache 生成 ttadapter.mlir
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 1: 跑 kernel → 生成 ttadapter.mlir"

export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0
$PY run_and_profile.py > "$LOG_DIR/run.log" 2>&1
R1=$?
echo "退出码: $R1"
grep "PASS\|FAIL\|max_err" "$LOG_DIR/run.log" 2>/dev/null | head -5 || true

# ── 找 ttadapter.mlir ──
TTADAPTER=$(find ~/.triton/cache -name "*ttadapter*" -type f 2>/dev/null | head -1)
if [ -z "$TTADAPTER" ]; then
    _err "未找到 ttadapter.mlir! cache 内容:"
    find ~/.triton/cache -type f 2>/dev/null | head -30
    exit 1
fi
_ok "ttadapter: $TTADAPTER ($(wc -c < "$TTADAPTER") bytes)"

# 拷贝 mlir 到 hivmir (语义数据用)
find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" 2>/dev/null | while read f; do cp "$f" "$HERE/hivmir/" 2>/dev/null; done
_ok "hivmir: $(find "$HERE/hivmir" -type f | wc -l) files"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 2: ttadapter.mlir → bishengir-compile → kernel.o
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 2: bishengir-compile → kernel.o"

_inf "输入: $TTADAPTER"
bishengir-compile "$TTADAPTER" \
    --enable-hivm-compile \
    -o "$HERE/sim_build/kernel.o" \
    > "$LOG_DIR/compile.log" 2>&1
R2=$?

if [ "$R2" != "0" ]; then
    _err "编译失败! 退出码=$R2"
    tail -30 "$LOG_DIR/compile.log"
    grep -i "error\|cannot\|fatal" "$LOG_DIR/compile.log" 2>/dev/null | head -10 || true
    exit 1
fi
_ok "kernel.o: $(wc -c < "$HERE/sim_build/kernel.o") bytes"
file "$HERE/sim_build/kernel.o" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 3: bisheng 链接 simulator libs → kernel_app
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 3: bisheng 链接 → kernel_app"

bisheng -Wl,--disable-new-dtags \
    -L"$SIM_LIB" -Wl,-rpath,"$SIM_LIB" \
    -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
    -lascendcl -lascendc_runtime -lprofapi -lunified_dlog \
    -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv \
    "$HERE/sim_build/kernel.o" \
    -o "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/link.log" 2>&1
R3=$?

if [ "$R3" != "0" ]; then
    _err "链接失败! 退出码=$R3"
    grep -i "error\|undefined\|cannot find" "$LOG_DIR/link.log" 2>/dev/null | head -20 || tail -20 "$LOG_DIR/link.log"
    _inf "SIM_LIB 中 .so 文件:"
    ls "$SIM_LIB"/*.so 2>/dev/null | head -20 || echo "  (无)"
    exit 1
fi
_ok "kernel_app: $(wc -c < "$HERE/sim_build/kernel_app") bytes"
file "$HERE/sim_build/kernel_app" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 4: msprof op simulator → trace.json
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 4: msprof op simulator → trace.json"

export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
rm -rf "$HERE/msprof_sim"

msprof op simulator \
    --soc-version=Ascend910B3 \
    --output="$HERE/msprof_sim" \
    "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/msprof.log" 2>&1
R4=$?
echo "退出码: $R4"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
if [ -z "$OPPROF" ]; then
    _err "OPPROF 未生成"
    tail -40 "$LOG_DIR/msprof.log"
    grep -i "error\|fail\|cannot\|UNKNOWN" "$LOG_DIR/msprof.log" 2>/dev/null | head -15
    exit 1
fi

_ok "OPPROF: $(basename $OPPROF)"
echo ""
echo "=== 全部文件 ==="
find "$OPPROF" -type f | while read f; do echo "  $f ($(wc -c < "$f") bytes)"; done

TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)

if [ "$TRACES" -gt 0 ] || [ "$INSTRS" -gt 0 ]; then
    _ok "★★★ 成功! trace.json: $TRACES, instr_exe.csv: $INSTRS ★★★"
else
    _err "OPPROF 存在但无 trace/instr_exe"
    grep -i "error\|fail\|cannot\|assert\|UNKNOWN" "$LOG_DIR/msprof.log" 2>/dev/null | head -15
fi

echo ""
echo "══════════ 汇总 ══════════"
echo "hivmir:     $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
echo "sim_build:  $(find "$HERE/sim_build" -type f 2>/dev/null | wc -l) files"
echo "msprof_sim: $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
find "$HERE" -name "trace.json" 2>/dev/null | while read f; do echo "★★★ $f ($(wc -c < "$f") bytes)"; done
echo "日志: $LOG_DIR"
