#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton kernel .py → msprof op simulator → trace.json
#  直接找 cache 里的 .o, 不依赖 TRITON_DEBUG dump (3.5.x 已知 bug)
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法: bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#
#  ★ 先清理:
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ sim_build/
#    rm -rf ~/.triton/cache/
# ═════════════════════════════════════════════════════════════════════════════════

KERNEL="${1:-softmax_kernel}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

RED='\033[31m'; GRN='\033[32m'; YLW='\033[33m'; CYN='\033[36m'; NC='\033[0m'
_err() { echo -e "${RED}[ERR]${NC} $*" | tee -a "$LOG_DIR/summary.log"; }
_wrn() { echo -e "${YLW}[WARN]${NC} $*" | tee -a "$LOG_DIR/summary.log"; }
_ok()  { echo -e "${GRN}[OK]${NC}  $*" | tee -a "$LOG_DIR/summary.log"; }
_inf() { echo -e "${CYN}[INFO]${NC} $*"; }
_hdr() { echo ""; echo "══════════ $* ══════════"; }

# ── 找工具 ──
CANN_HOME="${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME}}"
[ -z "$CANN_HOME" ] && for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do [ -d "$d" ] && { CANN_HOME="$d"; break; }; done

BISHENG=$(command -v bisheng 2>/dev/null || echo "")
MSPROF=$(command -v msprof 2>/dev/null || echo "")
PY=$(command -v python3 2>/dev/null || echo "")

SIM_LIB=""
for d in "$CANN_HOME/tools/simulator/Ascend910B3/lib"; do [ -f "$d/libruntime_camodel.so" ] && { SIM_LIB="$d"; break; }; done
[ -z "$SIM_LIB" ] && SIM_LIB=$(dirname "$(find /usr/local/Ascend -name "libruntime_camodel.so" -type f 2>/dev/null | head -1)" 2>/dev/null || echo "")

echo "CANN:    ${CANN_HOME:-未找到}"
echo "bisheng: ${BISHENG:-未找到}"
echo "msprof:  ${MSPROF:-未找到}"
echo "SIM lib: ${SIM_LIB:-未找到}"

[ -z "$BISHENG" ] && { _err "bisheng 未找到!"; exit 1; }
[ -z "$MSPROF" ] && { _err "msprof 未找到!"; exit 1; }
[ -z "$SIM_LIB" ] && { _err "SIM lib 未找到!"; exit 1; }

# ── 清理 ──
rm -rf "$HERE/hivmir" "$HERE/sim_build" "$HERE/msprof_sim" 2>/dev/null || true
rm -rf ~/.triton/cache/ 2>/dev/null || true
mkdir -p "$HERE/hivmir" "$HERE/sim_build"

# ═════════════════════════════════════════════════════════════════════════════════
#  1. 跑 kernel, 让 triton-ascend JIT 编译产生 .o 在 ~/.triton/cache/
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "1. 编译 Triton kernel (产生 .o 到 ~/.triton/cache/)"

export TRITON_ALWAYS_COMPILE=1
_inf "运行 run_and_profile.py ..."
$PY run_and_profile.py > "$LOG_DIR/run.log" 2>&1
R=$?
echo "退出码: $R"
grep "PASS\|FAIL\|max_err" "$LOG_DIR/run.log" 2>/dev/null | head -5 || true

# ═════════════════════════════════════════════════════════════════════════════════
#  2. 从 cache 找 .o 文件
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "2. 从 ~/.triton/cache/ 找 .o 和 .mlir"

CACHE_COUNT=$(find ~/.triton/cache -maxdepth 0 -type d 2>/dev/null | wc -l)
echo "cache 子目录: $CACHE_COUNT"

if [ "$CACHE_COUNT" -eq 0 ]; then
    _err "cache 为空! 编译可能失败了"
    _inf "运行日志:"
    tail -30 "$LOG_DIR/run.log"
    exit 1
fi

# 找并列出所有 .o
ALL_O=$(find ~/.triton/cache -name "*.o" -type f 2>/dev/null)
O_COUNT=$(echo "$ALL_O" | grep -c '.' 2>/dev/null || echo 0)

echo ""
if [ "$O_COUNT" -gt 0 ]; then
    _ok "找到 $O_COUNT 个 .o 文件:"
    echo "$ALL_O" | while read f; do
        echo "  $(basename "$f") ($(wc -c < "$f") bytes) → $(dirname "$f")"
    done
else
    _wrn "cache 中无 .o 文件"
    echo "cache 中的文件类型:"
    find ~/.triton/cache -type f 2>/dev/null | head -20 | while read f; do
        echo "  $(basename "$f") ($(wc -c < "$f") bytes)"
    done
fi

# 找 .mlir (语义数据用)
ALL_MLIR=$(find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" 2>/dev/null)
MLIR_COUNT=$(echo "$ALL_MLIR" | grep -c '.' 2>/dev/null || echo 0)
[ "$MLIR_COUNT" -gt 0 ] && _ok "MLIR: $MLIR_COUNT 个" || _wrn "MLIR: 0 个"
echo "$ALL_MLIR" | while read f; do [ -f "$f" ] && cp "$f" "$HERE/hivmir/" 2>/dev/null; done

if [ "$O_COUNT" -eq 0 ]; then
    _err "没有 .o 文件, 无法继续"
    _inf "尝试手动查找整个文件系统:"
    find / -name "*.o" -newer "$HERE/run_and_profile.py" -type f 2>/dev/null | head -10 || echo "  (未找到)"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  3. 链接 simulator libs → 可执行文件
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "3. bisheng 链接 → kernel_app"

FIRST_O=$(echo "$ALL_O" | head -1)
_inf "输入: $FIRST_O"

bisheng -Wl,--disable-new-dtags \
    -L"$SIM_LIB" \
    -Wl,-rpath,"$SIM_LIB" \
    -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
    -lascendcl -lascendc_runtime -lprofapi -lunified_dlog \
    -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv \
    "$FIRST_O" \
    -o "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/link.log" 2>&1
LINK_RC=$?

if [ "$LINK_RC" != "0" ]; then
    _err "链接失败! 退出码=$LINK_RC"
    grep -i "error\|undefined\|cannot find\|fatal" "$LOG_DIR/link.log" 2>/dev/null | head -20 || tail -20 "$LOG_DIR/link.log"
    exit 1
fi

_ok "kernel_app: $(wc -c < "$HERE/sim_build/kernel_app") bytes"
file "$HERE/sim_build/kernel_app" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  4. msprof op simulator
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "4. msprof op simulator → trace.json"

export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
rm -rf "$HERE/msprof_sim"

msprof op simulator \
    --soc-version=Ascend910B3 \
    --output="$HERE/msprof_sim" \
    "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/msprof.log" 2>&1
MS_RC=$?
echo "退出码: $MS_RC"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
if [ -z "$OPPROF" ]; then
    _err "OPPROF 未生成"
    tail -40 "$LOG_DIR/msprof.log"
    exit 1
fi

_ok "OPPROF: $(basename $OPPROF)"
echo "=== 全部文件 ==="
find "$OPPROF" -type f 2>/dev/null | while read f; do
    echo "  $f ($(wc -c < "$f") bytes)"
done

TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)

if [ "$TRACES" -gt 0 ] || [ "$INSTRS" -gt 0 ]; then
    _ok "★★★ 成功! trace.json: $TRACES, instr_exe.csv: $INSTRS ★★★"
else
    _err "OPPROF 存在但无 trace/instr_exe"
    grep -i "error\|fail\|cannot\|assert\|UNKNOWN" "$LOG_DIR/msprof.log" 2>/dev/null | head -15 || echo "(msprof 日志无错误关键词)"
fi

# ── 汇总 ──
echo ""
echo "══════════ 汇总 ══════════"
echo "hivmir:     $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
echo "sim_build:  $(find "$HERE/sim_build" -type f 2>/dev/null | wc -l) files"
echo "msprof_sim: $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
find "$HERE" -name "trace.json" 2>/dev/null | while read f; do echo "★★★ $f ($(wc -c < "$f") bytes)"; done
echo "日志: $LOG_DIR"
