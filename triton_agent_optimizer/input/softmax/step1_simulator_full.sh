#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → trace.json (经过中间编译, 多次处理)
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法:
#    cd triton_agent_optimizer/input/softmax
#    bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#
# ═════════════════════════════════════════════════════════════════════════════════
#  ★ 跑之前先清理:
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ msprof_hw/ msprof_timing/ sim_build/ sim_config.json kernel_meta/
#    rm -rf ~/.triton/cache/ ~/.triton/dump/
# ═════════════════════════════════════════════════════════════════════════════════
#
#  流程 (4步, 每步独立验证):
#    Step A: Triton .py → dump .mlir (TRITON_DEBUG=1 → ~/.triton/dump/)
#    Step B: ttadapter.mlir → bishengir-compile → kernel.o
#    Step C: kernel.o + simulator libs → bisheng 链接 → kernel_app
#    Step D: msprof op simulator ./kernel_app → trace.json + instr_exe.csv
#
#  来源:
#   架构: https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/architecture_design_and_core_features.html
#   编译: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/AscendNPUIR/ir_003.html
#   模拟: https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00059.html
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

# ── 环境 ──
CANN_HOME="${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME}}"
[ -z "$CANN_HOME" ] && for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do [ -d "$d" ] && { CANN_HOME="$d"; break; }; done

BISHENG=$(command -v bisheng 2>/dev/null || echo "")
BISHENGIR=$(command -v bishengir-compile 2>/dev/null || echo "")
MSPROF=$(command -v msprof 2>/dev/null || echo "")
PY=$(command -v python3 2>/dev/null || echo "")

SIM_LIB=""
for d in "$CANN_HOME/tools/simulator/Ascend910B3/lib" "/usr/local/Ascend/cann/tools/simulator/dav_2201/lib"; do
    [ -f "$d/libruntime_camodel.so" ] && { SIM_LIB="$d"; break; }
done

echo "CANN:     ${CANN_HOME:-未找到}"
echo "bisheng:  ${BISHENG:-未找到}"
echo "bishengir: ${BISHENGIR:-未找到}"
echo "msprof:   ${MSPROF:-未找到}"
echo "SIM lib:  ${SIM_LIB:-未找到}"
echo "python3:  ${PY:-未找到}"

# ── 清理 ──
rm -rf "$HERE/hivmir" "$HERE/sim_build" "$HERE/msprof_sim" 2>/dev/null || true
rm -rf ~/.triton/cache/ ~/.triton/dump/ 2>/dev/null || true
mkdir -p "$HERE/hivmir" "$HERE/sim_build"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step A: Triton .py → dump .mlir 到 ~/.triton/dump/
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step A: Triton .py → .mlir dump"

export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0

$PY run_and_profile.py > "$LOG_DIR/stepA_run.log" 2>&1
A_RC=$?
echo "Triton 退出码: $A_RC"
grep "Dumping intermediate results" "$LOG_DIR/stepA_run.log" 2>/dev/null || true

DUMP_DIR=$(ls -dt ~/.triton/dump/*/ 2>/dev/null | head -1)

if [ -z "$DUMP_DIR" ]; then
    _err "Step A: ~/.triton/dump/ 为空!"
    _inf "triton 运行日志最后20行:"
    tail -20 "$LOG_DIR/stepA_run.log"
    exit 1
fi

_ok "dump: $DUMP_DIR"
echo "dump 文件:"
find "$DUMP_DIR" -maxdepth 1 -type f 2>/dev/null | while read f; do
    echo "  $(basename "$f") ($(wc -c < "$f") bytes)"
done

# 拷贝 .mlir 到 hivmir
find "$DUMP_DIR" -maxdepth 1 -type f \( -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" \) 2>/dev/null | while read f; do
    cp "$f" "$HERE/hivmir/" && _ok "HIVM: $(basename "$f")"
done

# 找 ttadapter.mlir (这是 bishengir-compile 的输入)
TTADAPTER=$(find "$DUMP_DIR" -name "*ttadapter*" -o -name "*ttadapter.mlir" 2>/dev/null | head -1)
if [ -z "$TTADAPTER" ]; then
    # 也可能在 cache 里
    TTADAPTER=$(find ~/.triton/cache -name "*ttadapter*" -type f 2>/dev/null | head -1)
fi
[ -n "$TTADAPTER" ] && _ok "ttadapter: $TTADAPTER ($(wc -c < "$TTADAPTER") bytes)" || _err "ttadapter: 未找到!"

# 也找 npuir.mlir (可能是 HIVM 格式)
NPUIR=$(find "$DUMP_DIR" ~/.triton/cache -name "*npuir*" -type f 2>/dev/null | head -1)
[ -n "$NPUIR" ] && _ok "npuir: $NPUIR ($(wc -c < "$NPUIR") bytes)" || _wrn "npuir: 未找到 (3.5.x pass rename bug)"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step B: .mlir → bishengir-compile → kernel.o
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step B: .mlir → bishengir-compile → kernel.o"

# 选输入: 优先 npuir, 其次 ttadapter, 最后任意 .mlir
MLIR_INPUT="${NPUIR:-${TTADAPTER:-$(find "$DUMP_DIR" -name "*.mlir" -type f 2>/dev/null | head -1)}}"

if [ -z "$MLIR_INPUT" ]; then
    _err "Step B: 没有可用的 .mlir 输入!"
    exit 1
fi

_inf "输入: $MLIR_INPUT ($(wc -c < "$MLIR_INPUT") bytes)"
_inf "编译中..."

bishengir-compile "$MLIR_INPUT" \
    --enable-hivm-compile \
    -o "$HERE/sim_build/kernel.o" \
    > "$LOG_DIR/stepB_compile.log" 2>&1
B_RC=$?

if [ "$B_RC" != "0" ]; then
    _err "Step B: bishengir-compile 失败! 退出码=$B_RC"
    _dump_log() { echo "=== $1 (最后20行) ==="; tail -20 "$1" 2>/dev/null; }
    _dump_log "$LOG_DIR/stepB_compile.log"
    grep -i "error\|cannot\|fatal\|undefined" "$LOG_DIR/stepB_compile.log" 2>/dev/null | head -10 || true
    exit 1
fi

if [ -f "$HERE/sim_build/kernel.o" ]; then
    _ok "kernel.o: $(wc -c < "$HERE/sim_build/kernel.o") bytes"
    file "$HERE/sim_build/kernel.o" 2>/dev/null || true
else
    _err "Step B: kernel.o 未生成!"
    exit 1
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  Step C: kernel.o + simulator libs → bisheng 链接 → kernel_app
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step C: .o + simulator libs → kernel_app"

if [ -z "$SIM_LIB" ] || [ -z "$BISHENG" ]; then
    _err "Step C: 缺 simulator lib 或 bisheng"
    exit 1
fi

bisheng -Wl,--disable-new-dtags \
    -L"$SIM_LIB" \
    -Wl,-rpath,"$SIM_LIB" \
    -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
    -lascendcl -lascendc_runtime -lprofapi -lunified_dlog \
    -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv \
    "$HERE/sim_build/kernel.o" \
    -o "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/stepC_link.log" 2>&1
C_RC=$?

if [ "$C_RC" != "0" ]; then
    _err "Step C: 链接失败! 退出码=$C_RC"
    grep -i "error\|undefined\|cannot find\|fatal" "$LOG_DIR/stepC_link.log" 2>/dev/null | head -20 || tail -20 "$LOG_DIR/stepC_link.log"
    _inf "SIM_LIB 中的 .so:"
    ls -la "$SIM_LIB"/*.so 2>/dev/null | head -20 || echo "  (无)"
    exit 1
fi

_ok "kernel_app: $(wc -c < "$HERE/sim_build/kernel_app") bytes"
file "$HERE/sim_build/kernel_app" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  Step D: msprof op simulator → trace.json + instr_exe.csv
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step D: msprof op simulator → trace.json"

export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
rm -rf "$HERE/msprof_sim"

msprof op simulator \
    --soc-version=Ascend910B3 \
    --output="$HERE/msprof_sim" \
    "$HERE/sim_build/kernel_app" \
    > "$LOG_DIR/stepD_msprof.log" 2>&1
D_RC=$?
echo "msprof 退出码: $D_RC"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)

if [ -n "$OPPROF" ]; then
    _ok "OPPROF: $(basename $OPPROF)"
    echo ""
    echo "=== 完整目录树 ==="
    find "$OPPROF" -type f 2>/dev/null | while read f; do
        sz=$(wc -c < "$f" 2>/dev/null || echo 0)
        echo "  $f ($sz bytes)"
    done

    TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
    INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)

    if [ "$TRACES" -gt 0 ] || [ "$INSTRS" -gt 0 ]; then
        _ok "★★★ 成功! trace.json: $TRACES, instr_exe.csv: $INSTRS ★★★"
    else
        _err "OPPROF 有目录但无 trace/instr_exe 文件"
        echo "msprof 日志错误:"
        grep -i "error\|fail\|cannot\|assert\|UNKNOWN" "$LOG_DIR/stepD_msprof.log" 2>/dev/null | head -15 || echo "(无)"
    fi
else
    _err "OPPROF 未生成"
    echo "msprof 日志最后30行:"
    tail -30 "$LOG_DIR/stepD_msprof.log"
    echo ""
    echo "msprof 错误关键词:"
    grep -i "error\|fail\|ERROR\|FAIL\|cannot\|not found\|assert\|UNKNOWN\|ERR" "$LOG_DIR/stepD_msprof.log" 2>/dev/null | head -15 || echo "(无)"
fi

# ── 汇总 ──
_hdr "汇总"
echo "hivmir:     $(find "$HERE/hivmir" -type f | wc -l) files"
echo "sim_build:  $(find "$HERE/sim_build" -type f | wc -l) files"
echo "msprof_sim: $(find "$HERE/msprof_sim" -type f | wc -l) files"

TRACES=$(find "$HERE" -name "trace.json" 2>/dev/null)
[ -n "$TRACES" ] && echo "$TRACES" | while read f; do echo "★★★ $f ($(wc -c < "$f") bytes)"; done || echo "!! trace.json: 未找到"
echo "日志: $LOG_DIR"
