#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → 编译 → 可执行文件 → msprof op simulator → trace.json
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法:
#    cd triton_agent_optimizer/input/softmax
#    bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#
# ═════════════════════════════════════════════════════════════════════════════════
#  ★ 跑之前先清理之前的垃圾 (复制粘贴到终端执行):
# ═════════════════════════════════════════════════════════════════════════════════
#
#    cd triton_agent_optimizer/input/softmax
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ msprof_hw/ msprof_timing/ sim_build/ sim_config.json kernel_meta/
#    rm -rf ~/.triton/cache/ ~/.triton/dump/
#    ls -la  # 确认只剩 .sh .py .md 文件
#
# ═════════════════════════════════════════════════════════════════════════════════
#  核心思路:
#   1. ACL_OP_DEBUG_LEVEL=3 → CANN 把 .o 文件输出到 kernel_meta/
#   2. bisheng 链接 simulator libs → 可执行文件
#   3. msprof op simulator ./可执行文件 → trace.json + instr_exe.csv
#
#  来源:
#   ACL_OP_DEBUG_LEVEL: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha003/devaids/atctool/atlasatcparam_16_0086.html
#   bisheng simulator: https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00059.html
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

# ═════════════════════════════════════════════════════════════════════════════════
#  0. 环境诊断
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "0. 环境"

CANN_HOME="${ASCEND_HOME:-${ASCEND_TOOLKIT_HOME}}"
[ -z "$CANN_HOME" ] && for d in /usr/local/Ascend/ascend-toolkit/latest /usr/local/Ascend/cann; do [ -d "$d" ] && { CANN_HOME="$d"; break; }; done
echo "CANN: ${CANN_HOME:-未找到}"
echo "架构: $(uname -m)"

BISHENG=$(command -v bisheng 2>/dev/null || echo "")
MSPROF=$(command -v msprof 2>/dev/null || echo "")
[ -n "$BISHENG" ] && _ok "bisheng: $BISHENG" || _err "bisheng: 未找到!"
[ -n "$MSPROF" ] && _ok "msprof: $MSPROF" || _err "msprof: 未找到!"

# 找 simulator lib
SIM_LIB=""
for d in "$CANN_HOME/tools/simulator/Ascend910B3/lib" "/usr/local/Ascend/cann/tools/simulator/dav_2201/lib"; do
    [ -f "$d/libruntime_camodel.so" ] && { SIM_LIB="$d"; break; }
done
[ -z "$SIM_LIB" ] && SIM_LIB=$(dirname "$(find /usr/local/Ascend -name "libruntime_camodel.so" -type f 2>/dev/null | head -1)" 2>/dev/null || echo "")
[ -n "$SIM_LIB" ] && _ok "SIM lib: $SIM_LIB" || _err "SIM lib: 未找到!"
ls "$SIM_LIB"/lib*_camodel.so 2>/dev/null | head -5

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: 编译 Triton kernel → 拿到 .o 文件
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 1: 提取 .o 编译产物"

# 清理
rm -rf ~/.triton/cache/ kernel_meta/ "$HERE/kernel_meta" "$HERE/sim_build" "$HERE/msprof_sim" "$HERE/hivmir" 2>/dev/null || true
mkdir -p "$HERE/hivmir"

# 关键环境变量: 强制 CANN 输出 .o 到 kernel_meta/
export ACL_OP_DEBUG_LEVEL=3
export TRITON_DEBUG=1
export TRITON_ALWAYS_COMPILE=1
export TRITON_DISABLE_LINE_INFO=0

_inf "ACL_OP_DEBUG_LEVEL=3 (输出 .o 到 kernel_meta/)"
_inf "运行 python3 run_and_profile.py ..."

python3 run_and_profile.py > "$LOG_DIR/step1_run.log" 2>&1
R1=$?
echo "退出码: $R1"

# 找 kernel_meta 目录 (可能在当前目录或 HOME 下)
KM_DIR=""
for d in "$HERE/kernel_meta" "./kernel_meta" "$HOME/kernel_meta"; do
    [ -d "$d" ] && { KM_DIR="$d"; break; }
done

if [ -n "$KM_DIR" ]; then
    _ok "kernel_meta: $KM_DIR"
    echo "kernel_meta 内容:"
    find "$KM_DIR" -type f 2>/dev/null | head -30
    O_FILES=$(find "$KM_DIR" -name "*.o" -type f 2>/dev/null)
    O_COUNT=$(echo "$O_FILES" | grep -c '.' 2>/dev/null || echo 0)
    [ "$O_COUNT" -gt 0 ] && _ok "找到 $O_COUNT 个 .o 文件" || _wrn "kernel_meta 中无 .o 文件"
else
    _wrn "kernel_meta 目录不存在"
    _inf "搜索其他位置..."
    find / -maxdepth 4 -name "kernel_meta" -type d 2>/dev/null | head -5
    find ~/.triton -name "*.o" -type f 2>/dev/null | head -10
fi

# 也找 HIVM MLIR
MLIRS=$(find ~/.triton -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" -o -name "*.npuir" 2>/dev/null)
MLIR_CNT=$(echo "$MLIRS" | grep -c '.' 2>/dev/null || echo 0)
echo "$MLIRS" | while read f; do [ -f "$f" ] && cp "$f" "$HERE/hivmir/" 2>/dev/null; done
[ "$MLIR_CNT" -gt 0 ] && _ok "HIVM: $MLIR_CNT 个 → hivmir/" || _wrn "HIVM: 0 个"

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 2: bisheng 链接 simulator libs → 可执行文件
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 2: 链接 simulator libs → 可执行文件"

if [ "$O_COUNT" -gt 0 ] && [ -n "$SIM_LIB" ] && [ -n "$BISHENG" ]; then
    rm -rf "$HERE/sim_build" && mkdir -p "$HERE/sim_build"

    # 取第一个 .o
    FIRST_O=$(echo "$O_FILES" | head -1)
    _inf "链接: $FIRST_O"

    bisheng -Wl,--disable-new-dtags \
        -L"$SIM_LIB" \
        -Wl,-rpath,"$SIM_LIB" \
        -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++ \
        -lascendcl -lascendc_runtime -lprofapi -lunified_dlog \
        -lmmpa -lascend_dump -lc_sec -lerror_manager -lnpu_drv \
        "$FIRST_O" \
        -o "$HERE/sim_build/kernel_app" \
        > "$LOG_DIR/step2_link.log" 2>&1
    R2=$?

    if [ "$R2" = "0" ] && [ -f "$HERE/sim_build/kernel_app" ]; then
        _ok "kernel_app: $(wc -c < "$HERE/sim_build/kernel_app") bytes"
        file "$HERE/sim_build/kernel_app" 2>/dev/null || true
    else
        _err "链接失败! 退出码=$R2"
        echo "=== 链接日志 ==="
        grep -i "error\|undefined\|cannot find\|fatal" "$LOG_DIR/step2_link.log" 2>/dev/null | head -20 || tail -20 "$LOG_DIR/step2_link.log"
        echo "=== SIM_LIB 内容 ==="
        ls -la "$SIM_LIB"/*.so 2>/dev/null | head -20
    fi
else
    _err "Skip: 缺 .o(${O_COUNT:-0}) 或 SIM lib(${SIM_LIB:+有}/${SIM_LIB:-无}) 或 bisheng(${BISHENG:+有}/${BISHENG:-无})"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 3: msprof op simulator → trace.json
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 3: msprof op simulator"

if [ -f "$HERE/sim_build/kernel_app" ] && [ -n "$MSPROF" ]; then
    export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
    rm -rf "$HERE/msprof_sim"

    _inf "执行 msprof op simulator ..."
    msprof op simulator \
        --soc-version=Ascend910B3 \
        --output="$HERE/msprof_sim" \
        "$HERE/sim_build/kernel_app" \
        > "$LOG_DIR/step3_msprof.log" 2>&1
    R3=$?
    echo "退出码: $R3"

    OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
    if [ -n "$OPPROF" ]; then
        _ok "OPPROF: $(basename $OPPROF)"
        echo ""
        echo "=== 完整目录树 ==="
        find "$OPPROF" -type f 2>/dev/null | while read f; do
            echo "  $f ($(wc -c < "$f") bytes)"
        done

        TRACES=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
        INSTRS=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
        [ "$TRACES" -gt 0 ] && _ok "★★★ trace.json: $TRACES 个 ★★★"
        [ "$INSTRS" -gt 0 ] && _ok "★★★ instr_exe.csv: $INSTRS 个 ★★★"

        if [ "$TRACES" = "0" ] && [ "$INSTRS" = "0" ]; then
            echo ""
            echo "=== msprof 日志错误 ==="
            grep -i "error\|fail\|ERROR\|FAIL\|cannot\|not found\|assert\|ERR\|UNKNOWN" "$LOG_DIR/step3_msprof.log" 2>/dev/null | head -20 || echo "(无)"
        fi
    else
        _err "OPPROF 未生成"
        echo "=== msprof 日志(最后30行) ==="
        tail -30 "$LOG_DIR/step3_msprof.log"
    fi
else
    _wrn "Skip: 无可执行文件或 msprof 不可用"
fi

# ═════════════════════════════════════════════════════════════════════════════════
#  汇总
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "汇总"

echo "产物:"
echo "  hivmir:       $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
echo "  kernel_meta:  $(find "$HERE/kernel_meta" -type f 2>/dev/null | wc -l) files"
echo "  sim_build:    $(find "$HERE/sim_build" -type f 2>/dev/null | wc -l) files"
echo "  msprof_sim:   $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
echo "  日志:         $LOG_DIR"

TRACES=$(find "$HERE" -name "trace.json" 2>/dev/null)
if [ -n "$TRACES" ]; then
    echo ""
    echo "★★★ trace.json: ★★★"
    echo "$TRACES" | while read f; do echo "  $f ($(wc -c < "$f") bytes)"; done
else
    echo ""
    echo "未找到 trace.json"
    echo "各步骤日志: ls $LOG_DIR/"
fi
