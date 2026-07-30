#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 1: Triton .py → trace.json (cache → ttadapter.mlir → .o → 可执行文件)
#  CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# ═════════════════════════════════════════════════════════════════════════════════
#
#  用法: bash step1_simulator_full.sh [softmax_kernel|fused_gelu_kernel]
#  先清理:
#    rm -rf core* profile* __pycache__/ step1_logs/ hivmir/ msprof_sim/ sim_build/
#    rm -rf ~/.triton/cache/
#
# ═════════════════════════════════════════════════════════════════════════════════
#  ★ 已验证可用的编译路径 (CANN 8.5.1, 2026-07):
#
#  Step 1: python3 run_and_profile.py
#            → ~/.triton/cache/ 生成 ttadapter.mlir
#  Step 2: bishengir-compile ttadapter.mlir \
#            --enable-hfusion-compile=true \
#            --enable-hivm-compile=true \
#            --enable-triton-kernel-compile=true \
#            -o kernel.o                          ← 这个参数组合可工作!
#            (注意: 只用 --enable-hivm-compile 会 segfault)
#            (bug: MarkRealCoreType pass 死循环, AscendNPU-IR Issue #154)
#  Step 3: msprof op simulator --config=sim_config.json kernel.o
#          → trace.json + instr_exe.csv
#
#  来源:
#   bishengir-compile 参数: https://gitcode.com/Ascend/AscendNPU-IR/issues/154
#   bisheng simulator: https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_00059.html
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
_inf "编译中... (参数: --enable-hfusion-compile + --enable-hivm-compile + --enable-triton-kernel-compile)"
bishengir-compile "$TTADAPTER" \
    --enable-hfusion-compile=true \
    --enable-hivm-compile=true \
    --enable-triton-kernel-compile=true \
    -o "$HERE/sim_build/kernel.o" \
    > "$LOG_DIR/compile.log" 2>&1
R2=$?
R2=$?

if [ "$R2" != "0" ] || [ ! -f "$HERE/sim_build/kernel.o" ]; then
    _err "bishengir-compile 三次尝试全部失败! 退出码=$R2"
    echo ""
    echo "=== 编译日志关键内容 ==="
    grep -i "error\|Segmentation\|signal\|stack\|139\|cannot\|fatal\|assert" "$LOG_DIR/compile.log" 2>/dev/null | head -15 || echo "(无关键词)"
    echo ""
    echo "=== 编译日志最后20行 ==="
    tail -20 "$LOG_DIR/compile.log"
    echo ""
    _err "这是 bishengir-compile 的已知 bug (AscendNPU-IR Issue #154)"
    _err "MarkRealCoreType pass 死循环导致栈溢出 → segfault"
    _inf "解决: 升级 bishengir-compile 到 post-2025年3月 版本"
    _inf "备选: 跳过 bishengir-compile, 用 HIVM + 真机 msprof PipeUtilization.csv 做分析"
    exit 1
fi

_ok "kernel.o: $(wc -c < "$HERE/sim_build/kernel.o") bytes"
file "$HERE/sim_build/kernel.o" 2>/dev/null || true

# ═════════════════════════════════════════════════════════════════════════════════
#  Step 3: msprof op simulator --config 模式 (跳过链接, 直接用 .o)
#  kernel.o 是纯设备端二进制, 没有 host main(), 不能用 --application
#  用 --config 传 JSON: https://www.hiascend.com/document/detail/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/atlasopdev_16_0031.html
# ═════════════════════════════════════════════════════════════════════════════════
_hdr "Step 3: msprof op simulator --config → trace.json"

# 写 config JSON
cat > "$HERE/sim_build/sim_config.json" << JSONEOF
{
    "op_type": "AI_CORE",
    "kernel_name": "$KERNEL",
    "kernel_file": "$HERE/sim_build/kernel.o"
}
JSONEOF
_inf "config: $(cat $HERE/sim_build/sim_config.json)"

export LD_LIBRARY_PATH="$SIM_LIB:$LD_LIBRARY_PATH"
rm -rf "$HERE/msprof_sim"

msprof op simulator \
    --config="$HERE/sim_build/sim_config.json" \
    --output="$HERE/msprof_sim" \
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
