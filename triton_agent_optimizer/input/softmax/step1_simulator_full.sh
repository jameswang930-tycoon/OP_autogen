#!/bin/bash
# Step 1: matmul kernel -> msprof op simulator -> trace.json + HIVM MLIR
# 已验证可用: CANN 8.5.1 + triton-ascend 3.5.x + Ascend 910B3
# 用法: bash step1_simulator_full.sh
# 先清理: rm -rf core* step1_logs/ hivmir/ msprof_sim/ && rm -rf ~/.triton/cache/

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
LOG_DIR="$HERE/step1_logs"
rm -rf "$LOG_DIR" && mkdir -p "$LOG_DIR"

RED='\033[31m'; GRN='\033[32m'; YLW='\033[33m'; NC='\033[0m'
_ok()  { echo -e "${GRN}[OK]${NC}  $*"; }
_err() { echo -e "${RED}[ERR]${NC} $*"; }

echo "kernel: matmul_kernel"

# clean
rm -rf "$HERE/msprof_sim" "$HERE/hivmir" 2>/dev/null || true
rm -rf ~/.triton/cache/ 2>/dev/null || true
mkdir -p "$HERE/hivmir"

# === 1. msprof op simulator -> trace.json ===
echo ""
echo "=== 1. msprof op simulator ==="

msprof op simulator \
    --application="python run_and_profile.py" \
    --kernel-name="matmul_kernel" \
    --soc-version=Ascend910B3 \
    --launch-count=5 \
    --core-id=0 \
    --output=./msprof_sim \
    2>&1 | tee "$LOG_DIR/msprof.log"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)

if [ -n "$OPPROF" ]; then
    _ok "OPPROF: $(basename $OPPROF)"
    find "$OPPROF" -type f | while read f; do echo "  $f ($(wc -c < "$f") bytes)"; done
    T=$(find "$OPPROF" -name "trace.json" 2>/dev/null | wc -l)
    I=$(find "$OPPROF" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
    [ "$T" -gt 0 ] && _ok "trace.json: $T" || echo "  (no trace.json)"
    [ "$I" -gt 0 ] && _ok "instr_exe.csv: $I"
else
    _err "OPPROF not found"
fi

# === 2. HIVM MLIR from cache ===
echo ""
echo "=== 2. HIVM MLIR ==="

export TRITON_ALWAYS_COMPILE=1
python3 run_and_profile.py > "$LOG_DIR/hivm.log" 2>&1

MLIRS=$(find ~/.triton/cache -name "*.mlir" -o -name "*.ttir" -o -name "*.ttadapter" 2>/dev/null)
MLIR_CNT=$(echo "$MLIRS" | grep -c '.' 2>/dev/null || echo 0)
echo "$MLIRS" | while read f; do [ -f "$f" ] && cp "$f" "$HERE/hivmir/"; done
[ "$MLIR_CNT" -gt 0 ] && _ok "HIVM: $MLIR_CNT files -> hivmir/" || _err "HIVM: 0 files"

# === summary ===
echo ""
echo "=== Summary ==="
echo "msprof_sim: $(find "$HERE/msprof_sim" -type f 2>/dev/null | wc -l) files"
echo "hivmir:     $(find "$HERE/hivmir" -type f 2>/dev/null | wc -l) files"
find "$HERE" -name "trace.json" 2>/dev/null | while read f; do echo "*** $f ($(wc -c < "$f") bytes)"; done
echo "logs: $LOG_DIR"
