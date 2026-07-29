#!/bin/bash
# ============================================================================
#  Ascend C → msprof op simulator (CMake 路径)
# ============================================================================
# 用法: 在 WSL 终端中执行
#   cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/msprof_simulator_test
#   bash build_and_profile.sh
# ============================================================================
set -e

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC} $1"; }

source ~/Ascend/cann-8.5.1/set_env.sh

WORKDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$WORKDIR/build"
OUTDIR="$WORKDIR/output"

# ── Step 1: CMake 编译 ──
log "Step 1/3: CMake 编译 (仿真模式) ..."
rm -rf "$BUILDDIR"
mkdir -p "$BUILDDIR" && cd "$BUILDDIR"

cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1 | tail -5
log "cmake 完成"

make -j$(nproc) 2>&1 | tail -10
log "编译完成"
ls -lh vecadd_app

# ── Step 2: msprof op simulator ──
log ""
log "Step 2/3: msprof op simulator ..."
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

SIMDIR="$HOME/Ascend/cann-8.5.1/x86_64-linux/simulator/Ascend910B3"
export LD_LIBRARY_PATH="$SIMDIR/lib:$LD_LIBRARY_PATH"

msprof op simulator \
    --soc-version=Ascend910B3 \
    --output="$OUTDIR" \
    --timeout=5 \
    "$BUILDDIR/vecadd_app" 2>&1 | tail -5

# ── Step 3: 解析结果 ──
log ""
log "Step 3/3: 查找 trace.json ..."

TRACE=$(find "$OUTDIR" -name "trace.json" 2>/dev/null | head -3)
if [ -z "$TRACE" ]; then
    log "未找到 trace.json。列出输出目录:"
    find "$OUTDIR" -type f 2>/dev/null | head -20
    log "msprof op simulator 可能需要 NPU 硬件或特定编译选项。"
    exit 0
fi

for t in $TRACE; do
    SIZE=$(stat -c%s "$t" 2>/dev/null || echo "?")
    log "  $t (${SIZE} bytes)"
done

# 解析 trace.json 结构
MAIN_TRACE=$(echo "$TRACE" | head -1)
log ""
log "=== trace.json 内容 ==="
python3 -c "
import json
with open('$MAIN_TRACE') as f:
    data = json.load(f)
events = data if isinstance(data, list) else data.get('traceEvents', data.get('events', []))
print(f'Total events: {len(events)}')
cats = {}
for e in events:
    cat = str(e.get('cat','?'))
    cats[cat] = cats.get(cat, 0) + 1
print('Event categories:')
for cat, cnt in sorted(cats.items(), key=lambda x:-x[1]):
    print(f'  {cat}: {cnt}')
complete = [e for e in events if e.get('ph')=='X' and float(e.get('dur',0))>0]
if complete:
    print(f'Complete events (with duration): {len(complete)}')
    for e in complete[:15]:
        print(f'  {e.get(\"name\",\"?\"):20s} dur={float(e.get(\"dur\",0)):.3f}us  cat={e.get(\"cat\",\"?\")}')
"

log "=== 完成 ==="
