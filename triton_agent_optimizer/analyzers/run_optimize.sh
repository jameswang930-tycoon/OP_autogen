#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  优化流程主脚本 — 判断算子数 → 分流 (多算子融合 / 单算子 roofline)
# ═══════════════════════════════════════════════════════════════════════════════
#  流程:
#    ① 采集 msprof + msprof op (真机, 快/真) → 判断 kernel 数
#    ② 多算子 (>1 kernel)?
#        ├─ 是 → 编译+HIVM → filter_hivm_for_fusion → 融合视图 (给 LLM 做算法+融合)
#        └─ 否 → msprof+op → diagnosis.json (roofline) → Tier3-6 逐级优化
#    ③ 输出:
#        - diagnosis.json  (始终有, roofline 核心, 单算子优化用)
#        - hivm_fusion_view.txt (多算子时有, 融合用)
#
#  用法: bash run_optimize.sh [M] [N] [K]
#  产物: input/matmul/e2e_run/
#    04_board/   msprof op 8 CSV (真实带宽/L2/cube)
#    05_task/    通用 msprof op_summary (每kernel耗时/核数)
#    06_diagnosis/  board.json + task.json + diagnosis.json (+ hivm.json, 多算子时)
#    hivm_fusion_view.txt (多算子时, 融合专用)
#  环境: ENABLE_HIVM=1 强制跑 hivm (多算子融合需要); 默认自动判断
# ═══════════════════════════════════════════════════════════════════════════════
set -u

M=${1:-64}; N=${2:-64}; K=${3:-64}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATMUL_DIR="$REPO_ROOT/input/matmul"
OUT="$MATMUL_DIR/e2e_run"
[ -f "$MATMUL_DIR/test_matmul.py" ] || { echo "❌ 找不到 test_matmul.py"; exit 1; }
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 + 清理旧产物 ──
echo "══════ 环境 ══════"
echo "M=$M N=$N K=$K  输出=$OUT  (ENABLE_HIVM=${ENABLE_HIVM:-auto})"
echo "  → 清理旧产物"
rm -rf "$OUT"
mkdir -p "$OUT"/04_board "$OUT"/05_task "$OUT"/06_diagnosis
if command -v conda >/dev/null 2>&1; then
  CB=$(conda info --base 2>/dev/null || echo ""); [ -n "$CB" ] && source "$CB/etc/profile.d/conda.sh" 2>/dev/null
  conda activate triton-npu 2>/dev/null && echo "  ✅ conda triton-npu" || echo "  ⚠ 手动 activate"
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "  ✅ set_env.sh" || echo "  ⚠ 手动 source"
cd "$MATMUL_DIR"

# ═══════════ 阶段 1/3: 采集 msprof + msprof op (真机主源) ═══════════
echo ""; echo "══ 阶段 1/3: 采集 msprof op + 通用 msprof ══"

# 1a. msprof op (★主源, 默认全量 8 CSV)
BOARD_OUT="$OUT/04_board/board_prof"; rm -rf "$BOARD_OUT"; mkdir -p "$BOARD_OUT"
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof op --kernel-name=matmul_kernel \
  --output="$BOARD_OUT" --warm-up=10 $PY test_matmul.py > "$OUT/04_board/board_run.txt" 2>&1
echo "  msprof op 退出码=$? (8 CSV)"
[ -n "$(find "$BOARD_OUT" -name 'Memory.csv' 2>/dev/null | head -1)" ] && pass "msprof op 8 CSV (真实带宽)" || fail "缺 Memory.csv"

# 1b. 通用 msprof (任务级 op_summary → 判断 kernel 数)
TASK_OUT="$OUT/05_task/task_prof"; rm -rf "$TASK_OUT"; mkdir -p "$TASK_OUT"
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof --output="$TASK_OUT" \
  --application="$PY test_matmul.py" --ai-core=on > "$OUT/05_task/task_run.txt" 2>&1
OPSUM=$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | wc -l)
echo "  通用 msprof 退出码=$? op_summary x$OPSUM (8.5.1 用 --ai-core=on)"
[ "$OPSUM" -gt 0 ] && pass "op_summary (判断 kernel 数)" || fail "无 op_summary"

# ═══════════ 阶段 2/3: 解析 + 整合 → diagnosis.json (roofline) ═══════════
echo ""; echo "══ 阶段 2/3: 解析 + 整合 → diagnosis.json ══"
D="$OUT/06_diagnosis"
"$PY" "$SCRIPT_DIR/pipeline_parse_board.py" "$BOARD_OUT" "$D/board.json" || true
"$PY" "$SCRIPT_DIR/pipeline_parse_task.py" "$TASK_OUT" "$D/task.json" || true
echo '{}' > "$D/empty.json"
[ -f "$D/board.json" ] || cp "$D/empty.json" "$D/board.json"
[ -f "$D/task.json" ] || cp "$D/empty.json" "$D/task.json"
"$PY" "$SCRIPT_DIR/integrate.py" "$D/board.json" "$D/task.json" "$D/diagnosis.json"
[ -f "$D/diagnosis.json" ] && pass "diagnosis.json ✓ (roofline)" || fail "diagnosis.json 未生成"

# ═══════════ 阶段 3/3: 判断算子数 → 分流 ═══════════
echo ""; echo "══ 阶段 3/3: 判断算子数 → 分流 ══"
N_KERNELS=$("$PY" - "$D/task.json" <<'PYEOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    tk = json.loads(p.read_text(encoding='utf-8'))
    print(tk.get('execution_summary', {}).get('num_kernels', 0))
except Exception:
    print(0)
PYEOF
)
echo "  检测到 kernel 数: $N_KERNELS"
MULTI=0
if [ "${ENABLE_HIVM:-auto}" = "1" ]; then MULTI=1; echo "  (ENABLE_HIVM=1 强制走多算子路径)"
elif [ "$N_KERNELS" -gt 1 ]; then MULTI=1; echo "  (多个 kernel → 多算子: 需看结构做融合)"
else echo "  (单个 kernel → 单算子: roofline 诊断够用, 走 Tier3-6)"; fi

if [ "$MULTI" = "1" ]; then
  echo ""; echo "══ 多算子路径: 编译 HIVM → 融合视图 ══"
  rm -rf ~/.triton
  export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY test_matmul.py > "$OUT/04_board/compile.txt" 2>&1
  cp ~/.triton/dump/*/kernel.*.mlir "$MATMUL_DIR/" 2>/dev/null || true
  HIVM_OK=0
  for P in hivm-inject-sync hivm-graph-sync-solver; do
    (cd "$MATMUL_DIR" && bishengir-compile --target=Ascend910B3 \
      --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
      --enable-hfusion-compile=true --enable-hivm-compile=true \
      --enable-triton-kernel-compile=true \
      --bishengir-print-ir-after=$P kernel.ttadapter.mlir -o /tmp/k.o > hivm_try.txt 2>&1)
    CNT=$(grep -c 'hivm.hir' "$MATMUL_DIR/hivm_try.txt" 2>/dev/null || echo 0)
    if [ "$CNT" -gt 0 ]; then echo "  pass=$P → hivm.hir x$CNT"; HIVM_OK=1; break; fi
  done
  if [ "$HIVM_OK" -eq 1 ]; then
    pass "hivm_try.txt 生成"
    "$PY" "$SCRIPT_DIR/filter_hivm_for_fusion.py" "$MATMUL_DIR/hivm_try.txt" --out "$MATMUL_DIR/hivm_fusion_view.txt"
    pass "hivm_fusion_view.txt ✓ (融合专用: op+同步+依赖)"
    echo "  → LLM 读 hivm_fusion_view.txt: 找 RAW 链相邻逐元素 op → 融合; WAR → 换 buffer"
  else
    fail "hivm 生成失败 (看 hivm_try.txt)"
  fi
fi

# ═══════════ 字段校验 ═══════════
echo ""; echo "══════════ 字段校验 ══════════"
"$PY" - "$D" <<'PYEOF'
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
def load(n):
    p = D / n
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
bd, tk = load('board.json'), load('task.json')
b_norm, t_norm = bd.get('normalized', {}), tk.get('normalized', {})
print("board: ", end='')
print(f"total_ns={bd.get('execution_summary',{}).get('total_ns')} cores={bd.get('execution_summary',{}).get('num_cores')} "
      f"bandwidth={sum(1 for v in b_norm.get('bandwidth_gb_s',{}).values() if v)}条 L2={b_norm.get('l2_hit_rate')} "
      f"engine={len(b_norm.get('engine_utilization',{}))}种")
print("task: ", end='')
print(f"kernels={tk.get('execution_summary',{}).get('num_kernels')} api={len(t_norm.get('api_overhead',[]))}")
if Path(D/'diagnosis.json').exists():
    dg = json.load(open(D/'diagnosis.json',encoding='utf-8'))
    print("diagnosis: roofline=", dg['roofline']['bottleneck_type'],
          " 通路=", len(dg['transfer_paths']),
          " hints=", sum(1 for v in dg['bottlenecks'].values() if v.get('hint')), "/5")
PYEOF

echo ""; echo "══════════ 完成 (PASS=$PASS FAIL=$FAIL) ══════════"
echo "  diagnosis.json     : $D/diagnosis.json  (单算子优化用)"
echo "  hivm_fusion_view   : $MATMUL_DIR/hivm_fusion_view.txt  (多算子融合用, 若生成)"
echo "  看诊断报告: python3 input/matmul/real_report.py"
