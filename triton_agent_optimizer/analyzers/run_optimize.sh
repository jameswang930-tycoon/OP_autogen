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
#  用法: bash run_optimize.sh <input_dir> [output_dir] [M] [N] [K]
#    input_dir  算子目录 (含 kernel_op.py 单文件 或 test_matmul.py + triton_kernel.py)
#    output_dir 输出目录 (调度器每轮指向不同 round_dir; 默认 input_dir/e2e_run)
#  产物: <output_dir>/
#    05_task/    通用 msprof op_summary (骨架: 每kernel耗时/核数/形状) → task.json
#    04_board/   逐 kernel 的 msprof op 8 CSV (真实带宽/L2/cube) → board_<i>.json
#    06_diagnosis/  task.json + board_*.json + diagnosis.json (骨架+deep 合并) (+ hivm, 多算子时)
#    hivm_fusion_view.txt (多算子时, 融合专用)
#  流程: 通用 msprof → 骨架 → 逐 distinct kernel 采 msprof op → 按名合并 (见 docx/aggregation_rules.md)
#  单文件: 优先跑 input_dir/kernel_op.py (v4 单文件), 否则 test_matmul.py
#  环境: ENABLE_HIVM=1 强制跑 hivm (多算子融合需要); 默认自动判断
#
#  ═══════════════ 服务器运行步骤 (从 clone 到出诊断) ═══════════════
#   1. 环境:
#      conda activate triton-npu
#      source /usr/local/Ascend/ascend-toolkit/set_env.sh
#      cd triton_agent_optimizer
#   2. 单文件能不能跑 (自包含, 无 import 依赖):
#      cd input/matmul && python3 kernel_op.py && cd ../..
#      # 预期: [info] launch grid=... ; [info] kernel launched & synced OK
#   3. 采集+解析 (参数是 <input_dir> <output_dir>):
#      bash analyzers/run_optimize.sh input/matmul input/matmul/e2e_run
#      # 预期: run=kernel_op.py; ✅ op_summary; ✅ msprof op [matmul_kernel] 8 CSV;
#      #        ✅ diagnosis.json ✓; 字段校验: 列名不匹配 0 个
#   4. 看诊断:
#      python3 input/matmul/real_report.py input/matmul/e2e_run/06_diagnosis/diagnosis.json
#      # 预期: summary(num_kernels/filled) + kernel task/deep + roofline(一针见血)
#   5. 完整优化循环 (一键, 每轮: 采集→提取当前tier字段→planner→coder→msprof端到端→加速比):
#      LLM_CLI_COMMAND="nga run" python3 main.py input/matmul
#      # 本地无 LLM 先试流程: python3 main.py input/matmul --max-rounds 3 --stub
#   6. 核对当前策略筛字段 (解析完 07 已自动产出, 再手动核对一遍规则):
#      python3 analyzers/test_tier_extract.py input/matmul/e2e_run/06_diagnosis/diagnosis.json
#      # 预期: 每 tier 只显示自己策略的字段; X/Y 有数据 越大越好
#   ★ 07 产物: 解析完自动产出 <out_dir>/07_tier<N>_fields/{tier<N>_fields.txt,.json}
#      = 当前优化阶段筛选后的字段 (planner 只读这个; TIER 环境变量指定阶段, 默认 1)
#
#  ⚠ 参数变化: 原来 `bash run_optimize.sh 512 512 512` (M/N/K) 已改成
#    `bash run_optimize.sh input/matmul <out_dir> 512 512 512` (input/output 在前)
#  ⚠ 单文件 kernel_op.py 是源文件 (config+kernel+test 一体), coder 只改它; 已入库 (不在 .gitignore)
# ═══════════════════════════════════════════════════════════════════════════════
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="${1:-$REPO_ROOT/input/matmul}"
OUT="${2:-$INPUT_DIR/e2e_run}"
M=${3:-512}; N=${4:-512}; K=${5:-512}
TIER=${TIER:-1}    # 当前优化阶段 1~6 (调度器传; 决定 07 筛哪层字段)

# 要运行的文件: 优先单文件 kernel_op.py (v4), 否则 test_matmul.py
RUN_PY="test_matmul.py"
[ -f "$INPUT_DIR/kernel_op.py" ] && RUN_PY="kernel_op.py"
[ -f "$INPUT_DIR/$RUN_PY" ] || { echo "❌ 找不到 $INPUT_DIR/{kernel_op.py,test_matmul.py}"; exit 1; }
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 + 清理旧产物 ──
echo "══════ 环境 ══════"
echo "input=$INPUT_DIR  run=$RUN_PY  M=$M N=$N K=$K  输出=$OUT  (ENABLE_HIVM=${ENABLE_HIVM:-auto})"
echo "  → 清理旧产物"
rm -rf "$OUT"
mkdir -p "$OUT"/04_board "$OUT"/05_task "$OUT"/06_diagnosis
# 自包含: 本轮运行的源快照 → $OUT/input/, 从那里跑 (每轮独立可复现, 输入在 outputs 里)
mkdir -p "$OUT/input"
cp "$INPUT_DIR/$RUN_PY" "$OUT/input/$RUN_PY"
RUN_DIR="$OUT/input"
if command -v conda >/dev/null 2>&1; then
  CB=$(conda info --base 2>/dev/null || echo ""); [ -n "$CB" ] && source "$CB/etc/profile.d/conda.sh" 2>/dev/null
  conda activate triton-npu 2>/dev/null && echo "  ✅ conda triton-npu" || echo "  ⚠ 手动 activate"
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "  ✅ set_env.sh" || echo "  ⚠ 手动 source"
cd "$RUN_DIR"

# ═══════════ 阶段 1/3: 通用 msprof(骨架) → 逐 kernel 采 msprof op ═══════════
echo ""; echo "══ 阶段 1/3: 通用 msprof(骨架) + 逐 kernel msprof op ══"
D="$OUT/06_diagnosis"

# 1a. 通用 msprof 先跑 (任务级 op_summary → distinct kernel 名 → 骨架)
TASK_OUT="$OUT/05_task/task_prof"; rm -rf "$TASK_OUT"; mkdir -p "$TASK_OUT"
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof --output="$TASK_OUT" \
  --application="$PY $RUN_PY" --ai-core=on > "$OUT/05_task/task_run.txt" 2>&1
OPSUM=$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | wc -l)
echo "  通用 msprof 退出码=$? op_summary x$OPSUM (8.5.1 用 --ai-core=on)"
[ "$OPSUM" -gt 0 ] && pass "op_summary (骨架/kernel 数)" || fail "无 op_summary"
"$PY" "$SCRIPT_DIR/pipeline_parse_task.py" "$TASK_OUT" "$D/task.json" || true
[ -f "$D/task.json" ] || echo '{}' > "$D/task.json"

# 1b. 从 task.json 取 distinct kernel 名 → 逐 kernel 跑 msprof op → board_<i>.json
#     默认跳过 torch 框架 kernel (aclnn* 数据准备), 除非 FORCE_ALL_KERNELS=1
echo "  → 读取 kernel 名, 逐 kernel 采 msprof op (规则 O6, 默认跳过 aclnn* 框架 kernel)"
KERNELS=()
while IFS= read -r line; do
  [ -n "$line" ] || continue
  case "$line" in
    aclnn*) [ "${FORCE_ALL_KERNELS:-0}" = "1" ] && KERNELS+=("$line") || echo "  ⏭ 跳过框架 kernel: $line";;
    *) KERNELS+=("$line");;
  esac
done < <(
  "$PY" - "$D/task.json" <<'PYEOF'
import json, sys
from pathlib import Path
try:
    tk = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    for s in tk.get('normalized', {}).get('kernel_slots', []):
        n = s.get('kernel_name')
        if n:
            print(n)
except Exception:
    pass
PYEOF
)
if [ "${#KERNELS[@]}" -eq 0 ]; then KERNELS=(matmul_kernel); echo "  ⚠ 无 kernel 名 → 回退 matmul_kernel"; fi
echo "  待采 kernel: ${KERNELS[*]}"

IDX=0; BOARD_OK=0
for KNAME in "${KERNELS[@]}"; do
  IDX=$((IDX+1))
  BOUT="$OUT/04_board/op_${IDX}"
  rm -rf "$BOUT"; mkdir -p "$BOUT"
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof op --kernel-name="$KNAME" \
    --output="$BOUT" --warm-up=10 $PY "$RUN_PY" > "$OUT/04_board/board_${IDX}.txt" 2>&1
  MC=$(find "$BOUT" -name 'Memory.csv' 2>/dev/null | head -1)
  if [ -n "$MC" ]; then
    pass "msprof op [$KNAME] 8 CSV"
    "$PY" "$SCRIPT_DIR/pipeline_parse_board.py" "$BOUT" "$D/board_${IDX}.json" || true
    [ -f "$D/board_${IDX}.json" ] && BOARD_OK=$((BOARD_OK+1))
  else
    fail "msprof op [$KNAME] 缺 Memory.csv (kernel 名不匹配? 看 board_${IDX}.txt)"
  fi
done

# ═══════════ 阶段 2/3: 整合 → diagnosis.json (骨架 + deep 按 kernel 名合并) ═══════════
echo ""; echo "══ 阶段 2/3: 整合骨架 + deep → diagnosis.json ══"
BOARDS=( "$D"/board_*.json )
if [ "${#BOARDS[@]}" -ge 1 ] && [ -f "${BOARDS[0]}" ]; then
  "$PY" "$SCRIPT_DIR/integrate.py" "$D/task.json" "$D/diagnosis.json" "${BOARDS[@]}"
else
  echo "  ⚠ 无 board.json → 骨架保留 (filled=0)"
  "$PY" "$SCRIPT_DIR/integrate.py" "$D/task.json" "$D/diagnosis.json"
fi
[ -f "$D/diagnosis.json" ] && pass "diagnosis.json ✓ (骨架+deep, 规则M11)" || fail "diagnosis.json 未生成"

# 07: 解析完 → 按当前阶段筛字段, 产出到 07 (planner 读这个)
if [ -f "$D/diagnosis.json" ]; then
  "$PY" - "$REPO_ROOT" "$D/diagnosis.json" "$OUT" "$TIER" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from agents.scheduler import extract_tier_fields, TIER_FIELDS, _get
d = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
tier = int(sys.argv[4]); out = Path(sys.argv[3])
txt = extract_tier_fields(d, tier)
d7 = out / f"07_tier{tier}_fields"
d7.mkdir(parents=True, exist_ok=True)
(d7 / f"tier{tier}_fields.txt").write_text(txt, encoding="utf-8")
vals = {desc: _get(d, path) for path, desc in TIER_FIELDS.get(tier, [])}
(d7 / f"tier{tier}_fields.json").write_text(
    json.dumps(vals, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(f"  ✅ 07_tier{tier}_fields: {sum(1 for l in txt.splitlines() if l.startswith('-'))} 字段 → {d7}")
PYEOF
fi

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
  cd "$INPUT_DIR"     # hivm 编译/产物都在源目录 (自包含 input/ 只用于 msprof 采集)
  rm -rf ~/.triton
  export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY "$RUN_PY" > "$OUT/04_board/compile.txt" 2>&1
  cp ~/.triton/dump/*/kernel.*.mlir "$INPUT_DIR/" 2>/dev/null || true
  HIVM_OK=0
  for P in hivm-inject-sync hivm-graph-sync-solver; do
    (cd "$INPUT_DIR" && bishengir-compile --target=Ascend910B3 \
      --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
      --enable-hfusion-compile=true --enable-hivm-compile=true \
      --enable-triton-kernel-compile=true \
      --bishengir-print-ir-after=$P kernel.ttadapter.mlir -o /tmp/k.o > hivm_try.txt 2>&1)
    CNT=$(grep -c 'hivm.hir' "$INPUT_DIR/hivm_try.txt" 2>/dev/null || echo 0)
    if [ "$CNT" -gt 0 ]; then echo "  pass=$P → hivm.hir x$CNT"; HIVM_OK=1; break; fi
  done
  if [ "$HIVM_OK" -eq 1 ]; then
    pass "hivm_try.txt 生成"
    "$PY" "$SCRIPT_DIR/filter_hivm_for_fusion.py" "$INPUT_DIR/hivm_try.txt" --out "$INPUT_DIR/hivm_fusion_view.txt"
    pass "hivm_fusion_view.txt ✓ (融合专用: op+同步+依赖)"
    echo "  → LLM 读 hivm_fusion_view.txt: 找 RAW 链相邻逐元素 op → 融合; WAR → 换 buffer"
  else
    fail "hivm 生成失败 (看 hivm_try.txt)"
  fi
fi

# ═══════════ 字段校验 (check_fields.py: OK / 列名不匹配 / 源无) ═══════════
echo ""; echo "══════════ 字段校验 ══════════"
CF_RC=0
if [ -f "$D/task.json" ]; then
  DGN=""
  [ -f "$D/diagnosis.json" ] && DGN="$D/diagnosis.json"
  if ls "$D"/board_*.json >/dev/null 2>&1; then
    for b in "$D"/board_*.json; do
      echo "── 校验 $(basename "$b") ──"
      "$PY" "$SCRIPT_DIR/check_fields.py" "$b" "$D/task.json" $DGN || CF_RC=1
    done
  else
    echo "── 校验 task.json (无 board) ──"
    "$PY" "$SCRIPT_DIR/check_fields.py" "$D/task.json" "$D/task.json" $DGN || CF_RC=1
  fi
  [ "$CF_RC" -eq 0 ] && pass "字段校验: 无列名不匹配 (合法缺属正常)" || fail "字段校验: 有列名不匹配 (看 raw 修 parser)"
else
  fail "字段校验: task.json 缺失"
fi

echo ""; echo "══════════ 完成 (PASS=$PASS FAIL=$FAIL) ══════════"
echo "  diagnosis.json     : $D/diagnosis.json  (单算子优化用)"
echo "  hivm_fusion_view   : $INPUT_DIR/hivm_fusion_view.txt  (多算子融合用, 若生成)"
echo "  看诊断报告: python3 input/matmul/real_report.py"
