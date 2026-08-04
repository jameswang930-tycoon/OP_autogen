#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  服务器端一键采集+解析+整合 → diagnosis.json  — 定稿版
#  ★ 主流程 = msprof + msprof op + 通用 msprof (真机, 快/真/够用)
#    hivm/sim 默认跳过 (复杂场景用不到/字段易出错); ENABLE_HIVM=1 / ENABLE_SIM=1 才跑
#    diagnosis.json 无 hivm 时: transfer_paths 直接从 board Memory.csv 建真实带宽
# ═══════════════════════════════════════════════════════════════════════════════
#  用法: bash run_server_flow.sh [M] [N] [K]     (默认 64³, 避免 simulator 卡)
#  产物: input/matmul/e2e_run/  (每阶段一个子目录, 运行前自动清理)
#    01_compile/  ttir.mlir + ttadapter.mlir + run_debug.txt
#    02_hivm/     hivm_try.txt (真实 HIVM)
#    03_sim/      sim_prof/  (OPPROF_*/simulator/*instr_exe.csv + trace.json)
#    04_board/    board_prof/ (8 个 CSV: OpBasic/Memory/L2Cache/Arithmetic/PipeUtil...)
#    05_task/     task_prof/ (PROF_*/mindstudio_profiler_output/op_summary)
#    06_diagnosis/ hivm.json + sim.json + board.json + task.json + diagnosis.json
#
#  ── 经验教训 (8.5.1 / 910B3 实测, 已内嵌) ──────────────────────────────
#  1. npuir.mlir 不生成 → 手动 bishengir D 打印 HIVM (pass 名两个都试)
#  2. simulator 的 running_time(us) 常为 0 → parser 用 cycles÷1.9GHz 兜底
#  3. simulator 大尺寸会卡 → 默认 64³ + timeout
#  4. msprof op 不要指定 --aic-metrics (指定会限制/报错); 默认全量 8 CSV
#  5. 通用 msprof 的 --aic-metrics 在 8.5.1 不认 → 用 --ai-core=on 拿 op_summary
#  6. op_summary 目录名可能是 mind_studio_profile_output (拼写不一) → find 宽找
#  7. 每次运行自动清理 e2e_run, 避免旧数据混淆
#  8. 路径全部基于仓库结构 (BASH_SOURCE), 任何目录可跑
#
#  ── 字段预期 (哪些该有值 / 哪些合法缺失, 防误判) ─────────────────────────
#  hivm.json   该有: op_type/engine/dst/src/size_kb/region/deps/attrs
#              合法缺: duration/cycles (HIVM 无时序)
#  sim.json    该有: op_name/pipe/duration_ns/cycles/call_count
#              合法缺: data_size (detail 无搬运信息时)
#  board.json  该有: total_ns/num_cores; Memory.csv 若在 → real_bw
#              合法缺: L2Cache/ArithmeticUtilization (若 8 CSV 不全)
#  task.json   该有: Task Duration/Block Dim (每 kernel)
#              合法缺: aicore_time/aiv_time/total_cycles (8.5.1 --task-time 限制)
#  diagnosis.json 该有: summary/ops/transfer_paths/deps/bottlenecks.hint
#              合法缺: bw_utilization/regime (需 peak 校准), 部分 op 的 real_bw
#
#  环境变量: SIM_TIMEOUT=秒  SKIP_SIM=1  SKIP_BOARD=1  SKIP_TASK=1
# ═══════════════════════════════════════════════════════════════════════════════
set -u

M=${1:-64}; N=${2:-64}; K=${3:-64}

# ── 路径 (仓库结构相对) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MATMUL_DIR="$REPO_ROOT/input/matmul"
OUT="$MATMUL_DIR/e2e_run"
SIM_TIMEOUT=${SIM_TIMEOUT:-1800}
# ★ 主流程 = msprof + msprof op + 通用 msprof (真机, 快/真/够用)
#   hivm / simulator 默认跳过 (复杂场景用不到, 字段解析易出错) — ENABLE_HIVM=1 / ENABLE_SIM=1 才跑
ENABLE_HIVM=${ENABLE_HIVM:-0}
ENABLE_SIM=${ENABLE_SIM:-0}
[ -f "$MATMUL_DIR/test_matmul.py" ] || { echo "❌ 找不到 $MATMUL_DIR/test_matmul.py — 仓库结构不对"; exit 1; }
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 + 清理 ──
echo "══════ 环境 ══════"
echo "M=$M N=$N K=$K  输出=$OUT  (ENABLE_HIVM=$ENABLE_HIVM ENABLE_SIM=$ENABLE_SIM)"
echo "  → 清理旧产物"
rm -rf "$OUT"
mkdir -p "$OUT"/01_compile "$OUT"/02_hivm "$OUT"/03_sim "$OUT"/04_board "$OUT"/05_task "$OUT"/06_diagnosis
if command -v conda >/dev/null 2>&1; then
  CB=$(conda info --base 2>/dev/null || echo ""); [ -n "$CB" ] && source "$CB/etc/profile.d/conda.sh" 2>/dev/null
  conda activate triton-npu 2>/dev/null && echo "  ✅ conda triton-npu" || echo "  ⚠ 手动: conda activate triton-npu"
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "  ✅ set_env.sh" || echo "  ⚠ 手动 source set_env.sh"
cd "$MATMUL_DIR"

# ═══════════ 阶段 1/6: 编译 → ttir/ttadapter ═══════════
if [ "$ENABLE_HIVM" = "1" ]; then
echo ""; echo "══ 阶段 1/6 (可选, ENABLE_HIVM=1): 编译 → ttir/ttadapter ══"
rm -rf ~/.triton
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY test_matmul.py > "$OUT/01_compile/run_debug.txt" 2>&1
echo "  编译退出码=$? (日志 01_compile/run_debug.txt)"
cp ~/.triton/dump/*/kernel.*.mlir "$OUT/01_compile/" 2>/dev/null || true
[ -f "$OUT/01_compile/kernel.ttir.mlir" ] && pass "ttir.mlir" || fail "缺 ttir.mlir (看 run_debug.txt)"
[ -f "$OUT/01_compile/kernel.ttadapter.mlir" ] && pass "ttadapter.mlir" || fail "缺 ttadapter.mlir"

# ═══════════ 阶段 2/6 (可选): 真实 HIVM (流程 D) ═══════════
echo ""; echo "══ 阶段 2/6 (可选): 真实 HIVM (bishengir 打印) ══"
TTADAPTER="$OUT/01_compile/kernel.ttadapter.mlir"
if [ -f "$TTADAPTER" ]; then
  HIVM_OK=0
  for P in hivm-inject-sync hivm-graph-sync-solver; do
    (cd "$OUT/02_hivm" && bishengir-compile --target=Ascend910B3 \
      --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
      --enable-hfusion-compile=true --enable-hivm-compile=true \
      --enable-triton-kernel-compile=true \
      --bishengir-print-ir-after=$P "$TTADAPTER" -o /tmp/k.o > hivm_try.txt 2>&1)
    CNT=$(grep -c 'hivm.hir' "$OUT/02_hivm/hivm_try.txt" 2>/dev/null || echo 0)
    if [ "$CNT" -gt 0 ]; then echo "  pass=$P → hivm.hir x$CNT"; HIVM_OK=1; break; fi
  done
  [ "$HIVM_OK" -eq 1 ] && pass "hivm_try.txt (hivm.hir>0)" || fail "无 hivm.hir (看 02_hivm/hivm_try.txt)"
else
  fail "无 ttadapter.mlir"
fi
else
  echo "  ⚠ ENABLE_HIVM=0, 跳过编译+HIVM (主流程只用 msprof+op)"
fi

# ═══════════ 阶段 3/6 (可选): simulator (指令级时序) ═══════════
echo ""; echo "══ 阶段 3/6 (可选, ENABLE_SIM=1): msprof op simulator ══"
if [ "$ENABLE_SIM" != "1" ]; then
  echo "  ⚠ ENABLE_SIM=0, 跳过 simulator (真机 msprof+op 已够诊断)"
else
  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
  SIMOUT="$OUT/03_sim/sim_prof"; rm -rf "$SIMOUT"; mkdir -p "$SIMOUT"
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K timeout "$SIM_TIMEOUT" msprof op simulator \
    --kernel-name=matmul_kernel --soc-version=Ascend910B3 --output="$SIMOUT" \
    $PY test_matmul.py > "$OUT/03_sim/sim_run.txt" 2>&1
  RC=$?
  [ "$RC" -eq 124 ] && fail "simulator 超时(>${SIM_TIMEOUT}s) → 更小尺寸/加 SIM_TIMEOUT"
  NCSV=$(find "$SIMOUT" -name "*_instr_exe.csv" 2>/dev/null | wc -l)
  NTR=$(find "$SIMOUT" -name "trace.json" 2>/dev/null | wc -l)
  if [ "$NCSV" -gt 0 ] && [ "$NTR" -gt 0 ]; then pass "instr_exe x$NCSV + trace.json"; \
  elif [ "$NCSV" -gt 0 ]; then pass "instr_exe x$NCSV (缺 trace)"; \
  else fail "无 instr_exe (看 03_sim/sim_run.txt: LD_LIBRARY_PATH? 卡?)"; fi
fi

# ═══════════ 阶段 4/6: msprof op (真机全量 8 CSV) ═══════════
echo ""; echo "══ 阶段 4/6: msprof op (默认全量: 带宽/L2/cube) ══"
if [ "${SKIP_BOARD:-0}" = "1" ]; then
  echo "  ⚠ SKIP_BOARD=1, 跳过"
else
  BOARD_OUT="$OUT/04_board/board_prof"; rm -rf "$BOARD_OUT"; mkdir -p "$BOARD_OUT"
  # 不指定 --aic-metrics = 默认全量 8 CSV (指定会限制/报错); --warm-up 防降频
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof op --kernel-name=matmul_kernel \
    --output="$BOARD_OUT" --warm-up=10 $PY test_matmul.py > "$OUT/04_board/board_run.txt" 2>&1
  echo "  退出码=$? (日志 04_board/board_run.txt)"
  MEM=$(find "$BOARD_OUT" -name "Memory.csv" 2>/dev/null | wc -l)
  L2=$(find "$BOARD_OUT" -name "L2Cache.csv" 2>/dev/null | wc -l)
  AR=$(find "$BOARD_OUT" -name "ArithmeticUtilization.csv" 2>/dev/null | wc -l)
  OB=$(find "$BOARD_OUT" -name "OpBasicInfo.csv" 2>/dev/null | wc -l)
  if [ "$OB" -gt 0 ] && [ "$MEM" -gt 0 ]; then pass "OpBasic+Memory(带宽) L2 x$L2 Arithmetic x$AR"; \
  elif [ "$OB" -gt 0 ]; then pass "OpBasic ✓ 但缺 Memory/L2 → 看 board_run.txt"; \
  else fail "缺 OpBasicInfo → 看 04_board/board_run.txt"; fi
fi

# ═══════════ 阶段 5/6: 通用 msprof (任务级 op_summary) ═══════════
echo ""; echo "══ 阶段 5/6: msprof 通用 (--ai-core=on, op_summary) ══"
if [ "${SKIP_TASK:-0}" = "1" ]; then
  echo "  ⚠ SKIP_TASK=1, 跳过"
else
  TASK_OUT="$OUT/05_task/task_prof"; rm -rf "$TASK_OUT"; mkdir -p "$TASK_OUT"
  # 8.5.1 通用 msprof 不认 --aic-metrics (实测 255) → 用 --ai-core=on
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof --output="$TASK_OUT" \
    --application="$PY test_matmul.py" --ai-core=on > "$OUT/05_task/task_run.txt" 2>&1
  RC=$?; OPSUM=$(find "$OUT/05_task" -name "op_summary*.csv" 2>/dev/null | wc -l)
  echo "  退出码=$RC op_summary x$OPSUM (日志 05_task/task_run.txt)"
  if [ "$OPSUM" -gt 0 ]; then pass "op_summary ✓ (每kernel耗时/核数)"; \
  else echo "  ❌ 诊断: $(tail -8 "$OUT/05_task/task_run.txt" 2>/dev/null | tr '\n' ' ')"; fail "无 op_summary"; fi
fi

# ═══════════ 阶段 6/6: board+task 解析 + 整合 → diagnosis.json ═══════════
echo ""; echo "══ 阶段 6/6: 解析 + 整合 (msprof+op → diagnosis.json) ══"
D="$OUT/06_diagnosis"
HAVE=""
[ -n "$(find "$OUT/04_board" -name 'OpBasicInfo.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE board"
[ -n "$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE task"
# 可选源 (若 ENABLE_HIVM/ENABLE_SIM 开了)
[ -f "$OUT/02_hivm/hivm_try.txt" ] && HAVE="$HAVE hivm"
[ -n "$(find "$OUT/03_sim" -name '*_instr_exe.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE sim"
echo "  已有源:${HAVE:- 无}"
if [ -n "$HAVE" ]; then
  [ -n "$(find "$OUT/04_board" -name 'OpBasicInfo.csv' 2>/dev/null | head -1)" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_board.py" "$OUT/04_board" "$D/board.json" || true
  [ -n "$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | head -1)" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_task.py" "$OUT/05_task" "$D/task.json" || true
  echo '{}' > "$D/empty.json"
  [ -f "$D/board.json" ] || cp "$D/empty.json" "$D/board.json"
  [ -f "$D/task.json" ] || cp "$D/empty.json" "$D/task.json"
  "$PY" "$SCRIPT_DIR/integrate.py" "$D/board.json" "$D/task.json" "$D/diagnosis.json"
  [ -f "$D/diagnosis.json" ] && pass "diagnosis.json ✓ (roofline 核心)" || fail "diagnosis.json 未生成"
else
  fail "无 board/task 源 (阶段4/5 至少一个 ✅ 才整合)"
fi

# ═══════════ 字段预期校验 ═══════════
echo ""; echo "══════════ 字段预期校验 (哪些该有/哪些合法缺失) ══════════"
"$PY" - "$D" <<'PYEOF'
import json, sys
from pathlib import Path
D = Path(sys.argv[1])
def load(n):
    p = D / n
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

bd, tk = load('board.json'), load('task.json')

print("board.json (msprof op) 真机: ", end='')
b_sum = bd.get('execution_summary', {})
b_norm = bd.get('normalized', {})
print(f"total_ns={b_sum.get('total_ns')} cores={b_sum.get('num_cores')} freq={b_sum.get('freq_mhz')} "
      f"engine_util={len(b_norm.get('engine_utilization', {}))}种 "
      f"bandwidth={sum(1 for v in b_norm.get('bandwidth_gb_s', {}).values() if v)}条 "
      f"L2={b_norm.get('l2_hit_rate')} 冲突={len(b_norm.get('conflict', {}))}")

print("task.json (通用 msprof): ", end='')
t_sum = tk.get('execution_summary', {})
t_norm = tk.get('normalized', {})
print(f"kernels={t_sum.get('num_kernels')} 多kernel统计={len(t_norm.get('multi_kernel', []))} "
      f"api_overhead={len(t_norm.get('api_overhead', []))} L2={t_norm.get('l2_hit_rate')}")

if Path(D / 'diagnosis.json').exists():
    dg = json.load(open(D / 'diagnosis.json', encoding='utf-8'))
    print("diagnosis.json: ", end='')
    ro = dg.get('roofline', {})
    nt = len(dg.get('transfer_paths', []))
    nh = sum(1 for v in dg.get('bottlenecks', {}).values() if v.get('hint'))
    print(f"roofline={ro.get('bottleneck_type')} 通路={nt} 瓶颈hint {nh}/5")
    print(f"  访存利用率={ro.get('memory_utilization')} 算力利用率={ro.get('compute_utilization')}")
    print("  合法缺失: 无 hivm → 无 per-op/依赖 (Tier2 融合需临时 ENABLE_HIVM=1)")
PYEOF

echo ""; echo "══════════ 检查清单 (PASS=$PASS FAIL=$FAIL) ══════════"
echo "  阶段1 编译   : [ttir/ttadapter]      → 01_compile/"
echo "  阶段2 HIVM   : [hivm_try.txt]        → grep -c 'hivm.hir' 02_hivm/hivm_try.txt"
echo "  阶段3 sim    : [instr_exe+trace]     → 03_sim/sim_prof/OPPROF_*/simulator/"
echo "  阶段4 board  : [8 CSV 含 Memory/L2]  → 04_board/board_prof/OPPROF_*/"
echo "  阶段5 task   : [op_summary]          → 05_task/task_prof/PROF_*/"
echo "  阶段6 诊断   : [diagnosis.json]      → python3 -m json.tool 06_diagnosis/diagnosis.json"
echo "  看报告: cd input/matmul && python3 real_report.py [--llm]"
