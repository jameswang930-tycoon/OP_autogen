#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  服务器端一键采集+解析+整合 (6 阶段 → diagnosis.json)  — 定稿版
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
[ -f "$MATMUL_DIR/test_matmul.py" ] || { echo "❌ 找不到 $MATMUL_DIR/test_matmul.py — 仓库结构不对"; exit 1; }
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 + 清理 ──
echo "══════ 环境 ══════"
echo "M=$M N=$N K=$K  输出=$OUT"
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
echo ""; echo "══ 阶段 1/6: 编译 → ttir/ttadapter ══"
rm -rf ~/.triton
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY test_matmul.py > "$OUT/01_compile/run_debug.txt" 2>&1
echo "  编译退出码=$? (日志 01_compile/run_debug.txt)"
cp ~/.triton/dump/*/kernel.*.mlir "$OUT/01_compile/" 2>/dev/null || true
[ -f "$OUT/01_compile/kernel.ttir.mlir" ] && pass "ttir.mlir" || fail "缺 ttir.mlir (看 run_debug.txt)"
[ -f "$OUT/01_compile/kernel.ttadapter.mlir" ] && pass "ttadapter.mlir" || fail "缺 ttadapter.mlir"

# ═══════════ 阶段 2/6: 真实 HIVM (流程 D) ═══════════
echo ""; echo "══ 阶段 2/6: 真实 HIVM (bishengir 打印) ══"
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

# ═══════════ 阶段 3/6: simulator (指令级时序) ═══════════
echo ""; echo "══ 阶段 3/6: msprof op simulator (64³ 应几十秒内) ══"
if [ "${SKIP_SIM:-0}" = "1" ]; then
  echo "  ⚠ SKIP_SIM=1, 跳过"
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

# ═══════════ 阶段 6/6: 4 源解析 + 整合 → diagnosis.json ═══════════
echo ""; echo "══ 阶段 6/6: 4 源解析 + 整合 ══"
D="$OUT/06_diagnosis"
HAVE=""
[ -f "$OUT/02_hivm/hivm_try.txt" ] && HAVE="$HAVE hivm"
[ -n "$(find "$OUT/03_sim" -name '*_instr_exe.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE sim"
[ -n "$(find "$OUT/04_board" -name 'OpBasicInfo.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE board"
[ -n "$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | head -1)" ] && HAVE="$HAVE task"
echo "  已有源:${HAVE:- 无}"
if [ -n "$HAVE" ]; then
  [ -f "$OUT/02_hivm/hivm_try.txt" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_hivm.py" "$OUT/02_hivm/hivm_try.txt" "$D/hivm.json" || true
  [ -n "$(find "$OUT/03_sim" -name '*_instr_exe.csv' 2>/dev/null | head -1)" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_sim.py" "$OUT/03_sim" "$D/sim.json" || true
  [ -n "$(find "$OUT/04_board" -name 'OpBasicInfo.csv' 2>/dev/null | head -1)" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_board.py" "$OUT/04_board" "$D/board.json" || true
  [ -n "$(find "$OUT/05_task" -name 'op_summary*.csv' 2>/dev/null | head -1)" ] && "$PY" "$SCRIPT_DIR/pipeline_parse_task.py" "$OUT/05_task" "$D/task.json" || true
  echo '{}' > "$D/empty.json"
  for f in hivm sim task board; do [ -f "$D/$f.json" ] || cp "$D/empty.json" "$D/$f.json"; done
  "$PY" "$SCRIPT_DIR/integrate.py" "$D/hivm.json" "$D/sim.json" "$D/task.json" "$D/board.json" "$D/diagnosis.json"
  [ -f "$D/diagnosis.json" ] && pass "diagnosis.json ✓" || fail "diagnosis.json 未生成"
else
  fail "无任何源产物 (阶段2-5 至少一个 ✅ 才整合)"
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

hv, sm, bd, tk = load('hivm.json'), load('sim.json'), load('board.json'), load('task.json')

print("hivm.json 结构字段: ", end='')
if hv.get('per_op_statistics'):
    o = hv['per_op_statistics'][0]
    have = {k for k in ('op_type','engine','dst','src','size_kb','memory_region','dependencies') if k in o}
    print(f"{len(have)}/7 该有 (op_type/dst/size/region/deps), 缺失={sorted({'op_type','engine','dst','src','size_kb','memory_region','dependencies'}-have)}")
else:
    print("无 ops")

print("sim.json 时序: ", end='')
sm_ops = sm.get('per_op_statistics', [])
n_dur = sum(1 for o in sm_ops if (o.get('duration_ns') or 0) > 0)
print(f"{n_dur}/{len(sm_ops)} 指令有 duration (running_time=0 已用 cycles 兜底)")

print("board.json 真机: ", end='')
bw = bd.get('bandwidth_utilization', {})
print(f"total_ns={bd.get('execution_summary',{}).get('total_ns')} cores={bd.get('execution_summary',{}).get('num_cores')} "
      f"Memory={bool(bw.get('memory_bandwidth_gb_s'))} L2={bool(bw.get('l2_cache'))} Arithmetic={bool(bw.get('arithmetic'))}")

print("task.json: ", end='')
tk_ops = tk.get('per_op', [])
if tk_ops:
    t = tk_ops[0]
    print(f"Task Duration={t.get('task_duration_us')} Block Dim={t.get('block_num')} "
          f"aicore_time={t.get('aicore_time_us')}(合法缺=8.5.1 task-time限制)")
else:
    print("无 op")

if Path(D / 'diagnosis.json').exists():
    dg = json.load(open(D / 'diagnosis.json', encoding='utf-8'))
    print("diagnosis.json: ", end='')
    np = len(dg.get('ops', [])); nt = len(dg.get('transfer_paths', []))
    nbw = sum(1 for p in dg.get('transfer_paths', []) if p.get('real_bw_gb_s'))
    nh = sum(1 for k, v in dg.get('bottlenecks', {}).items() if v.get('hint'))
    print(f"ops={np} 通路={nt} (有真实带宽 {nbw}/{nt}) 瓶颈hint {nh}/6")
    print("  合法缺失: bw_utilization/regime (需 peak 校准); 无 Memory 的通路 real_bw=None")
PYEOF

echo ""; echo "══════════ 检查清单 (PASS=$PASS FAIL=$FAIL) ══════════"
echo "  阶段1 编译   : [ttir/ttadapter]      → 01_compile/"
echo "  阶段2 HIVM   : [hivm_try.txt]        → grep -c 'hivm.hir' 02_hivm/hivm_try.txt"
echo "  阶段3 sim    : [instr_exe+trace]     → 03_sim/sim_prof/OPPROF_*/simulator/"
echo "  阶段4 board  : [8 CSV 含 Memory/L2]  → 04_board/board_prof/OPPROF_*/"
echo "  阶段5 task   : [op_summary]          → 05_task/task_prof/PROF_*/"
echo "  阶段6 诊断   : [diagnosis.json]      → python3 -m json.tool 06_diagnosis/diagnosis.json"
echo "  看报告: cd input/matmul && python3 real_report.py [--llm]"
