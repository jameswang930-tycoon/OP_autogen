#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  服务器端一键采集+解析全流程 (小尺寸默认 64³, 避免 simulator 卡)
# ═══════════════════════════════════════════════════════════════════════════════
#  用法: bash run_server_flow.sh [M] [N] [K]
#    例: bash run_server_flow.sh             # 64 64 64
#        bash run_server_flow.sh 128 128 128 # 128³
#
#  产物全部收在: input/matmul/e2e_run/  (每阶段一个子目录, 不再散落)
#    01_compile/   ttir.mlir + ttadapter.mlir + run_debug.txt
#    02_hivm/      hivm_try.txt (真实 HIVM)
#    03_sim/       sim_prof/  (OPPROF_*/simulator/*instr_exe.csv + trace.json)
#    04_board/     board_prof/ (OpBasicInfo/PipeUtilization/ArithmeticUtilization/Memory)
#    05_task/      task_prof/ (PROF_*/mindstudio_profiler_output/op_summary 真机每op带宽/L2)
#    06_diagnosis/ hivm.json + sim.json + task.json + board.json + diagnosis.json(4源整合)
#
#  每个阶段跑完自动检查产物 → 打 ✅/❌; 末尾打印「检查清单」汇总全部 PASS/FAIL。
#  任何阶段失败不中断后续 (清单里看哪步没过, 单独重跑该阶段)。
#  ★ 每次运行开头自动清理 e2e_run 旧产物 (重新采集), 避免旧数据混淆。
#
#  ★ 路径解析 = 完全基于仓库结构 (不依赖服务器绝对位置):
#      脚本在 <仓库根>/analyzers/, 用例/产物在 <仓库根>/input/matmul/,
#      输出在 <仓库根>/input/matmul/e2e_run/。
#      同一仓库克隆到任何路径都能跑, 从任何目录调用均可 (BASH_SOURCE 解析)。
#      运行时会打印解析出的 仓库根/用例目录/输出目录, 先核对再继续。
#      仓库结构不对 (缺 test_matmul.py / run_all.sh) 会第一时间报错退出。
#
#  尺寸: 默认 64³ (simulator 指令级仿真, 大尺寸会极慢/卡)。
#       test_matmul.py 支持 MATMUL_M/N/K 环境变量覆盖 (本脚本自动传)。
#
#  前置: 脚本会尝试激活 conda triton-npu + set_env.sh; 失败就手动激活后再跑。
#  环境变量可调:
#    SIM_TIMEOUT=秒        simulator 超时 (默认 1800 = 30分钟, 64³ 通常几十秒内)
#    SKIP_SIM=1            跳过 simulator (不想等)
#    SKIP_BOARD=1          跳过真机 msprof op
#    SKIP_TASK=1           跳过真机 msprof 通用 (op_summary)
#    BOARD_METRICS=...     msprof op 指标 (默认全指标出 cube占比+带宽;
#                          若失败降级: BOARD_METRICS=PipeUtilization,ResourceConflictRatio)
#    TASK_AIC_METRICS=...  通用 msprof 的 aic-metrics (默认全指标出带宽/L2;
#                          逐级降级: 全指标→Memory,L2Cache→Memory→基础)
#  字段解析依据: analyzers/PIPELINE_FIELDS.md (字段来源/用途) + PIPELINE_README.md
# ═══════════════════════════════════════════════════════════════════════════════
set -u

M=${1:-64}; N=${2:-64}; K=${3:-64}

# ── 路径全部基于仓库结构解析 (不依赖服务器绝对位置, 任何目录可跑) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # 仓库根/analyzers
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"                       # 仓库根
MATMUL_DIR="$REPO_ROOT/input/matmul"                            # 产物/用例目录
OUT="$MATMUL_DIR/e2e_run"                                       # 采集输出目录
RUN_ALL="$SCRIPT_DIR/run_all.sh"
SIM_TIMEOUT=${SIM_TIMEOUT:-1800}

# 守卫: 仓库结构必须含 input/matmul (test_matmul.py) 和 analyzers/run_all.sh
[ -f "$MATMUL_DIR/test_matmul.py" ] || { echo "❌ 找不到 $MATMUL_DIR/test_matmul.py — 仓库结构不对, 请在正确仓库下运行"; exit 1; }
[ -f "$RUN_ALL" ] || { echo "❌ 找不到 $RUN_ALL — 脚本应与 run_all.sh 同目录"; exit 1; }

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 ──
echo "══════ 环境 ══════"
echo "M=$M N=$N K=$K"
echo "仓库根     = $REPO_ROOT   (脚本在: $SCRIPT_DIR)"
echo "用例/产物  = $MATMUL_DIR  (test_matmul.py + e2e_run/)"
echo "输出目录   = $OUT"
echo "  → 清理旧产物 (重新采集)"
rm -rf "$OUT"
mkdir -p "$OUT"/01_compile "$OUT"/02_hivm "$OUT"/03_sim "$OUT"/04_board "$OUT"/05_task "$OUT"/06_diagnosis
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
  [ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null
  conda activate triton-npu 2>/dev/null && echo "  ✅ conda triton-npu" || echo "  ⚠ conda activate 失败, 手动: conda activate triton-npu"
else
  echo "  ⚠ 无 conda, 请手动激活: conda activate triton-npu"
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null && echo "  ✅ set_env.sh" || echo "  ⚠ set_env.sh 未找到, 手动 source"
cd "$MATMUL_DIR"
echo "工作目录: $MATMUL_DIR"

# ═══════════════════════ 阶段 1/6: 编译 → ttir/ttadapter ═══════════════════════
echo ""
echo "══ 阶段 1/6: 编译 → ttir/ttadapter ══"
rm -rf ~/.triton
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY test_matmul.py > "$OUT/01_compile/run_debug.txt" 2>&1
echo "  编译运行退出码=$? (日志: 01_compile/run_debug.txt)"
cp ~/.triton/dump/*/kernel.*.mlir "$OUT/01_compile/" 2>/dev/null || true
if [ -f "$OUT/01_compile/kernel.ttir.mlir" ]; then pass "ttir.mlir 存在"; else fail "缺 ttir.mlir → 看 01_compile/run_debug.txt (缓存? 报错?)"; fi
if [ -f "$OUT/01_compile/kernel.ttadapter.mlir" ]; then pass "ttadapter.mlir 存在"; else fail "缺 ttadapter.mlir"; fi

# ═══════════════════════ 阶段 2/6: 真实 HIVM (流程 D) ═══════════════════════
echo ""
echo "══ 阶段 2/6: 真实 HIVM (bishengir 打印) ══"
TTADAPTER="$OUT/01_compile/kernel.ttadapter.mlir"
if [ -f "$TTADAPTER" ]; then
  HIVM_OK=0
  for P in hivm-inject-sync hivm-graph-sync-solver; do
    (cd "$OUT/02_hivm" && bishengir-compile --target=Ascend910B3 \
      --enable-auto-multi-buffer=True --enable-auto-bind-sub-block=True \
      --enable-hfusion-compile=true --enable-hivm-compile=true \
      --enable-triton-kernel-compile=true \
      --bishengir-print-ir-after=$P "$TTADAPTER" -o /tmp/k.o \
      > hivm_try.txt 2>&1)
    CNT=$(grep -c 'hivm.hir' "$OUT/02_hivm/hivm_try.txt" 2>/dev/null || echo 0)
    if [ "$CNT" -gt 0 ]; then echo "  pass=$P → hivm.hir x$CNT"; HIVM_OK=1; break; fi
  done
  if [ "$HIVM_OK" -eq 1 ]; then pass "hivm_try.txt (hivm.hir 指令>0)"; else fail "hivm_try.txt 无 hivm.hir → 看 02_hivm/hivm_try.txt 错误"; fi
else
  fail "跳过: 无 ttadapter.mlir"
fi

# ═══════════════════════ 阶段 3/6: simulator (指令级时序) ═══════════════════════
echo ""
echo "══ 阶段 3/6: msprof op simulator (64³ 应几十秒内) ══"
if [ "${SKIP_SIM:-0}" = "1" ]; then
  echo "  ⚠ SKIP_SIM=1, 跳过"
else
  export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/Ascend910B3/lib:$LD_LIBRARY_PATH
  SIMOUT="$OUT/03_sim/sim_prof"
  rm -rf "$SIMOUT"
  # 注意: 尺寸 env 必须放 msprof 之前 (整体传给其启动的 python)
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K timeout "$SIM_TIMEOUT" msprof op simulator \
    --kernel-name=matmul_kernel --soc-version=Ascend910B3 --output="$SIMOUT" \
    $PY test_matmul.py > "$OUT/03_sim/sim_run.txt" 2>&1
  SIMRC=$?
  if [ "$SIMRC" -eq 124 ]; then fail "simulator 超时 (>${SIM_TIMEOUT}s) → 换更小尺寸或加大 SIM_TIMEOUT"; \
  elif [ "$SIMRC" -ne 0 ]; then echo "  ⚠ simulator 退出码=$SIMRC (看 03_sim/sim_run.txt)"; fi
  NCSV=$(ls "$SIMOUT"/OPPROF_*/simulator/core*.cubecore*/*instr_exe.csv 2>/dev/null | wc -l)
  NTRACE=$(ls "$SIMOUT"/OPPROF_*/simulator/trace.json 2>/dev/null | wc -l)
  if [ "$NCSV" -gt 0 ] && [ "$NTRACE" -gt 0 ]; then pass "instr_exe.csv x$NCSV + trace.json ✓"; \
  elif [ "$NCSV" -gt 0 ]; then pass "instr_exe.csv x$NCSV (缺 trace.json)"; \
  else fail "无 instr_exe.csv → 看 03_sim/sim_run.txt (LD_LIBRARY_PATH? 卡住?)"; fi
fi

# ═══════════════════════ 阶段 4/6: 真机 msprof op ═══════════════════════
echo ""
echo "══ 阶段 4/6: 真机 msprof op (端到端/引擎占比) ══"
if [ "${SKIP_BOARD:-0}" = "1" ]; then
  echo "  ⚠ SKIP_BOARD=1, 跳过"
else
  BOARD_OUT="$OUT/04_board/board_prof"
  rm -rf "$BOARD_OUT"
  # 全指标才出 ArithmeticUtilization(Memory(MemoryL0/UB/L2Cache → board.json 的 cube 占比+带宽)
  BOARD_METRICS="${BOARD_METRICS:-PipeUtilization,ResourceConflictRatio,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache}"
  MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof op --kernel-name=matmul_kernel \
    --aic-metrics="$BOARD_METRICS" --output="$BOARD_OUT" \
    $PY test_matmul.py > "$OUT/04_board/board_run.txt" 2>&1
  echo "  退出码=$? (若全指标失败, 设 BOARD_METRICS=PipeUtilization,ResourceConflictRatio 重跑)"
  OBI=$(ls "$BOARD_OUT"/OPPROF_*/OpBasicInfo.csv 2>/dev/null | wc -l)
  PU=$(ls "$BOARD_OUT"/OPPROF_*/PipeUtilization.csv 2>/dev/null | wc -l)
  if [ "$OBI" -gt 0 ] && [ "$PU" -gt 0 ]; then pass "OpBasicInfo.csv + PipeUtilization.csv ✓"; \
  else fail "缺 OpBasicInfo/PipeUtilization → 看 04_board/board_run.txt"; fi
fi

# ═══════════════════════ 阶段 5/6: 真机 msprof 通用 (任务级 op_summary) ═══════════════════════
echo ""
echo "══ 阶段 5/6: msprof 通用 (任务级 op_summary, 真实每op带宽/L2/算力) ══"
if [ "${SKIP_TASK:-0}" = "1" ]; then
  echo "  ⚠ SKIP_TASK=1, 跳过"
else
  # 输出到 task_prof 子目录 (结构: 05_task/task_prof/PROF_xxx/mindstudio_profiler_output/op_summary)
  TASK_OUT="$OUT/05_task/task_prof"
  # 逐级降级: 全指标 → Memory,L2Cache → Memory → 基础 (碰到能出 op_summary 的停)
  # 注意: --task-time 取值 on/off; --aic-metrics 需 --ai-core=on
  FULL_METRICS="${TASK_AIC_METRICS:-PipeUtilization,ArithmeticUtilization,Memory,MemoryL0,MemoryUB,L2Cache,ResourceConflictRatio}"
  OPSUM=0; USED=""
  for METRICS in "$FULL_METRICS" "Memory,L2Cache" "Memory" ""; do
    rm -rf "$TASK_OUT"; mkdir -p "$TASK_OUT"
    FLAGS=""
    # 已验证模式 (CANN 8.5 910B3 成功案例): 只 --ai-core=on + --aic-metrics, 不显式 task-time/aic-mode (默认 on/task-based)
    [ -n "$METRICS" ] && FLAGS="--ai-core=on --aic-metrics=$METRICS"
    MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K msprof --output="$TASK_OUT" \
      --application="$PY test_matmul.py" $FLAGS \
      > "$OUT/05_task/task_run.txt" 2>&1
    RC=$?
    OPSUM=$(find "$OUT/05_task" -name "op_summary*.csv" 2>/dev/null | wc -l)
    echo "  try metrics=[${METRICS:-basic}] rc=$RC op_summary x$OPSUM"
    if [ "$OPSUM" -gt 0 ]; then USED="${METRICS:-basic}"; break; fi
  done
  if [ "$OPSUM" -gt 0 ]; then
    pass "op_summary x$OPSUM ✓ (metrics=[$USED])"
    [ "$USED" = "basic" ] && echo "  ⚠ 基础模式无带宽/L2 → 需定位全指标 flag (上面 t1~t5 命令)"
  else
    echo "  ❌ 所有 metrics 均无 op_summary → 诊断:"
    echo "    --- task_run.txt 末尾 20 行 ---"
    tail -20 "$OUT/05_task/task_run.txt" 2>/dev/null
    echo "    --- msprof 有效 flags (aic/task/metrics) ---"
    msprof --help 2>&1 | grep -iE 'aic|task-time|metric' | head -15
    fail "无 op_summary (诊断贴回给我, 即可定位)"
  fi
fi

# ═══════════════════════ 阶段 6/6: 4 源解析 + 整合诊断 ═══════════════════════
echo ""
echo "══ 阶段 6/6: 4 源解析 + 整合 (hivm/sim/task/board → diagnosis.json) ══"
HAVE=""
[ -f "$OUT/02_hivm/hivm_try.txt" ] && HAVE="$HAVE hivm"
[ -d "$OUT/03_sim/sim_prof" ] && HAVE="$HAVE sim"
[ -n "$(find "$OUT/05_task" -name "op_summary*.csv" 2>/dev/null | head -1)" ] && HAVE="$HAVE task"
[ -d "$OUT/04_board/board_prof" ] && HAVE="$HAVE board"
echo "  已有源:${HAVE:- 无}"
D=$OUT/06_diagnosis
if [ -f "$OUT/02_hivm/hivm_try.txt" ]; then
  "$PY" "$SCRIPT_DIR/pipeline_parse_hivm.py" "$OUT/02_hivm/hivm_try.txt" "$D/hivm.json" || true
fi
if [ -d "$OUT/03_sim/sim_prof" ]; then
  "$PY" "$SCRIPT_DIR/pipeline_parse_sim.py" "$OUT/03_sim/sim_prof" "$D/sim.json" || true
fi
if [ -n "$(find "$OUT/05_task" -name "op_summary*.csv" 2>/dev/null | head -1)" ]; then
  "$PY" "$SCRIPT_DIR/pipeline_parse_task.py" "$OUT/05_task" "$D/task.json" || true
fi
if [ -d "$OUT/04_board/board_prof" ]; then
  "$PY" "$SCRIPT_DIR/pipeline_parse_board.py" "$OUT/04_board/board_prof" "$D/board.json" || true
fi
# 整合 (缺的源用空占位文件; integrate 对空源自动跳过)
if [ -f "$D/hivm.json" ] || [ -f "$D/sim.json" ] || [ -f "$D/task.json" ] || [ -f "$D/board.json" ]; then
  echo '{}' > "$D/empty.json"
  for f in hivm sim task board; do
    [ -f "$D/$f.json" ] || cp "$D/empty.json" "$D/$f.json"
  done
  "$PY" "$SCRIPT_DIR/integrate.py" \
    "$D/hivm.json" "$D/sim.json" "$D/task.json" "$D/board.json" "$D/diagnosis.json"
  if [ -f "$D/diagnosis.json" ]; then pass "diagnosis.json 生成 ✓ (4源整合, 按优化策略组织)"; else fail "diagnosis.json 未生成"; fi
else
  fail "无任何源产物 (阶段2/3/4/5 至少一个 ✅ 才能整合)"
fi

# ═══════════════════════ 检查清单 ═══════════════════════
echo ""
echo "══════════════ 检查清单 (PASS=$PASS FAIL=$FAIL) ══════════════"
echo "  阶段1 编译   : [ttir/ttadapter]      → 看 01_compile/"
echo "  阶段2 HIVM   : [hivm_try.txt]        → grep -c 'hivm.hir' 02_hivm/hivm_try.txt"
echo "  阶段3 sim    : [instr_exe+trace]     → ls 03_sim/sim_prof/OPPROF_*/simulator/"
echo "  阶段4 board  : [OpBasic/PipeUtil]    → ls 04_board/board_prof/OPPROF_*/"
echo "  阶段5 task   : [op_summary]          → ls 05_task/task_prof/PROF_*/mindstudio_profiler_output/"
echo "  阶段6 诊断   : [diagnosis.json]      → python3 -m json.tool 06_diagnosis/diagnosis.json"
echo ""
echo "  全部 ✅ → 把 run_all.sh 末尾「验证摘要」的 4 行数字读回来"
echo "  有 ❌ → 按阶段日志排查 (每个阶段日志文件见上方提示), 单独重跑该阶段"
