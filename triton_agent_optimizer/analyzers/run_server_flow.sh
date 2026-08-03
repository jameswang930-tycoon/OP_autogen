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
#    05_merged/    hivm.json + sim.json + board.json + merged.json
#
#  每个阶段跑完自动检查产物 → 打 ✅/❌; 末尾打印「检查清单」汇总全部 PASS/FAIL。
#  任何阶段失败不中断后续 (清单里看哪步没过, 单独重跑该阶段)。
#
#  前置: 脚本会尝试激活 conda triton-npu + set_env.sh; 失败就手动激活后再跑。
#  环境变量可调:
#    SIM_TIMEOUT=秒     simulator 超时 (默认 1800 = 30分钟, 64³ 通常几十秒内)
#    SKIP_SIM=1         跳过 simulator (不想等)
#    SKIP_BOARD=1       跳过真机
# ═══════════════════════════════════════════════════════════════════════════════
set -u

M=${1:-64}; N=${2:-64}; K=${3:-64}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATMUL_DIR="$(cd "$SCRIPT_DIR/../input/matmul" && pwd)"
OUT="$MATMUL_DIR/e2e_run"
SIM_TIMEOUT=${SIM_TIMEOUT:-1800}

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

PASS=0; FAIL=0
pass(){ echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1"; FAIL=$((FAIL+1)); }

# ── 环境 ──
echo "══════ 环境 ══════"
echo "M=$M N=$N K=$K  输出=$OUT"
mkdir -p "$OUT"/01_compile "$OUT"/02_hivm "$OUT"/03_sim "$OUT"/04_board "$OUT"/05_merged
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

# ═══════════════════════ 阶段 1/5: 编译 → ttir/ttadapter ═══════════════════════
echo ""
echo "══ 阶段 1/5: 编译 → ttir/ttadapter ══"
rm -rf ~/.triton
export TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1
MATMUL_M=$M MATMUL_N=$N MATMUL_K=$K $PY test_matmul.py > "$OUT/01_compile/run_debug.txt" 2>&1
echo "  编译运行退出码=$? (日志: 01_compile/run_debug.txt)"
cp ~/.triton/dump/*/kernel.*.mlir "$OUT/01_compile/" 2>/dev/null || true
if [ -f "$OUT/01_compile/kernel.ttir.mlir" ]; then pass "ttir.mlir 存在"; else fail "缺 ttir.mlir → 看 01_compile/run_debug.txt (缓存? 报错?)"; fi
if [ -f "$OUT/01_compile/kernel.ttadapter.mlir" ]; then pass "ttadapter.mlir 存在"; else fail "缺 ttadapter.mlir"; fi

# ═══════════════════════ 阶段 2/5: 真实 HIVM (流程 D) ═══════════════════════
echo ""
echo "══ 阶段 2/5: 真实 HIVM (bishengir 打印) ══"
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

# ═══════════════════════ 阶段 3/5: simulator (指令级时序) ═══════════════════════
echo ""
echo "══ 阶段 3/5: msprof op simulator (64³ 应几十秒内) ══"
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

# ═══════════════════════ 阶段 4/5: 真机 msprof op ═══════════════════════
echo ""
echo "══ 阶段 4/5: 真机 msprof op (端到端/引擎占比) ══"
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

# ═══════════════════════ 阶段 5/5: 解析 + 合并 ═══════════════════════
echo ""
echo "══ 阶段 5/5: 三源解析 + 合并 (run_all.sh) ══"
if [ -f "$OUT/02_hivm/hivm_try.txt" ] && [ -d "$OUT/03_sim/sim_prof" ] && [ -d "$OUT/04_board/board_prof" ]; then
  bash "$SCRIPT_DIR/run_all.sh" "$OUT/02_hivm/hivm_try.txt" \
    "$OUT/03_sim/sim_prof" "$OUT/04_board/board_prof" "$OUT/05_merged"
  if [ -f "$OUT/05_merged/merged.json" ]; then pass "merged.json 生成 ✓"; else fail "merged.json 未生成"; fi
else
  echo "  ⚠ 缺前置产物, 跳过合并 (hivm/sim/board 都通过后再跑)"
  fail "缺前置 (需阶段2/3/4 都 ✅)"
fi

# ═══════════════════════ 检查清单 ═══════════════════════
echo ""
echo "══════════════ 检查清单 (PASS=$PASS FAIL=$FAIL) ══════════════"
echo "  阶段1 编译   : [ttir/ttadapter]   → 看 01_compile/"
echo "  阶段2 HIVM   : [hivm_try.txt]     → grep -c 'hivm.hir' 02_hivm/hivm_try.txt"
echo "  阶段3 sim    : [instr_exe+trace]  → ls 03_sim/sim_prof/OPPROF_*/simulator/"
echo "  阶段4 board  : [OpBasic/PipeUtil] → ls 04_board/board_prof/OPPROF_*/"
echo "  阶段5 合并   : [merged.json]      → python3 -m json.tool 05_merged/merged.json"
echo ""
echo "  全部 ✅ → 把 run_all.sh 末尾「验证摘要」的 4 行数字读回来"
echo "  有 ❌ → 按阶段日志排查 (每个阶段日志文件见上方提示), 单独重跑该阶段"
