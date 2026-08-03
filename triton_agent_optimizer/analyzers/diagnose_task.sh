#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  诊断: 通用 msprof 全指标为何 255 (逐 flag / 逐 metric 定位)
#  用法: bash diagnose_task.sh
#  输出: 每个组合的 rc + op_summary 数; 第一个 rc!=0 的就是问题项
# ═══════════════════════════════════════════════════════════════════════════════
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MATMUL_DIR="$SCRIPT_DIR/../input/matmul"
cd "$MATMUL_DIR" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
OUT=/tmp/msprof_diag
mkdir -p "$OUT"

echo "══════ 1. msprof 版本 + 有效 flags ══════"
msprof --version 2>&1 | head -2
echo "--- 相关 flags (aic/task-time/metric/ai-core) ---"
msprof --help 2>&1 | grep -iE 'aic|task-time|metric|ai-core' | head -20
echo ""

echo "══════ 2. 逐级加 flag/metric (第一个非 0 即问题项) ══════"
run() {
  local name="$1"; shift
  rm -rf "$OUT/$name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local n=$(find "$OUT/$name" -name "op_summary*.csv" 2>/dev/null | wc -l)
  echo "  [$name] rc=$rc op_summary x$n"
  echo "     cmd: $*"
  [ "$rc" -ne 0 ] && echo "     log尾: $(tail -3 "$OUT/$name.log" 2>/dev/null | tr '\n' ' ')"
  [ "$rc" -eq 0 ] && [ "$n" -gt 0 ] && echo "     ✅ 这个组合可用 (op_summary 生成)"
  echo ""
}

# 基准: 无任何 aic flag (肯定能出 op_summary)
run "00_basic" msprof --output="$OUT/00_basic" --application="$PY test_matmul.py"
# 加 --ai-core
run "01_aicore" msprof --output="$OUT/01_aicore" --application="$PY test_matmul.py" --ai-core=on
# 逐个加 metric (ArithmeticUtilization+PipeUtilization 基础, 再加其余)
run "02_2m"  msprof --output="$OUT/02_2m"  --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization"
run "03_mem" msprof --output="$OUT/03_mem" --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory"
run "04_l2"  msprof --output="$OUT/04_l2"  --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory,L2Cache"
run "05_meml0" msprof --output="$OUT/05_meml0" --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory,MemoryL0"
run "06_memub" msprof --output="$OUT/06_memub" --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory,MemoryUB"
run "07_cflt" msprof --output="$OUT/07_cflt" --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory,ResourceConflictRatio"
run "08_all" msprof --output="$OUT/08_all" --application="$PY test_matmul.py" --ai-core=on --aic-metrics="ArithmeticUtilization,PipeUtilization,Memory,L2Cache,MemoryL0,MemoryUB,ResourceConflictRatio"

echo "══════ 3. msprof op 详细指标 (带宽/L2/cube) ══════"
# 关键: msprof op 默认全量 8 CSV, 不要指定 --aic-metrics (指定会限制/报错)
oprun() {
  local name="$1"; shift
  rm -rf "$OUT/$name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  local n8=$(ls "$OUT/$name"/OPPROF_*/*.csv 2>/dev/null | wc -l)
  local mem=$(ls "$OUT/$name"/OPPROF_*/Memory.csv 2>/dev/null | wc -l)
  local l2=$(ls "$OUT/$name"/OPPROF_*/L2Cache.csv 2>/dev/null | wc -l)
  local arith=$(ls "$OUT/$name"/OPPROF_*/ArithmeticUtilization.csv 2>/dev/null | wc -l)
  echo "  [$name] rc=$rc csv x$n8  Memory x$mem  L2Cache x$l2  ArithmeticUtilization x$arith"
  [ "$rc" -ne 0 ] && echo "     log尾: $(tail -3 "$OUT/$name.log" 2>/dev/null | tr '\n' ' ')"
  echo ""
}
# 默认全量 (推荐, 应出 8 个 CSV)
oprun "op_default" msprof op --kernel-name=matmul_kernel --output="$OUT/op_default" $PY test_matmul.py
# 加 --warm-up (防降频, run_server_flow.sh 用它)
oprun "op_warmup"  msprof op --kernel-name=matmul_kernel --output="$OUT/op_warmup" --warm-up=10 $PY test_matmul.py

echo "══════ 4. 结论 ══════"
echo "  第2节: 通用 msprof 用 00/01 (--ai-core=on) 拿 op_summary。"
echo "  第3节: msprof op 默认全量 → 应出 8 个 CSV (Memory=带宽/L2Cache=L2/ArithmeticUtilization=cube)"
echo "    Memory>0 → 真实带宽; L2Cache>0 → L2; Arithmetic>0 → cube/vec 算力 → board.json 就全了"
echo "  把本脚本输出整体贴回, 我据此确认 stage4/5 命令。"
