#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#  一键流水线: 3 源解析 → 3 个 JSON → 合并 → 1 个 merged.json + 自动验证摘要
# ═══════════════════════════════════════════════════════════════════════════════
#  用法 (服务器):
#     bash run_all.sh <hivm_try.txt> <sim_prof目录> <board_prof目录> [输出目录]
#
#  前置: 必须先采集好 3 个真实产物 (命令在 input/matmul/test_matmul.py 头注释):
#     ① hivm_try.txt   — 真实 HIVM (标准流程 D: bishengir-compile 打印)
#     ② sim_prof/      — msprof op simulator --output=./sim_prof  (含 OPPROF_*/simulator/)
#     ③ board_prof/    — msprof op --output=./board_prof  (含 OPPROF_*/OpBasicInfo.csv 等)
#  环境: conda activate triton-npu && source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
#  产物 (4 个 JSON, 输出目录默认 ./outputs/matmul_e2e/):
#     hivm.json   结构字段 (op/engine/dst/src/size/region/依赖/attrs), 时序=None
#     sim.json    指令级真实时序 (per-call 耗时/cycles/call_count/搬运块大小), 按指令名+pipe 聚合
#     board.json  真机实测 (total_ns=Task Duration, cores=Block Dim, 引擎占比, 带宽)
#     merged.json 合并: per-op=HIVM语义op, 时序贴 sim, 端到端/占比用真机
#
#  验证方法:
#     跑完脚本末尾自动打印「验证摘要」, 看 4 行关键数字:
#       - hivm  ops=?       应该 ≈ 真实 IR 的语义 op 数 (含 sync)
#       - sim   指令组=?    关键 pipe 应含 MTE2/MTE3/CUBE/SET_FLAG/WAIT_FLAG/BAR
#       - board total_ns=?  应 ≈ 真机实测端到端; cube_ratio 应非 0
#       - merged 对齐=X/N   X 越大越好; 若 < N, 看「未对齐」列表的 op_type → 调 pipeline_merge.py 的映射
#     也可: python3 -m json.tool <out>/merged.json
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIVM=${1:?用法: bash run_all.sh <hivm_try.txt> <sim_prof目录> <board_prof目录> [输出目录]}
SIM=${2:?需要 sim_prof 目录}
BOARD=${3:?需要 board_prof 目录}
OUT=${4:-"$DIR/../outputs/matmul_e2e"}
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
[ -n "${PYTHON:-}" ] && PY=$PYTHON

# ── 前置检查: 3 源是否存在 ──
echo "══════ 前置检查 ══════"
[ -f "$HIVM" ] && echo "  ✅ HIVM   = $HIVM" || { echo "  ❌ 找不到 $HIVM (先跑 test_matmul.py 流程 D)"; exit 1; }
[ -d "$SIM" ] && echo "  ✅ SIM    = $SIM" || { echo "  ❌ 找不到 $SIM"; exit 1; }
[ -d "$BOARD" ] && echo "  ✅ BOARD  = $BOARD" || { echo "  ❌ 找不到 $BOARD"; exit 1; }

mkdir -p "$OUT"
echo ""
echo "══════ 流水线 start ══════"
echo "OUT   = $OUT"

echo ""
echo "== 1/4 解析 HIVM (hivm_try.txt) → hivm.json =="
echo "   预期: ops=真实语义op数, 含 sync; 每个 op 有 op_type/engine/dst/src/size_kb/memory_region/依赖"
"$PY" "$DIR/pipeline_parse_hivm.py" "$HIVM" "$OUT/hivm.json"

echo ""
echo "== 2/4 解析 simulator (instr_exe.csv+trace.json) → sim.json =="
echo "   预期: 指令组=数十~数百; 关键 pipe 含 MTE2/MTE3/CUBE/VECTOR + SET_FLAG/WAIT_FLAG/BAR;"
echo "         每指令组有 duration_ns(per-call)/cycles/call_count/data_size_bytes"
"$PY" "$DIR/pipeline_parse_sim.py" "$SIM" "$OUT/sim.json"

echo ""
echo "== 3/4 解析真机 msprof op (board_prof) → board.json =="
echo "   预期: total_ns=真机端到端(us×1000), num_cores=Block Dim; engine_util 含 CubeUnit/VecUnit 占比;"
echo "         bandwidth_utilization 含 Memory*.csv 带宽 + ArithmeticUtilization 的 cube fops"
"$PY" "$DIR/pipeline_parse_board.py" "$BOARD" "$OUT/board.json"

echo ""
echo "== 4/4 合并三源 → merged.json =="
echo "   预期: per-op=HIVM语义op(结构真实), 时序贴 sim(per-call), total_ns/cores/占比用真机"
"$PY" "$DIR/pipeline_merge.py" "$OUT/hivm.json" "$OUT/sim.json" "$OUT/board.json" "$OUT/merged.json"

echo ""
echo "══════ 验证摘要 ══════"
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
hv = json.load(open(out/"hivm.json", encoding="utf-8"))
sm = json.load(open(out/"sim.json", encoding="utf-8"))
bd = json.load(open(out/"board.json", encoding="utf-8"))
mg = json.load(open(out/"merged.json", encoding="utf-8"))

print(f"hivm  : ops={hv['execution_summary']['num_ops']}  deps={hv['dependencies_summary'].get('total','?')}")
pipes = sorted({o.get('pipeline_channel') for o in sm['per_op_statistics'] if o.get('pipeline_channel')})
print(f"sim   : 指令组={sm['execution_summary']['num_ops']}  pipes={pipes}")
print(f"        total_ns(trace)={sm['execution_summary'].get('total_ns')}")
b_sum = bd['execution_summary']
print(f"board : total_ns={b_sum.get('total_ns')}  cores={b_sum.get('num_cores')}  engine_util={json.dumps(bd['engine_utilization'], ensure_ascii=False)}")
m_sum = mg['execution_summary']
aligned = [o for o in mg['per_op_statistics'] if o.get('duration_ns') is not None]
missing = [f"{o['op_type']}" for o in mg['per_op_statistics'] if o.get('duration_ns') is None]
print(f"merged: ops={m_sum.get('num_ops')}  total_ns={m_sum.get('total_ns')}  cores={m_sum.get('num_cores')}")
print(f"        时序对齐={len(aligned)}/{len(mg['per_op_statistics'])}  未对齐={missing if missing else '无'}")
print(f"\n完整看: python3 -m json.tool {out}/merged.json")
PYEOF

echo ""
echo "══════ 完成: 4 个 JSON 在 $OUT ══════"
ls -la "$OUT"/*.json
