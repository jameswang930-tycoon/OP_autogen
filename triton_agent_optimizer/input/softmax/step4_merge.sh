#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 4: HIVM + msprof → merged_report.json (29 字段全填充)
# ═════════════════════════════════════════════════════════════════════════════════
#
#  执行:
#    cd triton_agent_optimizer
#    bash input/softmax/step4_merge.sh
#
#  输入:
#    input/softmax/hivmir/hivmir_report.json        (step2 产物, 11 字段)
#    input/softmax/msprof_sim/pipeline_report.json  (step3 产物, 14 字段)
#
#  产出:
#    input/softmax/merged_report.json     (29 字段 full report)
#    input/softmax/final_report_llm.txt   (LLM 可读文本)
#
# ═════════════════════════════════════════════════════════════════════════════════
#  29 字段 = 11 (HIVM 语义) + 14 (msprof 时序) + 4 (计算)
#
#  HIVM 11 字段:  op_id, op_type, instruction, dst, src, src2,
#                 size_kb, memory_region, variable_name, dependencies, dtype
#
#  msprof 14 字段: engine, pipeline_channel, duration_ns, start_ns, end_ns,
#                  time_ratio, cycles, total_ns, num_ops, execution_mode,
#                  num_cores, engine_utilization, parallel_pairs, critical_path
#
#  计算 4 字段:    effective_bw_gb_s, peak_bw_gb_s, bw_utilization, regime
#
#  SATURATION_PARAMS (910B3 硬件参数):
#    Engine 0 (GM→UB):   vpeak=121.08, k0=6.65KB, peak_clamp=80.83 GB/s
#    Engine 1 (UB→GM):   vpeak=190.19, k0=10.72KB, peak_clamp=76.67 GB/s
#    Engine 2 (VecUnit): vpeak=461.0,  k0=4.50KB, peak_clamp=404.0 GB/s
#    Engine 3 (GM→L1):   vpeak=37.5,   k0=6.65KB, peak_clamp=37.5 GB/s  (PLACEHOLDER)
#    Engine 4 (L1→L0):   vpeak=100.0,  k0=6.65KB, peak_clamp=100.0 GB/s (PLACEHOLDER)
#    Engine 5 (CubeUnit):vpeak=150.0,  k0=0,      peak_clamp=150.0 GB/s (PLACEHOLDER)
#    Engine 6 (L0→GM):   vpeak=37.5,   k0=6.65KB, peak_clamp=37.5 GB/s  (PLACEHOLDER)
#
#  regime 分类:
#    saturated: bw_util ≥ 95%
#    ramp:      50% < bw_util < 95%
#    floor:     bw_util ≤ 50%
# ═════════════════════════════════════════════════════════════════════════════════

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT="$(cd "$HERE/../.." && pwd)"
cd "$AGENT"

HIVM="$HERE/hivmir/hivmir_report.json"
MSPROF="$HERE/msprof_sim/pipeline_report.json"

if [ ! -f "$HIVM" ]; then
    echo "ERROR: $HIVM 不存在 → 请先执行 step2_parse_hivm.sh"
    exit 1
fi
if [ ! -f "$MSPROF" ]; then
    echo "ERROR: $MSPROF 不存在 → 请先执行 step3_parse_msprof.sh"
    exit 1
fi

echo "输入:"
echo "  HIVM:   $HIVM ($(wc -c < "$HIVM") bytes)"
echo "  msprof: $MSPROF ($(wc -c < "$MSPROF") bytes)"
echo ""

python3 -c "
import sys, json
sys.path.insert(0, '.')
from analyzers.dsl_merger import merge, format_llm

hivm = json.load(open('$HIVM', encoding='utf-8'))
msprof = json.load(open('$MSPROF', encoding='utf-8'))

merged = merge(hivm, msprof, tier=1)

out_json = '$HERE/merged_report.json'
out_llm  = '$HERE/final_report_llm.txt'

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

with open(out_llm, 'w', encoding='utf-8') as f:
    f.write(format_llm(merged))

import os
print(f'ops: {len(merged[\"per_op_statistics\"])}')
print(f'has_timing: {merged[\"meta\"][\"has_msprof_timing\"]}')
print(f'产出: {out_json} ({os.path.getsize(out_json)} bytes)')
print(f'产出: {out_llm} ({os.path.getsize(out_llm)} bytes)')

# 打印 per-op 摘要
print()
print('Per-op 摘要:')
for op in merged.get('per_op_statistics', [])[:15]:
    bw = op.get('bw_utilization', 0)
    reg = op.get('regime', '?')
    print(f'  op{op[\"op_id\"]:>2}  {op.get(\"engine\",\"?\"):<10}  bw={bw:.1%}  {reg}')
"
echo ""
echo "Step 4 完成 → merged_report.json + final_report_llm.txt"
