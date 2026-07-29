#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 2: HIVM MLIR → hivmir_report.json (11 语义字段)
# ═════════════════════════════════════════════════════════════════════════════════
#
#  执行:
#    cd triton_agent_optimizer
#    bash input/softmax/step2_parse_hivm.sh
#
#  输入: input/softmax/hivmir/*.mlir (step1 产物)
#  产出: input/softmax/hivmir/hivmir_report.json
#
# ═════════════════════════════════════════════════════════════════════════════════
#  11 字段定义 (hivmir_analyzer.py):
#    op_id          — 操作编号
#    op_type        — 操作类型 (gm_to_ub / ub_to_gm / vadd / vexp / ...)
#    instruction    — HIVM 操作原文 (hivm.hir.load / hivm.hir.vadd / ...)
#    dst            — SSA 目标 %buf
#    src            — SSA 源 %arg
#    src2           — SSA 第二源 (可选)
#    size_kb        — buffer 大小 (KB)
#    memory_region  — #hivm.address_space<ub/gm/l1>
#    variable_name  — SSA 变量名
#    dependencies   — RAW/WAR/WAW 依赖链 [{from_op_id, type}]
#    dtype          — 数据类型 (f32/f16)
# ═════════════════════════════════════════════════════════════════════════════════

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT="$(cd "$HERE/../.." && pwd)"
cd "$AGENT"

# 找第一个 .mlir 文件
MLIR=$(ls -t "$HERE/hivmir"/*.mlir 2>/dev/null | head -1)
if [ -z "$MLIR" ]; then
    echo "ERROR: 未找到 HIVM MLIR 文件 (input/softmax/hivmir/*.mlir)"
    echo "请先执行 step1_compile_simulator.sh"
    exit 1
fi

echo "输入: $MLIR ($(wc -c < "$MLIR") bytes)"
echo ""

python3 -c "
import sys, json
from pathlib import Path
sys.path.insert(0, '.')
from analyzers.hivmir_analyzer import HIVMIRAnalyzer

mlir_path = Path('$MLIR')
ha = HIVMIRAnalyzer()
report = ha.analyze_file(mlir_path)
data = ha.to_dict(report)

out = mlir_path.parent / 'hivmir_report.json'
out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

print(f'ops: {report.num_ops}')
print(f'RAW deps: {len(report.raw_deps)}')
print(f'WAR deps: {len(report.war_deps)}')
print(f'WAW deps: {len(report.waw_deps)}')
print(f'产出: {out} ({out.stat().st_size} bytes)')

# 打印 ops 摘要
print()
for op in data.get('ops', [])[:10]:
    print(f'  op{op[\"op_id\"]:>2}  {op[\"op_type\"]:<12}  {op.get(\"size_kb\",0):>6.2f}KB  mem={op.get(\"memory_region\",\"?\")}')
if len(data.get('ops', [])) > 10:
    print(f'  ... 共 {len(data[\"ops\"])} 个 ops')
"
echo ""
echo "Step 2 完成 → hivmir_report.json"
