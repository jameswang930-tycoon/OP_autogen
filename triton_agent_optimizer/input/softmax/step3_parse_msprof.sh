#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════════
#  Step 3: msprof simulator trace → pipeline_report.json (14 时序字段)
# ═════════════════════════════════════════════════════════════════════════════════
#
#  执行:
#    cd triton_agent_optimizer
#    bash input/softmax/step3_parse_msprof.sh
#
#  输入: input/softmax/msprof_sim/OPPROF_xxx/ (step1 产物)
#        └── simulator/
#            ├── core0.veccore0/core0.veccore0_instr_exe.csv  ← 指令级耗时
#            ├── core0.veccore0/trace.json                    ← 该核流水图
#            └── trace.json                                   ← 全核汇总
#  产出: input/softmax/msprof_sim/pipeline_report.json
#
# ═════════════════════════════════════════════════════════════════════════════════
#  instr_exe.csv 字段 (msprof 官方文档):
#    instr           — 硬件指令名称 (VADD / MOV_OUT_TO_UB / SET_FLAG / ...)
#    addr            — PC 地址
#    pipe            — 流水线通道 (MTE2/MTE3/VECTOR/CUBE/MTE1/SCALAR/ALL/FLOWCTRL/FIXP)
#    call_count      — 调用次数
#    cycles          — 执行总 cycle 数
#    running_time(us)— 执行时间 (微秒)
#    detail          — 指令详细参数
#
#  pipe → engine 映射 (triton-ascend 官方文档):
#    MTE2   → GM→UB  (vector path) 或 GM→L1 (cube/matmul path)
#    MTE3   → UB→GM
#    VECTOR → VecUnit
#    CUBE   → CubeUnit
#    MTE1   → L1→L0
#    FIXP   → L0→GM
#    SCALAR → 标量运算 (不映射到7-engine)
#    ALL    → 同步屏障
#    FLOWCTRL → 控制流
#
#  14 字段定义 (pipeline_report.json):
#    engine             — 7-engine 名称 (GM→UB / UB→GM / VecUnit / GM→L1 / L1→L0 / CubeUnit / L0→GM)
#    pipeline_channel   — 原始 pipe 值 (MTE2/MTE3/VECTOR/...)
#    duration_ns        — 该条指令耗时 (ns)
#    start_ns           — 起始时间 (ns)
#    end_ns             — 结束时间 (ns)
#    time_ratio         — 占总时间比例
#    cycles             — cycle 数
#    total_ns           — 总耗时 (ns)
#    num_ops            — 指令总数
#    execution_mode     — 执行模式 (parallel/sequential)
#    num_cores          — 核数
#    engine_utilization — 各 engine 耗时占比 {engine: ratio}
#    parallel_pairs     — 并行指令对 [{op_a, op_b, overlap_ns}]
#    critical_path      — 关键路径 (最长不重叠链) [op_ids]
# ═════════════════════════════════════════════════════════════════════════════════

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
AGENT="$(cd "$HERE/../.." && pwd)"
cd "$AGENT"

OPPROF=$(ls -dt "$HERE/msprof_sim"/OPPROF_*/ 2>/dev/null | head -1)
if [ -z "$OPPROF" ]; then
    echo "ERROR: 未找到 OPPROF 目录 (input/softmax/msprof_sim/OPPROF_*)"
    echo "请先执行 step1_compile_simulator.sh"
    exit 1
fi

echo "输入: $OPPROF"
echo ""

python3 -c "
import sys, json
from pathlib import Path
sys.path.insert(0, '.')
from analyzers.msprof_analyzer import MsprofAnalyzer

opprof = Path('$OPPROF')
ma = MsprofAnalyzer()
report = ma.parse_existing(opprof)
data = ma.to_dict(report)

out = opprof.parent / 'pipeline_report.json'
out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

print(f'指令数: {report.num_ops}')
print(f'核数: {report.num_cores}')
print(f'总耗时: {report.total_ns:.1f} ns ({report.total_ns/1000:.2f} us)')
print(f'执行模式: {report.execution_mode}')
print(f'产出: {out} ({out.stat().st_size} bytes)')

# 打印各 engine 耗时占比
print()
print('Engine 耗时占比:')
for eng, ratio in sorted(data.get('engine_utilization', {}).items(), key=lambda x: -x[1]):
    bar = '█' * int(ratio * 50)
    print(f'  {eng:<12} {ratio:>6.1%}  {bar}')
"
echo ""
echo "Step 3 完成 → pipeline_report.json"
