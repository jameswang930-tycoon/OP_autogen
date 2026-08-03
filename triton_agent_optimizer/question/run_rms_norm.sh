#!/bin/bash
set -e

# ============================================
# RMSNorm 完整优化迭代测试
# ============================================
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer

# 1. 环境
echo 'typedef unsigned int N;N a(void*x,void*y){return 0;}N b(void*x,void*y){return 0;}N c(void*x,void*y){return 0;}N d(void*x,void*y){return 0;}N e(void*x){return 0;}void f(void*x){}N g(void*x,void*y){return 0;}int h(void*x,void*y){return 0;}int i(void*x){return 0;}void j(void){}' | gcc -shared -fPIC -o /tmp/libstub_cuda.so -xc -
export LD_PRELOAD=/tmp/libstub_cuda.so
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# 2. 清理
rm -rf outputs/rms_norm

# 3. 运行 — 30轮, 目标2.0x
python3 main.py input/rms_norm/triton_kernel.py \
  --max-rounds 30 --target 2.0 \
  --msprof-dir ~/msprof_out2/OPPROF_* 2>&1 | tee /tmp/rms_norm_run.log

# 4. 分析结果
echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"

python3 << 'PYEOF'
import json, os, glob

base = "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/rms_norm"

# Trajectory
traj = json.load(open(f"{base}/optimization_trajectory.json"))
state = traj["state"]
print(f"Rounds: {state['round']}, Best: {state['best_speedup']:.3f}x, Tier: {state['tier']}")
print(f"Baseline: {traj['baseline']['total_ns']:.0f}ns, {traj['baseline']['num_ops']} ops")
print()

# History summary
print("History:")
for h in traj["history"]:
    sp = h.get("actual_speedup", 1.0)
    cs = h.get("cumulative_speedup", 1.0)
    d = h.get("decision", "?")
    s = h.get("strategy", "?")[:50]
    r = h.get("decision_reason", "")[:60]
    print(f"  R{h['round']:2d} T{h.get('tier',0)} {s:50s} sp={sp:.3f} {d:8s} {r}")

print()

# Per-round msprof traces
print("Per-round msprof:")
for d in sorted(glob.glob(f"{base}/*/round*")):
    rd = os.path.basename(d)
    csvs = glob.glob(f"{d}/msprof/**/*instr_exe.csv", recursive=True)
    merged_file = f"{d}/merged/merged_report.json"
    total_ns = "?"
    if os.path.exists(merged_file):
        mr = json.load(open(merged_file))
        total_ns = mr.get("execution_summary", {}).get("total_ns", "?")
    print(f"  {rd:20s}: {len(csvs)} csv, total_ns={total_ns}")

# Check errors
print()
errors_found = False
for h in traj["history"]:
    if h.get("decision") == "REVERT" and "emulator_passed" in h and not h["emulator_passed"]:
        errors_found = True
        print(f"⚠ R{h['round']}: Emulator FAIL — {h.get('decision_reason','')[:80]}")
    if h.get("code_lines_changed", 0) == 0 and h.get("decision") == "REVERT":
        errors_found = True
        print(f"⚠ R{h['round']}: Coder no-op — {h.get('decision_reason','')[:80]}")

if not errors_found:
    print("✅ No critical errors in iteration history")

# Check if any round actually improved
improvements = [h for h in traj["history"] if h.get("actual_speedup", 1.0) > 1.005]
if improvements:
    print(f"\n✅ {len(improvements)} rounds with speedup > 1.005x:")
    for h in improvements:
        print(f"   R{h['round']}: {h['actual_speedup']:.3f}x — {h['strategy'][:50]}")
else:
    print(f"\n⚠ No rounds with significant speedup")

print(f"\nFinal state: tier={state['tier']}, best={state['best_speedup']:.3f}x")
PYEOF
