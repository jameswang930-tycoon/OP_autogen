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
# ★bug 修复: op_dir 必须是目录 (input/rms_norm), 不能传 triton_kernel.py 文件;
#   --msprof-dir 不是 main.py 的合法参数 (调度器内部用 run_optimize.sh 采 msprof)
python3 main.py input/rms_norm \
  --max-rounds 30 --target 2.0 2>&1 | tee /tmp/rms_norm_run.log

# 4. 分析结果
echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"

python3 << 'PYEOF'
import json, os, glob

base = "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/outputs/rms_norm"

# Trajectory
# ★bug 修复: v4 trajectory 无 traj['baseline'] 子对象 — baseline 在 state['baseline_ns']/num_kernels
traj = json.load(open(f"{base}/optimization_trajectory.json"))
state = traj["state"]
print(f"Rounds: {state['round']}, Best: {state['best_speedup']:.3f}x, Tier: {state['tier']}")
print(f"Baseline: {state.get('baseline_ns', '?')}ns, {state.get('num_kernels', '?')} kernels")
print()

# History summary
# ★bug 修复: v4 hist 键是 speedup/prev_speedup/decision/result/error (无 actual_speedup 等)
print("History:")
for h in traj["history"]:
    sp = h.get("speedup", 1.0)
    d = h.get("decision", "?")
    s = h.get("strategy", "?")[:50]
    r = h.get("error", "")[:60] or h.get("change", "")[:60]
    print(f"  R{h['round']:2d} T{h.get('tier',0)} {s:50s} sp={sp:.3f} {d:8s} {r}")

print()

# Check errors
print()
errors_found = False
for h in traj["history"]:
    if h.get("decision") == "REVERT" and h.get("result") == "NOOP":
        errors_found = True
        print(f"⚠ R{h['round']}: Coder no-op (NOOP) — {h.get('error','')[:80]}")
if not errors_found:
    print("✅ No critical errors in iteration history")

# Check if any round actually improved
improvements = [h for h in traj["history"] if h.get("speedup", 1.0) > 1.005]
if improvements:
    print(f"\n✅ {len(improvements)} rounds with speedup > 1.005x:")
    for h in improvements:
        print(f"   R{h['round']}: {h['speedup']:.3f}x — {h['strategy'][:50]}")
else:
    print(f"\n⚠ No rounds with significant speedup")

print(f"\nFinal state: tier={state['tier']}, best={state['best_speedup']:.3f}x")
PYEOF
