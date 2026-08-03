#!/bin/bash
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer

# 重建 stub
echo 'typedef unsigned int N;N a(void*x,void*y){return 0;}N b(void*x,void*y){return 0;}N c(void*x,void*y){return 0;}N d(void*x,void*y){return 0;}N e(void*x){return 0;}void f(void*x){}N g(void*x,void*y){return 0;}int h(void*x,void*y){return 0;}int i(void*x){return 0;}void j(void){}' | gcc -shared -fPIC -o /tmp/libstub_cuda.so -xc -

export LD_PRELOAD=/tmp/libstub_cuda.so
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

rm -rf outputs/fam_e2e

python3 main.py input/fused_add_mul/triton_kernel.py \
  --max-rounds 3 --target 100.0 \
  --msprof-dir ~/msprof_out2/OPPROF_* 2>&1 | head -120

echo ""
echo "=== per-round msprof ==="
for d in outputs/fam_e2e/*/round*/ 2>/dev/null; do
  CSV=$(find "$d" -name "*instr_exe.csv" 2>/dev/null | wc -l)
  echo "  $d : $CSV instr_exe.csv"
done
echo "=== trajectory ==="
python3 -c "import json; d=json.load(open('outputs/fam_e2e/optimization_trajectory.json')); print(json.dumps({'rounds':d['state']['round'],'best':d['state']['best_speedup'],'tier':d['state']['tier']}))" 2>/dev/null || true
