#!/bin/bash
# 端到端测试: Triton .py → per-round msprof → speedup
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer

export LD_PRELOAD=/tmp/libstub_cuda.so
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# ★bug 修复: op_dir 必须是目录 (input/softmax); --msprof-dir 不是合法参数;
#   输出写到 outputs/<op_dir.name> = outputs/softmax (不是 softmax_e2e)
rm -rf outputs/softmax

echo "============================================"
echo " Test: softmax, 2 rounds, per-round msprof"
echo "============================================"

python3 main.py input/softmax \
  --max-rounds 2 --target 100.0 2>&1 | head -80

echo ""
echo "============================================"
echo " Check per-round msprof traces"
echo "============================================"

for d in outputs/softmax/*/round*/; do
  if [ -d "$d" ]; then
    CSV=$(find "$d" -name "*instr_exe.csv" 2>/dev/null | wc -l)
    echo "  $d : $CSV instr_exe.csv"
  fi
done

echo ""
echo "=== trajectory ==="
cat outputs/softmax/optimization_trajectory.json 2>/dev/null | python3 -m json.tool 2>/dev/null | head -20
