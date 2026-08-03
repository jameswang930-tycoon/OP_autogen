#!/bin/bash
# rms_norm_residual 完整 HIVM pipeline 测试 (比 softmax 更复杂)
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
source /usr/local/Ascend/cann/set_env.sh

echo "=== Generate HIVM ==="
python3 -c '
import sys; sys.path.insert(0,".")
from main import run_triton_to_hivm
from pathlib import Path
run_triton_to_hivm(Path("input/rms_norm_residual/triton_kernel.py"), Path("/tmp/hivm_rms"), "add_rms_norm_kernel")
'

echo "=== HIVM MLIR ==="
cat /tmp/hivm_rms/hivmir/compiler_output/hivmir_output.mlir

echo ""
echo "=== HIVM ops ==="
python3 -c 'import json; ops=json.load(open("/tmp/hivm_rms/hivmir/hivm_ops.json")); print(f"Total ops: {len(ops)}"); [print(f"  op{o[\"op_id\"]}: {o[\"op_type\"]:15s} size={o[\"size_kb\"]}KB  [{o[\"memory_region\"]}]") for o in ops]'

echo ""
echo "=== bishengir-opt ==="
bishengir-opt /tmp/hivm_rms/hivmir/compiler_output/hivmir_output.mlir 2>&1

echo ""
echo "=== bishengir-opt HIVM->STD ==="
bishengir-opt /tmp/hivm_rms/hivmir/compiler_output/hivmir_output.mlir --convert-hivm-to-std 2>&1

echo ""
echo "=== bishengir-compile ==="
bishengir-compile /tmp/hivm_rms/hivmir/compiler_output/hivmir_output.mlir -o /tmp/hivm_rms/kernel.o 2>&1
ls -la /tmp/hivm_rms/kernel.o 2>/dev/null && echo "kernel.o OK" || echo "kernel.o FAIL"
