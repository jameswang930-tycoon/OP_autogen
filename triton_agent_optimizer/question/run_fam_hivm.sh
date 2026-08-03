#!/bin/bash
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
source /usr/local/Ascend/cann/set_env.sh

python3 -c '
import sys; sys.path.insert(0,".")
from main import run_triton_to_hivm
from pathlib import Path
run_triton_to_hivm(Path("input/fused_add_mul/triton_kernel.py"), Path("/tmp/hivm_fam"), "fused_add_mul_kernel")
'

echo "=== HIVM ops ==="
python3 -c '
import json
ops=json.load(open("/tmp/hivm_fam/hivmir/hivm_ops.json"))
for i,o in enumerate(ops):
    print("  op%d: %-12s size=%.1fKB [%s]" % (i, o.get("op_type","?"), o.get("size_kb",0), o.get("memory_region","?")))
print("  TOTAL: %d ops" % len(ops))
'

echo "=== HIVM MLIR ==="
cat /tmp/hivm_fam/hivmir/compiler_output/hivmir_output.mlir

echo ""
echo "=== bishengir-opt ==="
bishengir-opt /tmp/hivm_fam/hivmir/compiler_output/hivmir_output.mlir 2>&1

echo ""
echo "=== HIVM→STD ==="
bishengir-opt /tmp/hivm_fam/hivmir/compiler_output/hivmir_output.mlir --convert-hivm-to-std 2>&1

echo ""
echo "=== compile .o ==="
bishengir-compile /tmp/hivm_fam/hivmir/compiler_output/hivmir_output.mlir -o /tmp/hivm_fam/kernel.o 2>&1
ls -la /tmp/hivm_fam/kernel.o 2>/dev/null && echo "OK" || echo "FAIL"
