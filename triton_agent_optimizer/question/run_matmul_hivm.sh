#!/bin/bash
# matmul 完整 HIVM pipeline 测试
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
source /usr/local/Ascend/cann/set_env.sh

python3 -c '
import sys; sys.path.insert(0,".")
from main import run_triton_to_hivm
from pathlib import Path
run_triton_to_hivm(Path("input/matmul/triton_kernel.py"), Path("/tmp/hivm_matmul"), "matmul_kernel")
print("HIVM GENERATED")
'

echo "=== HIVM MLIR ==="
cat /tmp/hivm_matmul/hivmir/compiler_output/hivmir_output.mlir

echo ""
echo "=== bishengir-opt parse ==="
bishengir-opt /tmp/hivm_matmul/hivmir/compiler_output/hivmir_output.mlir 2>&1

echo ""
echo "=== bishengir-opt HIVM->STD ==="
bishengir-opt /tmp/hivm_matmul/hivmir/compiler_output/hivmir_output.mlir --convert-hivm-to-std 2>&1

echo ""
echo "=== bishengir-compile ==="
bishengir-compile /tmp/hivm_matmul/hivmir/compiler_output/hivmir_output.mlir -o /tmp/hivm_matmul/kernel.o 2>&1
ls -la /tmp/hivm_matmul/kernel.o 2>/dev/null && echo "kernel.o OK" || echo "kernel.o FAIL"
