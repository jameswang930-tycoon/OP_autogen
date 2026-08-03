#!/bin/bash
# 用 main.py 自带的 run_triton_to_hivm 生成 HIVM，然后 bishengir-opt 解析
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
source /usr/local/Ascend/cann/set_env.sh

python3 -c '
import sys; sys.path.insert(0,".")
from main import run_triton_to_hivm
from pathlib import Path
run_triton_to_hivm(Path("input/softmax/triton_kernel.py"), Path("/tmp/hivm_out"), "softmax_kernel")
print("HIVM GENERATED")
'

echo "=== HIVM MLIR ==="
cat /tmp/hivm_out/hivmir/compiler_output/hivmir_output.mlir

echo ""
echo "=== bishengir-opt parse ==="
bishengir-opt /tmp/hivm_out/hivmir/compiler_output/hivmir_output.mlir 2>&1

echo ""
echo "=== bishengir-opt HIVM->STD lowering ==="
bishengir-opt /tmp/hivm_out/hivmir/compiler_output/hivmir_output.mlir --convert-hivm-to-std 2>&1

echo ""
echo "=== bishengir-compile ==="
bishengir-compile /tmp/hivm_out/hivmir/compiler_output/hivmir_output.mlir -o /tmp/hivm_out/kernel.o 2>&1
ls -la /tmp/hivm_out/kernel.o 2>/dev/null && echo "kernel.o OK" || echo "kernel.o FAIL"
