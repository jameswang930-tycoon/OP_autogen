#!/bin/bash
# 完整 4 层流水线演示: TTIR → HIVM → HIVM-STD → .o
set -e
cd /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
source /usr/local/Ascend/cann/set_env.sh

# ── Layer 1: TTIR ──
echo "============================================"
echo " LAYER 1: TTIR (Triton Compiler)"
echo "============================================"
python3 -c '
import sys; sys.path.insert(0,".")
from main import run_triton_to_hivm
from pathlib import Path
run_triton_to_hivm(Path("input/softmax/triton_kernel.py"), Path("/tmp/demo"), "softmax_kernel")
'

echo "--- TTIR ops (from HIVM) ---"
python3 /dev/stdin << 'PYEOF'
import json
ops = json.load(open("/tmp/demo/hivmir/hivm_ops.json"))
for o in ops:
    print("  %s" % json.dumps(o, ensure_ascii=False))
print("  TOTAL: %d ops" % len(ops))
PYEOF

# ── Layer 2: HIVM MLIR ──
echo ""
echo "============================================"
echo " LAYER 2: HIVM MLIR (bishengir-opt 解析验证)"
echo "============================================"
cat /tmp/demo/hivmir/compiler_output/hivmir_output.mlir

echo ""
echo "--- bishengir-opt parse ---"
bishengir-opt /tmp/demo/hivmir/compiler_output/hivmir_output.mlir 2>&1
echo "RESULT: bishengir-opt PARSE OK (零错误)"

# ── Layer 3: HIVM→STD ──
echo ""
echo "============================================"
echo " LAYER 3: HIVM→STD Lowering (指令级展开)"
echo "============================================"
bishengir-opt /tmp/demo/hivmir/compiler_output/hivmir_output.mlir --convert-hivm-to-std 2>&1

# ── Layer 4: Compile ──
echo ""
echo "============================================"
echo " LAYER 4: bishengir-compile → .o"
echo "============================================"
bishengir-compile /tmp/demo/hivmir/compiler_output/hivmir_output.mlir -o /tmp/demo/kernel.o 2>&1
ls -la /tmp/demo/kernel.o
echo ".o 编译成功, 纯CPU, 不需要NPU"
