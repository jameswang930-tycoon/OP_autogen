#!/bin/bash
# ============================================================================
# 完整 Triton → HIVM → bishengir-opt 指令级流水线测试
# 用法: 在 WSL2 终端中执行
#   bash /mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer/question/full_pipeline_test.sh
# ============================================================================
set -e

# ── 环境 ──
export LD_PRELOAD=/tmp/libstub_cuda.so
export LD_LIBRARY_PATH=$HOME/.local/lib/python3.9/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null

PROJ=/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer
OUT=/tmp/hivm_pipeline_test
rm -rf "$OUT" && mkdir -p "$OUT"
cd "$PROJ"

echo "============================================"
echo " Step 1: Triton .py → TTIR MLIR"
echo "============================================"

python3 << 'PYEOF'
import os, sys, json
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
sys.path.insert(0, "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer")

from unittest.mock import MagicMock
import triton.runtime.driver as _drv
_drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp
_comp.CompiledKernel = MagicMock()

import importlib.util
from triton.compiler import ASTSource
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "k", "input/softmax/triton_kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

src = ASTSource(fn=fn, signature={0: "*fp32", 2: "i32"}, constants={"BLOCK_SIZE": 256})
opts = SimpleNamespace(num_warps=4, num_stages=1, debug=False)
ttir = str(src.make_ir(opts))
print(f"TTIR: {len(ttir)} chars")

with open("/tmp/hivm_pipeline_test/ttir.mlir", "w") as f:
    f.write(ttir)
print("TTIR saved")
PYEOF

echo ""
echo "============================================"
echo " Step 2: TTIR → HIVM MLIR (自研转换器)"
echo "============================================"

python3 << 'PYEOF'
import sys, json
sys.path.insert(0, "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer")

ttir = open("/tmp/hivm_pipeline_test/ttir.mlir").read()
from analyzers.ttir_to_hivm import ttir_to_hivm
hivm_text, hivm_ops = ttir_to_hivm(ttir, "softmax_kernel")

with open("/tmp/hivm_pipeline_test/hivm.mlir", "w") as f:
    f.write(hivm_text)
with open("/tmp/hivm_pipeline_test/hivm_ops.json", "w") as f:
    json.dump(hivm_ops, f, indent=2)

print(f"HIVM ops: {len(hivm_ops)}")
print(f"HIVM text: {len(hivm_text)} chars")
print("--- HIVM MLIR ---")
print(hivm_text)
PYEOF

echo ""
echo "============================================"
echo " Step 3: bishengir-opt 解析 HIVM MLIR"
echo "============================================"

echo "--- 3a: Just parse (no pass) ---"
bishengir-opt "$OUT/hivm.mlir" 2>&1 || true

echo ""
echo "--- 3b: HIVM → STD lowering ---"
bishengir-opt "$OUT/hivm.mlir" --convert-hivm-to-std 2>&1 || true

echo ""
echo "--- 3c: Available HIVM passes ---"
bishengir-opt --help 2>&1 | grep -i "hivm\|hfusion\|convert" | grep -v "enable\|disable\|hivmc\|link\|align\|barrier\|buffer\|cross\|graph\|inject\|limit\|nd2nz\|storage\|sync\|tensor\|unit\|version\|workspace"

echo ""
echo "============================================"
echo " Step 4: bishengir-compile (完整编译)"
echo "============================================"

echo "--- 4a: compile to .o ---"
bishengir-compile "$OUT/hivm.mlir" -o "$OUT/kernel.o" 2>&1 || true
ls -la "$OUT/kernel.o" 2>/dev/null && echo "kernel.o OK" || echo "kernel.o FAIL"

echo ""
echo "--- 4b: compile with IR dump ---"
bishengir-compile "$OUT/hivm.mlir" -o "$OUT/kernel2.o" \
  --enable-hivm-compile=true \
  --mlir-print-ir-after-all 2>&1 | head -100 || true

echo ""
echo "=== DONE ==="
echo "Output: $OUT"
ls -la "$OUT/"
