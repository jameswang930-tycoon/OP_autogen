"""重新生成 HIVM MLIR（修复后的 ttir_to_hivm），保存到 /tmp/ir2/"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv
_drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp
_comp.CompiledKernel = MagicMock()
from triton.compiler import ASTSource
from types import SimpleNamespace
import importlib.util

spec = importlib.util.spec_from_file_location("k", "input/softmax/triton_kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

src = ASTSource(fn=fn, signature={0: "*fp32", 2: "i32"}, constants={"BLOCK_SIZE": 256})
ttir = str(src.make_ir(SimpleNamespace(num_warps=4, num_stages=1, debug=False)))
print(f"TTIR: {len(ttir)} chars")

from analyzers.ttir_to_hivm import ttir_to_hivm
hivm_text, hivm_ops = ttir_to_hivm(ttir, "softmax_kernel")
print(f"HIVM: {len(hivm_ops)} ops")

os.makedirs("/tmp/ir2", exist_ok=True)
with open("/tmp/ir2/hivm_output.mlir", "w") as f:
    f.write(hivm_text)
with open("/tmp/ir2/hivm_ops.json", "w") as f:
    json.dump(hivm_ops, f, indent=2)
print("SAVED to /tmp/ir2/")

# Check for scalars
has_scalars = any(s in hivm_text for s in ["%11", "%13", "%15"])
print(f"Has undeclared scalars: {has_scalars}")
print(hivm_text[:500])
