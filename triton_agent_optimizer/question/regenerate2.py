import os, sys, json
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock
import triton.runtime.driver as _drv
_drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp
_comp.CompiledKernel = MagicMock()

import importlib.util
from triton.compiler import ASTSource
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location(
    "k", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "input/softmax/triton_kernel.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

src = ASTSource(fn=fn, signature={0: "*fp32", 2: "i32"}, constants={"BLOCK_SIZE": 256})
opts = SimpleNamespace(num_warps=4, num_stages=1, debug=False)
ttir = str(src.make_ir(opts))
print(f"TTIR: {len(ttir)} chars")

from analyzers.ttir_to_hivm import ttir_to_hivm
hivm_text, hivm_ops = ttir_to_hivm(ttir, "softmax_kernel")
print(f"HIVM: {len(hivm_ops)} ops, text: {len(hivm_text)} chars")

os.makedirs("/tmp/ir3", exist_ok=True)
with open("/tmp/ir3/hivm.mlir", "w") as f: f.write(hivm_text)
with open("/tmp/ir3/hivm_ops.json", "w") as f: json.dump(hivm_ops, f, indent=2)
print("SAVED")
print(hivm_text)
