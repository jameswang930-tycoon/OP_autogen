"""Test triton 3.4.0 - pass target to compile"""
import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv
mock_driver = MagicMock()
mock_driver.get_current_target.return_value = ("cuda", 90)
_drv.active = mock_driver

from triton.compiler import ASTSource, compile as triton_compile
import importlib.util, sys

sys.path.insert(0, "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer")

spec = importlib.util.spec_from_file_location("k", "input/softmax/triton_kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

print("=== Test: compile with target param ===")
src = ASTSource(fn=fn, signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
                constexprs={"BLOCK_SIZE": 256})
try:
    result = triton_compile(src, target=("cuda", 90), options={"num_warps": 4, "num_stages": 1})
    print(f"compile OK! type={type(result).__name__}")
    if hasattr(result, "asm"):
        t = str(result.asm.get("ttir", ""))
        print(f"TTIR: {len(t)} chars")
except Exception as e:
    print(f"FAIL: {e}")

# If 3.4 doesn't work, revert to 2.3.1
print("\n=== Triton 3.4 conclusion ===")
print("API changed significantly. compile() needs proper GPUTarget.")
print("Downgrade back to 2.3.1 for now?")
