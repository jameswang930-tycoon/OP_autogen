"""Test triton 3.4.0 compilation API"""
import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv
_drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp
_comp.CompiledKernel = MagicMock()

import triton, triton.language as tl
from triton.compiler import compile as triton_compile, ASTSource
from types import SimpleNamespace
import importlib.util, sys

sys.path.insert(0, "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer")

# Load softmax kernel
spec = importlib.util.spec_from_file_location("k", "input/softmax/triton_kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

# Method 1: Try triton_compile directly
print("=== Method 1: triton.compiler.compile ===")
try:
    sig = {fn.arg_names[0]: "*fp32", fn.arg_names[2]: "i32"}
    # compile returns a CompiledKernel
    result = triton_compile(fn, signature=sig)
    print(f"type: {type(result).__name__}")
    if hasattr(result, 'ttir'):
        print(f"ttir: {len(str(result.ttir))} chars")
    elif hasattr(result, 'module'):
        print(f"module: {len(str(result.module))} chars")
    else:
        print(f"attrs: {[x for x in dir(result) if not x.startswith('_')]}")
except Exception as e:
    print(f"FAIL: {e}")

# Method 2: Check ASTSource new API
print("\n=== Method 2: ASTSource with codegen_fns ===")
try:
    sig = {fn.arg_names[0]: "*fp32", fn.arg_names[2]: "i32"}
    src = ASTSource(fn=fn, signature=sig)
    # Try to get what make_ir needs
    import triton.compiler.code_generator as cg
    print(f"code_generator attrs: {[x for x in dir(cg) if 'codegen' in x.lower() or 'make' in x.lower()]}")
except Exception as e:
    print(f"FAIL: {e}")

# Method 3: Try the kernel directly via JIT
print("\n=== Method 3: Check if we can dump TTIR via env var ===")
print("TRITON_DUMP_IR or similar env vars might exist in 3.4")
