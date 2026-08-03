import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv; _drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp; _comp.CompiledKernel = MagicMock()
import triton, triton.language as tl
from triton.compiler import ASTSource
from types import SimpleNamespace

def test(name, fn, sig, consts):
    try:
        ttir = str(ASTSource(fn=fn, signature=sig, constants=consts)
            .make_ir(SimpleNamespace(num_warps=4, num_stages=1, debug=False)))
        print(f"  ✅ {name} ({len(ttir)}c)")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:120]}")

@triton.jit
def z1(x): a=tl.zeros((64,),dtype=tl.float32); tl.store(x+tl.arange(0,64),a)
test("zeros(64,) hardcoded 1D", z1, {0: "*fp32"}, {})

@triton.jit
def z2(x): a=tl.zeros((64,64),dtype=tl.float32); tl.store(x+tl.arange(0,64),a)
test("zeros(64,64) hardcoded 2D", z2, {0: "*fp32"}, {})

@triton.jit
def z3(x, B: tl.constexpr): a=tl.zeros((B,),dtype=tl.float32); tl.store(x+tl.arange(0,B),a)
test("zeros((B,)) constexpr 1D", z3, {0: "*fp32"}, {"B": 64})

@triton.jit
def z4(x, B: tl.constexpr): a=tl.zeros((B,B),dtype=tl.float32); tl.store(x+tl.arange(0,B),a)
test("zeros((B,B)) constexpr 2D", z4, {0: "*fp32"}, {"B": 64})

print("\nDone.")
