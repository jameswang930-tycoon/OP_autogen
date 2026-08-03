"""验证 triton 2.3.1 中用硬编码数字 + tl.view 做 matmul 关键操作"""
import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv; _drv._obj = MagicMock(get_current_target=lambda: ("cuda", 90))
import triton.compiler.compiler as _comp; _comp.CompiledKernel = MagicMock()
import triton, triton.language as tl
from triton.compiler import ASTSource
from types import SimpleNamespace

def test(name, fn, sig, consts):
    try:
        src = ASTSource(fn=fn, signature=sig, constants=consts)
        ttir = str(src.make_ir(SimpleNamespace(num_warps=4, num_stages=1, debug=False)))
        print(f"  ✅ {name} ({len(ttir)}c)")
    except Exception as e:
        print(f"  ❌ {name}: {str(e)[:80]}")

@triton.jit
def t_view(x, S: tl.constexpr):
    a = tl.arange(0, 64)
    b = tl.view(a, (64, 1)) * S
    tl.store(x + a, a)
test("tl.view + constexpr mul", t_view, {0: "*fp32"}, {"S": 64})

@triton.jit
def t_zeros_hard(x):
    a = tl.zeros((64, 64), dtype=tl.float32)
    tl.store(x + tl.arange(0, 64), a)
test("tl.zeros(64,64) hardcoded", t_zeros_hard, {0: "*fp32"}, {})

@triton.jit
def t_dot_hard(x):
    a = tl.zeros((64, 64), dtype=tl.float32)
    b = tl.zeros((64, 64), dtype=tl.float32)
    c = tl.zeros((64, 64), dtype=tl.float32)
    d = tl.dot(a, b, c)
    tl.store(x + tl.arange(0, 64), d)
test("tl.dot hardcoded 64", t_dot_hard, {0: "*fp32"}, {})

@triton.jit
def t_matmul_mini(x, S: tl.constexpr):
    rm = tl.arange(0, 64)
    rk = tl.arange(0, 64)
    a_ptrs = x + tl.view(rm, (64, 1)) * S + tl.view(rk, (1, 64))
    a = tl.load(a_ptrs)
    b = tl.zeros((64, 64), dtype=tl.float32)
    c = tl.zeros((64, 64), dtype=tl.float32)
    d = tl.dot(a, b, c)
    tl.store(a_ptrs, d)
test("matmul pattern: view+dot", t_matmul_mini, {0: "*fp32"}, {"S": 64})

print("\nDone.")
