"""测试: tl.zeros(1D) + tl.view → 2D, 绕过 triton 2.3.1 的 tl.zeros(2D) 限制"""
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
        print(f"  ❌ {name}: {str(e)[:100]}")

# 核心技巧: zeros(1D) + view → 2D
@triton.jit
def t_zeros_2d_via_view(x):
    a = tl.zeros((64*64,), dtype=tl.float32)
    b = tl.view(a, (64, 64))
    tl.store(x + tl.arange(0, 64), b)
test("zeros(64*64) → view to (64,64)", t_zeros_2d_via_view, {0: "*fp32"}, {})

# tl.dot with this trick
@triton.jit
def t_dot_via_view(x):
    za = tl.zeros((64*64,), dtype=tl.float32)
    zb = tl.zeros((64*64,), dtype=tl.float32)
    zc = tl.zeros((64*64,), dtype=tl.float32)
    a = tl.view(za, (64, 64))
    b = tl.view(zb, (64, 64))
    c = tl.view(zc, (64, 64))
    d = tl.dot(a, b, c)
    tl.store(x + tl.arange(0, 64), d)
test("tl.dot via zeros+view", t_dot_via_view, {0: "*fp32"}, {})

# Mini matmul: 只用 arange + view + dot
@triton.jit
def t_mini_matmul(x, S: tl.constexpr):
    rm = tl.arange(0, 64)
    rk = tl.arange(0, 64)
    rn = tl.arange(0, 64)
    # ptr = base + row_offs[:,None] * S + col_offs[None,:]
    a_ptrs = x + tl.view(rm, (64, 1)) * S + tl.view(rk, (1, 64))
    a = tl.load(a_ptrs)
    # accum = zeros via view
    z = tl.zeros((64*64,), dtype=tl.float32)
    acc = tl.view(z, (64, 64))
    d = tl.dot(a, a, acc)
    tl.store(a_ptrs, d)
test("mini matmul: load+dot+store", t_mini_matmul, {0: "*fp32"}, {"S": 64})

print("\nDone.")
