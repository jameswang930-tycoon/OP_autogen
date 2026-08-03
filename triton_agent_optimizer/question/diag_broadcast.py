"""诊断 triton 2.3.1 中 [:, None] 和 stride 参数的兼容性"""
import os, sys
os.environ["TRITON_ALWAYS_COMPILE"] = "1"
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
        print(f"  {name}: OK ({len(ttir)} chars)")
    except Exception as e:
        print(f"  {name}: FAIL - {str(e)[:120]}")

# Test 1: [:,None] alone
@triton.jit
def t1(x_ptr, N, B: tl.constexpr):
    offs = tl.arange(0, B)
    a = offs[:, None]
    tl.store(x_ptr + offs, a)

test("[:,None] alone", t1, {0: "*fp32", 1: "i32"}, {"B": 64})

# Test 2: [:,None] × constexpr stride
@triton.jit
def t2(x_ptr, N, B: tl.constexpr, S: tl.constexpr):
    offs = tl.arange(0, B)
    a = offs[:, None] * S
    tl.store(x_ptr + offs, a)

test("[:,None] × constexpr", t2, {0: "*fp32", 1: "i32"}, {"B": 64, "S": 64})

# Test 3: [:,None] × runtime stride
@triton.jit
def t3(x_ptr, N, stride, B: tl.constexpr):
    offs = tl.arange(0, B)
    a = offs[:, None] * stride
    tl.store(x_ptr + offs, a)

test("[:,None] × runtime i32", t3, {0: "*fp32", 1: "i32", 2: "i32"}, {"B": 64})

# Test 4: pointer + [:,None] (分开加)
@triton.jit
def t4(x_ptr, N, B: tl.constexpr):
    offs = tl.arange(0, B)
    a = offs[:, None]
    p = x_ptr + a
    tl.store(p, a)

test("pointer + [:,None]", t4, {0: "*fp32", 1: "i32"}, {"B": 64})

# Test 5: pointer + ([,:,None] * constexpr) — 标准 matmul 模式
@triton.jit
def t5(x_ptr, N, B: tl.constexpr, S: tl.constexpr):
    offs = tl.arange(0, B)
    p = x_ptr + (offs[:, None] * S)
    tl.store(p, offs)

test("pointer + ([:,None] × constexpr)", t5, {0: "*fp32", 1: "i32"}, {"B": 64, "S": 64})

print("\nDone.")
