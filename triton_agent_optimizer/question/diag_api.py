"""系统诊断 triton 2.3.1 全部二维操作支持情况"""
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
        print(f"  ✅ {name} ({len(ttir)} chars)")
        return True
    except Exception as e:
        msg = str(e)[:100].replace("\n", " ")
        print(f"  ❌ {name}: {msg}")
        return False

print("=== 1. 基础操作 ===")

@triton.jit
def t_arange(x, B: tl.constexpr):
    a = tl.arange(0, B)
    tl.store(x + a, a)
test("tl.arange", t_arange, {0: "*fp32"}, {"B": 64})

@triton.jit
def t_zeros(x, B: tl.constexpr):
    a = tl.zeros((B, B), dtype=tl.float32)
    tl.store(x + tl.arange(0, B), a)
test("tl.zeros 2D", t_zeros, {0: "*fp32"}, {"B": 64})

@triton.jit
def t_dot(x, B: tl.constexpr):
    a = tl.zeros((B, B), dtype=tl.float32)
    b = tl.zeros((B, B), dtype=tl.float32)
    c = tl.dot(a, b, tl.zeros((B, B), dtype=tl.float32))
    tl.store(x + tl.arange(0, B), c)
test("tl.dot", t_dot, {0: "*fp32"}, {"B": 64})

print("\n=== 2. reshape/broadcast ===")

@triton.jit
def t_view(x, B: tl.constexpr):
    a = tl.arange(0, B)
    b = tl.view(a, (B, 1))
    tl.store(x + a, b)
test("tl.view", t_view, {0: "*fp32"}, {"B": 64})

@triton.jit
def t_reshape(x, B: tl.constexpr):
    a = tl.arange(0, B)
    b = tl.reshape(a, (B, 1))
    tl.store(x + a, b)
test("tl.reshape", t_reshape, {0: "*fp32"}, {"B": 64})

@triton.jit
def t_broadcast(x, B: tl.constexpr):
    a = tl.arange(0, B)
    b = tl.broadcast_to(a, (B, B))
    tl.store(x + tl.arange(0, B), b)
test("tl.broadcast_to", t_broadcast, {0: "*fp32"}, {"B": 64})

@triton.jit
def t_expand(x, B: tl.constexpr):
    a = tl.arange(0, B)
    b = tl.expand_dims(a, 1)
    tl.store(x + a, b)
test("tl.expand_dims", t_expand, {0: "*fp32"}, {"B": 64})

print("\n=== 3. 2D指针 + stride ===")

@triton.jit
def t_ptr2d(x, S: tl.constexpr):
    a = tl.arange(0, 64)
    b = tl.view(a, (64, 1))
    p = x + b * S
    tl.store(p, a)
test("view + stride", t_ptr2d, {0: "*fp32"}, {"S": 64})

@triton.jit
def t_full2d(x, S: tl.constexpr, B: tl.constexpr):
    offs_m = tl.arange(0, B)
    offs_k = tl.arange(0, B)
    rm = tl.view(offs_m, (B, 1))
    rk = tl.view(offs_k, (1, B))
    p = x + rm * S + rk
    a = tl.load(p)
    tl.store(p, a)
test("full 2D load/store", t_full2d, {0: "*fp32"}, {"S": 64, "B": 64})

print("\n=== 4. tl.load 2D mask ===")

@triton.jit
def t_load2d(x, N, B: tl.constexpr):
    offs_m = tl.arange(0, B)
    offs_k = tl.arange(0, B)
    rm = tl.view(offs_m, (B, 1))
    rk = tl.view(offs_k, (1, B))
    a = tl.load(x + rm * 64 + rk, mask=rk < N, other=0.0)
    tl.store(x + rm * 64 + rk, a)
test("load 2D with mask", t_load2d, {0: "*fp32", 1: "i32"}, {"B": 64})

print("\nDone.")
