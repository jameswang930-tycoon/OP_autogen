"""Test triton 3.4.0 driver mock + compile"""
import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from unittest.mock import MagicMock
import triton.runtime.driver as _drv

# Mock the active driver BEFORE anything else touches it
mock_driver = MagicMock()
mock_driver.get_current_target.return_value = ("cuda", 90)
_drv.active = mock_driver

from triton.compiler import ASTSource, compile as triton_compile
import importlib.util, sys

sys.path.insert(0, "/mnt/d/vscodeproject/huawei_work/OP_autogen/OP_autogen_hjkc/triton_agent_optimizer")

# Load softmax kernel
spec = importlib.util.spec_from_file_location(
    "k", "input/softmax/triton_kernel.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn = mod.softmax_kernel

# Test 1: compile with triton 3.4 API
print("=== Test 1: compile ===")
src = ASTSource(fn=fn, signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "N": "i32"},
                constexprs={"BLOCK_SIZE": 256})
try:
    result = triton_compile(src, options={"num_warps": 4, "num_stages": 1})
    print(f"compile OK, type={type(result).__name__}")
    if hasattr(result, "asm"):
        keys = list(result.asm.keys())
        print(f"asm keys: {keys}")
        if "ttir" in result.asm:
            t = str(result.asm["ttir"])
            print(f"TTIR: {len(t)} chars")
        if "ttgir" in result.asm:
            print(f"TTGIR: {len(str(result.asm['ttgir']))} chars")
except Exception as e:
    print(f"compile FAIL: {e}")

# Test 2: [:,None] matmul pattern
print("\n=== Test 2: matmul [:,None] ===")
import triton, triton.language as tl

@triton.jit
def matmul(a, b, c, M, N, K, sa, sb, sc, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(axis=0)
    gn = (N + BN - 1) // BN
    pm, pn = pid // gn, pid % gn
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    ap = a + (rm[:, None] * sa + rk[None, :])
    bp = b + (rk[:, None] * sb + rn[None, :])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a1 = tl.load(ap, mask=rk[None, :] < (K - k), other=0.0)
        b1 = tl.load(bp, mask=rk[:, None] < (K - k), other=0.0)
        acc = tl.dot(a1, b1, acc)
        ap += BK; bp += BK
    cp = c + (rm[:, None] * sc + rn[None, :])
    tl.store(cp, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

try:
    src2 = ASTSource(fn=matmul, signature={a: "*fp32" for a in matmul.arg_names[:3]})
    result2 = triton_compile(src2, options={"num_warps": 4, "num_stages": 1})
    t = str(result2.asm.get("ttir", ""))
    print(f"Matmul TTIR: {len(t)} chars")
except Exception as e:
    print(f"Matmul FAIL: {str(e)[:200]}")
