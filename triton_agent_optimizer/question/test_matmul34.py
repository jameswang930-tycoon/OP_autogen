"""Test matmul with triton 3.4.0"""
import os; os.environ["TRITON_ALWAYS_COMPILE"] = "1"
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, compile as triton_compile
import triton, triton.language as tl

@triton.jit
def matmul_kernel(a, b, c, M, N, K, sa, sb, sc,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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

# triton 3.4: ALL non-constexpr params need signature
sig = {}
for name in matmul_kernel.arg_names:
    if name in ("a", "b", "c"):
        sig[name] = "*fp32"
    else:
        sig[name] = "i32"  # M,N,K and strides are all int32
# triton 3.4: constexpr params must be passed explicitly
consts = {"BM": 64, "BN": 64, "BK": 32}
src = ASTSource(fn=matmul_kernel, signature=sig, constexprs=consts)
target = GPUTarget("cuda", 90, 32)
result = triton_compile(src, target=target, options={"num_warps": 4, "num_stages": 1})
ttir = str(result.asm["ttir"])
print(f"Matmul TTIR: {len(ttir)} chars")
print(f"Has tt.dot: {'tt.dot' in ttir}")
print(f"Has tt.load: {'tt.load' in ttir}")
print(f"Has tt.store: {'tt.store' in ttir}")
print()
print(ttir[:800])
