import triton
import triton.language as tl

@triton.jit
def fused_add_mul_kernel(x_ptr, y_ptr, w_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """z = (x + y) * w — fused element-wise with 3 loads, 1 add, 1 mul, 1 store"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    w = tl.load(w_ptr + offs, mask=mask)

    tmp = x + y
    result = tmp * w

    tl.store(out_ptr + offs, result, mask=mask)
