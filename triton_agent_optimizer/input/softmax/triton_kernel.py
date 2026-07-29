"""Online Softmax — classic optimization target with multiple pipeline stages."""
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """Online softmax: max→sub→exp→sum→div"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Stage 1: Load
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))

    # Stage 2: Find max (for numerical stability)
    x_max = tl.max(x, axis=0)

    # Stage 3: Subtract max + exp
    x_safe = x - x_max
    x_exp = tl.math.exp(x_safe)

    # Stage 4: Sum
    x_sum = tl.sum(x_exp, axis=0)

    # Stage 5: Normalize
    x_softmax = x_exp / x_sum

    # Stage 6: Store
    tl.store(out_ptr + offs, x_softmax, mask=mask)


@triton.jit
def fused_gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """GELU activation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x_sq = x * x
    x_cu = x_sq * x
    inner = 0.7978845608 * (x + 0.044715 * x_cu)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(out_ptr + offs, gelu, mask=mask)
