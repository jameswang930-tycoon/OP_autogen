"""Fused RMSNorm + Residual — LLaMA/Mistral 真实组件，triton 2.3.1 兼容。

residual = x + residual
rms = sqrt(mean(residual^2) + eps)
out = residual / rms * weight

优化空间:
  Tier 1 Algorithm:  Online one-pass vs two-pass
  Tier 2 Fusion:     3个操作融合: add + rms_norm + weight
  Tier 3 Tiling:     BLOCK_SIZE (256,512,1024,2048), num_warps (2,4,8)
  Tier 4 Memory:     3路 load (x,residual,weight) → 可合并传输
  Tier 5 Compute:    fp32 vs fp16 精度取舍, rsqrt 替换 1/sqrt
  Tier 6 910B3 Arch:  grid 分配, L2 驻留
"""

import triton
import triton.language as tl


@triton.jit
def rms_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
):
    """Fused RMSNorm + Residual"""
    pid = tl.program_id(0)
    row_start = pid * HIDDEN_DIM
    offs = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Stage 1: Load x, residual, weight (3路 DMA → MTE2)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE),
                mask=tl.arange(0, BLOCK_SIZE) < HIDDEN_DIM, other=0.0).to(tl.float32)

    # Stage 2: Residual add (VecUnit) — 可融合点
    combined = x + r

    # Stage 3: RMSNorm (VecUnit + reduction)
    ms = tl.sum(combined * combined, axis=0) / HIDDEN_DIM
    rms = tl.math.rsqrt(ms + 1e-6)

    # Stage 4: Normalize + weight (VecUnit)
    out = combined * rms * w

    # Stage 5: Store (MTE3)
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)
