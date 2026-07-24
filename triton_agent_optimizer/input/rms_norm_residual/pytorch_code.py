#!/usr/bin/env python3
"""
PyTorch Reference: RMSNorm + Residual Add
对标 Llama-2-7B Transformer Block 中的 Norm 层

数学: output = rms_norm(input + residual) * gamma
      rms = sqrt(mean(x^2) + eps)
      output = (x / rms) * gamma
"""

import torch
import torch.nn as nn


class RMSNormResidual(nn.Module):
    """RMSNorm with residual connection. Standard Llama architecture."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Step 1: Residual add
        y = x + residual

        # Step 2: RMSNorm (two-pass — naive implementation)
        rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True) + self.eps)
        y_normed = (y.float() / rms).to(x.dtype)

        # Step 3: Scale by gamma
        return y_normed * self.gamma


def reference_rms_norm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Functional reference implementation."""
    y = x + residual
    rms = torch.sqrt(torch.mean(y.float() ** 2, dim=-1, keepdim=True) + eps)
    return (y.float() / rms * gamma.float()).to(x.dtype)


# ── Performance measurement ──────────────────────────────────────────

def benchmark_pytorch(batch=1, seq_len=1024, hidden_dim=4096, warmup=30, repeat=200):
    """Measure PyTorch baseline latency."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device=device)
    residual = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16, device=device)
    gamma = torch.ones(hidden_dim, dtype=torch.float16, device=device)
    model = RMSNormResidual(hidden_dim).to(device)

    # Warmup
    for _ in range(warmup):
        model(x, residual)

    if device == "cuda":
        torch.cuda.synchronize()

    import time
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(x, residual)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / repeat * 1e3

    print(f"PyTorch RMSNorm+Residual: {elapsed:.3f} ms "
          f"(shape={batch}x{seq_len}x{hidden_dim}, fp16)")
    return elapsed


if __name__ == "__main__":
    benchmark_pytorch()
