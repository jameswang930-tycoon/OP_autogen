"""
GCN Integration Test: GCNConv = SpMM + Linear
================================================================
Composes gcn_spmm + matmul to emulate a full GCNConv layer.
Based on PyG GCN: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py

GCNConv math: X' = D^(-1/2) * A_hat * D^(-1/2) * X * W
Decomposed as:
  1. Linear: h = X @ W           → matmul
  2. SpMM:  out = norm_adj @ h   → gcn_spmm
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import verify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test.gcn_spmm import emulate_gcn_spmm, reference_gcn_spmm
from test.matmul import emulate_matmul, reference_matmul


def emulate_gcn_conv(x, weight, edge_src, edge_dst, edge_weight, N, F_in, F_out):
    """Emulate GCNConv: norm_adj @ (x @ W)"""
    h = emulate_matmul(x, weight)
    # Step 2: SpMM
    out = emulate_gcn_spmm(h, edge_src, edge_dst, edge_weight, N, F_out)
    return out


def reference_gcn_conv(x, weight, edge_src, edge_dst, edge_weight, N, F_in, F_out):
    """Numpy reference for GCNConv."""
    h = reference_matmul(x, weight)
    return reference_gcn_spmm(h, edge_src, edge_dst, edge_weight, N, F_out)


def _make_sym_norm(edge_src, edge_dst, N):
    """Symmetric normalization: D^(-1/2) * A * D^(-1/2)."""
    deg = np.zeros(N, dtype=np.float64)
    for s, d in zip(edge_src, edge_dst):
        deg[d] += 1
    deg = np.maximum(deg, 1.0)
    norm = (1.0 / np.sqrt(deg[edge_src]) * 1.0 / np.sqrt(deg[edge_dst]))
    return norm.astype(np.float32)


def _make_graph(N, avg_degree=4, seed=42):
    rng = np.random.RandomState(seed)
    E = N * avg_degree
    src = rng.randint(0, N, size=E)
    dst = rng.randint(0, N, size=E)
    src = np.concatenate([src, np.arange(N)])
    dst = np.concatenate([dst, np.arange(N)])
    w = _make_sym_norm(src, dst, N)
    return src.astype(np.int64), dst.astype(np.int64), w


def test():
    print("=" * 60)
    print(" GCN Integration Test (GCNConv = SpMM + matmul)")
    print("=" * 60)

    # ----------------------------------------------------------
    # Test 1: Single GCNConv layer
    # ----------------------------------------------------------
    print("\n--- Test 1: Single GCNConv (N=10, F_in=8, F_out=4) ---")
    N, F_in, F_out = 10, 8, 4
    src, dst, w = _make_graph(N, avg_degree=3)
    x = np.random.randn(N, F_in).astype(np.float32) * 0.1
    weight = np.random.randn(F_in, F_out).astype(np.float32) * 0.01

    out = emulate_gcn_conv(x, weight, src, dst, w, N, F_in, F_out)
    ref = reference_gcn_conv(x, weight, src, dst, w, N, F_in, F_out)
    verify(out, ref, "gcn_conv_single", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 2: Two GCNConv layers (like the PyG model)
    # ----------------------------------------------------------
    print("\n--- Test 2: Two GCNConv layers (N=20, F: 16->8->4) ---")
    N = 20
    src, dst, w = _make_graph(N, avg_degree=4)
    x = np.random.randn(N, 16).astype(np.float32) * 0.1
    w1 = np.random.randn(16, 8).astype(np.float32) * 0.01
    w2 = np.random.randn(8, 4).astype(np.float32) * 0.01

    # Layer 1 + relu
    h1 = emulate_gcn_conv(x, w1, src, dst, w, N, 16, 8)
    h1 = np.maximum(h1, 0)  # relu

    h1_ref = reference_gcn_conv(x, w1, src, dst, w, N, 16, 8)
    h1_ref = np.maximum(h1_ref, 0)

    # Layer 2
    out = emulate_gcn_conv(h1, w2, src, dst, w, N, 8, 4)
    ref = reference_gcn_conv(h1_ref, w2, src, dst, w, N, 8, 4)
    verify(out, ref, "gcn_two_layers", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 3: Cora-like dimensions (small N)
    # ----------------------------------------------------------
    print("\n--- Test 3: Cora-like (N=50, F_in=1433->16) ---")
    N, F_in, F_out = 50, 32, 16  # reduced F_in for speed
    src, dst, w = _make_graph(N, avg_degree=4, seed=77)
    x = np.random.randn(N, F_in).astype(np.float32) * 0.01
    weight = np.random.randn(F_in, F_out).astype(np.float32) * 0.01

    out = emulate_gcn_conv(x, weight, src, dst, w, N, F_in, F_out)
    ref = reference_gcn_conv(x, weight, src, dst, w, N, F_in, F_out)
    verify(out, ref, "gcn_cora_like", rtol=1e-2, atol=1e-3)

    print("\n" + "=" * 60)
    print(" GCN Integration: ALL TESTS PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    test()
