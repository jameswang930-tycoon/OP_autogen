"""
GCN SpMM Emulator: Sparse Matrix - Dense Matrix Multiplication
================================================================
Kernel: out[i, :] = sum_{j in neighbors(i)} norm_weight * node_feat[j, :]
Grid:   1D, each program handles one target node
Input:  CSR-format edge list + node features [N, F]
Output: [N, F]

This is the core operator of Graph Convolutional Networks (GCN).
GCNConv = normalize_adj + SpMM + linear_transform
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import tl, launch_kernel_1d, verify, EmulatorError


def gcn_spmm_kernel(node_feat_ptr, edge_src_ptr, edge_weight_ptr,
                    row_start_ptr, row_end_ptr, output_ptr,
                    N, F, E, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)

    # Get edge range for this target node
    idx = pid + tl.arange(0, 1)
    rs = int(tl.load(row_start_ptr, idx))
    re = int(tl.load(row_end_ptr, idx))

    # Feature offsets
    f_off = tl.arange(0, BLOCK_F)
    f_mask = f_off < F

    # Accumulator
    acc = tl.zeros((BLOCK_F,), dtype=tl.float32)

    # Iterate over incoming edges
    for _ in range(rs, re):
        src = int(tl.load(edge_src_ptr, _ + tl.arange(0, 1)))
        w = tl.load(edge_weight_ptr, _ + tl.arange(0, 1))
        feats = tl.load(node_feat_ptr, src * F + f_off, mask=f_mask)
        acc = acc + w * feats

    # Store result
    tl.store(output_ptr, pid * F + f_off, acc, mask=f_mask)


def emulate_gcn_spmm(node_feat, edge_src, edge_dst, edge_weight, N, F):
    """
    SpMM: output[i, :] = sum_{j in neighbors(i)} edge_weight * node_feat[j, :]

    Args:
        node_feat: [N, F] node features
        edge_src: [E] source node indices
        edge_dst: [E] destination node indices
        edge_weight: [E] normalized edge weights
        N, F: dimensions
    Returns:
        output: [N, F]
    """
    E = len(edge_src)
    node_feat_flat = node_feat.ravel().astype(np.float32)
    out_flat = np.zeros(N * F, dtype=np.float32)

    # Build CSR: sort edges by destination, compute row_start/end
    sorted_idx = np.argsort(edge_dst)
    sorted_src = edge_src[sorted_idx].astype(np.int64)
    sorted_weight = edge_weight[sorted_idx].astype(np.float32)
    sorted_dst = edge_dst[sorted_idx]

    row_start = np.zeros(N, dtype=np.int64)
    row_end = np.zeros(N, dtype=np.int64)
    pos = 0
    for i in range(N):
        row_start[i] = pos
        while pos < E and sorted_dst[pos] == i:
            pos += 1
        row_end[i] = pos

    BLOCK_F = ((F + 15) // 16) * 16  # round up to multiple of 16

    launch_kernel_1d(
        gcn_spmm_kernel,
        node_feat_flat, sorted_src, sorted_weight,
        row_start, row_end, out_flat,
        N, F, E, BLOCK_F,
        grid_size=N
    )
    return out_flat.reshape(N, F)


def reference_gcn_spmm(node_feat, edge_src, edge_dst, edge_weight, N, F):
    output = np.zeros((N, F), dtype=np.float32)
    for e in range(len(edge_src)):
        s, d, w = edge_src[e], edge_dst[e], edge_weight[e]
        output[d] += w * node_feat[s]
    return output


def _make_graph(N, avg_degree=4, seed=42):
    """Create a random graph for testing."""
    rng = np.random.RandomState(seed)
    E = N * avg_degree
    src = rng.randint(0, N, size=E)
    dst = rng.randint(0, N, size=E)
    # Add self-loops
    src = np.concatenate([src, np.arange(N)])
    dst = np.concatenate([dst, np.arange(N)])
    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
    deg = np.zeros(N)
    for s, d in zip(src, dst):
        deg[d] += 1
    deg = np.maximum(deg, 1)
    norm = 1.0 / np.sqrt(deg[src]) / np.sqrt(deg[dst])
    return src, dst, norm.astype(np.float32)


def test():
    print("=" * 60)
    print(" GCN SpMM Emulator Test")
    print("=" * 60)

    # ----------------------------------------------------------
    # Test 1: Basic 4x4 graph, F=4
    # ----------------------------------------------------------
    print("\n--- Test 1: Basic (N=4, F=4, dense graph) ---")
    N, F = 4, 4
    edge_src = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    edge_dst = np.array([1, 0, 3, 2, 0, 1, 2, 3, 2, 3, 0, 1], dtype=np.int64)
    edge_weight = np.ones(len(edge_src), dtype=np.float32)
    node_feat = np.random.randn(N, F).astype(np.float32)

    out = emulate_gcn_spmm(node_feat, edge_src, edge_dst, edge_weight, N, F)
    ref = reference_gcn_spmm(node_feat, edge_src, edge_dst, edge_weight, N, F)
    verify(out, ref, "gcn_spmm_basic", rtol=1e-3, atol=1e-5)

    # ----------------------------------------------------------
    # Test 2: Random graph with symmetric normalization
    # ----------------------------------------------------------
    print("\n--- Test 2: Random graph (N=10, F=8, avg_degree=4) ---")
    N, F = 10, 8
    src, dst, w = _make_graph(N, avg_degree=4)
    feat = np.random.randn(N, F).astype(np.float32)

    out = emulate_gcn_spmm(feat, src, dst, w, N, F)
    ref = reference_gcn_spmm(feat, src, dst, w, N, F)
    verify(out, ref, "gcn_spmm_random", rtol=1e-3, atol=1e-5)

    # ----------------------------------------------------------
    # Test 3: Single node (edge case)
    # ----------------------------------------------------------
    print("\n--- Test 3: Single node with self-loop ---")
    N, F = 1, 4
    src = np.array([0], dtype=np.int64)
    dst = np.array([0], dtype=np.int64)
    w = np.array([1.0], dtype=np.float32)
    feat = np.random.randn(N, F).astype(np.float32)

    out = emulate_gcn_spmm(feat, src, dst, w, N, F)
    ref = reference_gcn_spmm(feat, src, dst, w, N, F)
    verify(out, ref, "gcn_spmm_single_node", rtol=1e-3, atol=1e-5)

    # ----------------------------------------------------------
    # Test 4: Non-power-of-2 feature dim
    # ----------------------------------------------------------
    print("\n--- Test 4: Non-power-of-2 features (N=6, F=7) ---")
    N, F = 6, 7
    src, dst, w = _make_graph(N, avg_degree=3, seed=123)
    feat = np.random.randn(N, F).astype(np.float32)

    out = emulate_gcn_spmm(feat, src, dst, w, N, F)
    ref = reference_gcn_spmm(feat, src, dst, w, N, F)
    verify(out, ref, "gcn_spmm_odd_features", rtol=1e-3, atol=1e-5)

    # ----------------------------------------------------------
    # Test 5: Cora-scale (N=100, F=16, for timing)
    # ----------------------------------------------------------
    print("\n--- Test 5: Medium scale (N=100, F=16, avg_degree=4) ---")
    N, F = 100, 16
    src, dst, w = _make_graph(N, avg_degree=4, seed=99)
    feat = np.random.randn(N, F).astype(np.float32)

    out = emulate_gcn_spmm(feat, src, dst, w, N, F)
    ref = reference_gcn_spmm(feat, src, dst, w, N, F)
    verify(out, ref, "gcn_spmm_medium", rtol=1e-3, atol=1e-5)

    print("\n" + "=" * 60)
    print(" GCN SpMM: ALL TESTS PASSED")
    print("=" * 60)
    print()


if __name__ == "__main__":
    test()
