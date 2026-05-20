"""
Attention-ReLU Emulator: Scaled Dot-Product Attention with ReLU activation
============================================================================
Replaces softmax in standard attention with ReLU:
  scores = Q @ K^T / sqrt(d_k)
  attn = ReLU(scores)   # max(0, scores) — no row-wise normalization
  out = attn @ V

Key differences from standard (softmax) attention:
  - No normalization: output rows are NOT weighted averages
  - Sparse: negative Q·K correlations contribute zero weight
  - All-zero rows possible: when Q has negative dot products with all K
  - Scale-sensitive: output magnitude scales with input magnitude

Grid:   1D, each program computes one output row
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import tl, xarray, launch_kernel_1d, verify, EmulatorError


# ============================================================
# Triton-style Kernel
# ============================================================

def attention_relu_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    seq_len, d_k, d_v,
    stride_qs, stride_qd,
    stride_ks, stride_kd,
    stride_vs, stride_vd,
    stride_os, stride_od,
    scale,
    BLOCK_D: tl.constexpr,
):
    """
    Attention-ReLU kernel.

    Each program (pid = row_idx) computes one output row:
      out[row, :] = sum_j( ReLU( Q[row,:] . K[j,:] / scale ) * V[j,:] )

    BLOCK_D 统一补齐 Q/K 和 V 的维度差异, masked 部分填 0, 不影响结果。
    """
    pid = tl.program_id(0)
    row_idx = pid

    # 加载一行 Q: shape [BLOCK_D], 超出 d_k 的部分被 mask 置 0
    q_offsets = row_idx * stride_qs + tl.arange(0, BLOCK_D) * stride_qd
    q_row = tl.load(q_ptr, q_offsets, mask=tl.arange(0, BLOCK_D) < d_k, other=0.0)

    # 累加器: [BLOCK_D], 超出 d_v 部分始终为 0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # 遍历所有 key/value 位置
    for j in range(seq_len):
        # 加载一个 key 向量: [BLOCK_D]
        k_offsets = j * stride_ks + tl.arange(0, BLOCK_D) * stride_kd
        k_vec = tl.load(k_ptr, k_offsets, mask=tl.arange(0, BLOCK_D) < d_k, other=0.0)

        # 点积: sum(q_row * k_vec) → scalar [1]
        score = tl.sum(q_row * k_vec, axis=0)
        score_scaled = score / scale

        # ReLU — 替代 softmax! 负相关性直接置零
        attn_weight = tl.maximum(score_scaled, 0.0)

        # 加载一个 value 向量: [BLOCK_D]
        v_offsets = j * stride_vs + tl.arange(0, BLOCK_D) * stride_vd
        v_vec = tl.load(v_ptr, v_offsets, mask=tl.arange(0, BLOCK_D) < d_v, other=0.0)

        # 加权累加: acc += attn_weight * v_vec
        acc = acc + attn_weight * v_vec

    # 写出结果行: [d_v]
    out_offsets = row_idx * stride_os + tl.arange(0, BLOCK_D) * stride_od
    tl.store(out_ptr, out_offsets, acc, mask=tl.arange(0, BLOCK_D) < d_v)


# ============================================================
# Emulator 封装
# ============================================================

def emulate_attention_relu(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                           scale=None, BLOCK_D=None) -> np.ndarray:
    """
    在 CPU 上 emulate Attention-ReLU。

    参数:
      Q: query  [seq_len, d_k] 或 [batch, seq_len, d_k]
      K: key    [seq_len, d_k]
      V: value  [seq_len, d_v]
      scale:    缩放因子, 默认 sqrt(d_k)
      BLOCK_D:  特征维 block size, 默认自动取 >= max(d_k, d_v) 的 2 的幂

    返回:
      out: [seq_len, d_v] 或 [batch, seq_len, d_v]

    错误检查:
      - Q/K/V shape 不匹配
      - 空维度
      - 包含 NaN/Inf
    """
    # --- 输入校验 ---
    if Q.ndim == 2:
        Q = Q.reshape(1, *Q.shape)
    if Q.ndim != 3:
        raise EmulatorError("attention_relu_kernel",
            f"Q must be 2D [seq_len, d_k] or 3D [batch, seq_len, d_k], got shape {Q.shape}")

    batch, seq_len, d_k = Q.shape

    if K.ndim != 2 or K.shape != (seq_len, d_k):
        raise EmulatorError("attention_relu_kernel",
            f"K shape {K.shape} must be ({seq_len}, {d_k}), expected to match Q[0].shape={Q[0].shape}")
    if V.ndim != 2 or V.shape[0] != seq_len:
        raise EmulatorError("attention_relu_kernel",
            f"V shape {V.shape} must be ({seq_len}, d_v)")
    if seq_len == 0:
        raise EmulatorError("attention_relu_kernel", "seq_len is 0")
    if d_k == 0:
        raise EmulatorError("attention_relu_kernel", "d_k is 0")

    d_v = V.shape[1]
    if d_v == 0:
        raise EmulatorError("attention_relu_kernel", "d_v is 0")

    for name, arr in [("Q", Q), ("K", K), ("V", V)]:
        if np.any(np.isnan(arr)):
            raise EmulatorError("attention_relu_kernel",
                f"{name} contains {int(np.sum(np.isnan(arr)))} NaN values")
        if np.any(np.isinf(arr)):
            raise EmulatorError("attention_relu_kernel",
                f"{name} contains {int(np.sum(np.isinf(arr)))} Inf values")

    # --- 参数设置 ---
    if scale is None:
        scale = float(np.sqrt(d_k))

    if BLOCK_D is None:
        max_d = max(d_k, d_v)
        BLOCK_D = 1
        while BLOCK_D < max_d:
            BLOCK_D *= 2

    # 展开 stride (row-major)
    stride_qs, stride_qd = d_k, 1
    stride_ks, stride_kd = d_k, 1
    stride_vs, stride_vd = d_v, 1
    stride_os, stride_od = d_v, 1

    K_flat = K.ravel().astype(np.float32)
    V_flat = V.ravel().astype(np.float32)

    outs = []
    for b in range(batch):
        Q_b = Q[b].ravel().astype(np.float32)
        out_flat = np.zeros(seq_len * d_v, dtype=np.float32)

        launch_kernel_1d(
            attention_relu_kernel,
            Q_b, K_flat, V_flat, out_flat,
            seq_len, d_k, d_v,
            stride_qs, stride_qd,
            stride_ks, stride_kd,
            stride_vs, stride_vd,
            stride_os, stride_od,
            scale,
            BLOCK_D,
            grid_size=seq_len,
        )
        outs.append(out_flat.reshape(seq_len, d_v))

    result = np.stack(outs, axis=0).astype(np.float32)
    if batch == 1:
        result = result[0]
    return result


# ============================================================
# Reference (纯 NumPy)
# ============================================================

def reference_attention_relu(Q, K, V, scale=None):
    """
    纯 NumPy 实现 Attention-ReLU, 作为正确性基准。

    scores = Q @ K^T / scale
    attn = ReLU(scores)
    out = attn @ V
    """
    if Q.ndim == 2:
        Q = Q.reshape(1, *Q.shape)
    batch, seq_len, d_k = Q.shape

    if scale is None:
        scale = np.sqrt(d_k)

    Q = Q.astype(np.float64)
    K = K.astype(np.float64)
    V = V.astype(np.float64)

    outs = []
    for b in range(batch):
        scores = Q[b] @ K.T / scale          # [seq_len, seq_len]
        attn = np.maximum(scores, 0.0)       # ReLU — 替代 softmax
        out = attn @ V                        # [seq_len, d_v]
        outs.append(out)

    result = np.stack(outs, axis=0).astype(np.float32)
    if batch == 1:
        result = result[0]
    return result


# ============================================================
# Self-Test
# ============================================================

def test():
    print("=" * 60)
    print(" Attention-ReLU Emulator Test")
    print("=" * 60)

    # --- Test 1: 基本功能 (d_k == d_v) ---
    seq_len, d_k, d_v = 16, 32, 32
    Q = np.random.randn(seq_len, d_k).astype(np.float32)
    K = np.random.randn(seq_len, d_k).astype(np.float32)
    V = np.random.randn(seq_len, d_v).astype(np.float32)

    out = emulate_attention_relu(Q, K, V)
    ref = reference_attention_relu(Q, K, V)
    verify(out, ref, "attn_relu_basic", rtol=1e-3, atol=1e-4)

    # --- Test 2: d_k != d_v ---
    d_k2, d_v2 = 64, 32
    Q2 = np.random.randn(seq_len, d_k2).astype(np.float32)
    K2 = np.random.randn(seq_len, d_k2).astype(np.float32)
    V2 = np.random.randn(seq_len, d_v2).astype(np.float32)

    out2 = emulate_attention_relu(Q2, K2, V2)
    ref2 = reference_attention_relu(Q2, K2, V2)
    verify(out2, ref2, "attn_relu_unequal_dims", rtol=1e-3, atol=1e-4)

    # --- Test 3: 批处理 (3D Q) ---
    batch = 4
    Q3 = np.random.randn(batch, seq_len, d_k).astype(np.float32)
    K3 = np.random.randn(seq_len, d_k).astype(np.float32)
    V3 = np.random.randn(seq_len, d_v).astype(np.float32)

    out3 = emulate_attention_relu(Q3, K3, V3)
    ref3 = reference_attention_relu(Q3, K3, V3)
    verify(out3, ref3, "attn_relu_batched", rtol=1e-3, atol=1e-4)

    # --- Test 4: ReLU 稀疏性 ---
    # Q 全负 * K 全正 → 所有点积 < 0 → ReLU 清零 → 输出全零
    Q4 = -np.ones((4, 8), dtype=np.float32)
    K4 = np.ones((4, 8), dtype=np.float32)
    V4 = np.ones((4, 8), dtype=np.float32)

    out4 = emulate_attention_relu(Q4, K4, V4)
    if np.allclose(out4, 0.0, atol=1e-5):
        print("  [PASS] attn_relu_sparsity: All-negative scores correctly produce zero output")
    else:
        print(f"  [FAIL] attn_relu_sparsity: Expected all-zero, got max|out|={np.max(np.abs(out4)):.4e}")

    # --- Test 5: ReLU vs Softmax 差异验证 ---
    Q5 = np.random.randn(4, 16).astype(np.float32)
    K5 = np.random.randn(4, 16).astype(np.float32)
    V5 = np.random.randn(4, 16).astype(np.float32)
    scale5 = np.sqrt(16)

    out_relu = emulate_attention_relu(Q5, K5, V5)

    scores5 = Q5 @ K5.T / scale5
    attn_softmax = np.exp(scores5 - np.max(scores5, axis=-1, keepdims=True))
    attn_softmax = attn_softmax / np.sum(attn_softmax, axis=-1, keepdims=True)
    out_softmax = attn_softmax @ V5

    diff = np.max(np.abs(out_relu - out_softmax))
    print(f"  ReLU vs Softmax max delta: {diff:.4e}  "
         f"({'different ✓' if diff > 1e-3 else 'unexpectedly close ✗'})")

    # --- Test 6: 自定义 scale ---
    Q6 = np.random.randn(8, 16).astype(np.float32)
    K6 = np.random.randn(8, 16).astype(np.float32)
    V6 = np.random.randn(8, 16).astype(np.float32)

    out6_custom = emulate_attention_relu(Q6, K6, V6, scale=2.0)
    ref6 = reference_attention_relu(Q6, K6, V6, scale=2.0)
    verify(out6_custom, ref6, "attn_relu_custom_scale", rtol=1e-3, atol=1e-4)

    out6_default = emulate_attention_relu(Q6, K6, V6)
    diff_scale = np.max(np.abs(out6_default - out6_custom))
    print(f"  Scale sensitivity (default vs 2.0, max delta={diff_scale:.4e}): "
          f"{'✓ different' if diff_scale > 1e-5 else '✗ identical'}")

    # --- Test 7: 长序列 ---
    Q_large = np.random.randn(64, 32).astype(np.float32)
    K_large = np.random.randn(64, 32).astype(np.float32)
    V_large = np.random.randn(64, 32).astype(np.float32)

    out_large = emulate_attention_relu(Q_large, K_large, V_large)
    ref_large = reference_attention_relu(Q_large, K_large, V_large)
    verify(out_large, ref_large, "attn_relu_large_seq", rtol=1e-3, atol=1e-4)

    # --- Test 8: shape 不匹配应报错 ---
    try:
        emulate_attention_relu(
            np.zeros((4, 8), dtype=np.float32),
            np.zeros((4, 16), dtype=np.float32),  # K 的 d_k 不匹配
            np.zeros((4, 8), dtype=np.float32),
        )
        print("  [FAIL] Should have raised EmulatorError for K shape mismatch")
    except EmulatorError:
        print("  [PASS] Correctly caught K shape mismatch")

    # --- Test 9: NaN 检测 ---
    try:
        emulate_attention_relu(
            np.array([[np.nan, 0.0]], dtype=np.float32),
            np.array([[0.0, 0.0]], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
        )
        print("  [FAIL] Should have raised EmulatorError for NaN input")
    except EmulatorError:
        print("  [PASS] Correctly caught NaN input")

    print()


if __name__ == "__main__":
    test()
