# Missing Test Coverage

Test scenarios not yet covered by integration tests.

## Real pretrained weights

All current tests use random weights (np.random.randn * 0.01). Real ImageNet
pretrained weights have different statistical properties:
- Non-zero mean channels
- Correlated filters
- Larger value ranges in later layers

ONNX models with pretrained weights are available in models/:
- models/resnet18-v1-7.onnx (ImageNet pretrained)
- models/resnet34-v1-7.onnx (ImageNet pretrained)

- [ ] Extract weights from ONNX model and run emulator comparison
- [ ] Compare error profile vs random weights

## Batch size > 1

All tests use batch_size=1. Multi-batch exercises:
- Larger grid sizes (more programs launched)
- Potential for cross-batch interference if offsets are computed incorrectly

- [ ] Add batch_size=4 and batch_size=8 variants to existing tests

## Non-standard spatial sizes

Tests use 8x8 or 14x14. Real input is 224x224 but too slow for CPU serial.
Should test intermediate sizes and non-power-of-2 sizes:
- 7x7 (common conv output size)
- 13x13 (odd dimension)
- 56x56 (first layer output in real ResNet)

- [ ] Add spatial size sweep test for conv2d_resnet

## fc (Gemm) layer

ResNet ends with a fully-connected layer (Gemm in ONNX). This maps to matmul + add
in the emulator. Not tested in any integration test.

- [ ] Add fc layer test: [B, 512] -> [B, 1000] via matmul + bias

## Graph / sparse operators

GCN SpMM (gcn_spmm/) and GCN integration test (gcn/) added 2026-05-27.
Tested with random graphs (N=4..100, F=4..32). All pass.

Not yet covered:
- [ ] Real graph datasets (Cora: N=2708, F=1433, E=10556)
- [ ] Atomic add in parallel scenario (SpMM kernel uses serial execution, no race detection)
- [ ] Variable-degree nodes (some nodes with degree 0 or very high degree)
- [ ] GAT attention weights (different edge weight per feature channel)
- [ ] Mini-batch / subgraph sampling
