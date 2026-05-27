# Error Accumulation

Deep network composition causes numerical error to accumulate across layers.
Observed during ResNet18/34 integration tests.

## ResNet18 (2 blocks per layer, total ~8 blocks)

Max relative error stays within 1e-4 across all tests.

## ResNet34 (layer config [3,4,6,3], total ~16 blocks)

| Test | Blocks | max_abs | max_rel |
|------|--------|---------|---------|
| layer1 (64->64) | 3 | 4.77e-07 | 9.07e-05 |
| layer2 (64->128) | 4 | 3.58e-07 | 1.24e-04 |
| layer3 (128->256) | 6 | 1.79e-06 | 4.37e-04 |
| layer4 (256->512) | 3 | 2.62e-06 | 1.71e-03 |
| full chain | 16+ | 2.38e-07 | 8.23e-06 |
| 16 blocks stress | 16 | 9.54e-07 | 4.69e-05 |

### Key findings

- Single-layer rel_err grows with channel count (layer4: 1.71e-03)
- Full chain error is unexpectedly LOW due to random weight cancellation
- 16-block stress test passes rtol=1e-2

### TODO test cases

- [ ] ResNet50 (Bottleneck, ~50 layers) — deeper accumulation
- [ ] Real ImageNet weights — random weights mask real accumulation
- [ ] Quantify error growth rate: is it O(sqrt(N)) or O(N)?
