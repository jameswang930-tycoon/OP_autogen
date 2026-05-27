# Precision Gaps

Cases where emulator precision is insufficient or degraded.

## Channel dimension scaling

ResNet34 layer4 (256->512 channels) shows max_rel_err = 1.71e-03, significantly
higher than layer1 (64->64, 9.07e-05). Root cause: conv2d_resnet uses tiled
reduction across channels, accumulating floating point error.

- Observed: 2026-05-27, ResNet34 layer4 test
- Threshold: current rtol=1e-2, atol=1e-3 — passes but margin is thin
- Impact: could fail with real pretrained weights where channel correlations differ

### TODO test cases

- [ ] conv2d_resnet with increasing channel counts (64, 128, 256, 512, 1024) and fixed spatial
- [ ] Compare error with orthogonal vs random weights to isolate channel effect
- [ ] Test with fp16 dtype (emulator currently maps to float32, masking fp16 issues)

## fp16 not exercised

Emulator maps tl.float16 to numpy float32 internally. This means all fp16 precision
issues (overflow, underflow, reduced mantissa) are completely hidden.

- Status: NOT covered at all
- Priority: P1 — real Triton on NPU commonly uses fp16/bf16
- Blocker: numpy has no native fp16 compute (only storage)

### TODO test cases

- [ ] Add fp16 overflow detection: flag when values exceed fp16 range (±65504)
- [ ] Add fp16 precision warning: when computed values differ significantly
      between fp32 and fp16 simulation
