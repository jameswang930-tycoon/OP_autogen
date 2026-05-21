"""
ResNet18 Integration Test: BasicBlock + Stem composition
=========================================================
Composes conv2d_resnet + batchnorm2d + relu + maxpool2d + add
to validate the emulator framework with real ResNet18 data flow.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import verify

# Import individual operator modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test.conv2d_resnet import emulate_conv2d_resnet, reference_conv2d_resnet
from test.batchnorm2d import emulate_batchnorm2d, reference_batchnorm2d
from test.maxpool2d import emulate_maxpool2d, reference_maxpool2d
from test.adaptive_avgpool2d import emulate_adaptive_avgpool2d, reference_adaptive_avgpool2d
from test.relu import emulate_relu, reference_relu


def emulate_basic_block(x, conv1_w, bn1_params, conv2_w, bn2_params,
                        stride=1, downsample=None):
    """
    Emulate ResNet18 BasicBlock:
      identity = x
      out = relu(bn1(conv1(x)))
      out = bn2(conv2(out))
      if downsample: identity = downsample(x)
      out = relu(out + identity)
    """
    identity = x

    # conv1 + bn1 + relu
    out = emulate_conv2d_resnet(x, conv1_w, stride_h=stride, stride_w=stride,
                                 pad_h=1, pad_w=1)
    out = emulate_batchnorm2d(out, bn1_params[0], bn1_params[1],
                               bn1_params[2], bn1_params[3])
    out = emulate_relu(out)

    # conv2 + bn2
    out = emulate_conv2d_resnet(out, conv2_w, stride_h=1, stride_w=1,
                                 pad_h=1, pad_w=1)
    out = emulate_batchnorm2d(out, bn2_params[0], bn2_params[1],
                               bn2_params[2], bn2_params[3])

    # Downsample identity if needed
    if downsample is not None:
        ds_conv_w, ds_bn_params, ds_stride = downsample
        identity = emulate_conv2d_resnet(x, ds_conv_w,
                                          stride_h=ds_stride, stride_w=ds_stride,
                                          pad_h=0, pad_w=0)
        identity = emulate_batchnorm2d(identity, ds_bn_params[0], ds_bn_params[1],
                                        ds_bn_params[2], ds_bn_params[3])

    # Residual add + relu
    out = out + identity
    out = emulate_relu(out)
    return out


def reference_basic_block(x, conv1_w, bn1_params, conv2_w, bn2_params,
                           stride=1, downsample=None):
    """PyTorch reference for BasicBlock."""
    import torch
    import torch.nn.functional as F

    x_t = torch.tensor(x, dtype=torch.float32)
    identity = x_t

    # conv1 + bn1 + relu
    out = F.conv2d(x_t, torch.tensor(conv1_w, dtype=torch.float32),
                    stride=stride, padding=1)
    out = F.batch_norm(out,
                        torch.tensor(bn1_params[0], dtype=torch.float32),
                        torch.tensor(bn1_params[1], dtype=torch.float32),
                        torch.tensor(bn1_params[2], dtype=torch.float32),
                        torch.tensor(bn1_params[3], dtype=torch.float32),
                        training=False)
    out = F.relu(out)

    # conv2 + bn2
    out = F.conv2d(out, torch.tensor(conv2_w, dtype=torch.float32),
                    stride=1, padding=1)
    out = F.batch_norm(out,
                        torch.tensor(bn2_params[0], dtype=torch.float32),
                        torch.tensor(bn2_params[1], dtype=torch.float32),
                        torch.tensor(bn2_params[2], dtype=torch.float32),
                        torch.tensor(bn2_params[3], dtype=torch.float32),
                        training=False)

    # Downsample
    if downsample is not None:
        ds_conv_w, ds_bn_params, ds_stride = downsample
        identity = F.conv2d(x_t, torch.tensor(ds_conv_w, dtype=torch.float32),
                             stride=ds_stride, padding=0)
        identity = F.batch_norm(identity,
                                 torch.tensor(ds_bn_params[0], dtype=torch.float32),
                                 torch.tensor(ds_bn_params[1], dtype=torch.float32),
                                 torch.tensor(ds_bn_params[2], dtype=torch.float32),
                                 torch.tensor(ds_bn_params[3], dtype=torch.float32),
                                 training=False)

    out = F.relu(out + identity)
    return out.numpy()


def emulate_stem(x, conv1_w, bn1_params):
    """ResNet18 stem: conv1(7x7,s2,p3) -> bn1 -> relu -> maxpool(3x3,s2,p1)"""
    out = emulate_conv2d_resnet(x, conv1_w, stride_h=2, stride_w=2,
                                 pad_h=3, pad_w=3)
    out = emulate_batchnorm2d(out, bn1_params[0], bn1_params[1],
                               bn1_params[2], bn1_params[3])
    out = emulate_relu(out)
    out = emulate_maxpool2d(out, kH=3, kW=3, stride_h=2, stride_w=2,
                             pad_h=1, pad_w=1)
    return out


def reference_stem(x, conv1_w, bn1_params):
    """PyTorch reference for ResNet18 stem."""
    import torch
    import torch.nn.functional as F

    x_t = torch.tensor(x, dtype=torch.float32)
    out = F.conv2d(x_t, torch.tensor(conv1_w, dtype=torch.float32),
                    stride=2, padding=3)
    out = F.batch_norm(out,
                        torch.tensor(bn1_params[0], dtype=torch.float32),
                        torch.tensor(bn1_params[1], dtype=torch.float32),
                        torch.tensor(bn1_params[2], dtype=torch.float32),
                        torch.tensor(bn1_params[3], dtype=torch.float32),
                        training=False)
    out = F.relu(out)
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
    return out.numpy()


def _make_bn_params(C):
    """Create random BN parameters for C channels."""
    return (
        np.random.randn(C).astype(np.float32) * 0.3,   # running_mean
        np.abs(np.random.randn(C).astype(np.float32)) + 0.5,  # running_var
        np.random.randn(C).astype(np.float32),           # gamma
        np.random.randn(C).astype(np.float32) * 0.1,    # beta
    )


def test():
    print("=" * 70)
    print(" ResNet18 Integration Test")
    print("=" * 70)

    np.random.seed(42)

    # ----------------------------------------------------------
    # Test 1: Stem (conv1 + bn1 + relu + maxpool)
    # ----------------------------------------------------------
    print("\n--- Test 1: ResNet18 Stem (3->64, 7x7, s2, p3 + bn + relu + maxpool) ---")
    x_stem = np.random.randn(1, 3, 14, 14).astype(np.float32)
    conv1_w = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
    bn1_params = _make_bn_params(64)

    out_stem = emulate_stem(x_stem, conv1_w, bn1_params)
    ref_stem = reference_stem(x_stem, conv1_w, bn1_params)
    result1 = verify(out_stem, ref_stem, "resnet18_stem", rtol=1e-3, atol=1e-3)

    # ----------------------------------------------------------
    # Test 2: BasicBlock without downsample (layer1.0)
    # ----------------------------------------------------------
    print("\n--- Test 2: BasicBlock no-downsample (64->64, 3x3 s1 p1) ---")
    x_b1 = np.random.randn(1, 64, 8, 8).astype(np.float32)
    conv1_b1 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    conv2_b1 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    bn1_b1 = _make_bn_params(64)
    bn2_b1 = _make_bn_params(64)

    out_b1 = emulate_basic_block(x_b1, conv1_b1, bn1_b1, conv2_b1, bn2_b1, stride=1)
    ref_b1 = reference_basic_block(x_b1, conv1_b1, bn1_b1, conv2_b1, bn2_b1, stride=1)
    result2 = verify(out_b1, ref_b1, "resnet18_block_no_down", rtol=1e-3, atol=1e-3)

    # ----------------------------------------------------------
    # Test 3: BasicBlock with downsample (layer2.0)
    # ----------------------------------------------------------
    print("\n--- Test 3: BasicBlock with downsample (64->128, 3x3 s2 p1 + 1x1 proj) ---")
    x_b2 = np.random.randn(1, 64, 8, 8).astype(np.float32)
    conv1_b2 = np.random.randn(128, 64, 3, 3).astype(np.float32) * 0.01
    conv2_b2 = np.random.randn(128, 128, 3, 3).astype(np.float32) * 0.01
    bn1_b2 = _make_bn_params(128)
    bn2_b2 = _make_bn_params(128)

    # 1x1 projection shortcut
    ds_conv_w = np.random.randn(128, 64, 1, 1).astype(np.float32) * 0.01
    ds_bn_params = _make_bn_params(128)
    downsample = (ds_conv_w, ds_bn_params, 2)

    out_b2 = emulate_basic_block(x_b2, conv1_b2, bn1_b2, conv2_b2, bn2_b2,
                                  stride=2, downsample=downsample)
    ref_b2 = reference_basic_block(x_b2, conv1_b2, bn1_b2, conv2_b2, bn2_b2,
                                    stride=2, downsample=downsample)
    result3 = verify(out_b2, ref_b2, "resnet18_block_with_down", rtol=1e-3, atol=1e-3)

    # ----------------------------------------------------------
    # Test 4: Two consecutive BasicBlocks (layer1.0 + layer1.1)
    # ----------------------------------------------------------
    print("\n--- Test 4: Two consecutive BasicBlocks ---")
    x_two = np.random.randn(1, 64, 8, 8).astype(np.float32)

    # Block 1
    c1w_1 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    c2w_1 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    bn1_1 = _make_bn_params(64)
    bn2_1 = _make_bn_params(64)

    # Block 2
    c1w_2 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    c2w_2 = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    bn1_2 = _make_bn_params(64)
    bn2_2 = _make_bn_params(64)

    out_two = emulate_basic_block(x_two, c1w_1, bn1_1, c2w_1, bn2_1, stride=1)
    out_two = emulate_basic_block(out_two, c1w_2, bn1_2, c2w_2, bn2_2, stride=1)

    ref_two = reference_basic_block(x_two, c1w_1, bn1_1, c2w_1, bn2_1, stride=1)
    ref_two = reference_basic_block(ref_two, c1w_2, bn1_2, c2w_2, bn2_2, stride=1)

    result4 = verify(out_two, ref_two, "resnet18_two_blocks", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 5: Full chain (stem + block + adaptive_avgpool)
    # ----------------------------------------------------------
    print("\n--- Test 5: Full chain (stem -> block -> avgpool) ---")
    x_full = np.random.randn(1, 3, 14, 14).astype(np.float32)
    conv_stem_w = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
    bn_stem_params = _make_bn_params(64)

    # Stem
    out_full = emulate_stem(x_full, conv_stem_w, bn_stem_params)

    # Block (no downsample)
    c1w_f = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    c2w_f = np.random.randn(64, 64, 3, 3).astype(np.float32) * 0.01
    bn1_f = _make_bn_params(64)
    bn2_f = _make_bn_params(64)
    out_full = emulate_basic_block(out_full, c1w_f, bn1_f, c2w_f, bn2_f, stride=1)

    # Adaptive avg pool
    out_full = emulate_adaptive_avgpool2d(out_full)

    # Reference
    ref_full = reference_stem(x_full, conv_stem_w, bn_stem_params)
    ref_full = reference_basic_block(ref_full, c1w_f, bn1_f, c2w_f, bn2_f, stride=1)
    ref_full_t = np.array(ref_full)
    import torch
    ref_full = torch.nn.functional.adaptive_avg_pool2d(
        torch.tensor(ref_full_t, dtype=torch.float32), (1, 1)
    ).numpy()

    result5 = verify(out_full, ref_full, "resnet18_full_chain", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    all_passed = all(r["passed"] for r in [result1, result2, result3, result4, result5])
    print("\n" + "=" * 70)
    if all_passed:
        print(" ResNet18 Integration: ALL TESTS PASSED")
    else:
        print(" ResNet18 Integration: SOME TESTS FAILED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
