"""
ResNet34 Integration Test: BasicBlock composition with layer config [3,4,6,3]
=============================================================================
Extends the ResNet18 pattern to deeper architecture.
Input:  [B, 3, 224, 224]  (from shapes_registry)
Output: [B, 1000]
Layers: stem -> layer1(3 blocks) -> layer2(4 blocks) -> layer3(6 blocks) -> layer4(3 blocks) -> avgpool -> fc
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import verify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test.conv2d_resnet import emulate_conv2d_resnet, reference_conv2d_resnet
from test.batchnorm2d import emulate_batchnorm2d, reference_batchnorm2d
from test.maxpool2d import emulate_maxpool2d, reference_maxpool2d
from test.adaptive_avgpool2d import emulate_adaptive_avgpool2d, reference_adaptive_avgpool2d
from test.relu import emulate_relu, reference_relu


def _make_bn_params(C):
    return (
        np.random.randn(C).astype(np.float32) * 0.3,
        np.abs(np.random.randn(C).astype(np.float32)) + 0.5,
        np.random.randn(C).astype(np.float32),
        np.random.randn(C).astype(np.float32) * 0.1,
    )


def _make_block_weights(in_ch, out_ch):
    return {
        'conv1_w': np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32) * 0.01,
        'bn1': _make_bn_params(out_ch),
        'conv2_w': np.random.randn(out_ch, out_ch, 3, 3).astype(np.float32) * 0.01,
        'bn2': _make_bn_params(out_ch),
    }


def _make_downsample_weights(in_ch, out_ch, stride):
    return {
        'conv_w': np.random.randn(out_ch, in_ch, 1, 1).astype(np.float32) * 0.01,
        'bn': _make_bn_params(out_ch),
        'stride': stride,
    }


def emulate_basic_block(x, conv1_w, bn1_params, conv2_w, bn2_params,
                        stride=1, downsample=None):
    identity = x
    out = emulate_conv2d_resnet(x, conv1_w, stride_h=stride, stride_w=stride,
                                 pad_h=1, pad_w=1)
    out = emulate_batchnorm2d(out, bn1_params[0], bn1_params[1],
                               bn1_params[2], bn1_params[3])
    out = emulate_relu(out)
    out = emulate_conv2d_resnet(out, conv2_w, stride_h=1, stride_w=1,
                                 pad_h=1, pad_w=1)
    out = emulate_batchnorm2d(out, bn2_params[0], bn2_params[1],
                               bn2_params[2], bn2_params[3])
    if downsample is not None:
        ds_conv_w, ds_bn_params, ds_stride = downsample
        identity = emulate_conv2d_resnet(x, ds_conv_w,
                                          stride_h=ds_stride, stride_w=ds_stride,
                                          pad_h=0, pad_w=0)
        identity = emulate_batchnorm2d(identity, ds_bn_params[0], ds_bn_params[1],
                                        ds_bn_params[2], ds_bn_params[3])
    out = out + identity
    out = emulate_relu(out)
    return out


def reference_basic_block(x, conv1_w, bn1_params, conv2_w, bn2_params,
                           stride=1, downsample=None):
    import torch
    import torch.nn.functional as F
    x_t = torch.tensor(x, dtype=torch.float32)
    identity = x_t
    out = F.conv2d(x_t, torch.tensor(conv1_w, dtype=torch.float32),
                    stride=stride, padding=1)
    out = F.batch_norm(out,
                        torch.tensor(bn1_params[0], dtype=torch.float32),
                        torch.tensor(bn1_params[1], dtype=torch.float32),
                        torch.tensor(bn1_params[2], dtype=torch.float32),
                        torch.tensor(bn1_params[3], dtype=torch.float32),
                        training=False)
    out = F.relu(out)
    out = F.conv2d(out, torch.tensor(conv2_w, dtype=torch.float32),
                    stride=1, padding=1)
    out = F.batch_norm(out,
                        torch.tensor(bn2_params[0], dtype=torch.float32),
                        torch.tensor(bn2_params[1], dtype=torch.float32),
                        torch.tensor(bn2_params[2], dtype=torch.float32),
                        torch.tensor(bn2_params[3], dtype=torch.float32),
                        training=False)
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
    out = emulate_conv2d_resnet(x, conv1_w, stride_h=2, stride_w=2,
                                 pad_h=3, pad_w=3)
    out = emulate_batchnorm2d(out, bn1_params[0], bn1_params[1],
                               bn1_params[2], bn1_params[3])
    out = emulate_relu(out)
    out = emulate_maxpool2d(out, kH=3, kW=3, stride_h=2, stride_w=2,
                             pad_h=1, pad_w=1)
    return out


def reference_stem(x, conv1_w, bn1_params):
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


def _emulate_layer(x, blocks, in_channels, out_channels, stride, block_weights_list, ds_weights=None):
    """Compose N BasicBlocks. First block may have stride>1 + downsample."""
    out = x
    for i in range(blocks):
        bw = block_weights_list[i]
        s = stride if i == 0 else 1
        ds = None
        if i == 0 and ds_weights is not None:
            ds = (ds_weights['conv_w'], ds_weights['bn'], ds_weights['stride'])
        out = emulate_basic_block(out, bw['conv1_w'], bw['bn1'],
                                   bw['conv2_w'], bw['bn2'],
                                   stride=s, downsample=ds)
    return out


def _reference_layer(x, blocks, in_channels, out_channels, stride, block_weights_list, ds_weights=None):
    import torch
    import torch.nn.functional as F
    out = x
    for i in range(blocks):
        bw = block_weights_list[i]
        s = stride if i == 0 else 1
        ds = None
        if i == 0 and ds_weights is not None:
            ds = (ds_weights['conv_w'], ds_weights['bn'], ds_weights['stride'])
        out = reference_basic_block(out, bw['conv1_w'], bw['bn1'],
                                     bw['conv2_w'], bw['bn2'],
                                     stride=s, downsample=ds)
    return out


def test():
    print("=" * 70)
    print(" ResNet34 Integration Test")
    print(" Shape: [B,3,224,224] -> [B,1000]  (from shapes_registry)")
    print(" Layer config: [3, 4, 6, 3]")
    print("=" * 70)

    np.random.seed(42)

    # ----------------------------------------------------------
    # Test 1: Stem (same as ResNet18)
    # ----------------------------------------------------------
    print("\n--- Test 1: Stem (3->64, 7x7, s2, p3 + bn + relu + maxpool) ---")
    x_stem = np.random.randn(1, 3, 14, 14).astype(np.float32)
    conv1_w = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
    bn1_params = _make_bn_params(64)

    out_stem = emulate_stem(x_stem, conv1_w, bn1_params)
    ref_stem = reference_stem(x_stem, conv1_w, bn1_params)
    result1 = verify(out_stem, ref_stem, "resnet34_stem", rtol=1e-3, atol=1e-3)

    # ----------------------------------------------------------
    # Test 2: layer1 — 3 BasicBlocks, 64->64, no downsample
    # ----------------------------------------------------------
    print("\n--- Test 2: layer1 (3 blocks, 64->64, stride=1) ---")
    x_l1 = np.random.randn(1, 64, 8, 8).astype(np.float32)
    l1_blocks = [_make_block_weights(64, 64) for _ in range(3)]
    out_l1 = _emulate_layer(x_l1, 3, 64, 64, 1, l1_blocks)
    ref_l1 = _reference_layer(x_l1, 3, 64, 64, 1, l1_blocks)
    result2 = verify(out_l1, ref_l1, "resnet34_layer1", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 3: layer2 — 4 BasicBlocks, 64->128, first has downsample
    # ----------------------------------------------------------
    print("\n--- Test 3: layer2 (4 blocks, 64->128, stride=2) ---")
    x_l2 = np.random.randn(1, 64, 8, 8).astype(np.float32)
    l2_blocks = [_make_block_weights(128 if i == 0 else 128, 64 if i == 0 else 128) for i in range(4)]
    # Fix: first block conv1 takes 64ch in, rest take 128ch in
    l2_blocks[0] = _make_block_weights(64, 128)
    for i in range(1, 4):
        l2_blocks[i] = _make_block_weights(128, 128)
    l2_ds = _make_downsample_weights(64, 128, 2)
    out_l2 = _emulate_layer(x_l2, 4, 64, 128, 2, l2_blocks, l2_ds)
    ref_l2 = _reference_layer(x_l2, 4, 64, 128, 2, l2_blocks, l2_ds)
    result3 = verify(out_l2, ref_l2, "resnet34_layer2", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 4: layer3 — 6 BasicBlocks, 128->256, first has downsample
    # ----------------------------------------------------------
    print("\n--- Test 4: layer3 (6 blocks, 128->256, stride=2) ---")
    x_l3 = np.random.randn(1, 128, 8, 8).astype(np.float32)
    l3_blocks = [_make_block_weights(128, 256)] + [_make_block_weights(256, 256) for _ in range(5)]
    l3_ds = _make_downsample_weights(128, 256, 2)
    out_l3 = _emulate_layer(x_l3, 6, 128, 256, 2, l3_blocks, l3_ds)
    ref_l3 = _reference_layer(x_l3, 6, 128, 256, 2, l3_blocks, l3_ds)
    result4 = verify(out_l3, ref_l3, "resnet34_layer3", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 5: layer4 — 3 BasicBlocks, 256->512, first has downsample
    # ----------------------------------------------------------
    print("\n--- Test 5: layer4 (3 blocks, 256->512, stride=2) ---")
    x_l4 = np.random.randn(1, 256, 8, 8).astype(np.float32)
    l4_blocks = [_make_block_weights(256, 512)] + [_make_block_weights(512, 512) for _ in range(2)]
    l4_ds = _make_downsample_weights(256, 512, 2)
    out_l4 = _emulate_layer(x_l4, 3, 256, 512, 2, l4_blocks, l4_ds)
    ref_l4 = _reference_layer(x_l4, 3, 256, 512, 2, l4_blocks, l4_ds)
    result5 = verify(out_l4, ref_l4, "resnet34_layer4", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 6: Full chain — stem + all 4 layers + avgpool
    # ----------------------------------------------------------
    print("\n--- Test 6: Full chain (stem -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool) ---")
    x_full = np.random.randn(1, 3, 14, 14).astype(np.float32)
    conv_stem_w = np.random.randn(64, 3, 7, 7).astype(np.float32) * 0.01
    bn_stem_params = _make_bn_params(64)

    # Stem
    out_full = emulate_stem(x_full, conv_stem_w, bn_stem_params)
    ref_full = reference_stem(x_full, conv_stem_w, bn_stem_params)

    # layer1: 3 blocks, 64->64
    fl1_blocks = [_make_block_weights(64, 64) for _ in range(3)]
    out_full = _emulate_layer(out_full, 3, 64, 64, 1, fl1_blocks)
    ref_full = _reference_layer(ref_full, 3, 64, 64, 1, fl1_blocks)

    # layer2: 4 blocks, 64->128
    fl2_blocks = [_make_block_weights(64, 128)] + [_make_block_weights(128, 128) for _ in range(3)]
    fl2_ds = _make_downsample_weights(64, 128, 2)
    out_full = _emulate_layer(out_full, 4, 64, 128, 2, fl2_blocks, fl2_ds)
    ref_full = _reference_layer(ref_full, 4, 64, 128, 2, fl2_blocks, fl2_ds)

    # layer3: 6 blocks, 128->256
    fl3_blocks = [_make_block_weights(128, 256)] + [_make_block_weights(256, 256) for _ in range(5)]
    fl3_ds = _make_downsample_weights(128, 256, 2)
    out_full = _emulate_layer(out_full, 6, 128, 256, 2, fl3_blocks, fl3_ds)
    ref_full = _reference_layer(ref_full, 6, 128, 256, 2, fl3_blocks, fl3_ds)

    # layer4: 3 blocks, 256->512
    fl4_blocks = [_make_block_weights(256, 512)] + [_make_block_weights(512, 512) for _ in range(2)]
    fl4_ds = _make_downsample_weights(256, 512, 2)
    out_full = _emulate_layer(out_full, 3, 256, 512, 2, fl4_blocks, fl4_ds)
    ref_full = _reference_layer(ref_full, 3, 256, 512, 2, fl4_blocks, fl4_ds)

    # Adaptive avg pool
    out_full = emulate_adaptive_avgpool2d(out_full)
    import torch
    ref_full = torch.nn.functional.adaptive_avg_pool2d(
        torch.tensor(np.array(ref_full), dtype=torch.float32), (1, 1)
    ).numpy()

    result6 = verify(out_full, ref_full, "resnet34_full_chain", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 7: Error accumulation — 16 consecutive blocks
    # ----------------------------------------------------------
    print("\n--- Test 7: Error accumulation (16 consecutive blocks, 64->64) ---")
    x_err = np.random.randn(1, 64, 8, 8).astype(np.float32)
    err_blocks = [_make_block_weights(64, 64) for _ in range(16)]
    out_err = x_err
    ref_err = x_err
    for bw in err_blocks:
        out_err = emulate_basic_block(out_err, bw['conv1_w'], bw['bn1'],
                                       bw['conv2_w'], bw['bn2'], stride=1)
        ref_err = reference_basic_block(ref_err, bw['conv1_w'], bw['bn1'],
                                         bw['conv2_w'], bw['bn2'], stride=1)
    result7 = verify(out_err, ref_err, "resnet34_16_blocks_accum", rtol=1e-2, atol=1e-2)

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    all_passed = all(r["passed"] for r in [result1, result2, result3, result4, result5, result6, result7])
    print("\n" + "=" * 70)
    if all_passed:
        print(" ResNet34 Integration: ALL TESTS PASSED")
    else:
        print(" ResNet34 Integration: SOME TESTS FAILED")
    print("=" * 70)
    print()

    return {
        "stem": result1,
        "layer1": result2,
        "layer2": result3,
        "layer3": result4,
        "layer4": result5,
        "full_chain": result6,
        "error_accumulation": result7,
    }


if __name__ == "__main__":
    test()
