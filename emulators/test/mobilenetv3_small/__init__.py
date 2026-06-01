"""
MobileNetV3-Small Integration Test
====================================
Composes: conv2d_resnet + conv2d_depthwise + batchnorm2d + relu
          + hardswish + hardsigmoid + mul + adaptive_avgpool2d
Fixed shape: [1, 3, 224, 224] -> [1, 1000]

Architecture (from PyTorch mobilenet_v3_small):
  features.0:  Conv(3,16,k=3,s=2,p=1)+BN+Hardswish             -> [1,16,112,112]
  features.1:  (no expand) DW(16,k=3,s=2,p=1)+BN+ReLU          -> [1,16,56,56]
               SE(16,8) -> PW(16->16)+BN
  features.2:  PW(16->72)+BN+ReLU -> DW(72,k=3,s=2,p=1)+BN+ReLU -> PW(72->24)+BN
  features.3:  PW(24->88)+BN+ReLU -> DW(88,k=3,s=1,p=1)+BN+ReLU -> PW(88->24)+BN +res
  features.4:  PW(24->96)+BN+HS -> DW(96,k=5,s=2,p=2)+BN+HS -> SE(96,24) -> PW(96->40)+BN
  features.5:  PW(40->240)+BN+HS -> DW(240,k=5,s=1,p=2)+BN+HS -> SE(240,64) -> PW(240->40)+BN +res
  features.6:  PW(40->240)+BN+HS -> DW(240,k=5,s=1,p=2)+BN+HS -> SE(240,64) -> PW(240->40)+BN +res
  features.7:  PW(40->120)+BN+HS -> DW(120,k=5,s=1,p=2)+BN+HS -> SE(120,32) -> PW(120->48)+BN
  features.8:  PW(48->144)+BN+HS -> DW(144,k=5,s=1,p=2)+BN+HS -> SE(144,40) -> PW(144->48)+BN +res
  features.9:  PW(48->288)+BN+HS -> DW(288,k=5,s=2,p=2)+BN+HS -> SE(288,72) -> PW(288->96)+BN
  features.10: PW(96->576)+BN+HS -> DW(576,k=5,s=1,p=2)+BN+HS -> SE(576,144) -> PW(576->96)+BN +res
  features.11: PW(96->576)+BN+HS -> DW(576,k=5,s=1,p=2)+BN+HS -> SE(576,144) -> PW(576->96)+BN +res
  features.12: PW(96->576)+BN+Hardswish
  avgpool -> Linear(576,1024)+HS -> Linear(1024,1000)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from common import verify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from test.conv2d_resnet import emulate_conv2d_resnet
from test.conv2d_depthwise import emulate_conv2d_depthwise
from test.batchnorm2d import emulate_batchnorm2d
from test.relu import emulate_relu
from test.hardswish import emulate_hardswish
from test.hardsigmoid import emulate_hardsigmoid
from test.mul import emulate_mul
from test.adaptive_avgpool2d import emulate_adaptive_avgpool2d


# ---- Architecture config ----
# (in_c, hidden_c, out_c, kernel, stride, pad, use_hs, se_reduce, has_residual)
BLOCKS = [
    (16,  16,  16,  3, 2, 1, False, 8,   False),  # features.1
    (16,  72,  24,  3, 2, 1, False, None, False),  # features.2
    (24,  88,  24,  3, 1, 1, False, None, True),   # features.3
    (24,  96,  40,  5, 2, 2, True,  24,  False),   # features.4
    (40,  240, 40,  5, 1, 2, True,  64,  True),    # features.5
    (40,  240, 40,  5, 1, 2, True,  64,  True),    # features.6
    (40,  120, 48,  5, 1, 2, True,  32,  False),   # features.7
    (48,  144, 48,  5, 1, 2, True,  40,  True),    # features.8
    (48,  288, 96,  5, 2, 2, True,  72,  False),   # features.9
    (96,  576, 96,  5, 1, 2, True,  144, True),    # features.10
    (96,  576, 96,  5, 1, 2, True,  144, True),    # features.11
]


# ---- Helpers ----

def _bn_params(C):
    return (
        np.random.randn(C).astype(np.float32) * 0.3,
        np.abs(np.random.randn(C).astype(np.float32)) + 0.5,
        np.random.randn(C).astype(np.float32),
        np.random.randn(C).astype(np.float32) * 0.1,
    )

def _bn(x, params):
    return emulate_batchnorm2d(x, params[0], params[1], params[2], params[3])

def _pw(x, w):
    return emulate_conv2d_resnet(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0)

def _dw(x, w, stride=1, pad=0):
    return emulate_conv2d_depthwise(x, w, stride_h=stride, stride_w=stride, pad_h=pad, pad_w=pad)

def _act(x, use_hs):
    return emulate_hardswish(x) if use_hs else emulate_relu(x)

def _se(x, fc1_w, fc1_b, fc2_w, fc2_b):
    pooled = emulate_adaptive_avgpool2d(x)
    squeezed = emulate_conv2d_resnet(pooled, fc1_w, fc1_b)
    squeezed = emulate_relu(squeezed)
    scale = emulate_conv2d_resnet(squeezed, fc2_w, fc2_b)
    scale = emulate_hardsigmoid(scale)
    scale = np.broadcast_to(scale, x.shape).copy()
    return emulate_mul(x, scale)


# ---- Weight generation ----

def make_weights():
    sc = 0.01
    w = {}

    # Stem: Conv(3, 16, k=3, s=2, p=1)
    w['stem_w'] = np.random.randn(16, 3, 3, 3).astype(np.float32) * sc
    w['stem_bn'] = _bn_params(16)

    # Per-block weights
    for i, (in_c, hid, out_c, k, s, p, hs, se_r, res) in enumerate(BLOCKS):
        prefix = f'b{i}'
        if hid != in_c:
            w[f'{prefix}_exp_w'] = np.random.randn(hid, in_c, 1, 1).astype(np.float32) * sc
            w[f'{prefix}_exp_bn'] = _bn_params(hid)
        w[f'{prefix}_dw_w'] = np.random.randn(hid, 1, k, k).astype(np.float32) * sc
        w[f'{prefix}_dw_bn'] = _bn_params(hid)
        if se_r is not None:
            w[f'{prefix}_se1_w'] = np.random.randn(se_r, hid, 1, 1).astype(np.float32) * sc
            w[f'{prefix}_se1_b'] = np.random.randn(se_r).astype(np.float32) * sc
            w[f'{prefix}_se2_w'] = np.random.randn(hid, se_r, 1, 1).astype(np.float32) * sc
            w[f'{prefix}_se2_b'] = np.random.randn(hid).astype(np.float32) * sc
        w[f'{prefix}_pw_w'] = np.random.randn(out_c, hid, 1, 1).astype(np.float32) * sc
        w[f'{prefix}_pw_bn'] = _bn_params(out_c)

    # features.12: final pointwise 96->576
    w['final_pw_w'] = np.random.randn(576, 96, 1, 1).astype(np.float32) * sc
    w['final_pw_bn'] = _bn_params(576)

    # Classifier: Linear(576,1024)+HS -> Linear(1024,1000)
    w['fc1_w'] = np.random.randn(1024, 576).astype(np.float32) * sc
    w['fc1_b'] = np.random.randn(1024).astype(np.float32) * sc
    w['fc2_w'] = np.random.randn(1000, 1024).astype(np.float32) * sc
    w['fc2_b'] = np.random.randn(1000).astype(np.float32) * sc

    return w


# ---- Emulator forward ----

def run_block(x, w, i, cfg):
    in_c, hid, out_c, k, stride, pad, use_hs, se_r, has_res = cfg
    p = f'b{i}'
    out = x

    # Expand (skip if hidden == input)
    if hid != in_c:
        out = _bn(_pw(out, w[f'{p}_exp_w']), w[f'{p}_exp_bn'])
        out = _act(out, use_hs)

    # Depthwise + BN + act
    out = _bn(_dw(out, w[f'{p}_dw_w'], stride=stride, pad=pad), w[f'{p}_dw_bn'])
    out = _act(out, use_hs)

    # SE (optional)
    if se_r is not None:
        out = _se(out, w[f'{p}_se1_w'], w[f'{p}_se1_b'],
                      w[f'{p}_se2_w'], w[f'{p}_se2_b'])

    # Project + BN
    out = _bn(_pw(out, w[f'{p}_pw_w']), w[f'{p}_pw_bn'])

    # Residual
    if has_res:
        out = out + x

    return out


def emulate_forward(x, w):
    # Stem
    out = _bn(emulate_conv2d_resnet(x, w['stem_w'], stride_h=2, stride_w=2,
                                     pad_h=1, pad_w=1), w['stem_bn'])
    out = emulate_hardswish(out)

    # Blocks
    for i, cfg in enumerate(BLOCKS):
        out = run_block(out, w, i, cfg)

    # features.12: final pointwise
    out = _bn(_pw(out, w['final_pw_w']), w['final_pw_bn'])
    out = emulate_hardswish(out)

    # avgpool + classifier
    out = emulate_adaptive_avgpool2d(out)
    out = out.reshape(1, 576)

    # Linear(576, 1024) + HS
    out = (out @ w['fc1_w'].T + w['fc1_b']).astype(np.float32)
    out = emulate_hardswish(out.reshape(1, 1024, 1)).reshape(1, 1024)

    # Linear(1024, 1000)
    out = (out @ w['fc2_w'].T + w['fc2_b']).astype(np.float32)
    return out


# ---- PyTorch Reference ----

def reference_forward(x_np, w):
    import torch
    import torch.nn.functional as F

    x = torch.tensor(x_np, dtype=torch.float32)

    def _bn_t(t, p):
        return F.batch_norm(t,
            torch.tensor(p[0], dtype=torch.float32),
            torch.tensor(p[1], dtype=torch.float32),
            torch.tensor(p[2], dtype=torch.float32),
            torch.tensor(p[3], dtype=torch.float32),
            training=False)

    def _pw_t(t, ww):
        return F.conv2d(t, torch.tensor(ww, dtype=torch.float32))

    def _dw_t(t, ww, stride=1, pad=0):
        C = t.shape[1]
        return F.conv2d(t, torch.tensor(ww, dtype=torch.float32),
                        stride=stride, padding=pad, groups=C)

    def _se_t(t, fc1_w, fc1_b, fc2_w, fc2_b):
        pooled = F.adaptive_avg_pool2d(t, (1, 1))
        squeezed = F.relu(F.conv2d(pooled, torch.tensor(fc1_w, dtype=torch.float32),
                                    bias=torch.tensor(fc1_b, dtype=torch.float32)))
        scale = F.hardsigmoid(F.conv2d(squeezed, torch.tensor(fc2_w, dtype=torch.float32),
                                        bias=torch.tensor(fc2_b, dtype=torch.float32)))
        return t * scale

    # Stem
    out = _bn_t(F.conv2d(x, torch.tensor(w['stem_w'], dtype=torch.float32),
                          stride=2, padding=1), w['stem_bn'])
    out = F.hardswish(out)

    # Blocks
    for i, (in_c, hid, out_c, k, stride, pad, use_hs, se_r, has_res) in enumerate(BLOCKS):
        p = f'b{i}'
        identity = out

        if hid != in_c:
            out = _bn_t(_pw_t(out, w[f'{p}_exp_w']), w[f'{p}_exp_bn'])
            out = F.hardswish(out) if use_hs else F.relu(out)

        out = _bn_t(_dw_t(out, w[f'{p}_dw_w'], stride=stride, pad=pad), w[f'{p}_dw_bn'])
        out = F.hardswish(out) if use_hs else F.relu(out)

        if se_r is not None:
            out = _se_t(out, w[f'{p}_se1_w'], w[f'{p}_se1_b'],
                             w[f'{p}_se2_w'], w[f'{p}_se2_b'])

        out = _bn_t(_pw_t(out, w[f'{p}_pw_w']), w[f'{p}_pw_bn'])

        if has_res:
            out = out + identity

    # features.12
    out = _bn_t(_pw_t(out, w['final_pw_w']), w['final_pw_bn'])
    out = F.hardswish(out)

    # avgpool + classifier
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = out.flatten(1)
    out = F.linear(out, torch.tensor(w['fc1_w'], dtype=torch.float32),
                   bias=torch.tensor(w['fc1_b'], dtype=torch.float32))
    out = F.hardswish(out)
    out = F.linear(out, torch.tensor(w['fc2_w'], dtype=torch.float32),
                   bias=torch.tensor(w['fc2_b'], dtype=torch.float32))
    return out.numpy()


# ---- Self-test ----

def test():
    print("=" * 70)
    print(" MobileNetV3-Small Integration Test")
    print("=" * 70)

    np.random.seed(42)

    # ----------------------------------------------------------
    # Test 1: Single SE block (block 4 config: HS + SE + res)
    # ----------------------------------------------------------
    print("\n--- Test 1: SE block with residual (PW+HS -> DW+HS -> SE -> PW +res) ---")
    x_se = np.random.randn(1, 40, 14, 14).astype(np.float32)
    cfg_se = (40, 240, 40, 5, 1, 2, True, 64, True)
    w_se = {}
    w_se['b99_exp_w'] = np.random.randn(240, 40, 1, 1).astype(np.float32) * 0.01
    w_se['b99_exp_bn'] = _bn_params(240)
    w_se['b99_dw_w'] = np.random.randn(240, 1, 5, 5).astype(np.float32) * 0.01
    w_se['b99_dw_bn'] = _bn_params(240)
    w_se['b99_se1_w'] = np.random.randn(64, 240, 1, 1).astype(np.float32) * 0.01
    w_se['b99_se1_b'] = np.random.randn(64).astype(np.float32) * 0.01
    w_se['b99_se2_w'] = np.random.randn(240, 64, 1, 1).astype(np.float32) * 0.01
    w_se['b99_se2_b'] = np.random.randn(240).astype(np.float32) * 0.01
    w_se['b99_pw_w'] = np.random.randn(40, 240, 1, 1).astype(np.float32) * 0.01
    w_se['b99_pw_bn'] = _bn_params(40)

    out_se = run_block(x_se, w_se, 99, cfg_se)

    import torch
    import torch.nn.functional as F
    t = torch.tensor(x_se, dtype=torch.float32)
    t = F.batch_norm(F.conv2d(t, torch.tensor(w_se['b99_exp_w'], dtype=torch.float32)),
                      torch.tensor(w_se['b99_exp_bn'][0], dtype=torch.float32),
                      torch.tensor(w_se['b99_exp_bn'][1], dtype=torch.float32),
                      torch.tensor(w_se['b99_exp_bn'][2], dtype=torch.float32),
                      torch.tensor(w_se['b99_exp_bn'][3], dtype=torch.float32),
                      training=False)
    t = F.hardswish(t)
    t = F.batch_norm(F.conv2d(t, torch.tensor(w_se['b99_dw_w'], dtype=torch.float32),
                               stride=1, padding=2, groups=240),
                      torch.tensor(w_se['b99_dw_bn'][0], dtype=torch.float32),
                      torch.tensor(w_se['b99_dw_bn'][1], dtype=torch.float32),
                      torch.tensor(w_se['b99_dw_bn'][2], dtype=torch.float32),
                      torch.tensor(w_se['b99_dw_bn'][3], dtype=torch.float32),
                      training=False)
    t = F.hardswish(t)
    pooled = F.adaptive_avg_pool2d(t, (1, 1))
    squeezed = F.relu(F.conv2d(pooled, torch.tensor(w_se['b99_se1_w'], dtype=torch.float32),
                                bias=torch.tensor(w_se['b99_se1_b'], dtype=torch.float32)))
    scale = F.hardsigmoid(F.conv2d(squeezed, torch.tensor(w_se['b99_se2_w'], dtype=torch.float32),
                                    bias=torch.tensor(w_se['b99_se2_b'], dtype=torch.float32)))
    t = t * scale
    t = F.batch_norm(F.conv2d(t, torch.tensor(w_se['b99_pw_w'], dtype=torch.float32)),
                      torch.tensor(w_se['b99_pw_bn'][0], dtype=torch.float32),
                      torch.tensor(w_se['b99_pw_bn'][1], dtype=torch.float32),
                      torch.tensor(w_se['b99_pw_bn'][2], dtype=torch.float32),
                      torch.tensor(w_se['b99_pw_bn'][3], dtype=torch.float32),
                      training=False)
    t = t + torch.tensor(x_se, dtype=torch.float32)
    ref_se = t.numpy()
    verify(out_se, ref_se, "se_block_res", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 2: Block 0 (no expand, ReLU, SE, no residual)
    # ----------------------------------------------------------
    print("\n--- Test 2: Block 0 (no expand, ReLU, SE) ---")
    x_b0 = np.random.randn(1, 16, 56, 56).astype(np.float32)
    cfg_b0 = BLOCKS[0]  # (16, 16, 16, 3, 2, 1, False, 8, False)
    w_b0 = {}
    # No expand (hid == in_c)
    w_b0['b0_dw_w'] = np.random.randn(16, 1, 3, 3).astype(np.float32) * 0.01
    w_b0['b0_dw_bn'] = _bn_params(16)
    w_b0['b0_se1_w'] = np.random.randn(8, 16, 1, 1).astype(np.float32) * 0.01
    w_b0['b0_se1_b'] = np.random.randn(8).astype(np.float32) * 0.01
    w_b0['b0_se2_w'] = np.random.randn(16, 8, 1, 1).astype(np.float32) * 0.01
    w_b0['b0_se2_b'] = np.random.randn(16).astype(np.float32) * 0.01
    w_b0['b0_pw_w'] = np.random.randn(16, 16, 1, 1).astype(np.float32) * 0.01
    w_b0['b0_pw_bn'] = _bn_params(16)

    out_b0 = run_block(x_b0, w_b0, 0, cfg_b0)

    t0 = torch.tensor(x_b0, dtype=torch.float32)
    t0 = F.batch_norm(F.conv2d(t0, torch.tensor(w_b0['b0_dw_w'], dtype=torch.float32),
                                stride=2, padding=1, groups=16),
                       torch.tensor(w_b0['b0_dw_bn'][0], dtype=torch.float32),
                       torch.tensor(w_b0['b0_dw_bn'][1], dtype=torch.float32),
                       torch.tensor(w_b0['b0_dw_bn'][2], dtype=torch.float32),
                       torch.tensor(w_b0['b0_dw_bn'][3], dtype=torch.float32),
                       training=False)
    t0 = F.relu(t0)
    pooled0 = F.adaptive_avg_pool2d(t0, (1, 1))
    sq0 = F.relu(F.conv2d(pooled0, torch.tensor(w_b0['b0_se1_w'], dtype=torch.float32),
                           bias=torch.tensor(w_b0['b0_se1_b'], dtype=torch.float32)))
    sc0 = F.hardsigmoid(F.conv2d(sq0, torch.tensor(w_b0['b0_se2_w'], dtype=torch.float32),
                                  bias=torch.tensor(w_b0['b0_se2_b'], dtype=torch.float32)))
    t0 = t0 * sc0
    t0 = F.batch_norm(F.conv2d(t0, torch.tensor(w_b0['b0_pw_w'], dtype=torch.float32)),
                       torch.tensor(w_b0['b0_pw_bn'][0], dtype=torch.float32),
                       torch.tensor(w_b0['b0_pw_bn'][1], dtype=torch.float32),
                       torch.tensor(w_b0['b0_pw_bn'][2], dtype=torch.float32),
                       torch.tensor(w_b0['b0_pw_bn'][3], dtype=torch.float32),
                       training=False)
    ref_b0 = t0.numpy()
    verify(out_b0, ref_b0, "block0_no_expand_se", rtol=1e-2, atol=1e-3)

    # ----------------------------------------------------------
    # Test 3: Full forward [1, 3, 224, 224] -> [1, 1000]
    # ----------------------------------------------------------
    print("\n--- Test 3: Full MobileNetV3-Small [1,3,224,224] -> [1,1000] ---")
    w = make_weights()
    x_full = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print("  Running emulator forward...")
    out_full = emulate_forward(x_full, w)
    print(f"  Output shape: {out_full.shape}")

    print("  Running PyTorch reference...")
    ref_full = reference_forward(x_full, w)
    print(f"  Reference shape: {ref_full.shape}")

    verify(out_full, ref_full, "mobilenetv3_small_full", rtol=5e-2, atol=1e-2)

    print("\n" + "=" * 70)
    print(" MobileNetV3-Small Integration Test Complete")
    print("=" * 70)
    print()


if __name__ == "__main__":
    test()
