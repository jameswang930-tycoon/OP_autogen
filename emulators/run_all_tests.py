"""
全量 Emulator 自测
===================
运行: python run_all_tests.py
"""

import numpy as np
np.random.seed(42)

import sys, os, importlib
sys.path.insert(0, os.path.dirname(__file__))

print("╔" + "═" * 58 + "╗")
print("║   Triton CPU Emulator - All Atomic Ops Self-Test         ║")
print("╚" + "═" * 58 + "╝")

from add import test as test_add
from matmul import test as test_matmul
from transpose import test as test_transpose
from reshape import test as test_reshape
from softmax import test as test_softmax
from relu import test as test_relu
from rmsnorm import test as test_rmsnorm
from addrmsnormgamma import test as test_addrmsnormgamma
from conv2d import test as test_conv2d
from conv1d import test as test_conv1d
test_attention_relu = importlib.import_module('attention-relu').test

test_add()
test_matmul()
test_transpose()
test_reshape()
test_softmax()
test_relu()
test_rmsnorm()
test_addrmsnormgamma()
test_conv2d()
test_conv1d()
test_attention_relu()

print("=" * 60)
print(" All emulator tests completed.")
print("=" * 60)
