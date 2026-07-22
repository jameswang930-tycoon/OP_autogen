"""T7 gate: pre-sim 静态预筛。

Covers plan T7 "各门禁必须覆盖的内容":
  - shape 不自洽的 kernel 被挡下
  - check_extension_calls() 保持恒通过占位
Plus syntax check + consistent-contract passes + dtype check + matmul inner-dim.
No numerical simulation is performed.
"""
from control import presim_gate as gate

VALID_MM = (
    "import triton\nimport triton.language as tl\n"
    "@triton.jit\n"
    "def k(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK: tl.constexpr):\n"
    "    pass\n"
)


def test_check_extension_calls_is_placeholder_pass():
    assert gate.check_extension_calls("any kernel source") == []


def test_syntax_error_blocked():
    r = gate.check("def broken(:\n    pass\n")
    assert not r.ok
    assert any("syntax" in p.lower() for p in r.problems)


def test_valid_kernel_with_consistent_contract_passes():
    contract = {"kind": "matmul",
                "inputs": {"A": ["M", "K"], "B": ["K", "N"]},
                "output": ["M", "N"]}
    r = gate.check(VALID_MM, shape_contract=contract)
    assert r.ok, r.problems


def test_shape_inconsistent_matmul_output_blocked():
    # output declared [M,K] but matmul of [M,K]@[K,N] must be [M,N]
    contract = {"kind": "matmul",
                "inputs": {"A": ["M", "K"], "B": ["K", "N"]},
                "output": ["M", "K"]}
    r = gate.check(VALID_MM, shape_contract=contract)
    assert not r.ok
    assert r.problems


def test_matmul_inner_dim_mismatch_blocked():
    contract = {"kind": "matmul",
                "inputs": {"A": ["M", "K"], "B": ["L", "N"]},
                "output": ["M", "N"]}
    r = gate.check(VALID_MM, shape_contract=contract)
    assert not r.ok


def test_elementwise_shape_mismatch_blocked():
    contract = {"kind": "elementwise",
                "inputs": {"A": ["M", "N"], "B": ["M", "K"]},
                "output": ["M", "N"]}
    r = gate.check("def f(a, b): return a + b\n", shape_contract=contract)
    assert not r.ok


def test_dtype_mismatch_blocked():
    contract = {"kind": "elementwise",
                "inputs": {"A": ["M"], "B": ["M"]}, "output": ["M"],
                "dtypes": {"A": "fp16", "B": "fp32", "output": "fp16"}}
    r = gate.check("def f(a, b): return a + b\n", shape_contract=contract)
    assert not r.ok


def test_no_contract_skips_shape_check_only_syntax():
    r = gate.check(VALID_MM)  # no contract -> only syntax + extension (slot)
    assert r.ok


def test_unsupported_op_kind_blocked():
    contract = {"kind": "frobnicate", "inputs": {"A": ["M"]}, "output": ["M"]}
    r = gate.check("def f(a): return a\n", shape_contract=contract)
    assert not r.ok  # gate refuses to pass what it cannot verify
