"""任务 B gate: 槽位 5 check_extension_calls 拆成逻辑(5.2) + 数据(4.7) 两层。

逻辑不涉密（AST 解析、比对签名），涉密的只是签名表内容。用示例签名表测试，不需真实信息。
"""
from control.presim_gate import ExtensionSignature, check_extension_calls, load_signature_table
from control.build_signature_table import parse_inventory

TABLE = {
    "bulk_copy": ExtensionSignature("bulk_copy", [2, 3]),   # overloads: 2 or 3 args
    "tensor_mac": ExtensionSignature("tensor_mac", [4]),
}


def test_unknown_primitive_reported():
    src = "def k():\n    ext.unknown(1, 2)\n"
    probs = check_extension_calls(src, TABLE, "ext")
    assert any("unknown" in p.lower() for p in probs)


def test_arg_count_mismatch_reported():
    src = "def k():\n    ext.bulk_copy(1)\n"   # 1 arg, neither overload accepts
    assert check_extension_calls(src, TABLE, "ext")


def test_arg_count_matches_an_overload_passes():
    for n in (2, 3):
        src = f"def k():\n    ext.bulk_copy({', '.join(['1'] * n)})\n"
        assert check_extension_calls(src, TABLE, "ext") == [], f"{n} args should pass"


def test_legal_call_no_problems():
    src = "def k(a,b,c,d):\n    ext.tensor_mac(a,b,c,d)\n    x = ext.bulk_copy(a,b)\n"
    assert check_extension_calls(src, TABLE, "ext") == []


def test_non_extension_calls_ignored():
    src = "def k(a, b):\n    return a + b\n"
    assert check_extension_calls(src, TABLE, "ext") == []


def test_unparseable_source_returns_empty():
    # syntax is checked elsewhere (presim syntax gate); extension check degrades gracefully
    assert check_extension_calls("not valid python source !!", TABLE, "ext") == []


def test_example_signature_table_loads():
    table = load_signature_table()  # default public-branch example file
    assert isinstance(table, dict)


# ---- build_signature_table ----

def test_build_signature_table_parses_inventory_and_merges_overloads():
    inv = """
mod/path | bulk_copy | bulk_copy(dst, src, n) | bulk copy
mod/path | bulk_copy | bulk_copy(dst, src, n, stride) | bulk copy overload
mod/path | tensor_mac | tensor_mac(a, b, c, acc) | tensor mac
"""
    entries = parse_inventory(inv)
    by_name = {e["name"]: e["param_counts"] for e in entries}
    assert by_name["bulk_copy"] == [3, 4], "overloads must merge into one param_counts list"
    assert by_name["tensor_mac"] == [4]
