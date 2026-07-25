"""presim_gate —— 昂贵仿真前的静态预筛（T7）。

目的（架构文档 §1.3 / §3.6）：在花掉一次昂贵仿真之前，挡掉明显坏掉的 kernel。
大 kernel 场景下，因低级错误（shape 不匹配、语法问题、extension 调用非法）浪费一次
仿真最亏。**不做数值仿真**，只做廉价的静态检查：

  1. 语法合法（compile）
  2. shape / dtype 自洽（按声明的 shape_contract 校验；无 contract 则跳过）
  3. extension 原语调用合法（**逻辑由 5.2 实现**：AST 解析 + 比对签名表；
     涉密的只有"真实签名表"这份数据，4.7 用 build_signature_table.py 生成后加载）

shape_contract 是确定性检查的输入（gen'/4.7 随 kernel 一并提供），形如：
  {"kind": "matmul"|"elementwise"|"reduce",
   "inputs": {name: [dims...]},
   "output": [dims...],
   "dtypes": {name: "fp16"|...}}    # 可选

未知 op kind 一律挡下：预筛宁可保守（挡下后人工看一眼）也不放行无法校验的 kernel。
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class GateResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


# ---------------- extension signature table (data = 4.7-filled) ----------------

@dataclass
class ExtensionSignature:
    """一个 extension 原语的签名。param_counts 为接受的位置参数个数列表（支持重载）。"""
    name: str
    param_counts: list[int]


_DEFAULT_SIG_TABLE = Path(__file__).resolve().parent / "signature_table.example.yaml"


def load_signature_table(path: Optional[str | Path] = None) -> dict[str, ExtensionSignature]:
    """从 yaml 加载签名表 -> {primitive_name: ExtensionSignature}。"""
    p = Path(path) if path else Path(
        os.environ.get("PRESIM_SIGNATURE_TABLE") or _DEFAULT_SIG_TABLE)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    table: dict[str, ExtensionSignature] = {}
    for entry in data:
        table[entry["name"]] = ExtensionSignature(
            name=entry["name"], param_counts=list(entry["param_counts"]))
    return table


def _loaded_table() -> dict[str, ExtensionSignature]:
    return load_signature_table()


def _loaded_namespace() -> str:
    return os.environ.get("EXTENSION_NAMESPACE") or "ext"


def extract_extension_calls(kernel_src: str, namespace: str) -> list[tuple[str, int]]:
    """AST 解析 kernel_src，提取所有 ``namespace.name(...)`` 调用 -> [(name, n_positional_args)].

    语法错误时返回 []（语法由专门的语法闸门检查；extension 检查静默降级）。
    """
    try:
        tree = ast.parse(kernel_src)
    except SyntaxError:
        return []
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and f.value.id == namespace):
            calls.append((f.attr, len(node.args)))
    return calls


def check_extension_calls(
    kernel_src: str,
    signature_table: Optional[dict] = None,
    namespace: Optional[str] = None,
) -> list[str]:
    """检查 extension 原语调用是否合法（逻辑层，5.2 实现）。

    - 解析 kernel 中对 ``namespace`` 命名空间的调用；
    - 逐个比对签名表：原语名是否存在、位置参数个数是否匹配某个重载。
    签名表/命名空间缺省时从文件/环境加载（公开分支为示例表；保密环境为真实表）。
    """
    table = signature_table if signature_table is not None else _loaded_table()
    ns = namespace or _loaded_namespace()
    problems: list[str] = []
    for name, n_args in extract_extension_calls(kernel_src, ns):
        sig = table.get(name)
        if sig is None:
            problems.append(f"unknown extension primitive: {ns}.{name}")
        elif n_args not in sig.param_counts:
            problems.append(
                f"{ns}.{name} expects {sig.param_counts} positional args, got {n_args}")
    return problems


def _check_syntax(kernel_src: str) -> list[str]:
    try:
        compile(kernel_src, "<presim>", "exec")
        return []
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]


# ---------------- shape / dtype ----------------

def _check_matmul(inputs: dict, output: Optional[list]) -> list[str]:
    if len(inputs) != 2:
        return [f"matmul requires exactly 2 inputs, got {len(inputs)}"]
    names = list(inputs)
    a, b = inputs[names[0]], inputs[names[1]]
    probs: list[str] = []
    if len(a) != 2 or len(b) != 2:
        return ["matmul inputs must be 2-D"]
    if a[1] != b[0]:
        probs.append(f"matmul inner-dim mismatch: A[-1]={a[1]!r} vs B[0]={b[0]!r}")
    if output is not None:
        if len(output) != 2:
            probs.append("matmul output must be 2-D")
        elif output != [a[0], b[1]]:
            probs.append(f"matmul output {output} != expected [{a[0]!r}, {b[1]!r}]")
    return probs


def _check_elementwise(inputs: dict, output: Optional[list]) -> list[str]:
    if not inputs:
        return ["elementwise requires >=1 input"]
    probs: list[str] = []
    first = list(inputs.values())[0]
    for name, shape in inputs.items():
        # 保守规则：要求各输入形状精确相等（不做广播放宽，预筛宁可多挡）
        if shape != first:
            probs.append(f"elementwise input {name!r} shape {shape} != {first}")
    if output is not None and output != first:
        probs.append(f"elementwise output {output} != input shape {first}")
    return probs


def _check_reduce(inputs: dict, output: Optional[list]) -> list[str]:
    if len(inputs) != 1:
        return [f"reduce requires exactly 1 input, got {len(inputs)}"]
    a = list(inputs.values())[0]
    if output is None:
        return []
    probs: list[str] = []
    if len(output) != len(a):
        probs.append(f"reduce output rank {len(output)} != input rank {len(a)}")
        return probs
    for i, (ai, oi) in enumerate(zip(a, output)):
        if oi != ai and oi != 1:
            probs.append(f"reduce output dim {i}: {oi!r} must be {ai!r} or 1")
    return probs


def _check_dtypes(kind: str, inputs: dict, output: Optional[list], dtypes: dict) -> list[str]:
    in_dt = {n: dtypes[n] for n in inputs if n in dtypes}
    if len(set(in_dt.values())) > 1:
        return [f"input dtype mismatch: {in_dt}"]
    if not in_dt:
        return []
    base = next(iter(in_dt.values()))
    out_dt = dtypes.get("output")
    if out_dt is None:
        return []
    allowed = {base}
    if kind == "matmul":
        allowed.add("fp32")  # 累加可上转
    if out_dt not in allowed:
        return [f"output dtype {out_dt!r} not in allowed {sorted(allowed)} for {kind}"]
    return []


def check_shape_contract(contract: dict) -> list[str]:
    """按声明的 shape_contract 静态校验 shape/dtype 自洽。返回问题列表。"""
    kind = contract.get("kind")
    inputs = contract.get("inputs", {})
    output = contract.get("output")
    if kind == "matmul":
        probs = _check_matmul(inputs, output)
    elif kind == "elementwise":
        probs = _check_elementwise(inputs, output)
    elif kind == "reduce":
        probs = _check_reduce(inputs, output)
    else:
        return [f"unsupported op kind {kind!r}: cannot statically verify shapes"]
    if contract.get("dtypes") is not None:
        probs.extend(_check_dtypes(str(kind), inputs, output, contract["dtypes"]))
    return probs


# ---------------- top-level ----------------

def check(kernel_src: str, shape_contract: Optional[dict] = None) -> GateResult:
    """静态预筛：语法 + shape/dtype（若有 contract）+ extension（槽位）。不做数值仿真。"""
    problems: list[str] = []
    problems.extend(_check_syntax(kernel_src))
    if shape_contract is not None:
        problems.extend(check_shape_contract(shape_contract))
    problems.extend(check_extension_calls(kernel_src))
    return GateResult(ok=not problems, problems=problems)
