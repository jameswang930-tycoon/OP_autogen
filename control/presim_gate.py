"""presim_gate —— 昂贵仿真前的静态预筛（T7）。

目的（架构文档 §1.3 / §3.6）：在花掉一次昂贵仿真之前，挡掉明显坏掉的 kernel。
大 kernel 场景下，因低级错误（shape 不匹配、语法问题、extension 调用非法）浪费一次
仿真最亏。**不做数值仿真**，只做廉价的静态检查：

  1. 语法合法（compile）
  2. shape / dtype 自洽（按声明的 shape_contract 校验；无 contract 则跳过）
  3. extension 原语调用合法（槽位，4.7 实现；占位恒通过）

shape_contract 是确定性检查的输入（gen'/4.7 随 kernel 一并提供），形如：
  {"kind": "matmul"|"elementwise"|"reduce",
   "inputs": {name: [dims...]},
   "output": [dims...],
   "dtypes": {name: "fp16"|...}}    # 可选

未知 op kind 一律挡下：预筛宁可保守（挡下后人工看一眼）也不放行无法校验的 kernel。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GateResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


def check_extension_calls(kernel_src: str) -> list[str]:
    """槽位：检查 extension 原语调用是否合法。返回问题列表，空表示通过。

    由保密环境的 GLM 4.7 实现（需真实原语签名）。占位：保密环境实现前恒通过。
    """
    return []


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
