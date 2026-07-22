"""从算子规格计算算子特征(Fingerprint)。

接入点:成本模型给出瓶颈类别后,在此组装特征;检索与写回都用它做键。
"""

from __future__ import annotations

import math
from typing import Optional

from .schema import Fingerprint


def _shape_signature(shapes: dict) -> Optional[str]:
    """把各维度按数量级(2 的幂)分档,得到一个粗粒度形状特征。

    预留用途:当前不进主键,仅在将来需要按形状细分时启用。
    """
    if not shapes:
        return None
    buckets = []
    for name in sorted(shapes):
        v = shapes[name]
        if isinstance(v, int) and v > 0:
            buckets.append(f"{name}~2^{int(math.log2(v))}")
    return ",".join(buckets) if buckets else None


def compute_fingerprint(op_spec: dict, bottleneck: Optional[str] = None) -> Fingerprint:
    """op_spec 至少包含 op_kind;shapes 可选;bottleneck 来自成本模型。

    示例 op_spec: {"op_kind": "matmul", "shapes": {"M": 4096, "N": 4096, "K": 4096}}
    """
    return Fingerprint(
        op_kind=op_spec["op_kind"],
        bottleneck=bottleneck,
        shape_sig=_shape_signature(op_spec.get("shapes", {})),
    )


def fingerprint_from_plan_json(
    plan_json: dict, bottleneck: Optional[str] = None
) -> Fingerprint:
    """从 triton-plan 真实写出的 .plan.json 派生算子特征。

    .plan.json 的真实结构(见 .claude/commands/triton-plan.md):
      正常:{op, shapes, dtype, dsl, raw_llm}      # raw_llm 是 7 段纯文本
      失败:{mock: true, op, shapes, dtype, note}  # 模拟器失败的降级桩

    注意:瓶颈信息只以自然语言埋在 raw_llm 文本里,这里不做脆弱的文本解析。
    bottleneck 作为可选入参传入——推荐由 triton-gen 顺手吐出它已推断的瓶颈标签
    (它本就在读 TIME BREAKDOWN / CRITICAL PATH),再回喂到这里。
    缺失时主键自动退回 op_kind。
    """
    op_kind = plan_json.get("op") or plan_json.get("op_kind")
    shapes = plan_json.get("shapes", {}) or {}
    return Fingerprint(
        op_kind=op_kind,
        bottleneck=bottleneck,                       # None 时主键退回 op_kind
        shape_sig=_shape_signature(shapes),
    )
