"""Job 规格（T11 §1）：三种输入形态在边界处归一化为统一的 NormalizedJob。

本 pass 只完整实现 `triton_file`；`pytorch` 与 `shape_only` 留归一化骨架 +
NotImplementedError（job schema 已声明，后续接入不改架构）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Budget:
    max_rounds: int = 6
    epsilon: float = 0.03
    llm_retries: int = 3       # 解析失败重试，不计入轮数
    presim_retries: int = 2    # 未过静态闸门重试，不计入轮数
    sim_retries: int = 3       # 仿真设施故障退避重试，不计入轮数

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Budget":
        d = d or {}
        return cls(
            max_rounds=int(d.get("max_rounds", 6)),
            epsilon=float(d.get("epsilon", 0.03)),
            llm_retries=int(d.get("llm_retries", 3)),
            presim_retries=int(d.get("presim_retries", 2)),
            sim_retries=int(d.get("sim_retries", 3)),
        )


@dataclass
class NormalizedJob:
    op: str
    shapes: list
    dtype: str
    baseline_src: Optional[str]        # 用户提供的 Triton（triton_file）；其余形态为 None
    reference_src: Optional[str]       # 金标准（本 pass 由 gen 一并生成，故 None）
    has_baseline: bool
    budget: Budget
    form: str                          # triton_file | pytorch | shape_only

    @property
    def shape_sig(self) -> str:
        return "x".join(str(s) for s in self.shapes)


def normalize(job: dict, *, baseline_root: Optional[Path] = None) -> NormalizedJob:
    """把原始 job dict 归一化为 NormalizedJob。多态性只在这一层消化。"""
    if "input" not in job or "form" not in job["input"]:
        raise ValueError("job.input.form is required")
    form = job["input"]["form"]
    op = job["op"]
    shapes = list(job["shapes"])
    dtype = job.get("dtype", "fp32")
    budget = Budget.from_dict(job.get("budget"))

    if form == "triton_file":
        path = Path(job["input"]["path"])
        if not path.is_absolute():
            path = (baseline_root or Path.cwd()) / path
        baseline_src = path.read_text(encoding="utf-8")
        return NormalizedJob(
            op=op, shapes=shapes, dtype=dtype,
            baseline_src=baseline_src, reference_src=None,
            has_baseline=True, budget=budget, form=form,
        )
    if form in ("pytorch", "shape_only"):
        # 归一化骨架已就位；完整实现留待后续 pass（不改架构）。
        raise NotImplementedError(
            f"input form {form!r} not implemented in this pass (only triton_file)"
        )
    raise ValueError(f"unknown input form: {form!r}")


def load_job(path, *, baseline_root: Optional[Path] = None) -> NormalizedJob:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return normalize(data, baseline_root=baseline_root)
