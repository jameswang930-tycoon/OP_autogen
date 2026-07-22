"""四份冻结契约的定义与校验（T2 地基）。

字段名一旦冻结即不得更改——保密环境的 GLM 4.7 只填内容、不改形状。
四份契约：
  1. Event      —— 仿真事件中间结构（feedback_adapter 的输入）
  2. Verdict    —— 分析结论（adapter 输出，loop_controller / memory 的输入）
  3. SimResult  —— 发射脚本输出（正确性与性能是两个可分辨字段，架构文档 §3.6）
  4. vocabulary —— 见 control/vocabulary.yaml（词表契约，由 vocabulary.py 加载校验）

Event.stall_class 与 Verdict.bottleneck 必须取自瓶颈类别词表。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from . import vocabulary


def _check_int(value: Any, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be int, got {type(value).__name__}")


def _check_nonneg_int(value: Any, field: str) -> None:
    _check_int(value, field)
    if value < 0:
        raise ValueError(f"{field} must be >= 0, got {value}")


@dataclass
class Event:
    """仿真事件中间结构：feedback_adapter 的输入（来自 parse_raw）。

    字段冻结，4.7 不得更改：
      name        : 事件名（如某段流水/某次访存）
      start, end  : 事件起止（仿真时间单位，整数）
      duration    : 持续时长 = end - start（冗余但显式，便于 reduce）
      unit        : 执行单元（来自仿真侧的字段名，如 "MTE"）
      stall_class : 瓶颈类别，必须取自词表
      bytes       : 涉及字节数，可为空
    """

    name: str
    start: int
    end: int
    duration: int
    unit: str
    stall_class: str
    bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Event.name must be a non-empty str")
        _check_nonneg_int(self.start, "Event.start")
        _check_nonneg_int(self.end, "Event.end")
        _check_nonneg_int(self.duration, "Event.duration")
        if self.start > self.end:
            raise ValueError(f"Event.start ({self.start}) must be <= end ({self.end})")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("Event.unit must be a non-empty str")
        if self.bytes is not None:
            _check_nonneg_int(self.bytes, "Event.bytes")
        vocabulary.assert_label(self.stall_class)


@dataclass
class Verdict:
    """分析结论：adapter 输出，loop_controller / memory 的机器可读输入。

      bottleneck     : 主导瓶颈类别，必须取自词表
      lever          : 对应优化杠杆（供 sim-analyze 查表）
      cycles         : 本轮实测 cycles（来自 SimResult）
      expected_gain  : 预期改善（0~1 区间的相对增益估计，供排序参考）
    """

    bottleneck: str
    lever: str
    cycles: int
    expected_gain: float

    def __post_init__(self) -> None:
        vocabulary.assert_label(self.bottleneck)
        if not isinstance(self.lever, str) or not self.lever:
            raise ValueError("Verdict.lever must be a non-empty str")
        _check_nonneg_int(self.cycles, "Verdict.cycles")
        if not isinstance(self.expected_gain, (int, float)) or isinstance(self.expected_gain, bool):
            raise TypeError("Verdict.expected_gain must be a number")


@dataclass
class SimResult:
    """发射脚本输出。正确性与性能是两个可分辨字段（架构文档 §3.6）。

      correct    : 数值是否正确（True/False 二值）
      max_abs_err: 最大绝对误差（正确性信号）
      cycles     : 实测 cycles（性能信号）。correct=False 时为 None —— 性能数据作废
      pipeline   : 机器可读的流水分项（{unit: cycles}，供 adapter reduce）
    """

    correct: bool
    max_abs_err: float
    cycles: Optional[int]
    pipeline: dict

    def __post_init__(self) -> None:
        if not isinstance(self.correct, bool):
            raise TypeError("SimResult.correct must be bool")
        if not isinstance(self.max_abs_err, (int, float)) or isinstance(self.max_abs_err, bool):
            raise TypeError("SimResult.max_abs_err must be a number")
        if self.max_abs_err < 0:
            raise ValueError(f"SimResult.max_abs_err must be >= 0, got {self.max_abs_err}")
        if self.cycles is not None:
            _check_nonneg_int(self.cycles, "SimResult.cycles")
        if not isinstance(self.pipeline, dict):
            raise TypeError("SimResult.pipeline must be a dict")
