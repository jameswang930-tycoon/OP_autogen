"""launch_template —— 发射脚本模板（T6）。

发射到远端仿真器的文件**不止 kernel**，还自带 reference（金标准）与比对代码，
才构成可执行的功能+时序单元（架构文档 §1.0）。正确性比对由脚本内的 reference
承担，因此远端一次跑完同时给出「数值对不对」与「流水多快」两个可分辨信号（§3.6）。

本模块提供三样东西：
  1. LAUNCHABLE_TEMPLATE   —— 多段式（kernel / reference / compare）骨架，注明
                              compare 段必须吐出的规范化 raw_sim_output 字段。gen'/4.7 照此组装。
  2. launch(kernel_file)   —— 槽位：把组装好的文件发射到远端仿真器并取回原始输出。4.7 实现。
  3. build_sim_result(raw) —— 确定性：把 launch() 返回的规范化 dict 构造为 SimResult。
  4. run(kernel_file)      —— 端到端：launch() -> raw -> SimResult。

raw_sim_output 规范 schema（launch() 必须按此返回；多余键如 events 留给 feedback_adapter.parse_raw）:
  {"correct": bool, "max_abs_err": float, "cycles": int|None, "pipeline": {unit: cycles}}
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from .contracts import SimResult

# raw_sim_output 中 build_sim_result 直接读取的键
_REQUIRED_KEYS = ("correct", "max_abs_err", "pipeline")


def launch(kernel_file: str) -> dict:
    """槽位：把 kernel_file 发射到远端仿真器并取回原始输出。

    由保密环境的 GLM 4.7 实现。返回值须符合 raw_sim_output 规范 schema（见模块 docstring）。
    """
    raise NotImplementedError("待保密环境实现：把 kernel 发射到远端仿真器并取回原始输出")


def build_sim_result(raw_sim_output: dict) -> SimResult:
    """从规范化 raw_sim_output 构造 SimResult（确定性）。"""
    if not isinstance(raw_sim_output, dict):
        raise TypeError(f"raw_sim_output must be a dict, got {type(raw_sim_output).__name__}")
    for k in _REQUIRED_KEYS:
        if k not in raw_sim_output:
            raise ValueError(f"raw_sim_output missing required key {k!r}")
    return SimResult(
        correct=raw_sim_output["correct"],
        max_abs_err=raw_sim_output["max_abs_err"],
        cycles=raw_sim_output.get("cycles"),  # None on FAIL (perf voided, §3.6)
        pipeline=raw_sim_output["pipeline"],
    )


def run(
    kernel_file: str,
    *,
    launcher: Optional[Callable[[str], dict]] = None,
) -> SimResult:
    """端到端：launch(kernel_file) -> raw_sim_output -> SimResult。

    launcher 可注入：测试用本地假 launch 跑通全链路；缺省走真实 launch() 槽位。
    """
    fn = launcher or launch
    raw = fn(kernel_file)
    return build_sim_result(raw)


LAUNCHABLE_TEMPLATE = '''"""Launchable unit: kernel + reference + compare (架构文档 §1.0).

发射到远端仿真器的就是这一个文件。远端功能执行（真算数值）+ 时序采集（流水）兼具。
compare 段负责比对 kernel 与 reference，并吐出规范化 raw_sim_output。
"""
# === SEGMENT 1: kernel (Triton + extension) ===
# gen' 产物：默认标准 Triton，仅当 Verdict 瓶颈类别要求时叠加 extension 原语。
<kernel_src>

# === SEGMENT 2: reference (numpy / torch gold standard) ===
# 与 kernel 同输入、同输出的金标准实现，用于数值比对。
<reference_src>

# === SEGMENT 3: compare + emit result ===
# 跑 kernel 与 reference，计算最大绝对误差；采集时序；吐出规范化 raw_sim_output。
def _compare():
    # out_kernel = run_kernel(...)
    # out_ref    = run_reference(...)
    # max_abs_err = float(max abs diff)
    result = {
        "correct": bool(max_abs_err <= TOL),
        "max_abs_err": float(max_abs_err),
        "cycles": int(measured_cycles),      # None if correct == False (perf voided, §3.6)
        "pipeline": {unit: cycles},          # 机器可读流水分项，供 adapter reduce
        # "events": [...]                    # 可选：流水事件，供 feedback_adapter.parse_raw
    }
    return result
'''
