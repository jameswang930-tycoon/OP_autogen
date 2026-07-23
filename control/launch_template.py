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
  {"correct": bool, "max_abs_err": float, "cycles": int|None, "pipeline": {unit: cycles},
   "compiled": bool, "compile_log": str}        # T13-3：编译状态独立信号

目录式调用约束（T13-2，launch() 实现必须遵守）：
  ① 目录路径必须可配置——输入目录、输出目录、远程脚本路径全部走配置或环境变量，
     不得硬编码（公开分支不得出现真实路径）。
  ② 多轮结果隔离——编排器单作业跑 5–8 轮，每轮调一次 launch()。必须每次用 new_run_id()
     生成唯一 id，用 run id 区分文件名/子目录，并在读取结果时**校验结果确实属于本次 run id**
     （不匹配视为故障）。共享目录不做区分会让性能数据静默错位，极难排查。
  ③ 等待与超时——目录式提交是异步的：脚本返回不代表结果已写完。轮询等待完成标志 + 设置超时；
     超时或连接故障抛**可识别异常**（编排器按 sim_retries 退避重试，不计入轮数）。
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .contracts import SimResult

# raw_sim_output 中 build_sim_result 直接读取的键
_REQUIRED_KEYS = ("correct", "max_abs_err", "pipeline", "compiled")

_run_counter = 0


def new_run_id() -> str:
    """生成唯一 run id（时间戳 + 序号），供 launch() 做多轮结果隔离（T13-2 ②）。"""
    global _run_counter
    _run_counter += 1
    return f"run_{int(time.time())}_{_run_counter:04d}"


def launch(kernel_file: str) -> dict:
    """槽位：把 kernel_file 发射到远端仿真器并取回原始输出。

    由保密环境的 GLM 4.7 实现。返回值须符合 raw_sim_output 规范 schema（见模块 docstring）。
    实现须遵守「目录式调用约束」三条（见模块 docstring）：目录可配置、用 new_run_id()
    做多轮隔离并在读结果时校验 run id、轮询等待 + 超时并抛可识别异常。
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
        compiled=raw_sim_output["compiled"],
        compile_log=raw_sim_output.get("compile_log", ""),
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
        "compiled": bool(compiled_ok),       # T13-3：编译是否通过
        "compile_log": str(compile_log),     # T13-3：编译日志（失败时非空，成功时可空）
        # "events": [...]                    # 可选：流水事件，供 feedback_adapter.parse_raw
    }
    return result
'''
