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

import os
import time
from pathlib import Path
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


# ---------------- launchable template (file-loaded, 任务 C) ----------------

# Frozen placeholder contract for the launchable template. Confidential env's real
# triton.py template must use the same set; consistency is auto-checked (T10 mechanism).
LAUNCHABLE_PLACEHOLDERS = frozenset({
    "OP", "SHAPES", "DTYPE", "KERNEL_BODY", "REFERENCE",
})

_DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent / "launchable_template.example.py"


def load_launchable_template(path: Optional[str | Path] = None) -> str:
    """从文件加载可发射模板（公开分支默认占位模板；保密环境加载真实那份）。

    路径取 LAUNCHABLE_TEMPLATE_PATH 环境变量或默认示例文件。模板占位符契约冻结于
    LAUNCHABLE_PLACEHOLDERS；compare 段必须吐出规范 raw_sim_output 字段（固定契约）。
    """
    p = Path(path) if path else Path(
        os.environ.get("LAUNCHABLE_TEMPLATE_PATH") or _DEFAULT_TEMPLATE_PATH)
    return p.read_text(encoding="utf-8")


def assemble_launchable(template_str: str, values: dict) -> str:
    """把 values 填入模板的 {{VAR}} 占位符，产出可发射的 python 源码。"""
    out = template_str
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


# 模块级常量：默认加载的占位模板（保持向后兼容；test_t6 等仍可引用）。
LAUNCHABLE_TEMPLATE = load_launchable_template()
