"""GLM52 优化指导 P4 回归：核心模块失效不许静默——memory 未接线时报 warning。

旧: store/log=None 时 retrieve/record 静默 no-op，memory 全程没工作却无人发现。
新: 首次用到 memory 时显式 warning（每实例一次），把"失效"亮出来。
"""
import warnings  # noqa: F401  (documenting the module's new warning behavior)

import pytest

from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator

GEN = ('```python\ndef k(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": null, "extension_used": null, "notes": "x"}\n```')


class _LLM:
    def generate(self, p): return GEN
    def choose_lever(self, p): return ""


class _Launcher:
    def __init__(self): self.b = False

    def __call__(self, p):
        if not self.b:
            self.b = True
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000,
                    "pipeline": {}, "compiled": True, "compile_log": ""}
        return {"correct": True, "max_abs_err": 0.0, "cycles": 8000,
                "pipeline": {}, "compiled": True, "compile_log": ""}


def _job(max_rounds=1):
    return NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                         baseline_src="def baseline(): pass\n", reference_src=None,
                         has_baseline=True, budget=Budget(max_rounds=max_rounds),
                         form="triton_file")


def test_memory_disabled_emits_warning(tmp_path, recwarn):
    """store=None 跑一轮必须报 warning，不许静默 no-op。"""
    Orchestrator(_job(), llm=_LLM(), launcher=_Launcher(),
                 parse_raw_fn=lambda r: COMPUTE_BOUND,
                 output_dir=tmp_path / "out").run()
    msgs = [str(w.message) for w in recwarn.list]
    assert any("memory disabled" in m for m in msgs), (
        "store=None 时 retrieve/record 静默 no-op 必须报 warning（GLM52 第五部分）"
    )


def test_memory_disabled_warns_once_per_instance(tmp_path):
    """同一实例多轮（多轮 retrieve+record）只报一次，避免刷屏。"""
    with pytest.warns(UserWarning) as caught:
        Orchestrator(_job(max_rounds=2), llm=_LLM(), launcher=_Launcher(),
                     parse_raw_fn=lambda r: COMPUTE_BOUND,
                     output_dir=tmp_path / "out").run()
    mem_warns = [w for w in caught.list if "memory disabled" in str(w.message)]
    assert len(mem_warns) == 1, "memory-disengaged warning 应每实例只报一次"


def test_memory_wired_emits_no_such_warning(tmp_path, recwarn):
    """store+log 接好时不报 memory-disabled warning。"""
    from memory import ExperienceStore, RunLog
    Orchestrator(_job(), llm=_LLM(), launcher=_Launcher(),
                 parse_raw_fn=lambda r: COMPUTE_BOUND,
                 store=ExperienceStore(tmp_path / "e.json"), log=RunLog(tmp_path / "r.jsonl"),
                 output_dir=tmp_path / "out").run()
    msgs = [str(w.message) for w in recwarn.list]
    assert not any("memory disabled" in m for m in msgs)
