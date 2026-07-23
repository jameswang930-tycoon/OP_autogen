"""T13-3 gate: 编译结果作为独立信号（契约修订，已授权）。

- SimResult += compiled / compile_log
- raw_sim_output schema += compiled / compile_log; build_sim_result reads them
- COMPILE_FAIL: not counted as a round; compile_log fed back into the next gen prompt;
  compile_retries budget; exhaustion -> stop.reason == BUDGET_COMPILE
- compile OK but numerical FAIL -> still counts as a round (T3 semantics, regression)
- {{COMPILE_ERROR}} placeholder added to triton-gen template
"""
import json

import pytest

from control import contracts, placeholders
from control.job_spec import Budget, NormalizedJob
from control.launch_template import build_sim_result
from control.orchestrator import Orchestrator

GEN_RESP = """```python
def kernel(a, b, c):
    return a @ b
```
```json
{"lever": null, "extension_used": null, "notes": "ok"}
```"""


# ---- contract ----

def test_simresult_has_compiled_and_compile_log():
    fields = set(contracts.SimResult.__dataclass_fields__)
    assert "compiled" in fields and "compile_log" in fields


def test_simresult_compiled_defaults_and_validation():
    r = contracts.SimResult(correct=True, max_abs_err=0.0, cycles=10, pipeline={})
    assert r.compiled is True and r.compile_log == ""
    with pytest.raises(Exception):
        contracts.SimResult(correct=True, max_abs_err=0.0, cycles=10, pipeline={}, compiled=1)


def test_build_sim_result_reads_compiled_and_compile_log():
    raw = {"correct": False, "max_abs_err": 0.0, "cycles": None, "pipeline": {},
           "compiled": False, "compile_log": "SyntaxError: invalid extension call"}
    r = build_sim_result(raw)
    assert r.compiled is False
    assert "SyntaxError" in r.compile_log


def test_build_sim_result_requires_compiled():
    with pytest.raises(Exception):
        build_sim_result({"correct": True, "max_abs_err": 0.0, "cycles": 1, "pipeline": {}})


def test_compile_error_placeholder_added():
    assert "COMPILE_ERROR" in placeholders.TRITON_GEN_PLACEHOLDERS


# ---- orchestrator behavior ----

class _CaptureLLM:
    def __init__(self, gen=GEN_RESP):
        self.gen = gen
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.gen

    def choose_lever(self, prompt):
        return '```json\n{"lever": "x", "rationale": "r"}\n```'


class _CompileThenOKLauncher:
    """baseline compiles; round launches: N compile-fails then compile-OK."""

    def __init__(self, *, compile_fails_before_ok, ok_correct=True, ok_cycles=8000):
        self.fails = compile_fails_before_ok
        self.ok_correct = ok_correct
        self.ok_cycles = ok_cycles
        self._baseline_done = False
        self._round_fails = 0

    def __call__(self, path):
        if not self._baseline_done:
            self._baseline_done = True
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000, "pipeline": {},
                    "compiled": True, "compile_log": ""}
        if self._round_fails < self.fails:
            self._round_fails += 1
            return {"correct": False, "max_abs_err": 0.0, "cycles": None, "pipeline": {},
                    "compiled": False, "compile_log": f"compile error #{self._round_fails}: bad primitive"}
        return {"correct": self.ok_correct, "max_abs_err": 0.0 if self.ok_correct else 9.9,
                "cycles": self.ok_cycles if self.ok_correct else None, "pipeline": {},
                "compiled": True, "compile_log": ""}


def _fake_parse_raw(raw):
    from control.fixtures import COMPUTE_BOUND
    return COMPUTE_BOUND


def _job(max_rounds=3, compile_retries=2):
    return NormalizedJob(
        op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
        baseline_src="def baseline(): pass", reference_src=None, has_baseline=True,
        budget=Budget(max_rounds=max_rounds, compile_retries=compile_retries),
        form="triton_file",
    )


def _run(tmp_path, job, llm, launcher):
    return Orchestrator(job, llm=llm, launcher=launcher,
                        parse_raw_fn=_fake_parse_raw, output_dir=tmp_path / "out").run()


def test_compile_fail_does_not_count_as_round_and_feeds_back_log(tmp_path):
    llm = _CaptureLLM()
    launcher = _CompileThenOKLauncher(compile_fails_before_ok=1, ok_cycles=8000)
    report = _run(tmp_path, _job(max_rounds=1, compile_retries=3), llm, launcher)

    assert report["stop"]["rounds_used"] == 1, "compile fail must not increment round count"
    # the second gen prompt (after the compile fail) must contain the compile log
    assert any("compile error #1" in p for p in llm.prompts), "compile_log not fed back to gen"


def test_compile_retries_exhausted_stops_with_budget_compile(tmp_path):
    launcher = _CompileThenOKLauncher(compile_fails_before_ok=99)  # never compiles
    report = _run(tmp_path, _job(max_rounds=5, compile_retries=2), _CaptureLLM(), launcher)
    assert report["stop"]["reason"] == "BUDGET_COMPILE"
    assert report["stop"]["rounds_used"] == 0


def test_compile_ok_but_numerical_fail_counts_as_round(tmp_path):
    launcher = _CompileThenOKLauncher(compile_fails_before_ok=0, ok_correct=False)
    report = _run(tmp_path, _job(max_rounds=2, compile_retries=2), _CaptureLLM(), launcher)
    assert report["stop"]["reason"] == "max_rounds"
    assert report["stop"]["rounds_used"] == 2
    assert all(r["compiled"] for r in report["rounds"])
    assert all(r["correct"] is False for r in report["rounds"])
