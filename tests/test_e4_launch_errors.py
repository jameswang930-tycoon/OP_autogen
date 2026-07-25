"""E4 gate: launch 失败五分类（分清"谁的错"）。

前四类 = 框架外/基础设施 → sim_retries 退避重试、不计轮数、不重新生成 kernel。
ResultMismatch = 框架内 bug（run id 隔离失效）→ 立即停止、不重试。证据入报错。
"""
import pytest

from control.job_spec import Budget, NormalizedJob
from control.launch_template import (
    RemoteConnectionError, RemoteTimeout, RemoteScriptError, ResultNotFound, ResultMismatch,
)
from control.orchestrator import Orchestrator

GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": null, "extension_used": null, "notes": "ok"}\n```')


class _LLM:
    def generate(self, p):
        return GEN

    def choose_lever(self, p):
        return ""


class _RaisingLauncher:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def __call__(self, path):
        self.calls += 1
        raise self.exc


def _job(sim_retries=2):
    return NormalizedJob(op="matmul", shapes=[1024] * 3, dtype="fp16",
                        baseline_src=None, reference_src=None, has_baseline=False,
                        budget=Budget(max_rounds=5, sim_retries=sim_retries),
                        form="shape_only")


def _run(tmp_path, launcher, sim_retries=2):
    from control.fixtures import COMPUTE_BOUND
    return Orchestrator(_job(sim_retries=sim_retries), llm=_LLM(), launcher=launcher,
                        parse_raw_fn=lambda r: COMPUTE_BOUND,
                        output_dir=tmp_path / "out").run()


@pytest.mark.parametrize("exc", [
    RemoteConnectionError("endpoint_xyz", "conn refused"),
    RemoteTimeout(180, "run_abc"),
    RemoteScriptError(2, "stderr boom"),
    ResultNotFound("/out/run_abc/result.json", "run_abc", 30),
])
def test_retryable_failures_exhaust_sim_budget(tmp_path, exc):
    report = _run(tmp_path, _RaisingLauncher(exc), sim_retries=2)
    assert report["stop"]["reason"] == "BUDGET_SIM_RETRIES", report["stop"]
    assert report["stop"]["rounds_used"] == 0, "infra failures must not count as rounds"


def test_result_mismatch_stops_immediately_with_evidence(tmp_path):
    exc = ResultMismatch("run_expected", "run_actual")
    launcher = _RaisingLauncher(exc)
    report = _run(tmp_path, launcher, sim_retries=5)
    assert report["stop"]["reason"] == "RESULT_MISMATCH"
    detail = report["stop"]["detail"] or ""
    assert "run_expected" in detail and "run_actual" in detail, f"evidence missing: {detail}"
    # must NOT have retried away a framework bug
    assert launcher.calls == 1


def test_exception_types_carry_evidence_fields():
    assert RemoteConnectionError("e", "o").endpoint == "e"
    assert RemoteTimeout(180, "r").timeout_s == 180 and RemoteTimeout(180, "r").run_id == "r"
    assert RemoteScriptError(2, "s").exit_code == 2 and RemoteScriptError(2, "s").stderr == "s"
    rm = ResultMismatch("exp", "act")
    assert rm.expected_run_id == "exp" and rm.actual_run_id == "act"
