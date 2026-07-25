"""E1 gate: 每轮全量落盘（round transcript）。

成功轮：log/round_N/ 下 01-11 + meta.txt 全部存在且非空。
失败轮：已产生的部分仍落盘，meta.txt 标注失败阶段。
"""
import json

from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator

GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": null, "extension_used": null, "notes": "ok"}\n```')

ALL_FILES = [
    "01_prompt_generate.txt", "02_response_generate.txt", "03_kernel.py",
    "04_meta.json", "05_launch_input.py", "06_raw_sim.json", "07_sim_result.json",
    "08_events.json", "09_verdict.json", "10_summary.txt", "11_decision.json", "meta.txt",
]


class _LLM:
    def generate(self, p):
        return GEN

    def choose_lever(self, p):
        return '```json\n{"lever": "x", "rationale": "r"}\n```'


class _Launcher:
    def __init__(self, *, compiled=True, correct=True, cycles=8000):
        self.compiled = compiled
        self.correct = correct
        self.cycles = cycles
        self.c = 0

    def __call__(self, path):
        self.c += 1
        if self.c == 1:  # baseline
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000, "pipeline": {},
                    "compiled": True, "compile_log": ""}
        return {"correct": self.correct, "max_abs_err": 0.0 if self.correct else 9.9,
                "cycles": self.cycles if self.correct else None, "pipeline": {},
                "compiled": self.compiled, "compile_log": "" if self.compiled else "boom"}


def _job(max_rounds=1, compile_retries=3):
    return NormalizedJob(op="matmul", shapes=[1024] * 3, dtype="fp16",
                        baseline_src="def b(): pass", reference_src=None, has_baseline=True,
                        budget=Budget(max_rounds=max_rounds, compile_retries=compile_retries),
                        form="triton_file")


def _run(tmp_path, job, launcher):
    from control.fixtures import COMPUTE_BOUND
    return Orchestrator(job, llm=_LLM(), launcher=launcher,
                        parse_raw_fn=lambda r: COMPUTE_BOUND,
                        output_dir=tmp_path / "out").run()


def test_successful_round_writes_full_transcript(tmp_path):
    _run(tmp_path, _job(max_rounds=1), _Launcher(cycles=8000))
    rd = tmp_path / "out" / "log" / "round_1"
    for f in ALL_FILES:
        p = rd / f
        assert p.is_file() and p.read_text(encoding="utf-8").strip(), f"{f} missing or empty"
    meta = (rd / "meta.txt").read_text(encoding="utf-8")
    assert "round:" in meta


def test_failed_round_partial_transcript_marks_stage(tmp_path):
    # compile always fails, compile_retries=1 -> stop BUDGET_COMPILE; partial transcript kept
    report = _run(tmp_path, _job(max_rounds=5, compile_retries=1), _Launcher(compiled=False))
    assert report["stop"]["reason"] == "BUDGET_COMPILE"
    rd = tmp_path / "out" / "log" / "round_1"
    # files produced before the compile failure exist
    assert (rd / "01_prompt_generate.txt").is_file()
    assert (rd / "05_launch_input.py").is_file()
    # adapt-stage files were NOT reached
    assert not (rd / "11_decision.json").is_file()
    meta = (rd / "meta.txt").read_text(encoding="utf-8")
    assert "compile" in meta.lower(), f"meta must mark compile failure stage: {meta}"
