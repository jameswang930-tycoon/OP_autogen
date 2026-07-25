"""E7 gate: 实时进展（live progress）。

编排器执行时把结构化进度事件实时打到 stderr（人可读）并写入 progress.jsonl（机器可读）。
quiet 只出关键节点；normal 出各阶段；verbose 额外落盘路径。进度是旁路观测，不承载状态。
"""
import json

from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator

GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": null, "extension_used": null, "notes": "ok"}\n```')


class _LLM:
    def generate(self, p): return GEN
    def choose_lever(self, p): return ""


class _Launcher:
    def __init__(self): self.c = 0
    def __call__(self, path):
        self.c += 1
        if self.c == 1:
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000, "pipeline": {},
                    "compiled": True, "compile_log": ""}
        return {"correct": True, "max_abs_err": 0.0, "cycles": 8000, "pipeline": {},
                "compiled": True, "compile_log": ""}


def _run(tmp_path, progress, progress_path=None):
    from control.fixtures import COMPUTE_BOUND
    job = NormalizedJob(op="matmul", shapes=[1024] * 3, dtype="fp16",
                       baseline_src="def b(): pass", reference_src=None, has_baseline=True,
                       budget=Budget(max_rounds=1), form="triton_file")
    return Orchestrator(job, llm=_LLM(), launcher=_Launcher(),
                        parse_raw_fn=lambda r: COMPUTE_BOUND, output_dir=tmp_path / "out",
                        progress=progress, progress_path=progress_path).run()


def test_progress_normal_emits_stage_events_to_stderr(tmp_path, capsys):
    _run(tmp_path, progress="normal")
    err = capsys.readouterr().err
    assert "generate" in err
    assert "result" in err
    assert "stop" in err


def test_progress_writes_jsonl(tmp_path, capsys):
    pj = tmp_path / "progress.jsonl"
    _run(tmp_path, progress="normal", progress_path=pj)
    lines = [l for l in pj.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines, "progress.jsonl must have one json event per step"
    events = [json.loads(l) for l in lines]
    stages = [e["stage"] for e in events]
    assert "generate" in stages and "result" in stages and "stop" in stages


def test_progress_quiet_emits_fewer_stderr_lines(tmp_path, capsys):
    _run(tmp_path, progress="quiet")
    err = capsys.readouterr().err
    # quiet suppresses per-step detail but keeps key nodes
    assert "generate" not in err
    assert ("result" in err or "stop" in err)
