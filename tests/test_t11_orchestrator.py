"""T11 gate: 确定性编排器（全部离线，假 launch + 假 LLM，无任何保密信息）。

Covers plan T11 §7 的 7 个验收点：
  1. 端到端 triton_file 跑通 -> report/recommended/final 齐全
  2. 每轮都比 baseline 慢 -> recommended=baseline(speedup=1.0), final=最后一轮(<1)
  3. 解析失败重试不计入轮数（2 失败 + 1 成功 -> rounds_used=1）
  4. 数值 FAIL 计入轮数 -> 达 max_rounds 停
  5. 词表闭包：未知 bottleneck -> 立即停
  6. no-baseline 模式正常跑
  7. 占位符一致性：编排器注入变量 == 模板占位符
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from control import placeholders
from control.job_spec import Budget, NormalizedJob
from control.orchestrator import (
    Orchestrator, build_gen_prompt, build_analyze_prompt,
    parse_generate_response, parse_lever_response, ParseError,
)

REPO = Path(__file__).resolve().parent.parent

GEN_RESP = """```python
def kernel(a, b, c):
    return a @ b
```
```json
{"lever": "memory_underfilled", "extension_used": null, "notes": "ok"}
```"""

BAD_RESP = "the model rambles with no fenced blocks at all"


# ---------------- fakes ----------------

class FakeLLM:
    def __init__(self, gen=GEN_RESP, choose='```json\n{"lever": "x", "rationale": "r"}\n```',
                 fail_then_succeed=0):
        self.gen = gen
        self.choose = choose
        self._fails = fail_then_succeed

    def generate(self, prompt):
        if self._fails > 0:
            self._fails -= 1
            return BAD_RESP
        return self.gen

    def choose_lever(self, prompt):
        return self.choose


class FakeLauncher:
    """first call = baseline iff first_is_baseline; then round responses."""

    def __init__(self, *, first_is_baseline, baseline_cycles=12000,
                 round_cycles=8000, round_correct=True):
        self.first_is_baseline = first_is_baseline
        self.baseline_cycles = baseline_cycles
        self.round_cycles = list(round_cycles) if isinstance(round_cycles, list) else [round_cycles]
        self.round_correct = round_correct
        self._round_idx = 0
        self._baseline_done = False

    def __call__(self, path):
        if self.first_is_baseline and not self._baseline_done:
            self._baseline_done = True
            return {"correct": True, "max_abs_err": 0.0,
                    "cycles": self.baseline_cycles, "pipeline": {}}
        cyc = self.round_cycles[min(self._round_idx, len(self.round_cycles) - 1)]
        self._round_idx += 1
        correct = self.round_correct
        return {"correct": correct,
                "max_abs_err": 0.0 if correct else 9.9,
                "cycles": cyc if correct else None,
                "pipeline": {}}


def _fake_parse_raw(raw):
    # ignore raw; return a known fixture so adapt produces a valid verdict
    from control.fixtures import COMPUTE_BOUND
    return COMPUTE_BOUND


def _job(has_baseline=True, baseline_src="def baseline(): pass\n",
         max_rounds=4, epsilon=0.03):
    return NormalizedJob(
        op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
        baseline_src=baseline_src if has_baseline else None,
        reference_src=None, has_baseline=has_baseline,
        budget=Budget(max_rounds=max_rounds, epsilon=epsilon),
        form="triton_file" if has_baseline else "shape_only",
    )


def _run(tmp_path, job, *, llm, launcher, adapt_fn=None):
    orch = Orchestrator(
        job, llm=llm, launcher=launcher,
        parse_raw_fn=_fake_parse_raw, adapt_fn=adapt_fn,
        output_dir=tmp_path / "out",
    )
    return orch.run()


# ---------------- 1. end-to-end ----------------

def test_end_to_end_triton_file(tmp_path):
    job = _job(has_baseline=True, max_rounds=4)
    launcher = FakeLauncher(first_is_baseline=True, baseline_cycles=12000,
                            round_cycles=[10000, 9000, 8800])
    report = _run(tmp_path, job, llm=FakeLLM(), launcher=launcher)

    for key in ("job", "baseline", "recommended", "final_round", "stop", "rounds"):
        assert key in report
    assert report["baseline"] == {"cycles": 12000, "present": True}
    assert report["recommended"]["cycles"] <= 12000
    assert report["recommended"]["speedup_vs_baseline"] >= 1.0
    assert report["rounds"], "rounds list must be non-empty"
    # artifacts written
    assert (tmp_path / "out" / "report.json").is_file()
    assert (tmp_path / "out" / "recommended.py").is_file()
    assert (tmp_path / "out" / "final_round.py").is_file()


# ---------------- 2. fallback (all slower than baseline) ----------------

def test_all_slower_recommends_baseline(tmp_path):
    job = _job(has_baseline=True, max_rounds=5)
    launcher = FakeLauncher(first_is_baseline=True, baseline_cycles=12000,
                            round_cycles=13000)  # every round slower
    report = _run(tmp_path, job, llm=FakeLLM(), launcher=launcher)

    assert report["recommended"]["round"] == 0, "best-so-far must be the baseline"
    assert report["recommended"]["cycles"] == 12000
    assert report["recommended"]["speedup_vs_baseline"] == 1.0
    assert report["final_round"]["speedup_vs_baseline"] < 1.0
    assert report["final_round"]["round"] >= 1


# ---------------- 3. retry isolation (parse fails don't count as rounds) ----------------

def test_parse_retries_do_not_count_as_rounds(tmp_path):
    job = _job(has_baseline=True, max_rounds=1)
    launcher = FakeLauncher(first_is_baseline=True, round_cycles=8000)
    llm = FakeLLM(fail_then_succeed=2)  # 2 parse fails, then success
    report = _run(tmp_path, job, llm=llm, launcher=launcher)

    assert report["stop"]["rounds_used"] == 1, "parse-retries must not count as rounds"
    assert len(report["rounds"]) == 1
    assert report["rounds"][0]["correct"] is True


# ---------------- 4. numerical FAIL counts toward rounds ----------------

def test_numerical_fail_counts_toward_rounds(tmp_path):
    job = _job(has_baseline=True, max_rounds=3)
    launcher = FakeLauncher(first_is_baseline=True, round_correct=False)
    report = _run(tmp_path, job, llm=FakeLLM(), launcher=launcher)

    assert report["stop"]["reason"] == "max_rounds"
    assert report["stop"]["rounds_used"] == 3
    assert all(r["correct"] is False for r in report["rounds"])
    assert all(r["cycles"] is None for r in report["rounds"])


# ---------------- 5. vocab closure (unknown bottleneck stops) ----------------

def test_unknown_bottleneck_stops(tmp_path):
    job = _job(has_baseline=True, max_rounds=5)
    launcher = FakeLauncher(first_is_baseline=True, round_cycles=8000)
    fake_adapt = lambda events: SimpleNamespace(
        verdict=SimpleNamespace(bottleneck="totally_unknown_category"), summary="")
    report = _run(tmp_path, job, llm=FakeLLM(), launcher=launcher, adapt_fn=fake_adapt)

    assert report["stop"]["reason"] == "UNKNOWN_BOTTLENECK"
    assert report["stop"]["detail"] == "totally_unknown_category"


# ---------------- 6. no-baseline mode ----------------

def test_no_baseline_mode(tmp_path):
    job = _job(has_baseline=False, max_rounds=3)
    launcher = FakeLauncher(first_is_baseline=False, round_cycles=[10000, 9000, 8800])
    report = _run(tmp_path, job, llm=FakeLLM(), launcher=launcher)

    assert report["baseline"]["present"] is False
    assert report["baseline"]["cycles"] is None
    assert report["recommended"]["round"] >= 1, "best-so-far starts empty, then a round wins"
    assert report["recommended"]["speedup_vs_baseline"] is None


# ---------------- 7. placeholder consistency ----------------

def test_placeholder_consistency():
    job = _job(has_baseline=False)
    gen_prompt = build_gen_prompt(
        job, baseline_src="", verdict_json="", feedback_summary="",
        retrieved_experience="", extension_index="{}",
    )
    analyze_prompt = build_analyze_prompt("{}", "(summary)", "[leverA, leverB]")
    assert "{{" not in gen_prompt, "gen prompt has unsubstituted placeholders"
    assert "{{" not in analyze_prompt, "analyze prompt has unsubstituted placeholders"

    # orchestrator's canonical sets == skill body placeholders (single source of truth)
    def _body_ph(skill):
        text = (REPO / ".claude" / "skills" / skill / "SKILL.md").read_text(encoding="utf-8")
        return set(re.findall(r"{{([A-Z][A-Z0-9_]*)}}", text))

    assert _body_ph("triton-gen") == set(placeholders.TRITON_GEN_PLACEHOLDERS)
    assert _body_ph("sim-analyze") == set(placeholders.SIM_ANALYZE_PLACEHOLDERS)


# ---------------- parse gates (extra) ----------------

def test_parse_generate_rejects_missing_blocks():
    with pytest.raises(ParseError):
        parse_generate_response("no blocks", allowed_extensions=set())


def test_parse_generate_rejects_unknown_extension():
    resp = """```python
x = 1
```
```json
{"lever": null, "extension_used": "not_in_cheatsheet", "notes": "x"}
```"""
    with pytest.raises(ParseError):
        parse_generate_response(resp, allowed_extensions={"only_this_one"})


def test_parse_lever_rejects_out_of_set():
    with pytest.raises(ParseError):
        parse_lever_response('```json\n{"lever": "zzz", "rationale": "r"}\n```', ["a", "b"])
