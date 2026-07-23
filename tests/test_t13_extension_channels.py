"""T13-5 gate: 让模型理解私有 extension 原语（四+渠道）。

- 渠道① sample_entry.yaml 的 example 必须是完整、可编译的最小 kernel
- 渠道④ 编排器传给 record_attempt 的 extension_used 来自 gen 返回的 json
- 渠道⑤ triton-gen 由 vanilla-first 改为 extension-forward（已授权；反向回归保护）
- 渠道⑥ 探索偏好：多候选 !=off 选证据较少；off 确定可复现；单候选不引入随机性
- 负面经验区分：compiled=false vs compiled=true&correct=false 落库分类不同
"""
import json
from pathlib import Path

import pytest

from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator, pick_lever
from memory import AttemptRecord, ExperienceStore, RunLog, Fingerprint, add_experience, record_attempt

REPO = Path(__file__).resolve().parent.parent
SAMPLE_ENTRY = (REPO / ".claude/skills/extension-guide/references/sample_entry.yaml").read_text("utf-8")  # type: ignore[arg-type]
TRITON_GEN = (REPO / ".claude/skills/triton-gen/SKILL.md").read_text(encoding="utf-8")


# ---- 渠道①: example is a complete compilable kernel ----

def test_sample_example_is_complete_kernel():
    import yaml
    entry = yaml.safe_load(SAMPLE_ENTRY)
    ex = entry["example"]
    assert "import triton" in ex, "example must be a full kernel with imports"
    assert "@triton.jit" in ex and "def " in ex, "example must be a complete jit kernel"


# ---- 渠道⑤: vanilla-first removed, extension-forward in place (reverse regression) ----

def test_vanilla_first_wording_removed():
    assert "Defaults to standard Triton" not in TRITON_GEN
    assert "standard Triton by default" not in TRITON_GEN
    assert "Add an extension primitive ONLY when" not in TRITON_GEN


def test_extension_forward_rules_present():
    # body is English (T8/T10 no-CJK invariant), so check the English markers
    assert "do not hedge" in TRITON_GEN
    assert "structural base" in TRITON_GEN
    assert "First-round" in TRITON_GEN


# ---- 渠道④: orchestrator passes gen's extension_used to record_attempt ----

def test_record_attempt_extension_used_comes_from_gen_json(tmp_path):
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    job = NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                        baseline_src="def b(): pass", reference_src=None, has_baseline=True,
                        budget=Budget(max_rounds=1), form="triton_file")
    gen_resp = ('```python\ndef kernel(a,b,c):\n    return a @ b\n```\n'
                '```json\n{"lever": null, "extension_used": "sample_async_copy_template", "notes": "x"}\n```')

    class LLM:
        def generate(self, p): return gen_resp
        def choose_lever(self, p): return ""

    class Launcher:
        def __init__(s): s.c = 0
        def __call__(s, path):
            s.c += 1
            if s.c == 1:
                return {"correct": True, "max_abs_err": 0.0, "cycles": 12000, "pipeline": {},
                        "compiled": True, "compile_log": ""}
            return {"correct": True, "max_abs_err": 0.0, "cycles": 8000, "pipeline": {},
                    "compiled": True, "compile_log": ""}

    from control.fixtures import COMPUTE_BOUND
    Orchestrator(job, llm=LLM(), launcher=Launcher(), parse_raw_fn=lambda r: COMPUTE_BOUND,
                 store=store, log=log, output_dir=tmp_path / "out").run()
    entries = log.read_all()
    round_entries = [e for e in entries if e.get("extension_used") == "sample_async_copy_template"]
    assert round_entries, "extension_used from gen json must reach record_attempt/runlog"


# ---- 渠道⑥: exploration ----

def test_pick_lever_single_candidate_no_choice():
    assert pick_lever(["only"], evidence={}, exploration="mild") == "only"


def test_pick_lever_off_picks_most_evidence_deterministic():
    cands = ["a", "b", "c"]
    ev = {"a": 5, "b": 1, "c": 3}
    assert pick_lever(cands, ev, "off") == "a"   # exploit: most evidence
    assert pick_lever(cands, ev, "off") == "a"   # deterministic


def test_pick_lever_explore_picks_least_evidence():
    cands = ["a", "b", "c"]
    ev = {"a": 5, "b": 1, "c": 3}
    assert pick_lever(cands, ev, "mild") == "b"   # explore: least evidence
    assert pick_lever(cands, ev, "aggressive") == "b"


def test_pick_lever_off_with_no_evidence_is_deterministic():
    cands = ["a", "b"]
    # no evidence: off picks first (stable), explore also picks first (ties -> first)
    assert pick_lever(cands, {}, "off") == "a"
    assert pick_lever(cands, {}, "mild") == "a"


# ---- negative experience classification ----

def test_negative_experience_classification_differs(tmp_path):
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    fp = Fingerprint(op_kind="matmul", bottleneck="memory_underfilled")
    eid = add_experience(store, fp, text="use bulk copy", extension_used="sample_async_copy_template")

    # compile failure (compiled=false) -> low-value compile class
    record_attempt(log, store, fp, [eid], passed=False, compiled=False, cycles=None)
    # semantic misuse (compiled=true, correct=false) -> high-value negative
    record_attempt(log, store, fp, [eid], passed=False, compiled=True, cycles=None)

    entries = log.read_all()
    kinds = [e.get("failure_kind") for e in entries]
    assert "compile" in kinds and "semantic" in kinds
    # semantic bumps harmed (high-value negative); compile does not
    exp = store.get(eid)
    assert exp.harmed >= 1
