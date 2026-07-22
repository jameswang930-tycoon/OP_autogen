"""T5 gate: feedback-adapter 骨架 + 合成夹具。

Covers plan T5 "各门禁必须覆盖的内容":
  - 三份夹具（compute-bound / 传输欠填充 / 依赖 stall）均产出合法 7 段 + Verdict
  - 输出体量在上限内
  - parse_raw() 保持 NotImplementedError
Plus reduce/classify/critical-path coverage so the gate is not self-proving.
"""
import json
import re

import pytest

from control import feedback_adapter as fa
from control.contracts import Event, Verdict
from control.fixtures import COMPUTE_BOUND, MEMORY_UNDERFILLED, STALL_DEPENDENCY

FIXTURES = [
    (COMPUTE_BOUND, "compute_bound_at_peak"),
    (MEMORY_UNDERFILLED, "memory_underfilled"),
    (STALL_DEPENDENCY, "stall_dependency"),
]


def test_each_fixture_produces_valid_7sections_and_verdict():
    for events, expected_bn in FIXTURES:
        out = fa.adapt(events)
        # all 7 sections present
        for sec in fa.SECTIONS:
            assert sec in out.summary, f"{expected_bn}: missing section {sec!r}"
        # verdict is a valid Verdict whose bottleneck matches the fixture intent
        assert isinstance(out.verdict, Verdict)
        assert out.verdict.bottleneck == expected_bn
        # machine-readable verdict header at the very top
        assert out.summary.lstrip().startswith("<!-- VERDICT ")
        # output within budget
        assert len(out.summary) < fa.MAX_OUTPUT_CHARS, f"{expected_bn}: output too large"


def test_verdict_json_header_matches_verdict_object():
    out = fa.adapt(MEMORY_UNDERFILLED)
    m = re.search(r"<!-- VERDICT (\{.*\}) -->", out.summary)
    assert m, "verdict JSON header not found"
    head = json.loads(m.group(1))
    assert head["bottleneck"] == out.verdict.bottleneck
    assert head["cycles"] == out.verdict.cycles
    assert head["lever"] == out.verdict.lever


def test_parse_raw_is_slot():
    with pytest.raises(NotImplementedError):
        fa.parse_raw("any raw simulation output")


def test_classify_bottleneck_is_max_duration_class():
    r = fa.reduce_events(COMPUTE_BOUND)
    c = fa.classify(r)
    assert c.bottleneck == "compute_bound_at_peak"
    assert 0.0 <= c.expected_gain <= 1.0


def test_critical_path_is_non_overlapping_chain():
    for events, _ in FIXTURES:
        r = fa.reduce_events(events)
        assert len(r.critical_path) >= 1
        for a, b in zip(r.critical_path, r.critical_path[1:]):
            assert a.end <= b.start, "critical path must be a non-overlapping chain"


def test_reduce_caps_dominant_and_output_stays_bounded():
    # 60 overlapping events (parallel noise) — reduce must keep output compact
    events = [
        Event(f"u{i % 4}_op{i}", 0, (i % 30) + 5, (i % 30) + 5,
              f"U{i % 4}", "compute_bound_at_peak")
        for i in range(60)
    ]
    r = fa.reduce_events(events, k=5)
    assert len(r.dominant) <= 5
    out = fa.adapt(events)
    assert len(out.summary) < fa.MAX_OUTPUT_CHARS


def test_empty_events_rejected():
    with pytest.raises(Exception):
        fa.adapt([])
