"""T2 gate: 四份契约 + 词表注册表。

Covers (plan §3 "各门禁必须覆盖的内容" for T2):
  - 四份契约可校验：合法数据通过、非法数据报错
  - 词表中不存在的标签被拒绝

Four contracts: Event / Verdict / SimResult (in control/contracts.py) +
vocabulary.yaml (loaded/validated by control/vocabulary.py). The bottleneck
vocab is the single source of truth (架构文档 §5.6) shared by adapter / memory /
extension-cheatsheet.
"""
import pytest

from control import contracts, vocabulary
from control import check_vocab_consistency

EXAMPLE_IDS = {"compute_bound_at_peak", "memory_underfilled", "stall_dependency"}


# ---------------- Event ----------------

def _good_event(**over):
    base = dict(
        name="matmul_tile",
        start=10, end=130, duration=120, unit="MTE",
        stall_class="memory_underfilled", bytes=4096,
    )
    base.update(over)
    return contracts.Event(**base)


def test_event_valid_constructs():
    ev = _good_event()
    assert ev.stall_class == "memory_underfilled"
    assert ev.bytes == 4096
    assert ev.bytes is not None


def test_event_bytes_optional():
    ev = _good_event(bytes=None)
    assert ev.bytes is None


def test_event_rejects_unknown_stall_class():
    with pytest.raises(Exception):
        _good_event(stall_class="not_a_real_stall_class")


def test_event_rejects_start_after_end():
    with pytest.raises(Exception):
        _good_event(start=200, end=100)


def test_event_rejects_negative_duration():
    with pytest.raises(Exception):
        _good_event(duration=-5)


def test_event_rejects_bad_field_type():
    with pytest.raises(Exception):
        _good_event(start="100")  # start must be int


# ---------------- Verdict ----------------

def _good_verdict(**over):
    base = dict(
        bottleneck="compute_bound_at_peak",
        lever="introduce tensor/MAC extension primitive",
        cycles=148230,
        expected_gain=0.18,
    )
    base.update(over)
    return contracts.Verdict(**base)


def test_verdict_valid_constructs():
    v = _good_verdict()
    assert v.bottleneck == "compute_bound_at_peak"
    assert v.cycles == 148230


def test_verdict_rejects_unknown_bottleneck():
    with pytest.raises(Exception):
        _good_verdict(bottleneck="nonexistent_bottleneck")


def test_verdict_rejects_negative_cycles():
    with pytest.raises(Exception):
        _good_verdict(cycles=-1)


# ---------------- SimResult ----------------

def test_simresult_pass_with_cycles():
    r = contracts.SimResult(correct=True, max_abs_err=1.2e-6, cycles=148230,
                            pipeline={"MTE": 120})
    assert r.correct is True
    assert r.cycles == 148230


def test_simresult_fail_voids_perf():
    # §3.6: on numerical FAIL, perf data is voided -> cycles is None
    r = contracts.SimResult(correct=False, max_abs_err=9.9, cycles=None, pipeline={})
    assert r.correct is False
    assert r.cycles is None


def test_simresult_rejects_non_bool_correct():
    with pytest.raises(Exception):
        contracts.SimResult(correct=1, max_abs_err=0.0, cycles=10, pipeline={})


def test_simresult_rejects_negative_err():
    with pytest.raises(Exception):
        contracts.SimResult(correct=True, max_abs_err=-0.1, cycles=10, pipeline={})


# ---------------- vocabulary ----------------

def test_vocab_all_ids_are_examples():
    assert vocabulary.all_ids() == EXAMPLE_IDS


def test_vocab_assert_label_accepts_known():
    vocabulary.assert_label("stall_dependency")  # no raise


def test_vocab_assert_label_rejects_unknown():
    with pytest.raises(Exception):
        vocabulary.assert_label("totally_made_up_label")


def test_vocab_validate_format_accepts_good():
    good = [
        {"id": "a", "desc": "d", "lever": "l", "primitives": []},
        {"id": "b", "desc": "d", "lever": "l", "primitives": ["x"]},
    ]
    vocabulary.validate_format(good)  # no raise


def test_vocab_validate_format_rejects_missing_id():
    with pytest.raises(Exception):
        vocabulary.validate_format([{"desc": "d", "lever": "l", "primitives": []}])


def test_vocab_validate_format_rejects_duplicate_id():
    with pytest.raises(Exception):
        vocabulary.validate_format([
            {"id": "a", "desc": "d", "lever": "l", "primitives": []},
            {"id": "a", "desc": "d2", "lever": "l2", "primitives": []},
        ])


def test_vocab_validate_format_rejects_bad_primitives_type():
    with pytest.raises(Exception):
        vocabulary.validate_format([{"id": "a", "desc": "d", "lever": "l", "primitives": "not-a-list"}])


# ---------------- consistency script ----------------

def test_consistency_passes_on_real_vocab():
    assert check_vocab_consistency.main() == 0


def test_consistency_fails_on_malformed_vocab(tmp_path):
    bad = tmp_path / "vocabulary.yaml"
    bad.write_text("- id: a\n  desc: d\n  lever: l\n  primitives: []\n"
                   "- id: a\n  desc: dup\n  lever: l\n  primitives: []\n",
                   encoding="utf-8")
    assert check_vocab_consistency.main(str(bad)) != 0
