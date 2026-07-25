"""E3 gate: 组件边界断言（fail fast with context）。

坏数据在产生它的那一步就被拦下、带可操作的定位信息（哪条/哪个字段/来自哪）。
"""
import pytest

from control import feedback_adapter as fa
from control.contracts import Event, Verdict
from control.feedback_adapter import AdapterOutput, ParseError, validate_events, validate_output
from control.fixtures import COMPUTE_BOUND


def _v():
    return Verdict(bottleneck="compute_bound_at_peak", lever="l", cycles=100, expected_gain=0.1)


# ---- validate_events (parse_raw boundary) ----

def test_event_duration_mismatch_caught_with_index_and_field():
    e = Event(name="weird", start=0, end=100, duration=50, unit="U",
              stall_class="compute_bound_at_peak")
    with pytest.raises(ParseError) as exc:
        validate_events([e])
    msg = str(exc.value)
    assert "Event[0]" in msg and "duration" in msg, f"not actionable: {msg}"


def test_valid_events_pass():
    validate_events(COMPUTE_BOUND)  # fixtures are internally consistent -> no raise


def test_event_index_in_error_points_at_wrong_one():
    good = Event(name="g", start=0, end=10, duration=10, unit="U", stall_class="compute_bound_at_peak")
    bad = Event(name="b", start=0, end=10, duration=99, unit="U", stall_class="compute_bound_at_peak")
    with pytest.raises(ParseError) as exc:
        validate_events([good, bad])
    assert "Event[1]" in str(exc.value)


# ---- validate_output (adapt boundary) ----

def test_empty_summary_rejected():
    out = AdapterOutput(summary="", verdict=_v())
    with pytest.raises(ParseError) as exc:
        validate_output(out)
    assert "summary" in str(exc.value).lower()


def test_valid_output_passes():
    out = AdapterOutput(summary="makespan=...", verdict=_v())
    validate_output(out)  # no raise


# ---- adapt enforces the boundaries end-to-end ----

def test_adapt_rejects_inconsistent_events():
    bad = [Event(name="x", start=0, end=100, duration=50, unit="U",
                 stall_class="compute_bound_at_peak")]
    with pytest.raises(ParseError):
        fa.adapt(bad)
