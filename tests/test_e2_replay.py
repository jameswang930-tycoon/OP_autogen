"""E2 gate: 单轮重放入口（replay）——只读，不发射/不调 LLM/不写 outputs。

每个确定性组件有命令行重放入口；用 fixture 输入跑通；坏输入打印清晰错误而非静默。
"""
import json

from control import feedback_adapter as fa
from control import launch_template as lt
from control import loop_controller as lc
from control.fixtures import COMPUTE_BOUND


def _write_events(tmp_path):
    ev = [{
        "name": e.name, "start": e.start, "end": e.end, "duration": e.duration,
        "unit": e.unit, "stall_class": e.stall_class, "bytes": e.bytes,
    } for e in COMPUTE_BOUND]
    p = tmp_path / "events.json"
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_feedback_adapter_adapt_only(tmp_path, capsys):
    p = _write_events(tmp_path)
    rc = fa.main(["adapt-only", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "compute_bound_at_peak" in out  # the verdict bottleneck


def test_feedback_adapter_replay_with_mocked_parse_raw(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(fa, "parse_raw", lambda raw: COMPUTE_BOUND)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"correct": True, "cycles": 100}), encoding="utf-8")
    rc = fa.main(["replay", str(raw)])
    assert rc == 0
    assert "mac_0" in capsys.readouterr().out  # an event name from the fixture


def test_launch_template_assemble(tmp_path, capsys):
    k = tmp_path / "k.py"
    k.write_text("def kernel(a, b, c):\n    return a @ b\n", encoding="utf-8")
    rc = lt.main(["assemble", str(k)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "def kernel" in out  # kernel body embedded in assembled launchable


def test_loop_controller_replay(tmp_path, capsys):
    hist = [
        {"variant": "v1", "correct": True, "cycles": 100, "bottleneck": "compute_bound_at_peak",
         "lever": "l", "expected_gain": 0.1},
        {"variant": "v2", "correct": True, "cycles": 90, "bottleneck": "compute_bound_at_peak",
         "lever": "l", "expected_gain": 0.1},
    ]
    p = tmp_path / "hist.json"
    p.write_text(json.dumps(hist), encoding="utf-8")
    rc = lc.main(["replay", str(p)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "round 1" in out and "round 2" in out


def test_replay_bad_input_prints_clear_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {{", encoding="utf-8")
    rc = fa.main(["adapt-only", str(bad)])
    assert rc != 0
    err = capsys.readouterr().err
    assert err.strip(), "bad input must print a clear error, not fail silently"
