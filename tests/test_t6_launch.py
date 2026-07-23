"""T6 gate: 发射脚本模板。

Covers plan T6 "各门禁必须覆盖的内容":
  - 用一个本地假 launch（返回夹具数据）跑通全链路 -> 合法 SimResult
  - launch() 保持 NotImplementedError
Plus build_sim_result validation + multi-segment template shape (§1.0).
"""
import pytest

from control import contracts
from control import launch_template as launch


def _raw_pass():
    return {
        "correct": True, "max_abs_err": 1.2e-6, "cycles": 148230,
        "pipeline": {"MTE": 120, "ALU": 80},
        "compiled": True, "compile_log": "",
        "events": [],  # extra key for the adapter's parse_raw; ignored here
    }


def test_launch_is_slot():
    with pytest.raises(NotImplementedError):
        launch.launch("any_kernel.py")


def test_build_sim_result_pass():
    r = launch.build_sim_result(_raw_pass())
    assert isinstance(r, contracts.SimResult)
    assert r.correct is True
    assert r.cycles == 148230
    assert r.max_abs_err == 1.2e-6
    assert r.pipeline == {"MTE": 120, "ALU": 80}


def test_build_sim_result_fail_voids_cycles():
    raw = {"correct": False, "max_abs_err": 9.9, "cycles": None, "pipeline": {},
           "compiled": True, "compile_log": ""}
    r = launch.build_sim_result(raw)
    assert r.correct is False
    assert r.cycles is None  # §3.6: perf voided on FAIL


def test_build_sim_result_rejects_missing_required_field():
    # max_abs_err and compiled are required; cycles is optional (None on FAIL)
    with pytest.raises(Exception):
        launch.build_sim_result({"correct": True, "cycles": 1, "pipeline": {}})


def test_build_sim_result_rejects_bad_correct_type():
    with pytest.raises(Exception):
        launch.build_sim_result({"correct": "yes", "max_abs_err": 0.0, "cycles": 1,
                                 "pipeline": {}, "compiled": True})


def _fake_launcher(kernel_file):
    # a local fake launch returning fixture-shaped raw output
    return _raw_pass()


def test_run_pipeline_with_fake_launcher_produces_simresult():
    r = launch.run("kernel.py", launcher=_fake_launcher)
    assert isinstance(r, contracts.SimResult)
    assert r.correct is True
    assert r.cycles == 148230


def test_run_pipeline_defaults_to_launch_slot():
    # without an injected launcher, run() must delegate to the real (slot) launch()
    with pytest.raises(NotImplementedError):
        launch.run("kernel.py")


def test_template_is_multisegment_with_emit_contract():
    t = launch.LAUNCHABLE_TEMPLATE
    # §1.0: launchable unit = kernel + reference + compare
    for marker in ("kernel", "reference", "compare"):
        assert marker in t.lower(), f"template missing {marker!r} segment"
    # the compare segment must emit the canonical raw_sim_output keys
    for key in ("correct", "max_abs_err", "cycles", "pipeline"):
        assert key in t, f"template emit contract missing key {key!r}"
