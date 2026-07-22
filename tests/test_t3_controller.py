"""T3 gate: loop-controller 停止决策（全量确定性实现）。

Three required cases (plan §3 T3) + extra coverage of the other §3.5 stop
conditions, so the gate is not self-proving:
  1. 每轮仅 1% 改善 → 在 ε 处停，而非跑满 N
  2. 第 3 轮 cycles 变大 → 回滚，最终返回历史最优那版
  3. 连续 correct=false → 不参与改善计算但计入轮数、最终达上限停止
  + max_rounds cap, irreducible, variant ping-pong oscillation, env-var config,
    best-so-far preserved across a late regression.
"""
import pytest

from control import contracts
from control.loop_controller import (
    LoopController, Decision, BestVariant, StopReason,
)

BN = "compute_bound_at_peak"


def _sim(correct, cycles):
    return contracts.SimResult(
        correct=correct, max_abs_err=0.0 if correct else 9.9,
        cycles=cycles, pipeline={},
    )


def _verdict(bottleneck=BN, cycles=0):
    return contracts.Verdict(
        bottleneck=bottleneck, lever="some lever", cycles=cycles, expected_gain=0.1,
    )


# -------- case 1: ε stop --------

def test_epsilon_stop_before_max_rounds():
    ctrl = LoopController(epsilon=0.05, max_rounds=10)
    r1 = ctrl.update("v1", _sim(True, 100), _verdict(cycles=100))
    assert not r1.should_stop, "first best must not stop"
    r2 = ctrl.update("v2", _sim(True, 99), _verdict(cycles=99))  # 1% gain < 5%
    assert r2.should_stop
    assert r2.reason == StopReason.EPSILON
    assert r2.best.cycles == 99
    assert ctrl.round_count == 2, "must stop early, not run to max_rounds=10"


# -------- case 2: regression rollback + historical best --------

def test_regression_rollback_returns_historical_best():
    ctrl = LoopController(epsilon=0.05, max_rounds=10)
    ctrl.update("v1", _sim(True, 100), _verdict(cycles=100))   # best=100
    r2 = ctrl.update("v2", _sim(True, 90), _verdict(cycles=90))  # 10% gain >= 5%
    assert not r2.should_stop
    r3 = ctrl.update("v3", _sim(True, 200), _verdict(cycles=200))  # regression
    assert r3.rolled_back, "round 3 regressed -> rollback flagged"
    assert not r3.should_stop, "one non-improving round alone must not stop"
    assert r3.best.cycles == 90, "best stays at historical best, not 200"
    # drive to a stop and confirm the FINAL returned variant is still the historical best
    r4 = ctrl.update("v3b", _sim(True, 195), _verdict(cycles=195))  # still no improvement
    assert r4.should_stop
    assert r4.best.cycles == 90


# -------- case 3: FAIL rounds count toward N, stop at cap --------

def test_consecutive_fail_counts_toward_rounds():
    ctrl = LoopController(epsilon=0.05, max_rounds=3)
    r1 = ctrl.update("v1", _sim(False, None), None)
    r2 = ctrl.update("v2", _sim(False, None), None)
    assert not r1.should_stop and not r2.should_stop, "FAILs must not stop early by themselves"
    r3 = ctrl.update("v3", _sim(False, None), None)
    assert r3.should_stop
    assert r3.reason == StopReason.MAX_ROUNDS, "FAILs counted toward round budget -> stop at N"
    assert ctrl.round_count == 3
    assert r3.best is None, "never passed -> no best"


# -------- extra coverage --------

def test_max_rounds_cap_when_still_improving_above_epsilon():
    ctrl = LoopController(epsilon=0.001, max_rounds=3)  # tiny ε -> never ε-stops
    ctrl.update("v1", _sim(True, 100), _verdict(cycles=100))
    ctrl.update("v2", _sim(True, 90), _verdict(cycles=90))   # 10% >> ε
    r3 = ctrl.update("v3", _sim(True, 80), _verdict(cycles=80))  # still big gains
    assert r3.should_stop
    assert r3.reason == StopReason.MAX_ROUNDS
    assert r3.best.cycles == 80


def test_irreducible_signal_stops():
    ctrl = LoopController(epsilon=0.05, max_rounds=10)
    ctrl.update("v1", _sim(True, 100), _verdict(cycles=100))
    r2 = ctrl.update("v2", _sim(True, 60), _verdict(cycles=60), irreducible=True)
    assert r2.should_stop
    assert r2.reason == StopReason.IRREDUCIBLE


def test_variant_pingpong_oscillation():
    ctrl = LoopController(epsilon=0.001, max_rounds=10)
    # alternating two variants with alternating bottlenecks (so bottleneck-stuck
    # can't fire); none beats best after round 1 -> A,B,A ping-pong
    ctrl.update("A", _sim(True, 100), _verdict(bottleneck="compute_bound_at_peak", cycles=100))
    ctrl.update("B", _sim(True, 100), _verdict(bottleneck="memory_underfilled", cycles=100))
    r3 = ctrl.update("A", _sim(True, 100), _verdict(bottleneck="compute_bound_at_peak", cycles=100))
    assert r3.should_stop
    assert r3.reason == StopReason.OSCILLATION


def test_numerical_fail_only_on_explicit_regen_failed():
    # §3.5 #5: FAIL + regen-still-fails -> stop; but plain consecutive FAILs (test 3) must NOT.
    ctrl = LoopController(epsilon=0.05, max_rounds=10)
    r1 = ctrl.update("v1", _sim(False, None), None)
    assert not r1.should_stop
    r2 = ctrl.update("v1b", _sim(False, None), None, regen_failed=True)
    assert r2.should_stop
    assert r2.reason == StopReason.NUMERICAL_FAIL


def test_config_via_env_var(monkeypatch):
    monkeypatch.setenv("LOOP_MAX_ROUNDS", "2")
    monkeypatch.setenv("LOOP_EPSILON", "0.2")
    ctrl = LoopController()
    assert ctrl.max_rounds == 2
    assert ctrl.epsilon == 0.2


def test_best_is_min_cycles_over_pass_rounds():
    ctrl = LoopController(epsilon=0.001, max_rounds=10)
    ctrl.update("v1", _sim(True, 100), _verdict(cycles=100))
    ctrl.update("v2", _sim(True, 70), _verdict(cycles=70))   # best=70
    ctrl.update("v3", _sim(True, 120), _verdict(cycles=120))  # worse
    r4 = ctrl.update("v4", _sim(True, 80), _verdict(cycles=80))  # worse than 70
    assert ctrl.best.cycles == 70
    assert r4.best.cycles == 70
