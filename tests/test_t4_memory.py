"""T4 gate: memory 演进（价值信号从「正确性」转向「性能」）。

Covers plan T4 acceptance:
  - 同一 fingerprint 连续三次尝试（第二次刷新最优、第三次未刷新）→ score() 只在第二次上升
  - correct=false 的尝试不影响 score 但出现在 runlog
Plus schema/best-cycles/extension_used/persistence coverage.
"""
import pytest

from memory import (
    AttemptRecord, Experience, ExperienceStore, Fingerprint, RunLog,
    add_experience, record_attempt,
)

FP = Fingerprint(op_kind="matmul", bottleneck="compute_bound_at_peak")


def _store(tmp_path):
    return ExperienceStore(tmp_path / "experience.json")


def _log(tmp_path):
    return RunLog(tmp_path / "runlog.jsonl")


# -------- case A: score rises only when historical best is refreshed --------

def test_score_rises_only_on_best_refresh(tmp_path):
    store, log = _store(tmp_path), _log(tmp_path)
    eid = add_experience(store, FP, text="tiling helps bandwidth")

    # round 1: pass, cycles=100 — first measurement, no prior best to beat
    record_attempt(log, store, FP, [eid], passed=True, cycles=100)
    e1 = store.get(eid)
    assert e1.helped == 0, "first round establishes baseline, must not bump helped"
    assert e1.used == 1
    s1 = e1.score()

    # round 2: pass, cycles=90 — refreshes best -> helped bumps, score rises
    record_attempt(log, store, FP, [eid], passed=True, cycles=90)
    e2 = store.get(eid)
    assert e2.helped == 1
    s2 = e2.score()
    assert s2 > s1, "score must rise on the best-refreshing (2nd) round"

    # round 3: pass, cycles=95 — does NOT beat best -> no helped bump, score drops
    record_attempt(log, store, FP, [eid], passed=True, cycles=95)
    e3 = store.get(eid)
    assert e3.helped == 1, "non-improving round must not bump helped"
    s3 = e3.score()
    assert s3 < s2, "score must not rise on a non-improving (3rd) round"

    # cycles recorded into the runlog
    entries = log.read_all()
    assert [e["cycles"] for e in entries] == [100, 90, 95]


# -------- case B: FAIL does not affect score but is logged --------

def test_fail_does_not_affect_score_but_logged(tmp_path):
    store, log = _store(tmp_path), _log(tmp_path)
    eid = add_experience(store, FP, text="x")
    record_attempt(log, store, FP, [eid], passed=True, cycles=100)
    before = store.get(eid)
    score_before, used_before, helped_before = before.score(), before.used, before.helped

    record_attempt(log, store, FP, [eid], passed=False)  # FAIL round, no cycles

    after = store.get(eid)
    assert after.used == used_before
    assert after.helped == helped_before
    assert after.score() == score_before, "FAIL must not affect score"
    assert after.failed == 1, "FAIL recorded as neutral-negative"

    entries = log.read_all()
    assert len(entries) == 2
    assert entries[-1]["passed"] is False, "FAIL must appear in runlog"


# -------- schema: cycles replaces latency_us; extension_used added --------

def test_attempt_record_uses_cycles_not_latency():
    rec = AttemptRecord(fingerprint="k", retrieved=[], passed=True, cycles=100)
    assert rec.cycles == 100
    assert "latency_us" not in AttemptRecord.__dataclass_fields__
    with pytest.raises(TypeError):
        AttemptRecord(fingerprint="k", retrieved=[], passed=True, latency_us=100)


def test_experience_carries_extension_used(tmp_path):
    store, log = _store(tmp_path), _log(tmp_path)
    fp = Fingerprint(op_kind="matmul", bottleneck="memory_underfilled")
    eid = add_experience(store, fp, text="use bulk copy", extension_used="bulk_copy")
    assert store.get(eid).extension_used == "bulk_copy"

    rec = record_attempt(
        log, store, fp, [eid], passed=True, cycles=100, extension_used="bulk_copy"
    )
    assert rec.extension_used == "bulk_copy"
    assert log.read_all()[-1]["extension_used"] == "bulk_copy"


# -------- per-fingerprint best-cycles tracking + persistence --------

def test_best_cycles_tracking(tmp_path):
    store = _store(tmp_path)
    k = "matmul|compute_bound_at_peak"
    assert store.update_best(k, 100) is False   # baseline
    assert store.update_best(k, 90) is True     # improved
    assert store.update_best(k, 95) is False    # not improved
    assert store.update_best(k, 90) is False    # equal, not strictly better
    assert store.best_cycles_for(k) == 90


def test_best_cycles_persists_across_reload(tmp_path):
    p = tmp_path / "experience.json"
    s1 = ExperienceStore(p)
    s1.update_best("matmul|compute_bound_at_peak", 100)
    s1.update_best("matmul|compute_bound_at_peak", 80)
    s2 = ExperienceStore(p)  # reload from disk
    assert s2.best_cycles_for("matmul|compute_bound_at_peak") == 80


def test_best_cycles_isolated_per_fingerprint(tmp_path):
    store = _store(tmp_path)
    store.update_best("opA|compute_bound_at_peak", 100)
    store.update_best("opB|memory_underfilled", 50)
    assert store.best_cycles_for("opA|compute_bound_at_peak") == 100
    assert store.best_cycles_for("opB|memory_underfilled") == 50
