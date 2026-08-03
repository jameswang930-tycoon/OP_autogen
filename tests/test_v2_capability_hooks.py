"""V2 预埋能力接口回归：optimization 知识 + 单元能力上限/利用率感知（骨架，降级兼容）。

mock 验证五条路径（不接真实 agent/数据）：
  1. opt-* skill 骨架可加载、references/ 空目录降级（无内容不报错）+ 瓶颈→skill 映射；
  2. memory opt_technique_ref 读（format_context 渲染）/写（record_attempt 落库）；缺省降级；
  3. gen prompt OPTIMIZATION_HINT：有瓶颈→填充、无瓶颈/无匹配→空（降级）；
  4. Event.unit_peak 缺省 → classify 纯占比（与现状一致）；
  5. Event.unit_peak + saturation_threshold → 利用率分支：饱和类标约束、改选未饱和类作真瓶颈；
     无阈值 → 即便有 unit_peak 也降级纯占比。
"""
from pathlib import Path

from control.contracts import Event
from control.feedback_adapter import classify, reduce_events
from control.orchestrator import _optimization_hint, _optimization_skill_for
from memory import ExperienceStore, Fingerprint, RunLog, add_experience, record_attempt
from memory.retrieve import format_context

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".claude" / "skills"


# ---- 1. optimization skill 骨架 + 空目录降级 ----
def test_opt_skills_skeleton_and_mapping():
    for s in ("opt-compute-bound", "opt-memory-bound", "opt-stall-dependency"):
        assert (SKILLS / s / "SKILL.md").is_file(), f"{s} 缺 SKILL.md"
        assert (SKILLS / s / "references").is_dir(), f"{s} 缺 references/（空目录待环境侧填）"
    assert _optimization_skill_for("compute_bound_at_peak") == "opt-compute-bound"
    assert _optimization_skill_for("memory_underfilled") == "opt-memory-bound"
    assert _optimization_skill_for("stall_dependency") == "opt-stall-dependency"
    assert _optimization_skill_for("nonexistent") is None


# ---- 2. memory opt_technique_ref 读写（+ 缺省降级）----
def test_opt_technique_ref_written_and_rendered(tmp_path):
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    fp = Fingerprint("matmul", "memory_underfilled")
    eid = add_experience(store, fp, text="use async copy", opt_technique_ref="async_bulk_copy")
    ctx = format_context([store.get(eid)])
    assert "关联优化技巧" in ctx and "async_bulk_copy" in ctx
    record_attempt(log, store, fp, [eid], passed=True, cycles=100, opt_technique_ref="async_bulk_copy")
    assert log.read_all()[-1]["opt_technique_ref"] == "async_bulk_copy"


def test_opt_technique_ref_absent_degrades(tmp_path):
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    fp = Fingerprint("matmul", "compute_bound_at_peak")
    eid = add_experience(store, fp, text="plain tip")  # 不传 opt_technique_ref
    assert "关联优化技巧" not in format_context([store.get(eid)])
    record_attempt(log, store, fp, [eid], passed=True, cycles=100)
    assert log.read_all()[-1]["opt_technique_ref"] is None


# ---- 3. gen prompt OPTIMIZATION_HINT 段填充/降级 ----
def test_optimization_hint_filled_or_empty():
    assert "opt-memory-bound" in _optimization_hint("matmul", "memory_underfilled")
    assert _optimization_hint("matmul", None) == ""          # 无瓶颈→空
    assert _optimization_hint("matmul", "nonexistent") == ""  # 无匹配 skill→空


# ---- 4. unit_peak 缺省 → classify 纯占比 ----
def test_classify_pure_ratio_when_no_unit_peak():
    events = [
        Event("c1", 0, 100, 100, "COMPUTE", "compute_bound_at_peak"),
        Event("m1", 0, 30, 30, "MEMORY", "memory_underfilled", bytes=1024),
    ]
    c = classify(reduce_events(events))
    assert c.bottleneck == "compute_bound_at_peak"   # 纯占比：compute(100) > memory(30)
    assert c.constraints == []


# ---- 5. unit_peak + 阈值 → 利用率分支 ----
def _ev(name, unit, cls, dur, bytes_, peak, start=0):
    return Event(name, start, start + dur, dur, unit, cls, bytes=bytes_, unit_peak=peak)


def test_classify_utilization_marks_saturated_and_picks_unsaturated():
    # memory 占比大但已饱和(util=1.0≥0.9)；compute 占比小未饱和。
    # 纯占比会选 memory；利用率分支应把 memory 标约束、改选 compute 作真瓶颈。
    events = [
        _ev("m1", "MEMORY", "memory_underfilled", 100, 10000, 100.0),  # util=10000/(100·100)=1.0
        _ev("c1", "COMPUTE", "compute_bound_at_peak", 40, None, None),
    ]
    c = classify(reduce_events(events), saturation_threshold=0.9)
    assert "memory_underfilled" in c.constraints
    assert c.bottleneck == "compute_bound_at_peak"


def test_classify_utilization_keeps_unsaturated_top():
    # memory 占比大且未饱和(util=0.1) → 仍是真瓶颈，不被误标约束。
    events = [
        _ev("m1", "MEMORY", "memory_underfilled", 100, 1000, 100.0),  # util=1000/10000=0.1
        _ev("c1", "COMPUTE", "compute_bound_at_peak", 40, None, None),
    ]
    c = classify(reduce_events(events), saturation_threshold=0.9)
    assert c.bottleneck == "memory_underfilled"
    assert c.constraints == []


def test_classify_no_threshold_degrades_even_with_unit_peak():
    """无 saturation_threshold → 即便有 unit_peak 也降级纯占比。"""
    events = [
        _ev("m1", "MEMORY", "memory_underfilled", 100, 10000, 100.0),
        _ev("c1", "COMPUTE", "compute_bound_at_peak", 40, None, None),
    ]
    c = classify(reduce_events(events))   # 默认无阈值
    assert c.bottleneck == "memory_underfilled"   # 纯占比选最大
    assert c.constraints == []
