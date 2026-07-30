"""GLM52 优化指导 P2 回归：memory 两个 bug 的编排器接线修复（一 bug 一测，互不掩盖）。

Bug1（fingerprint key 错位）：_retrieve_experience 旧用 bottleneck=None → key 落到
    "op|unknown"，与 record 存的 "op|<真实瓶颈>" 错位，by_key 永远 miss、全靠 op_kind 回退，
    丢掉瓶颈区分度。修复后 retrieve 用上一轮 verdict 的已知瓶颈。
Bug2（retrieved_ids 硬编码 []）：record_attempt 旧传 retrieved_ids=[] → store.bump 永不
    更新 used/helped → 经验分数恒为初始值、好坏不分。修复后把 retrieve 实际命中的 ID 回传。

retrieve 的语义（memory/retrieve.py）：先 by_key 精确匹配，**不足 n 条才** by_op_kind 回退，
    再按分数取 top-n。故 by_key 命中 ≥n 时根本不回退——瓶颈区分度正体现在这里。

- test_bug2_retrieved_ids_thread_to_record：1 条经验 / 1 轮。round1 靠 op_kind 回退必捞到
  它；只有把命中 ID 回传 record（Bug2 修复）才会 used+1。Bug2 回归（retrieved_ids=[]）则 used 恒 0。
- test_bug1_retrieve_uses_known_bottleneck：3 条同瓶颈经验（≥n → round2 不回退）+ 1 条高分
  异瓶颈经验。round2 用已知瓶颈 by_key 命中 3 条、不回退 → 异瓶颈那条被排除；Bug1 回归
  （用 None）则 by_key 落空、回退把高分异瓶颈那条捞进 top3。
"""
from pathlib import Path

from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator
from memory import ExperienceStore, Fingerprint, RunLog
from memory.schema import Experience

REPO = Path(__file__).resolve().parent.parent

# extension_used=null：memory 测试不依赖 extension 速查表内容（避免与 P3 的 sample 条目耦合）。
GEN_RESP = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
            '```json\n{"lever": null, "extension_used": null, "notes": "x"}\n```')


class _LLM:
    def generate(self, p): return GEN_RESP
    def choose_lever(self, p): return ""


class _Launcher:
    """first call = baseline (iff has_baseline); then per-round correct results."""

    def __init__(self, *, has_baseline, round_cycles=8000):
        self.has_baseline = has_baseline
        self.round_cycles = round_cycles
        self._baseline_done = False

    def __call__(self, path):
        if self.has_baseline and not self._baseline_done:
            self._baseline_done = True
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000,
                    "pipeline": {}, "compiled": True, "compile_log": ""}
        return {"correct": True, "max_abs_err": 0.0, "cycles": self.round_cycles,
                "pipeline": {}, "compiled": True, "compile_log": ""}


def _job(max_rounds, has_baseline=True):
    return NormalizedJob(
        op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
        baseline_src="def baseline(): pass\n" if has_baseline else None,
        reference_src=None, has_baseline=has_baseline,
        budget=Budget(max_rounds=max_rounds, epsilon=0.03),
        form="triton_file" if has_baseline else "shape_only",
    )


def test_bug2_retrieved_ids_thread_to_record(tmp_path):
    """Bug2：retrieve 命中的 ID 必须回传 record_attempt，分数才会迭代。"""
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    fp = Fingerprint("matmul", "compute_bound_at_peak")
    eid = store.add(Experience(text="tip", applies_to=fp.key()))

    Orchestrator(_job(max_rounds=1), llm=_LLM(), launcher=_Launcher(has_baseline=True),
                 parse_raw_fn=lambda raw: COMPUTE_BOUND,
                 store=store, log=log, output_dir=tmp_path / "out").run()

    assert store.get(eid).used > 0, (
        "经验被检索到就应 used+1；若 record_attempt 仍传 retrieved_ids=[]（Bug2），used 恒 0"
    )


def test_bug1_retrieve_uses_known_bottleneck(tmp_path):
    """Bug1：round2 retrieve 用已知瓶颈 by_key，命中 ≥n 时不回退，排除异瓶颈经验。"""
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    fp_comp = Fingerprint("matmul", "compute_bound_at_peak")
    fp_mem = Fingerprint("matmul", "memory_underfilled")
    # 5 条同瓶颈（compute）经验：≥ retrieve 的 n(=3) → round2 by_key 命中后不再回退。
    # 取 5（而非恰好 n）留余量：即便 n 上调也不会误触发回退、把高分 memory 经验捞进 top3。
    for i in range(5):
        store.add(Experience(text=f"compute tip {i}", applies_to=fp_comp.key()))
    # 1 条高分异瓶颈（memory）经验：只有回退才会被捞进 top3。
    other_id = store.add(Experience(
        text="mem tip", applies_to=fp_mem.key(), helped=10, used=0))

    Orchestrator(_job(max_rounds=2), llm=_LLM(), launcher=_Launcher(has_baseline=True),
                 parse_raw_fn=lambda raw: COMPUTE_BOUND,
                 store=store, log=log, output_dir=tmp_path / "out").run()

    # 末条 runlog 记录即 round2 的 AttemptRecord；其 retrieved 不应含异瓶颈经验。
    round2_retrieved = log.read_all()[-1]["retrieved"]
    assert other_id not in round2_retrieved, (
        "round2 应按 compute_bound 瓶颈 by_key 检索、不回退；若 retrieve 仍用 bottleneck=None"
        "（Bug1），by_key 落空、回退会把高分 memory 经验捞进 top3"
    )
