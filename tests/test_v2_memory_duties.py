"""V2-P4 回归：memory 三职责（真跑，不只 pytest 绿）。

职责②：迭代基准用 best-so-far kernel——round1 prompt 含原始 baseline 标记、不含 round
产物标记；round2 prompt 换成 round1 产物标记、不含原始 baseline 标记。
职责③：避坑清单——预置 helped=0/failed=2/extension_used=tlext_bad 的经验，run 后某轮 prompt
含 `tlext_bad`。
CapturingLLM 记录每轮 gen prompt，直接断言 prompt 内容（证明真生效）。
"""
from pathlib import Path

from control import vocabulary
from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator
from memory import ExperienceStore, Fingerprint, RunLog
from memory.schema import Experience

# baseline 与每轮 kernel 各放一个唯一锚点，据此判断迭代基准换了没
BASELINE_SRC = "def baseline():\n    return None  # BASELINE_ANCHOR\n"
GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b  # ROUND_KERNEL_ANCHOR\n```\n'
       '```json\n{"lever": "memory_underfilled", "extension_used": null, "notes": "x"}\n```')


class CapturingLLM:
    def __init__(self): self.prompts = []
    def generate(self, prompt):
        self.prompts.append(prompt)
        return GEN
    def choose_lever(self, prompt): return ""


class Launcher:
    """baseline=12000；round1=8000（刷新 best）；round2=7900（再刷新）。best 持续推进。"""
    def __init__(self): self.c = 0
    def __call__(self, path):
        self.c += 1
        if self.c == 1:
            return {"correct": True, "max_abs_err": 0.0, "cycles": 12000,
                    "pipeline": {}, "compiled": True, "compile_log": ""}
        cyc = [8000, 7900][min(self.c - 2, 1)]
        return {"correct": True, "max_abs_err": 0.0, "cycles": cyc,
                "pipeline": {}, "compiled": True, "compile_log": ""}


def _job():
    return NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                         baseline_src=BASELINE_SRC, reference_src=None, has_baseline=True,
                         budget=Budget(max_rounds=2, epsilon=0.03), form="triton_file")


def test_duty2_iterates_on_best_so_far_not_original_baseline(tmp_path):
    llm = CapturingLLM()
    Orchestrator(_job(), llm=llm, launcher=Launcher(),
                 parse_raw_fn=lambda r: COMPUTE_BOUND, output_dir=tmp_path / "out").run()
    assert len(llm.prompts) >= 2, f"应至少跑 2 轮，实得 {len(llm.prompts)}"
    p1, p2 = llm.prompts[0], llm.prompts[1]
    # round1：迭代基准 = 原始 baseline
    assert "BASELINE_ANCHOR" in p1 and "ROUND_KERNEL_ANCHOR" not in p1, \
        "round1 应以原始 baseline 为迭代基准"
    # round2：迭代基准 = round1 产物（best-so-far），原始 baseline 已被换掉
    assert "ROUND_KERNEL_ANCHOR" in p2 and "BASELINE_ANCHOR" not in p2, \
        "round2 应以 best-so-far（round1 产物）为迭代基准，不再用原始 baseline"


def test_duty3_avoid_list_in_prompt(tmp_path):
    """预置一条失败经验，run 后某轮 prompt 应含该原语名（避坑清单生效）。"""
    store = ExperienceStore(tmp_path / "e.json")
    log = RunLog(tmp_path / "r.jsonl")
    store.add(Experience(text="bad tip", applies_to=Fingerprint("matmul", "compute_bound_at_peak").key(),
                         helped=0, failed=2, extension_used="tlext_bad"))
    llm = CapturingLLM()
    Orchestrator(_job(), llm=llm, launcher=Launcher(), parse_raw_fn=lambda r: COMPUTE_BOUND,
                 store=store, log=log, output_dir=tmp_path / "out").run()
    assert any("tlext_bad" in p for p in llm.prompts), \
        "避坑清单应把曾失败原语 tlext_bad 拼进某轮 gen prompt"
