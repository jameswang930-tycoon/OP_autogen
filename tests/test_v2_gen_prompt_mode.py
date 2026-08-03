"""V2-P5.2 回归：gen prompt 双模式（env GEN_PROMPT_MODE）。

- nga 模式（缺省）：EXTENSION_INDEX 塞按场景检索的原语子集（含 sample 名）。
- agent 模式：EXTENSION_INDEX 只给场景提示、不塞全量 index，原语靠 agent 触发 skill 加载。
用 CapturingLLM 抓每轮 gen prompt 直接断言。"""
from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.orchestrator import Orchestrator

GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": "memory_underfilled", "extension_used": null, "notes": "x"}\n```')


class CapturingLLM:
    def __init__(self): self.prompts = []
    def generate(self, p):
        self.prompts.append(p)
        return GEN
    def choose_lever(self, p): return ""


class Launcher:
    def __init__(self): self.c = 0
    def __call__(self, path):
        self.c += 1
        cyc = 12000 if self.c == 1 else 8000
        return {"correct": True, "max_abs_err": 0.0, "cycles": cyc,
                "pipeline": {}, "compiled": True, "compile_log": ""}


def _job():
    return NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                         baseline_src="def baseline(): pass\n", reference_src=None,
                         has_baseline=True, budget=Budget(max_rounds=1), form="triton_file")


def _run(tmp_path):
    llm = CapturingLLM()
    Orchestrator(_job(), llm=llm, launcher=Launcher(),
                 parse_raw_fn=lambda r: COMPUTE_BOUND, output_dir=tmp_path / "out").run()
    return llm.prompts[0]


def test_nga_mode_includes_full_index(tmp_path):
    prompts = [_run(tmp_path)]  # default mode
    assert any("sample_async_copy_template" in p for p in prompts), \
        "nga 模式应把按场景检索的原语子集（含 sample 名）塞进 EXTENSION_INDEX"


def test_agent_mode_only_scene_hint_no_full_index(tmp_path, monkeypatch):
    monkeypatch.setenv("GEN_PROMPT_MODE", "agent")
    p = _run(tmp_path)
    assert "sample_async_copy_template" not in p, "agent 模式不应塞全量 index"
    assert "extension-guide skill" in p and "matmul" in p, \
        "agent 模式应给场景提示（算子 + 查 extension-guide skill）"
