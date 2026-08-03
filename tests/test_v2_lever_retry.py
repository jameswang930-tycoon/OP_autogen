"""V2-P1 回归：choose_lever 超时/失败按 lever_retries 重试，耗尽则回退 vocabulary lever，
不让一次子进程超时炸掉整个 run。"""
import control.orchestrator as orch
from control import vocabulary
from control.contracts import Verdict
from control.job_spec import Budget, NormalizedJob
from control.llm_backend import LLMTimeout


def test_choose_lever_timeout_falls_back_to_vocabulary(tmp_path, monkeypatch):
    job = NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                        baseline_src="def b(): pass\n", reference_src=None,
                        has_baseline=True, budget=Budget(max_rounds=1), form="triton_file")
    # 强制多候选 -> 走 choose_lever 分支（签名 P2 后带 strict，用 *a/**k 前向兼容）
    monkeypatch.setattr(orch, "_relevant_entries",
                        lambda *a, **k: [{"name": "p1"}, {"name": "p2"}])

    class LLM:
        last_model_echo = None
        def generate(self, p): raise AssertionError("generate 不应被调用")
        def choose_lever(self, p): raise LLMTimeout("timeout")

    o = orch.Orchestrator(job, llm=LLM(), output_dir=tmp_path / "out")
    verdict = Verdict(bottleneck="compute_bound_at_peak", lever="x",
                      cycles=100, expected_gain=0.1)
    lever, ext = o._resolve_lever(verdict)

    assert lever == vocabulary.lever_for("compute_bound_at_peak"), "耗尽应回退 vocabulary lever"
    assert ext is None


def test_choose_lever_succeeds_after_retries(tmp_path, monkeypatch):
    """前两次超时、第三次成功：应返回第三次的 lever，不回退。"""
    job = NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                        baseline_src="def b(): pass\n", reference_src=None,
                        has_baseline=True, budget=Budget(max_rounds=1, lever_retries=3),
                        form="triton_file")
    monkeypatch.setattr(orch, "_relevant_entries",
                        lambda *a, **k: [{"name": "p1"}, {"name": "p2"}])

    class LLM:
        last_model_echo = None
        def __init__(self): self.n = 0
        def generate(self, p): raise AssertionError("generate 不应被调用")
        def choose_lever(self, p):
            self.n += 1
            if self.n < 3:
                raise LLMTimeout("timeout")
            return '```json\n{"lever": "p1", "rationale": "r"}\n```'

    o = orch.Orchestrator(job, llm=LLM(), output_dir=tmp_path / "out")
    verdict = Verdict(bottleneck="compute_bound_at_peak", lever="x",
                      cycles=100, expected_gain=0.1)
    lever, ext = o._resolve_lever(verdict)
    assert lever == "p1"
    assert ext == "p1"
