"""V2-P5.1 回归：AgentBackend（类 claude-code agent 子进程后端，接口同 NgaBackend）。

公开分支不接真实 agent，用 mock 验证：
  - AgentBackend 类：命令拼接、runner 注入、超时->LLMTimeout、非零退出->LLMInvocationError；
  - FakeAgentBackend（模拟 agent 收精简 prompt → 触发 extension-guide skill → 返回 kernel）
    接到 orchestrator 上，跑通调用链，且 extension-guide skill 目录结构存在。
orchestrator 主体不动——AgentBackend 与 NgaBackend 接口一致（generate/choose_lever -> str）。
"""
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.llm_backend import AgentBackend, LLMInvocationError, LLMTimeout
from control.orchestrator import Orchestrator

REPO = Path(__file__).resolve().parent.parent
SKILLS = REPO / ".claude" / "skills"


# ---------------- AgentBackend 类单测 ----------------

def test_agent_backend_builds_cmd_and_returns_stdout():
    seen = []

    def runner(cmd, *, timeout):
        seen.append(cmd)
        return SimpleNamespace(returncode=0,
                               stdout='> build · agent-m · s\n```python\nk\n```\n```json\n{}\n```',
                               stderr="")

    b = AgentBackend(config={"cmd": ["agent"], "generate": {"model": "m1"}}, runner=runner)
    out = b.generate("do X")
    assert seen[0] == ["agent", "--model", "m1", "do X"]
    assert "```python" in out
    assert b.call_count == 1


def test_agent_backend_timeout_raises():
    def runner(cmd, *, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    b = AgentBackend(config={"cmd": ["agent"]}, runner=runner)
    with pytest.raises(LLMTimeout):
        b.generate("x")


def test_agent_backend_nonzero_raises():
    def runner(cmd, *, timeout):
        return SimpleNamespace(returncode=2, stdout="", stderr="boom")
    b = AgentBackend(config={"cmd": ["agent"]}, runner=runner)
    with pytest.raises(LLMInvocationError):
        b.generate("x")


def test_agent_backend_needs_cmd_config():
    # 无 config 且无 AGENT_CMD env -> 友好报错（env 侧适配点）
    import os
    saved = os.environ.pop("AGENT_CMD", None)
    try:
        # 缺省 cmd=["agent"]，所以不报错；验证缺省命令前缀
        b = AgentBackend(runner=lambda c, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
        assert b.cmd == ["agent"]
    finally:
        if saved is not None:
            os.environ["AGENT_CMD"] = saved


# ---------------- FakeAgentBackend 接 orchestrator（调用链 + skill 目录） ----------------

class FakeAgentBackend:
    """模拟 agent：收到 prompt -> 假装触发 extension-guide skill（验证其存在）-> 返回 kernel。"""
    def __init__(self, skills_dir):
        self.skills_dir = Path(skills_dir)
        self.generate_calls = []

    def generate(self, prompt):
        self.generate_calls.append(prompt)
        # agent 会按 description 隐式触发 extension-guide skill——验证该 skill 目录结构在
        assert (self.skills_dir / "extension-guide" / "SKILL.md").is_file(), \
            "extension-guide skill 应存在（agent 按其 description 触发）"
        return ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
                '```json\n{"lever": "memory_underfilled", "extension_used": null, "notes": "x"}\n```')

    def choose_lever(self, prompt):
        return ""


def _job():
    return NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                         baseline_src="def baseline(): pass\n", reference_src=None,
                         has_baseline=True, budget=Budget(max_rounds=1), form="triton_file")


def test_fake_agent_backend_orchestrator_call_chain(tmp_path):
    """AgentBackend 接口的 mock 接到 orchestrator，跑通一整轮（调用链 + skill 目录）。"""
    agent = FakeAgentBackend(SKILLS)

    class Launcher:
        def __init__(self): self.c = 0
        def __call__(self, p):
            self.c += 1
            if self.c == 1:
                return {"correct": True, "max_abs_err": 0.0, "cycles": 12000,
                        "pipeline": {}, "compiled": True, "compile_log": ""}
            return {"correct": True, "max_abs_err": 0.0, "cycles": 8000,
                    "pipeline": {}, "compiled": True, "compile_log": ""}

    report = Orchestrator(_job(), llm=agent, launcher=Launcher(),
                          parse_raw_fn=lambda r: COMPUTE_BOUND,
                          output_dir=tmp_path / "out").run()
    assert agent.generate_calls, "agent.generate 应被调用（调用链通）"
    assert report["rounds"], "应完成至少一轮"
