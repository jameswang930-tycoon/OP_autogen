"""V2 backend 重构回归：统一 NgaBackend（配置驱动 + 可扩展，**无哑后端/双模式**）。

mock 验证三条路径（不接真实 agent）：
  1. 配置→命令行映射：options dict→flags（bool/scalar/None）、extra_args 透传；
  2. 结构化输出解析：config 声明结构化能力→agent 返 json→规范成 fenced block→orchestrator 解析；
  3. 降级：什么都不配→仅 cmd+prompt 的最基础调用，不报错。
另验 gen prompt 统一精简（场景提示、不塞全量 index）——双模式已删，唯一路径。
"""
import json
from types import SimpleNamespace

from control.fixtures import COMPUTE_BOUND
from control.job_spec import Budget, NormalizedJob
from control.llm_backend import NgaBackend
from control.orchestrator import Orchestrator, parse_generate_response


class FakeRunner:
    def __init__(self, stdout="", returncode=0, exc=None):
        self.stdout = stdout
        self.returncode = returncode
        self.exc = exc
        self.calls = []

    def __call__(self, cmd, timeout=None):
        self.calls.append(list(cmd))
        if self.exc is not None:
            raise self.exc
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


# ---- 1. 配置→命令行映射 ----
def test_config_to_cli_mapping_options_and_extra_args():
    r = FakeRunner(stdout="out")
    cfg = {"cmd": ["agent", "run"],
           "generate": {"model": "m1",
                        "options": {"thinking": True, "off": False,
                                    "fmt": "json", "skip": None},
                        "extra_args": ["--foo", "bar"]}}
    NgaBackend(cfg, runner=r).generate("PROMPT")
    cmd = r.calls[0]
    assert cmd[:2] == ["agent", "run"]               # 命令前缀
    assert "--model" in cmd and "m1" in cmd
    assert "--thinking" in cmd                        # bool True → flag
    assert "--off" not in cmd and "--skip" not in cmd  # False/None → 省略
    assert "--fmt" in cmd and "json" in cmd           # scalar → --fmt json
    assert "--foo" in cmd and "bar" in cmd            # extra_args 原样透传
    assert cmd[-1] == "PROMPT"                        # prompt 位置最后
    assert "--continue" not in cmd and "--session" not in cmd  # 无状态


# ---- 2. 结构化输出解析 ----
def test_structured_output_normalized_to_fenced_blocks():
    agent_json = json.dumps({"kernel": "def kernel(a, b, c):\n    return a @ b",
                             "meta": {"lever": "memory_underfilled", "extension_used": None,
                                      "notes": "ok"}})
    r = FakeRunner(stdout=agent_json)
    cfg = {"generate": {"model": "m"},
           "structured": {"enabled": True, "request": {"output_format": "json"},
                          "kernel_key": "kernel", "meta_key": "meta"}}
    out = NgaBackend(cfg, runner=r).generate("p")
    assert "--output-format" in r.calls[0]            # 请求侧：结构化选项进了命令行
    py, meta = parse_generate_response(out, allowed_extensions=set())  # 响应侧：json→fenced，可解析
    assert "def kernel" in py
    assert meta["notes"] == "ok"


def test_structured_parse_failure_degrades_to_free_text():
    """agent 返回非合法 json → 降级原样返回（自由文本解析），不报错。"""
    r = FakeRunner(stdout="not json at all")
    cfg = {"generate": {"model": "m"}, "structured": {"enabled": True}}
    out = NgaBackend(cfg, runner=r).generate("p")
    assert out == "not json at all"


# ---- 3. 降级：什么都不配 ----
def test_degrade_minimal_config_basic_call(monkeypatch):
    monkeypatch.delenv("AGENT_CMD", raising=False)
    r = FakeRunner(stdout="ok")
    NgaBackend(runner=r).generate("p")                # 完全无 config
    cmd = r.calls[0]
    assert cmd == ["agent", "run", "p"]               # 仅默认前缀 + prompt，无 model/options
    assert "--model" not in cmd


# ---- gen prompt 统一精简（场景提示，不塞全量 index）----
GEN = ('```python\ndef kernel(a, b, c):\n    return a @ b\n```\n'
       '```json\n{"lever": "memory_underfilled", "extension_used": null, "notes": "x"}\n```')


def _job():
    return NormalizedJob(op="matmul", shapes=[1024, 1024, 1024], dtype="fp16",
                         baseline_src="def baseline(): pass\n", reference_src=None,
                         has_baseline=True, budget=Budget(max_rounds=1), form="triton_file")


def test_gen_prompt_is_concise_scene_hint_no_full_index(tmp_path):
    """双模式已删——gen prompt 唯一路径：场景提示、不塞全量 extension index。"""
    prompts = []

    class LLM:
        def generate(self, p):
            prompts.append(p)
            return GEN
        def choose_lever(self, p):
            return ""

    class Launcher:
        def __init__(self): self.c = 0
        def __call__(self, path):
            self.c += 1
            return {"correct": True, "max_abs_err": 0.0,
                    "cycles": 12000 if self.c == 1 else 8000,
                    "pipeline": {}, "compiled": True, "compile_log": ""}

    Orchestrator(_job(), llm=LLM(), launcher=Launcher(),
                 parse_raw_fn=lambda raw: COMPUTE_BOUND, output_dir=tmp_path / "out").run()
    assert prompts
    p = prompts[0]
    assert "sample_async_copy_template" not in p, "不再塞全量 extension index"
    assert "ext-* skill" in p, "应给场景提示（触发 ext-* skill lazy-load）"
