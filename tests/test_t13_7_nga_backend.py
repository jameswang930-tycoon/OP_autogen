"""T13-7 / 任务 A gate: NgaBackend（subprocess 调 nga run 的 LLMProvider）。

开发机无 nga，用 mock runner 验证逻辑。覆盖 spec §3 六项。
"""
import subprocess
from types import SimpleNamespace

import pytest

from control.llm_backend import NgaBackend, LLMInvocationError, LLMTimeout
from control.orchestrator import NoLLMBackend, parse_generate_response

CONFIG = {
    "generate": {"model": "strong/model", "options": {"variant": "high"}, "timeout_s": 180},
    "choose_lever": {"model": "w3/GLM-4.7", "options": {}, "timeout_s": 60},
}

# real nga output shape: a '> ...' header line + fenced python + json blocks
SAMPLE_STDOUT = """> build · CAC-GLM-4.7-cj · ses_abc123
```python
def kernel(a, b, c):
    return a @ b
```
```json
{"lever": null, "extension_used": null, "notes": "hello"}
```
"""


class FakeRunner:
    def __init__(self, stdout=SAMPLE_STDOUT, returncode=0, raise_exc=None):
        self.stdout = stdout
        self.returncode = returncode
        self.raise_exc = raise_exc
        self.calls = []

    def __call__(self, cmd, timeout=None):
        self.calls.append(list(cmd))
        if self.raise_exc is not None:
            raise self.raise_exc
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr="")


def test_dual_block_parse_ignores_header():
    r = FakeRunner()
    b = NgaBackend(CONFIG, runner=r)
    out = b.generate("p")
    kernel, meta = parse_generate_response(out, allowed_extensions=set())
    assert "def kernel" in kernel
    assert meta["notes"] == "hello"


def test_model_echo_traced_from_header():
    r = FakeRunner()
    b = NgaBackend(CONFIG, runner=r)
    b.generate("p")
    assert b.last_model_echo == "CAC-GLM-4.7-cj"


def test_command_per_call_type_and_stateless():
    r = FakeRunner()
    b = NgaBackend(CONFIG, runner=r)
    b.generate("gp")
    b.choose_lever("cp")
    gen_cmd, choose_cmd = r.calls[0], r.calls[1]
    assert gen_cmd[0] == "nga" and gen_cmd[1] == "run"
    assert "strong/model" in gen_cmd and "high" in gen_cmd
    assert "w3/GLM-4.7" in choose_cmd and "--variant" not in choose_cmd
    for cmd in r.calls:
        assert "--continue" not in cmd and "--session" not in cmd, "must be stateless"


def test_config_driven_not_hardcoded():
    r = FakeRunner()
    cfg = {"generate": {"model": "OTHER/model", "timeout_s": 10},
           "choose_lever": {"model": "x"}}
    b = NgaBackend(cfg, runner=r)
    b.generate("p")
    assert "OTHER/model" in r.calls[0]


def test_nonzero_exit_raises_and_no_self_retry():
    r = FakeRunner(returncode=2)
    b = NgaBackend(CONFIG, runner=r)
    with pytest.raises(LLMInvocationError):
        b.generate("p")
    assert len(r.calls) == 1, "NgaBackend must not retry on its own"


def test_timeout_raises():
    r = FakeRunner(raise_exc=subprocess.TimeoutExpired(cmd=["nga"], timeout=1))
    b = NgaBackend(CONFIG, runner=r)
    with pytest.raises(LLMTimeout):
        b.generate("p")
    assert len(r.calls) == 1


def test_nollmbackend_regression():
    with pytest.raises(NotImplementedError):
        NoLLMBackend().generate("p")
    with pytest.raises(NotImplementedError):
        NoLLMBackend().choose_lever("p")
