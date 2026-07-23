"""T13-1 gate: LLM 后端接入（第六个槽位）。

- 配置缺失时报错清晰
- 注入假后端后 generate 可用，且输出过解析闸门
- NoLLMBackend 行为不变（回归保护）
- temperature 默认低值；可配置覆盖；端点/模型走配置不硬编码
"""
import os

import pytest

from control.llm_backend import ConfigurableLLMBackend
from control.orchestrator import NoLLMBackend, parse_generate_response


def test_missing_config_raises_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(ValueError) as exc:
        ConfigurableLLMBackend()
    msg = str(exc.value)
    assert "LLM_BASE_URL" in msg and "LLM_MODEL" in msg, f"error not clear: {msg}"


def test_config_via_constructor_no_hardcoding():
    b = ConfigurableLLMBackend(base_url="https://example.invalid/v1", model="m")
    assert b.base_url == "https://example.invalid/v1"
    assert b.model == "m"
    assert b.temperature <= 0.2, "default temperature must be low for reproducibility"


def test_config_via_env(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://env.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "env-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.3")
    b = ConfigurableLLMBackend()
    assert b.base_url == "https://env.invalid/v1"
    assert b.model == "env-model"
    assert b.temperature == 0.3


def test_nollmbackend_unchained_regression():
    # NoLLMBackend still raises (default orchestrator backend unchanged)
    with pytest.raises(NotImplementedError):
        NoLLMBackend().generate("p")
    with pytest.raises(NotImplementedError):
        NoLLMBackend().choose_lever("p")


class _FakeBackend:
    """A duck-typed LLMProvider whose generate returns a parse-gate-passing response."""

    def generate(self, prompt):
        return (
            "```python\ndef kernel(a, b, c):\n    return a @ b\n```\n"
            '```json\n{"lever": null, "extension_used": null, "notes": "ok"}\n```'
        )

    def choose_lever(self, prompt):
        return '```json\n{"lever": "x", "rationale": "r"}\n```'


def test_injected_backend_generate_passes_parse_gate():
    resp = _FakeBackend().generate("anything")
    kernel, meta = parse_generate_response(resp, allowed_extensions=set())
    assert "def kernel" in kernel
    assert meta["notes"] == "ok"
