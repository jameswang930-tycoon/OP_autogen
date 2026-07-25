"""可配置的 LLM 后端（T13-1，第六个槽位）。

交接包五个槽位填完后，编排器仍需一个能真正调通的 LLM 后端（否则第一次 generate
即失败）。本模块提供一个具体的、可配置的 ``LLMProvider`` 实现：

- 后端地址 / 模型名 / temperature / 超时全部走**构造参数或环境变量**，公开分支不硬编码
  真实端点。
- temperature 默认低值（0.0）以提升可复现性；可配置覆盖。
- 请求体采用 OpenAI 兼容的 chat 格式；若真实后端接口不同，在保密环境调整
  ``_post`` 的请求/响应解析即可（见 HANDOFF 槽位 0）。

环境变量：``LLM_BASE_URL`` / ``LLM_MODEL`` / ``LLM_TEMPERATURE`` / ``LLM_TIMEOUT``。
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional


class ConfigurableLLMBackend:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = base_url if base_url is not None else os.environ.get("LLM_BASE_URL")
        self.model = model if model is not None else os.environ.get("LLM_MODEL")
        self.temperature = float(
            temperature if temperature is not None
            else os.environ.get("LLM_TEMPERATURE", 0.0)
        )
        self.timeout = float(
            timeout if timeout is not None
            else os.environ.get("LLM_TIMEOUT", 30)
        )
        missing = [
            name for name, val in (("LLM_BASE_URL", self.base_url), ("LLM_MODEL", self.model))
            if not val
        ]
        if missing:
            raise ValueError(
                "ConfigurableLLMBackend missing required config: "
                + ", ".join(missing)
                + " (set the env vars or pass them to the constructor)"
            )

    def _post(self, prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"LLM backend request failed: {exc}") from exc
        data = json.loads(body)
        # OpenAI-compatible: choices[0].message.content
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str) -> str:
        return self._post(prompt)

    def choose_lever(self, prompt: str) -> str:
        return self._post(prompt)


# ---------------- NgaBackend (subprocess `nga run`) ----------------

class LLMInvocationError(Exception):
    """nga run exited non-zero (or otherwise failed). Orchestrator handles retries."""


class LLMTimeout(Exception):
    """nga run exceeded the configured timeout."""


@dataclass
class NgaCallConfig:
    model: str
    variant: Optional[str]
    timeout_s: float


def _default_runner(cmd: list[str], *, timeout: float):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_model_echo(stdout: str) -> Optional[str]:
    """从 nga 输出的 '> build · <model 回显> · <session>' 头行解析 model 回显名。"""
    first = stdout.splitlines()[0] if stdout else ""
    if first.startswith(">") and "·" in first:
        parts = [p.strip() for p in first.split("·")]
        if len(parts) >= 2:
            return parts[1]
    return None


class NgaBackend:
    """LLMProvider via `nga run` subprocess (OpenCode CLI; no HTTP API exists).

    - 模型 / variant / timeout 全部配置驱动（config dict 或环境变量），不硬编码。
    - generate 与 choose_lever 各自可配（难任务配强模型+高 variant，易任务配轻量模型）。
    - 无状态：绝不传 --continue / --session。
    - 解析只认 fenced block、忽略 '>' 头；model 回显名记录到 last_model_echo 供溯源。
    - 失败抛可识别异常（LLMInvocationError / LLMTimeout），不在内部重试（重试归编排器）。
    """

    def __init__(self, config: Optional[dict] = None, *, runner: Callable = _default_runner):
        self.generate_cfg = self._mk_cfg("generate", config)
        self.choose_lever_cfg = self._mk_cfg("choose_lever", config)
        self.runner = runner
        self.last_model_echo: Optional[str] = None
        self.call_count = 0

    @staticmethod
    def _mk_cfg(kind: str, config: Optional[dict]) -> NgaCallConfig:
        section = (config or {}).get(kind, {}) if config else {}
        env_kind = kind.upper()
        model = section.get("model") or os.environ.get(f"NGA_{env_kind}_MODEL")
        if not model:
            raise ValueError(
                f"NgaBackend: {kind} model not configured "
                f"(set config llm.{kind}.model or env NGA_{env_kind}_MODEL)"
            )
        variant = section.get("variant", None)
        if variant is None:
            variant = os.environ.get(f"NGA_{env_kind}_VARIANT") or None
        timeout = float(section.get(
            "timeout_s", os.environ.get(f"NGA_{env_kind}_TIMEOUT_S", 120),
        ))
        return NgaCallConfig(model=model, variant=variant or None, timeout_s=timeout)

    def _run(self, prompt: str, cfg: NgaCallConfig) -> str:
        # list-form args: 避免 shell 注入；绝不传 --continue / --session
        cmd = ["nga", "run", "--model", cfg.model]
        if cfg.variant:
            cmd += ["--variant", cfg.variant]
        cmd.append(prompt)
        try:
            result = self.runner(cmd, timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise LLMTimeout(f"nga run timed out after {cfg.timeout_s}s") from exc
        self.call_count += 1
        if result.returncode != 0:
            raise LLMInvocationError(
                f"nga run exited {result.returncode}: {(result.stderr or '').strip()}"
            )
        stdout = result.stdout or ""
        self.last_model_echo = _parse_model_echo(stdout)
        return stdout

    def generate(self, prompt: str) -> str:
        return self._run(prompt, self.generate_cfg)

    def choose_lever(self, prompt: str) -> str:
        return self._run(prompt, self.choose_lever_cfg)
