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
import urllib.error
import urllib.request
from typing import Optional


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
