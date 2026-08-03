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


def _default_runner(cmd: list[str], *, timeout: float):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_model_echo(stdout: str) -> Optional[str]:
    """从 nga 输出的 '> build · <model 回显> · <session>' 头行解析 model 回显名。
    非 nga agent（无此头行）返回 None——backend 对输出格式不作假设。"""
    first = stdout.splitlines()[0] if stdout else ""
    if first.startswith(">") and "·" in first:
        parts = [p.strip() for p in first.split("·")]
        if len(parts) >= 2:
            return parts[1]
    return None


def _map_options_to_cli(opts: dict) -> list[str]:
    """通用 options→命令行映射（**不硬编码任何具体选项名**，V2 重构）：
    {key: value} → ['--key', 'value']（key 中 `_`→`-`）；True → ['--key']；False/None → 省略。
    具体 key 由环境侧 config 决定，框架不假设 agent 支持哪些选项。"""
    args: list[str] = []
    for k, v in opts.items():
        flag = "--" + str(k).replace("_", "-")
        if v is True:
            args.append(flag)
        elif v is False or v is None:
            continue
        else:
            args += [flag, str(v)]
    return args


class NgaBackend:
    """单轮命令行 LLM backend（V2 重构：配置驱动 + 可扩展，**不预设 agent 能力**）。

    nga 与目标 agent 调用形式本质一致（`xxx run --model xxx` 单轮命令行），单轮 run 本就
    能触发 skill——**只有一个 backend，无"哑后端/双模式"**。不硬编码 agent 支持哪些命令行选项
    （model/thinking/output_format/...）；环境侧探测能力后填 config，框架只提供 options→命令行的
    通用映射 + extra_args 透传口 + 可选结构化输出路径。generate/choose_lever 对 orchestrator 不变。

    config（dict 或 env；generate/choose_lever 各自可配）：
      cmd            命令前缀 list，默认 ["nga","run"]，或 env AGENT_CMD（空格分隔）。
      generate/choose_lever: {"model", "options": {...}, "extra_args": [...], "timeout_s"}
      structured     {"enabled", "request": {...}, "kernel_key", "meta_key"}  可选结构化输出路径。
      timeout_s      全局默认（被 per-kind 覆盖）。
    环境变量：AGENT_CMD（命令前缀）、NGA_GENERATE_MODEL / NGA_CHOOSE_LEVER_MODEL（兼容）。
    未填任何 options→仅 cmd + model + prompt 的最基础调用（不报错，mock/开源可跑）。
    """

    def __init__(self, config: Optional[dict] = None, *, runner: Callable = _default_runner):
        cfg = config or {}
        self.cmd = self._cmd_prefix(cfg)
        self.generate_cfg = self._kind_cfg(cfg, "generate")
        self.choose_lever_cfg = self._kind_cfg(cfg, "choose_lever")
        self.structured: dict = cfg.get("structured") or {}
        self.runner = runner
        self.last_model_echo: Optional[str] = None
        self.call_count = 0

    @staticmethod
    def _cmd_prefix(cfg: dict) -> list[str]:
        cmd = cfg.get("cmd")
        if cmd is None:
            env = os.environ.get("AGENT_CMD")
            cmd = env.split() if env else ["nga", "run"]
        return list(cmd)

    @staticmethod
    def _kind_cfg(cfg: dict, kind: str) -> dict:
        sec = cfg.get(kind) or {}
        return {
            "model": sec.get("model") or os.environ.get(f"NGA_{kind.upper()}_MODEL"),
            "options": dict(sec.get("options") or {}),
            "extra_args": list(sec.get("extra_args") or []),
            "timeout_s": float(sec.get("timeout_s", cfg.get("timeout_s", 120))),
        }

    def _build_cmd(self, prompt: str, kcfg: dict) -> list[str]:
        opts = {}
        if kcfg["model"]:
            opts["model"] = kcfg["model"]
        opts.update(kcfg["options"])                      # 环境侧填的真实选项透传
        if self.structured.get("enabled"):                # 结构化输出请求选项
            opts.update(self.structured.get("request") or {})
        # list-form args 避免 shell 注入；绝不传 --continue / --session（无状态）
        return (list(self.cmd) + _map_options_to_cli(opts)
                + list(kcfg["extra_args"]) + [prompt])

    def _invoke(self, prompt: str, kcfg: dict) -> str:
        cmd = self._build_cmd(prompt, kcfg)
        try:
            result = self.runner(cmd, timeout=kcfg["timeout_s"])
        except subprocess.TimeoutExpired as exc:
            raise LLMTimeout(f"agent timed out after {kcfg['timeout_s']}s") from exc
        self.call_count += 1
        if result.returncode != 0:
            raise LLMInvocationError(
                f"agent exited {result.returncode}: {(result.stderr or '').strip()}"
            )
        self.last_model_echo = _parse_model_echo(result.stdout or "")
        return result.stdout or ""

    def _maybe_normalize_structured(self, raw: str) -> str:
        """结构化输出路径（config 声明 agent 支持时启用）：把 agent 返回的 json 规范成
        orchestrator 期望的 fenced-block 形式——比从自由文本抠可靠、解决 kernel/json 混排。
        未启用 / json 解析失败 / schema 不符 → 原样返回（降级到自由文本 fenced-block 解析）。"""
        if not self.structured.get("enabled"):
            return raw
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001 - 解析失败则降级
            return raw
        if not isinstance(data, dict):
            return raw
        kk = self.structured.get("kernel_key", "kernel")
        mk = self.structured.get("meta_key", "meta")
        if kk in data and mk in data:                     # generate 形式：kernel + meta
            return (f"```python\n{data[kk]}\n```\n"
                    f"```json\n{json.dumps(data[mk], ensure_ascii=False)}\n```")
        return f"```json\n{json.dumps(data, ensure_ascii=False)}\n```"  # 单 json 块（choose_lever 等）

    def generate(self, prompt: str) -> str:
        return self._maybe_normalize_structured(self._invoke(prompt, self.generate_cfg))

    def choose_lever(self, prompt: str) -> str:
        return self._maybe_normalize_structured(self._invoke(prompt, self.choose_lever_cfg))
