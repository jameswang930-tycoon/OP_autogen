#!/usr/bin/env python3
"""
LLM 客户端 — 统一抽象层，支持 3 种调用模式。

模式 (优先级从高到低):
  1. API: DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY → 直接调 API
  2. CLI: LLM_CLI_COMMAND 环境变量 → 管道调用本地 agent
  3. STUB: 无 API key 也无 CLI → 返回占位结果

CLI 模式用法:
  export LLM_CLI_COMMAND="nga run"
  # planner/coder 会自动: echo "<prompt>" | nga run
"""

from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


class LLMClient:
    """统一 LLM 调用接口。

    Usage:
        client = LLMClient()
        response = client.chat(
            system="You are a helpful assistant.",
            user="Write a hello world function.",
            max_tokens=2048,
        )
        # response 是纯文本字符串
    """

    def __init__(self):
        self.mode = self._detect_mode()

    def _detect_mode(self) -> str:
        if os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
            return "api"
        if os.environ.get("LLM_CLI_COMMAND"):
            return "cli"
        return "stub"

    def chat(self, system: str, user: str, max_tokens: int = 2048) -> str:
        """发送 chat 请求，返回 LLM 响应文本。"""
        if self.mode == "api":
            return self._call_api(system, user, max_tokens)
        elif self.mode == "cli":
            return self._call_cli(system, user)
        else:
            return self._call_stub(system, user)

    # ── API 模式 ──────────────────────────────────────────────────────────────

    def _call_api(self, system: str, user: str, max_tokens: int) -> str:
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

        if deepseek_key:
            from openai import OpenAI
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com",
                           timeout=120.0)
            resp = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""

        elif anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text

        raise RuntimeError("No API key (BUG: mode=api but no key found)")

    # ── CLI 模式 ──────────────────────────────────────────────────────────────

    def _call_cli(self, system: str, user: str) -> str:
        cmd = os.environ.get("LLM_CLI_COMMAND", "nga run")
        # 合并 system prompt 和 user prompt 为一个输入
        full_prompt = f"{system}\n\n---\n\n{user}"

        print(f"  [LLM/CLI] piping {len(full_prompt)} chars to: {cmd}")

        try:
            result = subprocess.run(
                cmd, shell=True,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=int(os.environ.get("LLM_CLI_TIMEOUT", "300")),
            )
            if result.returncode != 0:
                stderr = result.stderr[:500] if result.stderr else ""
                raise RuntimeError(
                    f"CLI command '{cmd}' exited with {result.returncode}: {stderr}"
                )
            output = result.stdout.strip()
            if not output:
                raise RuntimeError(f"CLI command '{cmd}' returned empty output")
            print(f"  [LLM/CLI] response: {len(output)} chars")
            return output
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"CLI command '{cmd}' timed out")

    # ── Stub 模式 ─────────────────────────────────────────────────────────────

    def _call_stub(self, system: str, user: str) -> str:
        print("  [LLM/STUB] no API key or CLI command configured — returning placeholder")
        return "(STUB: no LLM configured. Set DEEPSEEK_API_KEY or LLM_CLI_COMMAND in .env)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Planner 专用: 从 LLM 响应中提取 JSON
# ═══════════════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON 对象（容错处理）。"""
    if not text or not text.strip():
        raise ValueError("Empty LLM response")

    text = text.strip()

    # 去掉 markdown 代码块
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                text = p
                break

    # 找第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]

    for attempt in range(3):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            if attempt == 0:
                text = re.sub(r':\s*([^{}"\s,]+)(?=\s*[,}])', r': "\1"', text)
            elif attempt == 1:
                text = re.sub(r'(?<=:)\s*([^"{}\[\],\s]+)(?=\s*[,}\]])', r' "\1"', text)

    return json.loads(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    client = LLMClient()
    print(f"LLM mode: {client.mode}")
    if client.mode == "stub":
        resp = client.chat("You are helpful.", "Say hi.")
        print(f"Stub response: {resp[:100]}")
    else:
        print(f"Ready (use client.chat() to call LLM)")
