#!/usr/bin/env python3
"""
上下文管理器 — Token 估算 + 裁剪 + 窗口构建。

═══════════════════════════════════════════════════════════════════════════════
  职责
═══════════════════════════════════════════════════════════════════════════════

  1. 构建 Planner prompt 上下文 (diagnosis + extracted + playbook + history + cases)
  2. 估算 token 数 (保守: chars/1.5 for 中英混合)
  3. 裁剪: 超限时优先保留诊断+Playbook+热层, 裁剪案例+温层
  4. 窗口管理: 对接 SlidingWindow

═══════════════════════════════════════════════════════════════════════════════
  裁剪优先级 (从高到低)
═══════════════════════════════════════════════════════════════════════════════

  1. Playbook 章节 (最高优先级, 不能裁)
  2. 瓶颈诊断数据 (diagnosis + extracted)
  3. 热层上下文 (最近 5 轮完整)
  4. 当前 kernel 代码
  5. 温层摘要 (6~15 轮)
  6. 相似案例 (experience retrieval)
  7. 冷层数据点
"""

from __future__ import annotations

from typing import List, Optional


def estimate_tokens(text: str) -> int:
    """估算 token 数。

    保守估计: 中英混合 ~1 char = 0.67 token (chars / 1.5)
    纯英文: ~1 char = 0.25 token (chars / 4)
    这里用折中: chars / 2
    """
    return max(1, len(text) // 2)


def build_context(
    diagnosis_text: str = "",
    extracted_text: str = "",
    playbook_text: str = "",
    history_text: str = "",
    similar_cases_text: str = "",
    kernel_code: str = "",
    max_tokens: int = 800_000,
) -> str:
    """构建完整 Planner prompt 上下文, 超限自动裁剪。

    Args:
        diagnosis_text: 瓶颈诊断 (BottleneckDiagnosis JSON 格式化文本)
        extracted_text: 按需提取的精简数据 (~2KB)
        playbook_text: 当前 Tier 的 Playbook 全文
        history_text: 从 SlidingWindow 构建的历史文本
        similar_cases_text: 从 experience_retriever 检索的经验文本
        kernel_code: 当前 kernel 源码
        max_tokens: 上下文窗口上限 (默认 800K, 1M 窗口留 20% 余量)

    Returns:
        拼接好的上下文文本
    """
    # 必须保留的部分 (不可裁剪)
    mandatory = f"{diagnosis_text}\n\n{extracted_text}\n\n{playbook_text}"
    mandatory_tokens = estimate_tokens(mandatory)

    # 重要但可裁剪的部分
    important = f"{history_text}\n\n{kernel_code}"
    important_tokens = estimate_tokens(important)

    # 可裁剪的部分
    optional = similar_cases_text
    optional_tokens = estimate_tokens(optional)

    total = mandatory_tokens + important_tokens + optional_tokens

    # 不超限 → 完整返回
    if total <= max_tokens:
        parts = [diagnosis_text, extracted_text, playbook_text]
        if history_text:
            parts.append(history_text)
        if kernel_code:
            parts.append(f"## Current Kernel Code\n```python\n{kernel_code}\n```")
        if similar_cases_text:
            parts.append(similar_cases_text)
        return "\n\n".join(parts)

    # 超限 → 裁剪
    # 优先保留: mandatory + important
    if mandatory_tokens + important_tokens <= max_tokens:
        # 裁剪 optional
        remaining = max_tokens - mandatory_tokens - important_tokens
        trimmed_optional = _trim_text(optional, remaining)
        parts = [diagnosis_text, extracted_text, playbook_text]
        if history_text:
            parts.append(history_text)
        if kernel_code:
            parts.append(f"## Current Kernel Code\n```python\n{kernel_code}\n```")
        if trimmed_optional:
            parts.append(trimmed_optional)
        return "\n\n".join(parts)

    # 严重超限 → 裁 kernel_code 和 history
    remaining = max_tokens - mandatory_tokens
    half = remaining // 2
    trimmed_history = _trim_text(history_text, half)
    trimmed_kernel = _trim_text(kernel_code, half)

    parts = [diagnosis_text, extracted_text, playbook_text]
    if trimmed_history:
        parts.append(trimmed_history)
    if trimmed_kernel:
        parts.append(f"## Current Kernel Code (truncated)\n```python\n{trimmed_kernel}\n```")
    return "\n\n".join(parts)


def _trim_text(text: str, max_tokens: int) -> str:
    """裁剪文本到指定 token 数以内。保留开头和结尾。"""
    if not text or max_tokens <= 0:
        return ""
    max_chars = max_tokens * 2
    if len(text) <= max_chars:
        return text
    # 保留前 70% 和后 20%
    head = int(max_chars * 0.7)
    tail = int(max_chars * 0.2)
    return text[:head] + f"\n\n...(truncated {len(text) - head - tail} chars)...\n\n" + text[-tail:]


def format_diagnosis(diag) -> str:
    """将 BottleneckDiagnosis 格式化为 prompt 文本。"""
    lines = [
        "## Bottleneck Diagnosis",
        f"- **Tier**: {getattr(diag, 'current_tier', '?')} ({getattr(diag, 'tier_name', '?')})",
        f"- **Bottleneck op**: op{getattr(diag, 'bottleneck_op_id', '?')} "
        f"({getattr(diag, 'bottleneck_op_type', '?')}, {getattr(diag, 'bottleneck_engine', '?')})",
        f"- **Type**: {getattr(diag, 'bottleneck_type', '?')} "
        f"({getattr(diag, 'bottleneck_category', '?')})",
        f"- **Headroom**: {getattr(diag, 'optimization_headroom', '?')}",
        f"- **Time ratio**: {getattr(diag, 'bottleneck_time_ratio', 0):.2%}",
        f"- **BW utilization**: {getattr(diag, 'bottleneck_bw_utilization', 0):.2%}",
        f"- **Regime**: {getattr(diag, 'bottleneck_regime', '?')}",
        f"- **Suggested strategies**: {getattr(diag, 'suggested_strategies', [])}",
        f"- **Structural issues**: {getattr(diag, 'structural_issues', [])}",
    ]
    return "\n".join(lines)
