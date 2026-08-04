#!/usr/bin/env python3
"""
Coder Agent — Prompt 编排器 + LLM 调用。

═══════════════════════════════════════════════════════════════════════════════
  职责: 读 Plan + 当前代码 → 调用 LLM 做最小化代码改动

  输入 (从 Orchestrator):
    - plan: RoundPlan (Planner 产出)
    - kernel_code: str (当前 kernel 源码)
    - previous_error: str (Verifier 回传的错误, 用于重试)
    - round_dir: Path (本轮输出目录, 用于保存 diff)

  输出:
    - CoderResult (optimized_code, diff, lines_changed)

═══════════════════════════════════════════════════════════════════════════════
  约束
═══════════════════════════════════════════════════════════════════════════════

  可以读:
    ✅ kernel.py (当前代码)
    ✅ plan.md (优化计划)
    ✅ Playbook (参考知识)

  只能写:
    ✅ optimized kernel.py (修改后代码)
    ✅ diff.patch (变更记录)

  绝对不能碰:
    ❌ 任何 msprof/ hivmir/ merged 数据
    ❌ optimization_trajectory.json
    ❌ 其他 round 目录的文件

═══════════════════════════════════════════════════════════════════════════════
  回退机制
═══════════════════════════════════════════════════════════════════════════════

  Coder 不负责回退。回退由 Orchestrator 实现:
    - Coder 把优化代码写入 round_N/kernel.py
    - Verifier 验证 → 如果 REVERT:
      Orchestrator.current_kernel 保持不变 (还是上一轮的代码)
      round_N/kernel.py 保留在目录中作为记录 (不删除)
    → 下一轮从上一轮代码开始, 自然实现了 "回退"
"""

from __future__ import annotations

import json
import os
import re
import sys
import difflib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoderResult:
    """Coder 产出的代码修改结果。"""
    success: bool
    optimized_code: str
    diff: str
    lines_changed: int = 0
    error_message: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 构建
# ═══════════════════════════════════════════════════════════════════════════════

def _load_coding_guide() -> str:
    """加载编码指南文档。"""
    import os
    guide_path = Path(__file__).resolve().parent.parent / "docx" / "CODING_GUIDE.md"
    if guide_path.exists():
        return guide_path.read_text(encoding="utf-8")[:4000]
    return ""

def _build_system_prompt(plan_text: str = "", tier: int = 1) -> str:
    guide = _load_coding_guide()

    # 加载当前 Tier 的 Playbook
    playbook = ""
    tier_files = {
        1: "playbook_tier1_algorithm.md", 2: "playbook_tier2_fusion.md",
        3: "playbook_tier3_tiling.md", 4: "playbook_tier4_memory.md",
        5: "playbook_tier5_compute.md", 6: "playbook_tier6_architecture.md",
    }
    pb_path = Path(__file__).resolve().parent.parent / "docx" / tier_files.get(tier, "")
    if pb_path.exists():
        playbook = pb_path.read_text(encoding="utf-8")[:2000]

    parts = []
    parts.append("You are a Triton kernel code modifier. Output the COMPLETE modified Python file.")

    # ★ 注入 CODING_GUIDE + Tier Playbook (裁剪, 只留关键段; 主要走确定性 changes[] 替换)
    if guide:
        parts.append(f"## Coding Guide (MUST READ)\n{guide[:1200]}")
    if playbook:
        parts.append(f"## Tier {tier} Optimization Strategy (MUST READ)\n{playbook[:1500]}")

    parts.append(f"""## CRITICAL: Your output MUST be DIFFERENT from the input code.
Make at least ONE concrete change from this list:
- Increase BLOCK_SIZE to next power-of-2 (256→512→1024→2048)
- Fuse two adjacent simple ops into one expression
- Eliminate a redundant tl.load (same pointer loaded twice)
- Reorder operations to reduce intermediate variables

## IRON RULES (violating any = FAILURE)
1. NEVER put num_warps or num_stages inside @triton.jit() — these do NOT go there
2. NEVER use @triton.autotune — ONLY plain @triton.jit
3. NEVER change function name, parameter names, or parameter count
4. NEVER modify the mathematical formula
5. NEVER add new imports (no torch, numpy, etc.)
6. Keep exact indentation and code style

## Output: COMPLETE modified Python code. No markdown, no explanation.""")

    return "\n\n".join(parts)


def _build_user_prompt(
    plan_text: str,
    kernel_code: str,
    previous_error: str = "",
) -> str:
    parts = []

    parts.append("## Optimization Plan")
    parts.append(plan_text)
    parts.append("")

    if previous_error:
        parts.append("## Previous Attempt Failed")
        parts.append(f"The last code change caused this error:")
        parts.append(f"```")
        parts.append(previous_error[:1000])
        parts.append(f"```")
        parts.append("Please fix this error while still implementing the plan.")
        parts.append("")

    parts.append("## Current Kernel Code")
    parts.append("```python")
    parts.append(kernel_code)
    parts.append("```")
    parts.append("")
    parts.append("---")
    parts.append("Output the COMPLETE modified kernel code (no explanation, no markdown).")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  代码验证
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_python(code: str) -> tuple[bool, str]:
    """Python 语法检查。"""
    try:
        compile(code, "<kernel>", "exec")
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _generate_diff(original: str, optimized: str) -> str:
    """生成 unified diff。"""
    orig_lines = original.splitlines(keepends=True)
    opt_lines = optimized.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, opt_lines,
        fromfile="kernel.py (original)",
        tofile="kernel.py (optimized)",
    )
    return "".join(diff)


def _count_lines_changed(diff_text: str) -> int:
    """统计 diff 中的行数变更。"""
    additions = sum(1 for line in diff_text.split("\n") if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff_text.split("\n") if line.startswith("-") and not line.startswith("---"))
    return additions + deletions


def _extract_changes(plan_text: str) -> list:
    """从 plan JSON 提取 changes[] (planner 输出, 见 skills/triton-op-planner)。"""
    if not plan_text or not plan_text.strip():
        return []
    import json
    text = plan_text.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.startswith("```"))
    start = text.find("{"); end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        d = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return d.get("changes", []) if isinstance(d, dict) else []


def _apply_plan_changes(code: str, changes: list):
    """确定性应用 changes[]: old_code → new_code 精确替换。返回 (新代码, applied, missing)。"""
    applied, missing = [], []
    for ch in changes:
        old = (ch or {}).get("old_code", "")
        new = (ch or {}).get("new_code", "")
        if not old:
            continue
        if old in code:
            code = code.replace(old, new, 1)
            applied.append(ch)
        else:
            missing.append(old[:60])
    return code, applied, missing


# ═══════════════════════════════════════════════════════════════════════════════
#  Coder Agent
# ═══════════════════════════════════════════════════════════════════════════════

class CoderAgent:
    """编码智能体 — Prompt 编排器。

    支持 3 种 LLM 调用模式 (通过 LLMClient):
      - api: DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
      - cli: LLM_CLI_COMMAND="nga run"
      - stub: 无配置时返回原代码

    Usage:
        coder = CoderAgent()
        result = coder.apply(...)
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def apply(
        self,
        kernel_code: str,
        plan_text: str = "",
        previous_error: str = "",
        tier: int = 1,
    ) -> CoderResult:
        """应用优化计划到代码。

        Args:
            kernel_code: 当前 kernel 源码
            plan_text: Planner 产出的计划文本 (JSON 格式)
            previous_error: 上一轮失败的错误信息 (Verifier 重试时传入)

        Returns:
            CoderResult (optimized_code + diff)
        """

        # ★Step 0: 确定性应用 plan 的 changes[] (old_code→new_code 精确替换, 不靠 LLM, 最稳)
        #    仅当无 previous_error (有报错时才走 LLM 修复)
        if not previous_error:
            changes = _extract_changes(plan_text)
            if changes:
                new_code, applied, missing = _apply_plan_changes(kernel_code, changes)
                if missing:
                    return CoderResult(
                        success=False, optimized_code=kernel_code, diff="",
                        error_message=f"plan old_code 未在代码中找到: {missing} (请让 planner 输出精确 old_code)")
                if applied:
                    ok, err = _validate_python(new_code)
                    if not ok:
                        return CoderResult(success=False, optimized_code=kernel_code,
                                           diff="", error_message=f"替换后语法错: {err}")
                    diff = _generate_diff(kernel_code, new_code)
                    return CoderResult(success=True, optimized_code=new_code, diff=diff,
                                       lines_changed=_count_lines_changed(diff))

        # Step 1: 调用 LLM (或 stub)
        if self.use_llm:
            optimized = self._call_llm(kernel_code, plan_text, previous_error, tier)
        else:
            optimized = self._stub_apply(kernel_code, plan_text)

        # Step 2: 清理 LLM 输出
        optimized = self._clean_output(optimized, kernel_code)

        # 如果之前有错误，且这次成功了 (代码不同)，记录解决方案
        if previous_error and optimized.strip() != kernel_code.strip():
            try:
                from memory.codeerror import CodeErrorMemory
                mem = CodeErrorMemory(os.path.basename(os.getcwd()) or "kernel")
                mem.record_solution(previous_error[:200],
                    f"成功修改: {plan_text[:100]}")
            except Exception:
                pass

        # Step 3: Python 语法检查
        valid, err = _validate_python(optimized)
        if not valid:
            return CoderResult(
                success=False,
                optimized_code=kernel_code,
                diff="",
                error_message=err,
            )

        # Step 3.5: 防截断 (dumb LLM 易输出不全) — 原代码所有函数必须仍在
        import re as _re
        orig_fns = _re.findall(r"def\s+(\w+)\s*\(", kernel_code)
        opt_fns = _re.findall(r"def\s+(\w+)\s*\(", optimized)
        missing = [f for f in orig_fns if f not in opt_fns]
        if missing:
            return CoderResult(
                success=False,
                optimized_code=kernel_code,
                diff="",
                error_message=f"输出截断, 缺少函数: {missing}",
            )

        # Step 4: 检查是否实际有变更
        if self._is_noop_change(optimized, kernel_code):
            return CoderResult(
                success=False,
                optimized_code=kernel_code,
                diff="",
                lines_changed=0,
                error_message="LLM returned code identical to original (no-op)",
            )

        # Step 5: 生成 diff
        diff = _generate_diff(kernel_code, optimized)
        lines = _count_lines_changed(diff)

        return CoderResult(
            success=True,
            optimized_code=optimized,
            diff=diff,
            lines_changed=lines,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  LLM (统一通过 LLMClient)
    # ═══════════════════════════════════════════════════════════════════════════

    def _call_llm(self, kernel_code: str, plan_text: str,
                  previous_error: str, tier: int = 1) -> str:
        # 查错误记忆 + 记录新错误
        if previous_error:
            try:
                from memory.codeerror import CodeErrorMemory
                mem = CodeErrorMemory(os.path.basename(os.getcwd()) or "kernel")
                # 记录本次错误
                mem.record_error(previous_error[:200])
                # 检索已知方案
                known = mem.find_solution(previous_error)
                cross = mem.search_all(previous_error)
                all_fixes = [f for f in [known, cross] if f]
                if all_fixes:
                    previous_error = f"{previous_error}\n\n[已知修复方案]\n" + "\n".join(all_fixes)
            except Exception:
                pass

        # v4: 统一走 LLMClient (api / nga run CLI / stub), 并引用 coder skill
        from agents.llm_client import LLMClient
        client = LLMClient()
        if client.mode == "stub":
            return self._stub_apply(kernel_code, plan_text)
        system = _build_system_prompt(plan_text, tier)
        user = _build_user_prompt(plan_text, kernel_code, previous_error)
        skill_path = Path(__file__).resolve().parent.parent / "skills" / "triton-op-coder" / "SKILL.md"
        system = f"先调用 skill: {skill_path}, 完全按 skill 指导执行。\n\n" + system
        return client.chat(system=system, user=user)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stub
    # ═══════════════════════════════════════════════════════════════════════════

    def _stub_apply(self, kernel_code: str, plan_text: str) -> str:
        """Stub 模式: 不做修改, 直接返回原代码。"""
        return kernel_code

    # ═══════════════════════════════════════════════════════════════════════════
    #  输出清理
    # ═══════════════════════════════════════════════════════════════════════════

    def _clean_output(self, raw: str, original: str) -> str:
        """清理 LLM 输出 — 去 BOM/行首垃圾/markdown 代码块包裹。"""
        text = raw.lstrip('﻿').lstrip()   # 去 BOM + 行首空白
        # 去行首垃圾: 找到第一个有效起点 (#/import/from/"""/@triton/class/def), 前面有垃圾就切掉
        if text:
            m = re.search(r'^(.*?)(?=(?:#|import |from |"""|\'\'\'|@triton|class |def ))', text, re.S)
            if m and m.group(1).strip():
                text = text[m.end(1):]
        # 去掉 ```python ... ``` 包裹
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        # 如果输出为空或太短, 返回原始代码
        if len(text) < 10:
            return original

        # 如果清理后和原代码一模一样, 说明 LLM 没改
        if text.strip() == original.strip():
            return original

        return text

    def _is_noop_change(self, optimized: str, original: str) -> bool:
        """检查是否实际有代码变更 (忽略空白和注释差异)。"""
        import re
        def normalize(s):
            return re.sub(r'\s+', ' ', s).strip()
        return normalize(optimized) == normalize(original)


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    coder = CoderAgent()

    # 模拟 kernel
    kernel = """import triton, triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
"""
    plan_text = json.dumps({
        "strategy": "increase_tile_size",
        "specific_change": "BLOCK_SIZE: 256 → 8192",
        "expected_impact": "bw_util from 21% to 90%+",
    }, indent=2)

    # Stub 测试
    r = coder.apply(kernel, plan_text)
    print(f"Coder stub test: success={r.success}, diff_len={len(r.diff)}")
    assert r.success

    # 语法检查测试
    valid, err = _validate_python(kernel)
    print(f"Syntax check: valid={valid}, err='{err}'")
    assert valid

    # 损坏代码测试
    bad = "def foo(\n"
    valid, err = _validate_python(bad)
    print(f"Bad code check: valid={valid}, err='{err[:50]}'")
    assert not valid

    # Diff 测试
    modified = kernel.replace("BLOCK_SIZE: tl.constexpr",
                               "BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr")
    diff = _generate_diff(kernel, modified)
    lines = _count_lines_changed(diff)
    print(f"Diff test: {lines} lines changed")
    assert lines == 2

    print("\n[Coder] All tests passed")


if __name__ == "__main__":
    _self_test()
