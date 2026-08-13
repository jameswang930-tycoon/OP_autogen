#!/usr/bin/env python3
"""
Coder Agent — 读 Plan + 当前代码 → 做最小化代码改动。

两步策略:
  Step 0 (首选, 确定性): 从 plan 的 changes[] 做 old_code→new_code 精确替换
    → _sanitize_unicode 清洗 new_code 的 Unicode 脏字符 → _validate_python 语法校验
  Step 1 (兜底, LLM): Step 0 失败/有 previous_error 时调 LLM 改码
    → _clean_output (去 BOM/前导垃圾/markdown/Unicode 脏字符) → _validate_python

Unicode 脏字符清洗 (_sanitize_unicode):
  - 15+ 类替换: 中文标点/箭头/dash/引号/数学符号/省略号/空格 → ASCII 等价
  - 兜底: compile 失败时逐行清非注释行的非 ASCII

防截断: 检查所有 def 仍在; 防 no-op: 检查输出 ≠ 原码。
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
    """构造 coder LLM prompt — ★只给文件路径, 让 nga 自己读, 不内嵌大段内容。"""
    docx = Path(__file__).resolve().parent.parent / "docx"
    tier_files = {
        1: "playbook_tier1_algorithm.md", 2: "playbook_tier2_fusion.md",
        3: "playbook_tier3_tiling.md", 4: "playbook_tier4_memory.md",
        5: "playbook_tier5_compute.md", 6: "playbook_tier6_architecture.md",
    }
    parts = []
    parts.append("你是 Triton kernel 代码修改者。优先按 changes[] 精确替换; 需要修报错时才自行改码。")
    parts.append(f"## 读改码教程: {docx / 'CODING_GUIDE.md'} (前1200字符)")
    parts.append(f"## 读当前 tier({tier}) 策略: {docx / tier_files.get(tier, '')} (前1500字符速查卡; "
                 f"★若本次改动涉及算法/结构 (im2col/剪枝/单遍/online softmax), 必读该文档文末『结构层优化执行教学』"
                 f"的教学1/2/3完整代码, 照抄正确实现, 不要凭记忆重写)")
    parts.append("""## IRON RULES (违反 = 失败)
1. 只改 prompt 给的当前 kernel 文件, 不碰其他文件 (尤其不碰 input/<op>/kernel_op.py 源文件)
2. 不把 num_warps/num_stages 写进 @triton.jit() (triton-ascend 自动管理)
3. 不新增 import, 不改函数名/参数名/参数个数
4. 输出完整修改后的代码, 不要 markdown 代码块包裹
5. 若 changes[] 的 old_code 找不到 → 报告, 绝不猜测乱改""")
    return "\n\n".join(parts)


def _build_user_prompt(
    plan_text: str,
    kernel_code: str,
    previous_error: str = "",
    kernel_path: Optional[str] = None,
) -> str:
    parts = []

    parts.append("## Optimization Plan (含 changes[])")
    parts.append(plan_text)
    parts.append("")

    if previous_error:
        parts.append("## Previous Attempt Failed (★只读参考, 严禁抄写)")
        parts.append(f"The last code change caused this error:")
        parts.append(f"```")
        parts.append(previous_error[:1000])
        parts.append(f"```")
        parts.append("★这段报错/修复方案是**参考信息**, 不是代码 — 禁止把其中的解释文字、英文句子、")
        parts.append("  'vsel'/关键词、括号内容原样抄进输出代码 (会导致 unterminated string literal/语法错).")
        parts.append("  只能理解错误后修改代码本身; 输出必须是合法 Python 代码 (注释里也别抄整段).")
        parts.append("Please fix this error while still implementing the plan.")
        parts.append("")

    if kernel_path:
        # ★只给路径, 让 nga 读文件 (不内嵌几千字符)
        parts.append(f"## 当前要改的文件 (读它): {kernel_path}")
        parts.append("按 plan 的 changes[] 对文件做精确替换 (old_code→new_code); 若 old_code 找不到就报告, 不猜不改。")
    else:
        parts.append("## Current Kernel Code")
        parts.append("```python")
        parts.append(kernel_code[:2500])
        parts.append("```")
    parts.append("")
    parts.append("---")
    parts.append("Output the COMPLETE modified kernel code (no explanation, no markdown).")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
#  代码验证
# ═══════════════════════════════════════════════════════════════════════════════

def _validate_python(code: str) -> tuple[bool, str]:
    """Python 语法检查。
    ★修复: 报错信息带上出错行的真实内容 (repr) — 之前只给行号, 用户/planner 看到
    "line 13" 却不知道那行有什么 (常见: 千分位 1,024 / 全角字符 / markdown 残留),
    导致"文件里看着没非法字符"的困惑."""
    try:
        compile(code, "<kernel>", "exec")
        return True, ""
    except SyntaxError as e:
        line_txt = ""
        if e.lineno:
            lines = code.split("\n")
            if 1 <= e.lineno <= len(lines):
                line_txt = repr(lines[e.lineno - 1])[:150]
        return False, f"SyntaxError at line {e.lineno}: {e.msg} | 该行内容: {line_txt}"


# ★Unicode 脏字符 → ASCII 等价物 (LLM 常把 ASCII 写成 Unicode → 语法错/运行错/HIVM编译错)
#   覆盖: 各种 dash/引号/空格/数学符号/中文标点/箭头/省略号
_UNICODE_FIXES = {
    # ── dash/连字符 ──
    "—": "-", "–": "-", "―": "-", "‒": "-", "‑": "-",   # em/en/horizontal/figure/non-breaking hyphen
    # ── 引号 ──
    "‘": "'", "’": "'", "‚": "'", "‛": "'",              # 单引号变体
    "“": '"', "”": '"', "„": '"', "‟": '"',              # 双引号变体
    "«": "<", "»": ">", "‹": "<", "›": ">",              # 角引号
    # ── 空格 ──
    " ": " ", "　": " ", " ": " ", " ": " ",             # 不间断/全角/数字/窄不间断
    "⁠": "", "​": "",                                     # 零宽 (连接符/空格)
    # ── 省略号 ──
    "…": "...",
    # ── 数学符号 (LLM 常替代运算符) ──
    "×": "*", "÷": "/", "∙": "*", "·": "*",              # 乘除号 → */ /
    "−": "-",                                             # U+2212 minus sign → -
    "±": "+/-", "∓": "-/+",
    "≈": "~=", "≠": "!=", "≤": "<=", "≥": ">=",          # 比较运算符
    "∞": "float('inf')",
    "°": " deg",
    "²": "**2", "³": "**3",                              # 上标
    # ── 箭头 (注释里常见, 不影响运行但统一清) ──
    "→": "->", "←": "<-", "↑": "^", "↓": "v", "↔": "<->",
    "⇒": "=>", "⇐": "<=", "⇑": "^^", "⇓": "vv",
    # ── 中文标点 (LLM 中文上下文泄漏到代码) ──
    "。": ".", "，": ",", "；": ";", "：": ":",
    "（": "(", "）": ")", "【": "[", "】": "]",
    "「": "'", "」": "'", "『": '"', "』": '"',
    "？": "?", "！": "!", "、": ",",
    # ── 其他 ──
    "•": "*", "◦": "*", "▪": "*", "‣": "*",
    "©": "(c)", "®": "(r)", "™": "(tm)",
    "№": "No.", "§": "S.", "¶": "P.",
    # ── 制表符/框线 (注释里常见, 如 ═══ 分隔线; LLM 重吐整文件时最易改崩这片 → 统一转 ASCII) ──
    "═": "=", "║": "|", "╔": "=", "╗": "=", "╚": "=", "╝": "=",
    "╠": "=", "╣": "=", "╦": "=", "╩": "=", "╬": "=",
    "─": "-", "│": "|", "┌": "+", "┐": "+", "└": "+", "┘": "+",
    "├": "+", "┤": "+", "┬": "+", "┴": "+", "┼": "+",
    # ── 全角数字/字母 (LLM 常把 ASCII 数字写成全角 → 删除会留空, 必须转 ASCII) ──
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
    "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G", "Ｈ": "H",
    "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O", "Ｐ": "P",
    "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T", "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X",
    "Ｙ": "Y", "Ｚ": "Z", "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f",
    "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
    "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t", "ｕ": "u", "ｖ": "v",
    "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
    # ── 不可见/空字符 (LLM 常混入 → "空行 invalid character" 报错, 最常见于首/末行) ──
    #   BOM/零宽/方向符 → 删; 空格类变体 → 普通空格; 行/段分隔 → 空格 (防结构错位)
    "﻿": "", "‌": "", "‍": "", "‎": "", "‏": "",
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ",
}


_QUOTE_BADS = "“”‘’"
_QUOTE_MAP = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def _fix_quote_chars(text: str) -> str:
    """智能/中文引号两难处理:
      场景A 定界符 (print(“[info] OK”)) → 必须换成 ASCII 引号 print("[info] OK")
      场景B 字符串内容 (print("他说“你好”")) → 换成 ASCII 会破坏字符串, 必须删除
      字符级替换无法区分两者 → 两种候选都试, 取第一个 compile 通过的版本;
      都失败(文本还有其它语法问题) → 返回替换版 (至少引号已 ASCII 化, 其余交后续清洗)."""
    if not any(q in text for q in _QUOTE_BADS):
        return text
    del_v = text
    rep_v = text
    for bad in _QUOTE_BADS:
        del_v = del_v.replace(bad, "")
        rep_v = rep_v.replace(bad, _QUOTE_MAP[bad])
    for cand in (del_v, rep_v):
        try:
            compile(cand, "<check>", "exec")
            return cand
        except SyntaxError:
            continue
    return rep_v


def _sanitize_unicode(text: str) -> str:
    """把 LLM 输出里的 Unicode 脏字符替换成 ASCII 等价物.
    ★修复(引号两难): 智能引号先走 _fix_quote_chars (删/换取编译通过者),
       普通字符替换只处理定界符场景时反而会破坏字符串内部的引号."""
    text = _fix_quote_chars(text)
    for bad, good in _UNICODE_FIXES.items():
        if bad in _QUOTE_BADS:
            continue   # 引号已由 _fix_quote_chars 处理
        text = text.replace(bad, good)
    # ★千分位逗号归一: LLM 常把数值写成 1,024 / 1，024(全角已转半角) → Python 里
    #   `1,024` 是 `1, 024` → 前导零 → SyntaxError "leading zeros/invalid decimal literal".
    #   安全边界: 只匹配 \b数字1-3位(,数字3位)+ \b (标准千分位分组); 带空格的元组/赋值
    #   (64, 64, 32) 不匹配; 4 位以上数字(2048,1024 合法元组)不匹配.
    text = re.sub(r"(?<!\w)\d{1,3}(?:,\d{3})+(?!\w)", lambda m: m.group(0).replace(",", ""), text)
    # ★兜底: 如果替换后仍有非 ASCII 且 compile 失败, 暴力清掉所有非 ASCII (注释外)
    try:
        compile(text, "<check>", "exec")
    except SyntaxError:
        # 还有非 ASCII 导致语法错 → 逐行处理: 非 comment/string 行去掉非 ASCII
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("#"):
                cleaned.append(line)   # 注释行保留 Unicode (不影响运行)
            else:
                # 非注释行: 替换剩余非 ASCII 为 ASCII 最近邻 (保守: 只清确实非法的)
                cleaned.append("".join(ch if ord(ch) < 128 else _closest_ascii(ch) for ch in line))
        text = "\n".join(cleaned)
        # ★修复: 兜底清洗后必须再验证一次 — 清洗可能引入新错 (如全角数字被删 → "= , 64")
        #   (字符级删除无法保证语义, 但至少让调用方的报错信息指向真实残留行)
        try:
            compile(text, "<check>", "exec")
        except SyntaxError:
            pass   # 仍失败 → 交给调用方 _validate_python 拦 (报错信息已含出错行内容)
    return text


def _closest_ascii(ch: str) -> str:
    """非 ASCII 字符 → 最近 ASCII 替代 (保守兜底)."""
    # 常见 Unicode → ASCII (补充 _UNICODE_FIXES 没覆盖的)
    _map = {"²": "2", "³": "3", "¹": "1", "⁰": "0", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
            # ★全角数字/字母 (兜底路径也必须转, 不能删 — 删除会留空产生 "= , 64" 新语法错)
            "０": "0", "１": "1", "２": "2", "３": "3", "４": "4", "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
            "Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D", "Ｅ": "E", "Ｆ": "F", "Ｇ": "G", "Ｈ": "H",
            "Ｉ": "I", "Ｊ": "J", "Ｋ": "K", "Ｌ": "L", "Ｍ": "M", "Ｎ": "N", "Ｏ": "O", "Ｐ": "P",
            "Ｑ": "Q", "Ｒ": "R", "Ｓ": "S", "Ｔ": "T", "Ｕ": "U", "Ｖ": "V", "Ｗ": "W", "Ｘ": "X",
            "Ｙ": "Y", "Ｚ": "Z", "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f",
            "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l", "ｍ": "m", "ｎ": "n",
            "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r", "ｓ": "s", "ｔ": "t", "ｕ": "u", "ｖ": "v",
            "ｗ": "w", "ｘ": "x", "ｙ": "y", "ｚ": "z",
            "ɑ": "a", "ɛ": "e", "ɔ": "o", "ν": "v", "β": "B", "γ": "g", "δ": "d", "θ": "th", "λ": "l", "μ": "u",
            "π": "pi", "ρ": "r", "σ": "s", "τ": "t", "φ": "phi", "ω": "w", "Δ": "D", "Σ": "S", "Ω": "O",
            "√": "sqrt", "∑": "sum", "∏": "prod", "∫": "int", "∂": "d", "∇": "grad"}
    return _map.get(ch, "")  # 未知 → 删除 (比留非法字符安全)


def _preserve_header(text: str, original: str) -> str:
    """★保 shebang + coding 声明 (防 LLM 重吐整文件时吞/改 header → 'line 3 invalid syntax').
    原代码有 shebang(`#!...`) 或 coding(`# -*- coding -*-`), 清理后的 text 丢了 → 从 original 补回最前.
    清理后已含则不动. 注: coding 声明必须在第 1/2 行才有效, 所以只补这两类前缀."""
    import re as _re
    orig_lines = original.splitlines()
    # 取原代码前两行的 shebang / coding
    header = []
    for ln in orig_lines[:2]:
        if ln.startswith("#!") or "coding" in ln and ":" in ln:
            header.append(ln)
    if not header:
        return text   # 原代码无 header, 不补
    # 清理后的 text 已含这些 header 行 → 不重复补
    text_first2 = "\n".join(text.splitlines()[:2])
    missing = [h for h in header if h.strip() not in text_first2]
    if not missing:
        return text
    return "\n".join(missing) + "\n" + text


def _generate_diff(original: str, optimized: str) -> str:
    """生成 unified diff。"""
    orig_lines = original.splitlines(keepends=True)
    opt_lines = optimized.splitlines(keepends=True)
    diff = difflib.unified_diff(
        orig_lines, opt_lines,
        fromfile="kernel_op.py (original)",
        tofile="kernel_op.py (optimized)",
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
    """确定性应用 changes[]: old_code → new_code 替换（★全部出现处, 不只第一处）。
    1) 精确匹配优先 (old_code 为整行, planner 铁律);
    2) ★容错: 单行 old_code 精确没匹配上时, 按行去首尾空白/CRLF 归一化再匹配
       — 处理 planner 从旧版本/示例复制 old_code 导致的缩进/尾随空格/换行符差异
       (保持原行缩进 + 原换行符替换)。多行 old_code 只走精确匹配(安全)。
    ★D4 同名 kernel 防误伤: old_code 匹配 >1 处时打警告 — 若只想改一处
      (如 attention_mlp 多个 matmul_kernel 共用 BLOCK), 需带函数名/调用处上下文使唯一.
    """
    applied, missing = [], []
    for ch in changes:
        old = (ch or {}).get("old_code", "")
        new = (ch or {}).get("new_code", "")
        if not old:
            continue
        new = _sanitize_unicode(new)   # ★清 new_code 的 Unicode 脏字符 (em dash 等, compile 查不出但运行错)
        n_hits = code.count(old)
        if n_hits > 1:
            print(f"    ⚠ D4: old_code 匹配 {n_hits} 处 '{old.strip()[:50]}...' "
                  f"(全部替换; 若只想改一处请带函数名上下文)")
        if old in code:
            code = code.replace(old, new)   # replace all (old_code 为整行, 所有出现都该改)
            applied.append(ch)
            continue
        # ★容错: 单行归一化匹配
        old_norm = old.strip()
        if old_norm and "\n" not in old_norm:
            lines = code.splitlines(keepends=True)
            hit = None
            for i, line in enumerate(lines):
                if line.rstrip("\r\n").strip() == old_norm:
                    hit = i
                    break
            if hit is not None:
                nl = "\r\n" if "\r\n" in lines[hit] else "\n"
                indent = lines[hit][:len(lines[hit]) - len(lines[hit].lstrip())]
                lines[hit] = indent + new.strip() + nl
                code = "".join(lines)
                applied.append(ch)
                continue
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

    @staticmethod
    def _dump_failed(code: str, kernel_path: Optional[str], why: str):
        """★失败留证: 把 coder 产出的坏代码写到 round_dir/failed_kernel.py (不覆盖 kernel_op.py).
        便于事后排查真实坏字符 (用户常反映 'kernel_op.py 看着没问题' — 因为失败时写回的是原码,
        坏的中间产物被丢了; 留这份才能定位). 永不抛异常."""
        try:
            if not kernel_path:
                return
            from pathlib import Path as _P
            fp = _P(kernel_path).parent / "failed_kernel.py"
            fp.write_text((f"# failed_kernel.py — coder 输出未通过校验: {why}\n"
                           f"# (kernel_op.py 仍是上一轮的正确代码, 这份只是排查用)\n"
                           + code), encoding="utf-8")
        except Exception:
            pass

    def apply(
        self,
        kernel_code: str,
        plan_text: str = "",
        previous_error: str = "",
        tier: int = 1,
        kernel_path: Optional[str] = None,
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
                        self._dump_failed(new_code, kernel_path, f"Step0 替换后语法错: {err}")
                        return CoderResult(success=False, optimized_code=kernel_code,
                                           diff="", error_message=f"替换后语法错: {err}")
                    diff = _generate_diff(kernel_code, new_code)
                    return CoderResult(success=True, optimized_code=new_code, diff=diff,
                                       lines_changed=_count_lines_changed(diff))

        # Step 1: 调用 LLM (或 stub)
        if self.use_llm:
            try:
                optimized = self._call_llm(kernel_code, plan_text, previous_error, tier, kernel_path)
            except Exception as e:
                # ★超时/调用失败 → 不崩整个循环: 返回原代码 + 报错信息 (调度器标 NOOP/FAIL 继续)
                print(f"  [Coder] ⚠ LLM 调用失败: {str(e)[:200]}")
                return CoderResult(success=False, optimized_code=kernel_code, diff="",
                                   error_message=f"LLM 调用失败: {str(e)[:200]}")
        else:
            optimized = self._stub_apply(kernel_code, plan_text)

        # Step 2: 清理 LLM 输出
        optimized = self._clean_output(optimized, kernel_code)

        # 如果之前有错误，且这次成功了 (代码不同)，回填失败案例库 (solved)
        if previous_error and optimized.strip() != kernel_code.strip():
            try:
                from memory.failed_cases import mark_solved
                mark_solved(tier, previous_error, f"成功修改: {plan_text[:200]}")
            except Exception:
                pass

        # Step 3: Python 语法检查
        valid, err = _validate_python(optimized)
        if not valid:
            self._dump_failed(optimized, kernel_path, f"Step3 LLM 输出语法错: {err}")
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
                  previous_error: str, tier: int = 1,
                  kernel_path: Optional[str] = None) -> str:
        # 查失败案例库: 记录本次错误 + 检索已知方案 (两级: 指纹精确 + 关键词相似)
        if previous_error:
            try:
                from memory.failed_cases import retrieve as _retr_fc, format_for_coder as _fmt_fc
                _inj = _fmt_fc(_retr_fc(tier, previous_error, ""), tier)
                if _inj:
                    # ★2026-08-12: 已知方案是"参考信息" — 弱模型会把这段含 'vsel' 等英文单引号的
                    #   解释文本抄进输出代码 → unterminated string literal 语法错. 明确标注只读.
                    previous_error = (f"{previous_error}\n\n{_inj}")
            except Exception:
                pass

        # v4: 统一走 LLMClient (api / nga run CLI / stub), 并引用 coder skill
        from agents.llm_client import LLMClient
        client = LLMClient()
        if client.mode == "stub":
            return self._stub_apply(kernel_code, plan_text)
        system = _build_system_prompt(plan_text, tier)
        user = _build_user_prompt(plan_text, kernel_code, previous_error, kernel_path)
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
        """清理 LLM 输出 — 去 BOM/行首垃圾/markdown 代码块包裹/★Unicode 脏字符/保 header。
        ★修复: markdown 剥离必须先于"去行首垃圾" — 否则 ```python 开头会被垃圾正则吃掉,
        text.startswith('```') 失活, 结尾 ``` 残留 → SyntaxError at line N (看代码没非法字符)."""
        text = raw.lstrip('﻿').lstrip()   # 去 BOM + 行首空白
        # 先剥 markdown ```python ... ``` 包裹 (含 ``` 后无语言标注 / 行首空格变体)
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            while lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        # 去行首垃圾: 找到第一个有效起点 (#/import/from/"""/@triton/class/def), 前面有垃圾就切掉
        if text:
            m = re.search(r'^(.*?)(?=(?:#|import |from |"""|\'\'\'|@triton|class |def ))', text, re.S)
            if m and m.group(1).strip():
                text = text[m.end(1):]

        # ★Unicode 脏字符清洗 (LLM 常把 ASCII 写成 Unicode 等价物 → 语法错/运行错):
        #   em dash — (U+2014) / en dash – (U+2013) → ASCII - (减号/连字符)
        #   智能引号 ' ' " " (U+2018/9/C/D) → ASCII ' "
        #   省略号 … (U+2026) → ...
        #   不间断空格 (U+00A0) / 全角空格 (U+3000) → 普通空格
        #   ★制表符 ═ ─ │ 等 (注释分隔线) → ASCII, 防 LLM 重吐整文件时改崩 header
        text = _sanitize_unicode(text)

        # ★保 header (防 LLM 吞/改 shebang + coding 声明 → "line 3 invalid syntax"):
        #   原代码有 shebang/coding, 清理后丢了 → 从原代码补回最前面.
        text = _preserve_header(text, original)

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
        # ★F5: 自测必须带 changes[] — 否则 Step0 无替换 → stub 返回原码 = no-op → success=False
        "changes": [{"old_code": "BLOCK_SIZE: tl.constexpr",
                     "new_code": "BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr",
                     "reason": "自测", "section": "② kernel"}],
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
