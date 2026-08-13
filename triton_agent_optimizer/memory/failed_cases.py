#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""失败案例库 — 轮内/跨轮错误记忆: 检索参考方案 + 防止重复踩同一方案.

★设计 (2026-08-13):
  - 按 tier 分文件 memory/tier{N}_failed_cases.json (错误类型与 tier 强相关,
    与 excellent_cases 的 tier{N}_cases.json 对称; entry 内记 op 作软权重).
  - 两级检索: L1 指纹精确 (归一化签名 hash, 同错必中) + L2 关键词交集 (近似).
  - 判定键 = (tier, stage, fingerprint): 同指纹不新增, attempts+1, 方案历史去重.
  - 生命周期: open → solved (被成功修复) | stuck (attempts≥STUCK_AFTER 仍未解决).
    ★stuck 只禁止"原样重试已失败方案", 不禁止新方案 — 弱模型也能继续试错,
    不会被"跳过跳过再跳过"锁死.
  - 环境性失败 (超时/欠采/设备重置/采集失败) 不入库 (scheduler 自己处理).
  - 永不抛异常 (损坏重置; 单进程无锁).
"""
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
CASES_DIR = _PROJECT / "memory"
# 同指纹尝试 ≥ STUCK_AFTER 次仍未解决 → stuck (env 可调)
STUCK_AFTER = int(os.environ.get("FAILED_CASE_STUCK_AFTER", "3"))
# 每 tier 上限, 超出淘汰最久未命中的 open 条目
MAX_CASES_PER_TIER = int(os.environ.get("FAILED_CASE_MAX", "50"))

# 英文停用词 (normalize 时过滤)
_STOP_WORDS = {
    "this", "that", "with", "from", "been", "were", "when", "will", "have",
    "they", "them", "then", "than", "also", "into", "just", "like", "make",
    "more", "over", "some", "such", "take", "very", "your", "here", "each",
    "error", "errorcode", "failed", "failure", "exception", "traceback",
    "raise", "called", "while", "during", "after", "before", "what", "which",
    "where", "there", "these", "those", "because", "could", "would", "should",
    "file", "value", "values", "return", "line", "self", "none", "true",
    "false", "check", "found", "using", "usage", "about", "above", "again",
}
# 环境性失败关键词 → 不入库 (scheduler 自己处理, 不是"代码知识")
_ENV_KEYS = ("timed out", "timeout", "超时", "采集失败", "欠采", "under-sampl",
             "device reset", "npu-smi", "连接失败",
             "connection", "msprof 漏采", "漏采", "无法启动", "terminated")
# 编译期错误 → code_compile (编译器接地证据: vsel/MLIR 等, 喂 planner 定位"编译器为何没实现")
_COMPILE_KEYS = ("syntaxerror", "invalid syntax", "syntax error", "unsupported op",
                 "not supported", "not implemented", "compilation error", "compile failed",
                 "compile error", "mlir", "vsel", "语法错", "语法错误", "编译失败")
# 数值校验错误 → code_numeric (结果不对, 不是跑不起来)
_NUMERIC_KEYS = ("result check", "check failed", "mismatch", "correctness", "atol",
                 "rtol", "nan", "inf", "数值", "不匹配", "校验失败", "result mismatch")


def cases_path(tier: int) -> Path:
    """tier 1~6 各自的失败案例文件."""
    return CASES_DIR / f"tier{int(tier)}_failed_cases.json"


def normalize_error(err: str):
    """报错 → (归一化签名, 关键词列表, 指纹).
    去行号/地址/路径/数字序列, 保留英文主体 + 中文片段 (规范化使同错必同签)."""
    try:
        t = (err or "")[:800]
        t = re.sub(r"0x[0-9a-fA-F]+", " ", t)
        t = re.sub(r"line\s+\d+", " ", t)
        t = re.sub(r":\d+:\d+", " ", t)
        t = re.sub(r"[\w\-\\/.]*\.py\w*", " ", t)
        t = re.sub(r"\b\d+\b", " ", t)
        words = [w.lower() for w in re.findall(r"[a-zA-Z_]{4,}", t)
                 if w.lower() not in _STOP_WORDS]
        cjk = re.findall(r"[\u4e00-\u9fff]{2,}", t)
        keywords = list(dict.fromkeys(words[:20] + cjk[:8]))
        sig = " ".join(keywords)
        fp = hashlib.sha1(sig.encode("utf-8", errors="ignore")).hexdigest()[:12]
        return sig, keywords, fp
    except Exception:
        return "", [], "000000000000"


def classify_error(err: str) -> str:
    """失败分类 (统一入口, scheduler 各处共用):
    'env'=环境性(不入库) | 'code_compile'=编译期 | 'code_numeric'=数值校验 | 'code_runtime'=运行时."""
    e = (err or "").lower()
    if any(k in e for k in _ENV_KEYS):
        return "env"
    if any(k in e for k in _COMPILE_KEYS):
        return "code_compile"
    if any(k in e for k in _NUMERIC_KEYS):
        return "code_numeric"
    return "code_runtime"


def is_code_error(err: str) -> bool:
    """是否代码类失败 (可入库): 编译/数值/运行时均可, 环境性不入."""
    return classify_error(err) != "env"


def load(tier: int) -> list:
    """读某 tier 失败案例; 缺失/损坏 → 空列表 (不抛)."""
    p = cases_path(tier)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, list):
                return d
        except Exception:
            pass
    return []


def _save(tier: int, cases: list) -> None:
    """写回 + 上限淘汰 (LRU: 最久未更新的 open 条目优先淘汰). 永不抛."""
    try:
        if len(cases) > MAX_CASES_PER_TIER:
            open_ = [c for c in cases if c.get("status") in ("open", "stuck")]
            excess = len(cases) - MAX_CASES_PER_TIER
            if open_:
                open_.sort(key=lambda c: c.get("updated_at", ""))
                drop = {id(c) for c in open_[:excess]}
                cases = [c for c in cases if id(c) not in drop]
            else:
                cases = cases[-MAX_CASES_PER_TIER:]
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        cases_path(tier).write_text(
            json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _find(cases: list, fp: str, stage: str):
    """同 (fingerprint, stage) 命中 (同一错误的同一阶段)."""
    for c in cases:
        if c.get("fingerprint") == fp and c.get("stage") == stage:
            return c
    return None


def record_failure(tier: int, stage: str, err: str, op: str = "",
                   change_summary: str = "", rn: int = 0) -> None:
    """记录一次失败. 同指纹同阶段 → attempts+1 + 方案历史去重追加;
    新错误 → open 入库. 自动流转: attempts≥STUCK_AFTER → stuck;
    ★solved 后再现 → 降级回 open (说明上次方案没根治). 永不抛."""
    try:
        if classify_error(err) == "env":
            return
        sig, _kw, fp = normalize_error(err)
        if not sig:
            return
        cases = load(tier)
        c = _find(cases, fp, stage)
        now = datetime.now().isoformat()
        if c:
            c["attempts"] = int(c.get("attempts", 0)) + 1
            c["updated_at"] = now
            c["error_class"] = classify_error(err)   # ★分类统一入口
            if rn and rn not in (c.get("related_rounds") or []):
                c.setdefault("related_rounds", []).append(rn)
            if change_summary:
                tried = c.setdefault("attempted_solutions", [])
                if not any(change_summary in t.get("s", "") for t in tried):
                    tried.append({"s": change_summary[:200], "round": rn})
                    tried = tried[-3:]
                    c["attempted_solutions"] = tried
            if c.get("status") == "solved":
                c["status"] = "open"          # ★没根治 → 重新开放
                c["solution"] = None
                c["attempts"] = 1             # ★新尝试周期重新计数 (旧方案历史仍在 attempted_solutions)
                c["stuck_at"] = None
            if c.get("status") == "open" and c["attempts"] >= STUCK_AFTER:
                c["status"] = "stuck"         # ★尝试够多了, 封原方案
                c["stuck_at"] = now
        else:
            cases.append({
                "fingerprint": fp, "op": op, "stage": stage,
                "error_sig": sig[:200], "keywords": _kw,
                "error_class": classify_error(err),   # ★统一分类入口
                "status": "open", "attempts": 1,
                "attempted_solutions": ([{"s": change_summary[:200], "round": rn}]
                                        if change_summary else []),
                "related_rounds": [rn] if rn else [],
                "solution": None, "fix_diff": "",
                "hit_count": 0, "hit_solved": 0,
                "created_at": now, "updated_at": now,
            })
        _save(tier, cases)
    except Exception:
        pass


def retrieve(tier: int, err: str, op: str = "", fallback_all: bool = False) -> list:
    """两级检索: L1 指纹精确 (必中) + L2 关键词交集 (近似).
    排序: 指纹命中 > 交集数 > 同op > solved > 最近更新. 返回 ≤3 条."""
    try:
        sig, _kw, fp = normalize_error(err)
        if not sig:
            return []
        cases = load(tier)
        hits = []
        for c in cases:
            score = 0
            if c.get("fingerprint") == fp:
                score += 100
                c["hit_count"] = int(c.get("hit_count", 0)) + 1   # ★命中反馈
            else:
                inter = len(set(c.get("keywords") or []) & set(_kw))
                if inter < 2:
                    continue
                score += inter * 10
            if op and c.get("op") == op:
                score += 5
            if c.get("status") == "solved":
                score += 3
            hits.append((score, c))
        hits.sort(key=lambda x: (-x[0], str(x[1].get("updated_at", ""))))
        top = [c for _, c in hits[:3]]
        # miss → 跨 tier 兜底 (同错在别的 tier 解决过)
        if not top and fallback_all:
            for t in range(1, 7):
                if t == tier:
                    continue
                for c in retrieve(t, err, op, fallback_all=False):
                    if c not in top:
                        top.append(c)
        _save(tier, cases)   # 落 hit_count
        return top
    except Exception:
        return []


def format_for_coder(hits: list, tier: int) -> str:
    """把检索到的失败案例格式化成注入修复器的文本 (含"禁止原样重试"黑名单语义).
    ★泛化提示: 显示 op/解决轮次 — 同类错误在别的算子解决过也提示 (权重稍低, 由检索排序控制)."""
    if not hits:
        return ""
    lines = [f"## 历史失败案例参考 (Tier{tier} 失败库, ★只读参考, 严禁原样复制修复方案代码)"]
    for c in hits:
        st = c.get("status", "open")
        sig = c.get("error_sig", "?")[:100]
        op = c.get("op") or "?"
        rr = c.get("related_rounds") or []
        solved_r = rr[-1] if (st == "solved" and rr) else "?"
        cls = c.get("error_class", "?")
        tried = c.get("attempted_solutions") or []
        _tried_s = "; ".join(t.get("s", "")[:80] for t in tried) or "(无)"
        if st == "solved" and c.get("solution"):
            lines.append(f"- [已解决/op:{op}/R{solved_r}/类型:{cls}] 错误: {sig} → 修复: {str(c['solution'])[:150]}")
            if c.get("fix_diff"):
                lines.append(f"    参考补丁: {str(c['fix_diff'])[:150]}")
        elif st == "stuck":
            lines.append(f"- [★此路不通/op:{op}/类型:{cls}] 错误: {sig} 已试 {c.get('attempts')} 次失败: {_tried_s}")
            lines.append("    ★禁止原样重试上述方案, 必须换思路")
        else:
            lines.append(f"- [未解决/op:{op}/类型:{cls}] 错误: {sig} 已试 {c.get('attempts')} 次: {_tried_s}")
            lines.append("    可换新方案再试, 但不要原样重试已失败方案")
    return "\n".join(lines)


def mark_solved(tier: int, err: str, solution: str, fix_diff: str = "") -> None:
    """失败被成功修复 → 回填 solved (solution/fix_diff). 按指纹定位. 永不抛."""
    try:
        sig, _kw, fp = normalize_error(err)
        if not sig:
            return
        cases = load(tier)
        hit = False
        for c in cases:
            if c.get("fingerprint") == fp:
                c["status"] = "solved"
                c["solution"] = (solution or "")[:400]
                if fix_diff:
                    c["fix_diff"] = fix_diff[:400]
                c["updated_at"] = datetime.now().isoformat()
                c["hit_solved"] = int(c.get("hit_solved", 0)) + 1
                hit = True
        if hit:
            _save(tier, cases)
    except Exception:
        pass


def build_retry_context(attempt_log: list, new_err: str, tier: int,
                        op: str = "", stage: str = "code", rn: int = 0) -> str:
    """构建"重试上下文" — 传给修复器 (coder) 的完整累积文本:
      ① 之前每次尝试的 (改动方案 + 报错) 全序列 → 修复器知道试过什么、为何失败
      ② 本次新报错
      ③ 失败案例库检索注入 (solved 方案 / stuck 黑名单 / 已试方案)
      ④ 禁止原样重试提示
    同时把本次失败入库 (record_failure). scheduler 每轮重试调这一个入口."""
    parts = []
    for it in attempt_log:
        parts.append(f"【第{it.get('n')}次尝试】改动: {(it.get('change') or '?')[:120]}\n"
                     f"  结果: {(it.get('err') or '?')[:400]}")
    if parts:
        parts.append("── 以上尝试均失败 ★禁止原样重试上述方案, 必须换思路或修根因 ──")
    parts.append(f"【第{len(attempt_log) + 1}次结果】{str(new_err)[:600]}")
    try:
        if classify_error(new_err) != "env":
            # ★先检索再入库: 避免"首犯自命中" — 刚入的条目检索时命中自己 (无参考价值噪声)
            #   下次失败再命中时 attempts 已是历史值, 提示"已试过什么"才有收敛价值
            hits = retrieve(tier, new_err, op)
            inj = format_for_coder(hits, tier)
            if inj:
                parts.append(inj)
            record_failure(tier, stage, new_err, op,
                           change_summary=(attempt_log[-1].get("change") if attempt_log else ""),
                           rn=rn)
            # ★负正闭环: 本轮方案曾在本 tier 优秀案例成功过 → 提示可能上下文差异
            #   (同一 change 此一时成功彼一时失败 = 环境/尺寸/上下文不同, 别急着否定方案)
            if attempt_log:
                _cur = attempt_log[-1].get("change", "")
                try:
                    from memory.excellent_cases import load as _load_ec
                    for ec in reversed(_load_ec(tier)):
                        if ec.get("op") and op and ec.get("op") != op:
                            continue
                        for ch in (ec.get("changes") or [])[:2]:
                            _t = f"{ch.get('old_code','')}{ch.get('new_code','')}"
                            # 空格归一化匹配 (change 摘要 "BLOCK_K=32" ↔ 案例 "BLOCK_K = 32")
                            _c, _t2 = _cur.replace(" ", ""), _t.replace(" ", "")
                            if _c and len(_c) >= 6 and _c[:40] in _t2:
                                parts.append(
                                    f"★提示: 该方案曾在 R{ec.get('round')} 成功 "
                                    f"({ec.get('speedup_before')}x→{ec.get('speedup_after')}x), "
                                    f"本次失败可能因上下文/尺寸/环境不同 — 可对比当时代码, 但勿原样重放")
                                break
                        else:
                            continue
                        break
                except Exception:
                    pass
    except Exception:
        pass
    return "\n".join(parts)
