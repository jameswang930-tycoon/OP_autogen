#!/usr/bin/env python3
"""
Planner Agent — Prompt 编排器 + LLM 调用。

═══════════════════════════════════════════════════════════════════════════════
  职责: 读诊断 + Playbook + 历史 → 调用 LLM 生成本轮优化计划

  输入 (从 Orchestrator):
    - diagnosis: BottleneckDiagnosis (瓶颈是谁, 什么类型, 优化空间)
    - extracted_data: str (data_extractor 的精简文本, ~2KB)
    - tier: int (当前优化层级 1~6)
    - history: list (最近 N 轮记录, 从 trajectory.json 读)
    - kernel_code: str (当前 kernel 源码)
    - playbook_dir: Path (docx/ 目录)

  输出:
    - RoundPlan (strategy, target_speedup, specific_change, ...)
    - plan.md 写入本轮目录

═══════════════════════════════════════════════════════════════════════════════
  LLM 调用方式
═══════════════════════════════════════════════════════════════════════════════

  支持两种模式:
    - stub: 返回占位计划 (本地测试用)
    - anthropic: 调用 Claude API (生产环境)

  切换: 设置环境变量 ANTHROPIC_API_KEY 即启用真实模式
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoundPlan:
    """Planner 产出的优化计划。"""
    round_num: int
    tier: int
    tier_name: str
    strategy: str
    target_speedup: float
    specific_change: str
    expected_impact: str
    verification_method: str
    plan_text: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  Playbook 加载
# ═══════════════════════════════════════════════════════════════════════════════

PLAYBOOK_FILES = {
    1: "playbook_tier1_algorithm.md",
    2: "playbook_tier2_fusion.md",
    3: "playbook_tier3_tiling.md",
    4: "playbook_tier4_memory.md",
    5: "playbook_tier5_compute.md",
    6: "playbook_tier6_architecture.md",
}

TIER_NAMES = {
    1: "Algorithmic Structure", 2: "Operator Fusion",
    3: "Tiling & Block Config", 4: "Memory Access",
    5: "Compute & Occupancy", 6: "910B3 Architecture",
}


def _load_playbook(tier: int, playbook_dir: Optional[Path] = None) -> str:
    """加载当前 Tier 对应的 Playbook + CODING_GUIDE。"""
    if playbook_dir is None:
        playbook_dir = _PROJECT_DIR / "docx"
    fname = PLAYBOOK_FILES.get(tier)
    parts = []
    # CODING_GUIDE (所有 Tier 都读)
    guide_path = playbook_dir / "CODING_GUIDE.md"
    if guide_path.exists():
        parts.append("## Coding Guide (MUST FOLLOW)\n" + guide_path.read_text(encoding="utf-8")[:3000])
    # 主 Playbook
    if fname:
        fpath = playbook_dir / fname
        if fpath.exists():
            parts.append(fpath.read_text(encoding="utf-8"))
    return "\n\n".join(parts) if parts else "(no playbook for this tier)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Memory 检索 (对接 memory/ 模块)
# ═══════════════════════════════════════════════════════════════════════════════

def _retrieve_similar_cases(diagnosis, max_cases: int = 3) -> str:
    """从 memory/ 经验库检索相似案例。"""
    try:
        from memory.experience_retriever import retrieve, format_for_prompt

        cases = retrieve(
            op_type=getattr(diagnosis, "bottleneck_op_type", ""),
            bottleneck_type=getattr(diagnosis, "bottleneck_type", ""),
            engine=getattr(diagnosis, "bottleneck_engine", ""),
            tier=getattr(diagnosis, "current_tier", 1),
            max_results=max_cases,
        )
        if cases:
            return format_for_prompt(cases)
    except Exception:
        pass
    # Fallback: 尝试旧的 memory/ 模块
    try:
        from memory import compute_fingerprint, retrieve, format_context
        fp = compute_fingerprint({
            "op_type": getattr(diagnosis, "bottleneck_op_type", "?"),
            "bottleneck_type": getattr(diagnosis, "bottleneck_type", "?"),
            "engine": getattr(diagnosis, "bottleneck_engine", "?"),
        })
        from memory.store import ExperienceStore
        store_path = _PROJECT_DIR / "memory" / "experience" / "store.json"
        if store_path.exists():
            store = ExperienceStore(store_path)
            hits = retrieve(store, fp, n=max_cases)
            if hits:
                return format_context(hits)
    except Exception:
        pass
    return "(no similar cases found)"


# ═══════════════════════════════════════════════════════════════════════════════
#  Prompt 构建
# ═══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(tier: int, playbook: str) -> str:
    """构建 System Prompt。"""
    return f"""You are a Triton kernel optimizer for Ascend 910B3 NPU (triton 3.4.0, CANN 9.0).
Generate ONE concrete optimization change for Tier {tier}: {TIER_NAMES.get(tier, 'Unknown')}.

## Rules
1. ALWAYS output a real strategy name — NOT "algorithm_already_optimal"
2. "algorithm_already_optimal" ONLY allowed if ALL of these are true:
   - Tier 3+: bw_util > 90% on ALL ops AND
   - Tier 2: 0 RAW dependency chains AND
   - Tier 1: num_ops <= 3
3. Give exact code change (e.g. "change BLOCK_SIZE from 256 to 1024 on line 15")
4. Target speedup must be realistic (1.05-1.5x)

## Hardware
910B3: 20 AI Core + 40 Vec Core @ 1.8GHz, UB=192KB/core
GM→UB: 80.8 GB/s, UB→GM: 76.7 GB/s, VecUnit: 404 GB/s, CubeUnit: 150 GB/s

## Playbook (Tier {tier})
{playbook}

## Output JSON only:
```json
{{"strategy":"xxx","target_speedup":1.1,"specific_change":"...","expected_impact":"...","verification_method":"msprof"}}
```"""


def _format_history(history: list) -> str:
    """将 history 列表格式化为文本。"""
    if not history:
        return "(no history)"
    recent = history[-5:]
    lines = ["## Recent History (last 5 rounds)"]
    for r in recent:
        lines.append(
            f"- Round {r.get('round','?')}: {r.get('strategy','?')} → "
            f"{r.get('decision','?')} ({r.get('actual_speedup',1.0):.2f}x)")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Planner Agent
# ═══════════════════════════════════════════════════════════════════════════════

class PlannerAgent:
    """规划智能体 — Prompt 编排器。

    支持 3 种 LLM 调用模式 (通过 LLMClient):
      - api: DEEPSEEK_API_KEY / ANTHROPIC_API_KEY
      - cli: LLM_CLI_COMMAND="nga run"
      - stub: 无配置时返回占位计划

    Usage:
        planner = PlannerAgent(playbook_dir=Path("docx"))
        plan = planner.generate(...)
    """

    def __init__(self, playbook_dir: Optional[Path] = None,
                 use_llm: bool = True):
        self.playbook_dir = playbook_dir or (_PROJECT_DIR / "docx")
        self.use_llm = use_llm

    def generate(
        self,
        diagnosis,
        extracted_text: str,
        tier: int,
        history: list,
        kernel_code: str,
        round_num: int,
    ) -> RoundPlan:
        """生成本轮优化计划。"""

        # Step 1: 加载 Playbook
        playbook = _load_playbook(tier, self.playbook_dir)

        # Step 2: 检索相似案例
        similar_cases = _retrieve_similar_cases(diagnosis)

        # Step 3: 构建 Prompt (使用 ContextManager 做 token 管理)
        from memory.context_manager import build_context, format_diagnosis, estimate_tokens

        diagnosis_text = format_diagnosis(diagnosis)
        history_text = _format_history(history)
        full_prompt = build_context(
            diagnosis_text=diagnosis_text,
            extracted_text=extracted_text,
            playbook_text=playbook,
            history_text=history_text,
            similar_cases_text=similar_cases,
            kernel_code=kernel_code,
        )
        system_prompt = _build_system_prompt(tier, playbook)
        print(f"  [Planner] prompt ~{estimate_tokens(system_prompt + full_prompt):,} tokens")

        # Step 4: 调用 LLM (或 stub)
        if self.use_llm:
            try:
                plan_dict = self._call_llm(system_prompt, full_prompt)
            except Exception as e:
                print(f"  [Planner] LLM call failed: {e}, falling back to stub")
                plan_dict = self._stub_plan(diagnosis, tier)
        else:
            plan_dict = self._stub_plan(diagnosis, tier)

        # Step 5: 返回 RoundPlan
        return RoundPlan(
            round_num=round_num,
            tier=tier,
            tier_name=TIER_NAMES.get(tier, "?"),
            strategy=plan_dict.get("strategy", "unknown"),
            target_speedup=float(plan_dict.get("target_speedup", 1.05)),
            specific_change=str(plan_dict.get("specific_change", "")),
            expected_impact=str(plan_dict.get("expected_impact", "")),
            verification_method=str(plan_dict.get("verification_method", "")),
            plan_text=json.dumps(plan_dict, indent=2, ensure_ascii=False),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  LLM 调用
    # ═══════════════════════════════════════════════════════════════════════════

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        import os
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if deepseek_key:
            from openai import OpenAI
            client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-v4-pro", max_tokens=2048,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}])
            text = resp.choices[0].message.content or ""
        elif anthropic_key:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}])
            text = resp.content[0].text
        else:
            raise RuntimeError("No API key")

        print(f"  [Planner] LLM response: {len(text)} chars")
        if not text or not text.strip():
            raise ValueError("Empty LLM response")
        text = text.strip()
        if "```" in text:
            parts = text.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"): p = p[4:].strip()
                if p.startswith("{"): text = p; break
        start = text.find("{"); end = text.rfind("}")
        if start >= 0 and end > start: text = text[start:end+1]
        import re, json
        for attempt in range(3):
            try: return json.loads(text)
            except json.JSONDecodeError:
                if attempt == 0: text = re.sub(r':\s*([^{}"\s,]+)(?=\s*[,}])', r': "\1"', text)
                elif attempt == 1: text = re.sub(r'(?<=:)\s*([^"{}\[\],\s]+)(?=\s*[,}\]])', r' "\1"', text)
        return json.loads(text)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stub (本地测试)
    # ═══════════════════════════════════════════════════════════════════════════

    def _stub_plan(self, diagnosis, tier: int) -> dict:
        """根据 Tier 返回合理的 stub 计划 (启发式, 不依赖 LLM)。"""
        headroom = getattr(diagnosis, "optimization_headroom", "MEDIUM")
        btype = getattr(diagnosis, "bottleneck_type", "unknown")
        engine = getattr(diagnosis, "bottleneck_engine", "?")

        strategies = {
            1: ("evaluate_algorithm_choice",
                "检查当前算法是否最优 — 对照 playbook 算子→算法表"),
            2: ("identify_fusion_opportunities",
                "分析 RAW 链上的连续 VecUnit op, 找融合机会"),
            3: ("increase_tile_size" if headroom in ("HIGH", "MEDIUM") else "tune_block_config",
                "增大 BLOCK_SIZE 使传输进入饱和区" if headroom in ("HIGH", "MEDIUM")
                else "已饱和 — 检查 num_warps/num_stages"),
            4: ("merge_small_transfers" if "latency" in btype else "double_buffering",
                "合并小传输" if "latency" in btype
                else f"{engine} 已饱和 → double buffer 或减少数据量"),
            5: ("overlap_compute_transfer",
                "用 double buffer 让计算和传输重叠"),
            6: ("adjust_grid_count",
                "检查 engine_utilization 是否均衡, 调整 grid 分配"),
        }

        strat, change = strategies.get(tier, ("analyze_deeper", "进一步分析"))
        return {
            "strategy": f"[STUB] {strat}",
            "target_speedup": 1.05,
            "specific_change": f"[STUB] {change}",
            "expected_impact": f"[STUB] ~5% improvement on {engine}",
            "verification_method": "CPU emulator multi-shape test + simulator --llm",
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    from analyzers.bottleneck_diagnoser import diagnose_round

    outputs = _PROJECT_DIR / "outputs"
    rd = outputs / "vector_add_fp16_N65536" / "round0"

    if not (rd / "merged" / "merged_report.json").exists():
        print("[SKIP] merged_report.json not found")
        return

    planner = PlannerAgent()

    for tier in range(1, 7):
        diag = diagnose_round(rd, current_tier=tier)
        from analyzers.data_extractor import extract
        import json as _j
        with open(rd / "merged" / "merged_report.json", encoding="utf-8") as f:
            merged = _j.load(f)
        extracted = extract(merged, {
            "bottleneck": {"op_id": diag.bottleneck_op_id,
                           "op_type": diag.bottleneck_op_type,
                           "engine": diag.bottleneck_engine,
                           "type": diag.bottleneck_type,
                           "category": diag.bottleneck_category,
                           "headroom": diag.optimization_headroom,
                           "time_ratio": diag.bottleneck_time_ratio,
                           "bw_utilization": diag.bottleneck_bw_utilization,
                           "regime": diag.bottleneck_regime},
            "strategies": diag.suggested_strategies,
        }, tier=tier)

        plan = planner.generate(diag, extracted, tier, [], "// kernel code", 1)
        print(f"Tier {tier}: {plan.strategy}")
        print(f"  change: {plan.specific_change[:80]}...")

    print(f"\n[Planner] All 6 tiers OK (stub mode)")


if __name__ == "__main__":
    _self_test()
