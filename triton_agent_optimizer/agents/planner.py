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
    promote: bool = False          # planner 决策: 是否晋升下一 tier
    promote_reason: str = ""


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
    """加载当前 Tier 对应的 Playbook + CODING_GUIDE (裁剪, 控制 prompt 大小)。"""
    if playbook_dir is None:
        playbook_dir = _PROJECT_DIR / "docx"
    fname = PLAYBOOK_FILES.get(tier)
    parts = []
    # CODING_GUIDE (所有 Tier 都读, 只取前 1500)
    guide_path = playbook_dir / "CODING_GUIDE.md"
    if guide_path.exists():
        parts.append("## Coding Guide (MUST FOLLOW)\n" + guide_path.read_text(encoding="utf-8")[:1500])
    # 主 Playbook (只取前 2500, 含优化内容表 + 决策依据; 避免 18KB 全文拖慢 nga)
    if fname:
        fpath = playbook_dir / fname
        if fpath.exists():
            parts.append(fpath.read_text(encoding="utf-8")[:2500])
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


def _extract_kernel_section(code: str, max_chars: int = 2500) -> str:
    """只取 config ① + kernel ② 区 (跳过测试 main ③), 控制 prompt 大小。"""
    # 以真正的函数 def main(): 为截断点 (避免头部注释里提到"测试 main"提前截断)
    for marker in ("def main():", "if __name__", "#  ③ 测试 main"):
        idx = code.find(marker)
        if idx > 0:
            return code[:idx].rstrip()[:max_chars]
    return code[:max_chars]


def _extract_config_constants(kernel_code: str) -> str:
    """从单文件 kernel_op.py 的 ① config 区提取常量 (M/N/K/DTYPE/BLOCK_*)。
    只扫 @triton.jit 之前的 config 区, 避免抓到 launch 调用的关键字参数。"""
    import re
    head = kernel_code.split("@triton.jit")[0]   # ① config 区 (kernel 之前)
    out = []
    for m in re.finditer(r"^\s*([\w, ]+?)\s*=\s*([^#\n]+)", head, re.M):
        vars_ = [v.strip() for v in m.group(1).split(",") if v.strip()]
        if len(vars_) > 1:                        # BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            vals = [v.strip() for v in m.group(2).split(",")]
            if len(vars_) == len(vals):
                for vv, vv_val in zip(vars_, vals):
                    out.append(f"{vv} = {vv_val}")
        else:                                      # M = int(os.environ.get(...)) / DTYPE = ...
            out.append(f"{vars_[0]} = {m.group(2).strip()}")
    return "\n".join(out)


def _format_history(history: list) -> str:
    """将 history 格式化为紧凑梗概 (每轮一行: 改了啥 → 加速比 [结果])。
    ★读 speedup (不是 actual_speedup — 旧 bug 恒显示 1.00x)。
    ★REVERT(变慢回退) 轮标「↩回退」— planner 知道别重复这个坏改动。"""
    if not history:
        return "(no history)"
    recent = history[-5:]   # 只看最近5轮, 紧凑
    lines = ["## 历史梗概 (最近5轮: 改了啥 → 加速比 [结果])"]
    for r in recent:
        change = r.get("change") or r.get("strategy", "?")
        sp = r.get("speedup")
        sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "?"
        res = r.get("result", "")
        line = f"- R{r.get('round','?')} t{r.get('tier','?')}: {change} → {sp_s}"
        if r.get("decision") == "REVERT":
            line += " ↩回退"          # 变慢/未采纳 → 明确标记, 防重复优化
        if res:
            line += f" [{res}]"
        if r.get("error"):
            line += f" (err: {r['error'][:50]})"
        lines.append(line)
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
    #  v4: 只喂当前 tier 提取的字段段 + 策略文档 + 单文件 + config → plan + 晋升决策
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_v4(
        self,
        extracted: str,
        tier: int,
        history: list,
        kernel_code: str,
        round_num: int,
        op_dir: Optional[Path] = None,
        fusion_analysis: Optional[dict] = None,
        round_dir: Optional[Path] = None,
        current_kernel: Optional[Path] = None,
    ) -> RoundPlan:
        """v4 Planner: 输入 = 小指令 + 读文件路径 + 内联小字段 → plan + 晋升决策。

        ★不嵌入 playbook/kernel 全文, 让 nga run 自己读文件 (避免 prompt 超限/卡顿)。
        额外输出 promote: bool + promote_reason (当前瓶颈是否属于本 tier, 决定晋升)。"""
        history_text = _format_history(history)
        config_text = _extract_config_constants(kernel_code)   # ★从单文件 config 区提取
        fusion_text = (json.dumps(fusion_analysis, ensure_ascii=False)[:600]
                       if fusion_analysis else "(无融合分析)")

        # 读文件路径 (绝对路径, nga run 自己读)
        skill_path = self.playbook_dir.parent / "skills" / "triton-op-planner" / "SKILL.md"
        pb_fname = PLAYBOOK_FILES.get(tier, "")
        playbook_path = self.playbook_dir / pb_fname if pb_fname else self.playbook_dir
        d7_path = ""
        if round_dir:
            d7_path = round_dir / f"07_tier{tier}_fields" / f"tier{tier}_fields.txt"

        system_prompt = (
            f"你是 Triton 优化 Planner。按下面步骤执行, 只输出 JSON。\n\n"
            f"执行步骤 (调用 skill / 读文件, 不要复述大段内容):\n"
            f"1. 调用 skill: {skill_path}\n"
            f"2. 读优化策略文档: {playbook_path}  (只看『优化内容』表 + 『决策依据』段)\n"
            f"3. 读当前单文件 (★当前正在优化的版本, 不是源文件): {current_kernel or (op_dir / 'kernel_op.py' if op_dir else '(未给)')}  (重点 ② kernel 区)\n"
            f"4. 读诊断字段文件: {d7_path or '(未给, 用下面内联字段)'}\n\n"
            f"## 内联诊断字段 (当前 tier 筛好的, 若第4步文件读不到就用这些):\n{extracted}\n\n"
            f"## config 常量:\n{config_text or '(无)'}\n"
            f"## 历史 (最近几轮):\n{history_text}\n"
            f"## 融合分析 (Tier2 才有):\n{fusion_text}\n\n"
            f"## 任务:\n"
            f"1. 先判断: 当前瓶颈是否属于 Tier{tier}({TIER_NAMES.get(tier,'?')}) 的优化范畴?\n"
            f"   - 属于 → promote=false, 给出 changes[]\n"
            f"   - 不属于 → promote=true, promote_reason 说明该晋升到哪个 tier\n"
            f"2. changes[] 的 old_code 必须与 kernel_op.py 某段【逐字符】相同 (coder 精确替换用); 拿不准就不改, 宁缺勿错\n"
            f"3. 只改单文件 kernel_op.py ①config/②kernel, 不碰其他文件; 不引入 num_warps/num_stages 到 @triton.jit()\n"
            f"4. 目标加速比 1.05~1.5x\n\n"
            f"## 输出 JSON only (★先看真实示例再写):\n"
            f"例: 想把 BLOCK_K 增大 → \n"
            f'{{"strategy":"增大BLOCK_K减MTE1次数","target_speedup":1.1,\n'
            f'  "changes":[{{"old_code":"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32","new_code":"BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64","reason":"Tier3: mte1_ratio高, 增大BLOCK_K减少MTE1搬运","section":"① config"}}],\n'
            f'  "expected_impact":"MTE1搬运次数减半","promote":false,"promote_reason":""}}\n'
            f"格式 (old_code 必须逐字符匹配当前 kernel):\n"
            f'{{"strategy":"...","target_speedup":1.1,\n'
            f'  "changes":[{{"old_code":"被替换的整行","new_code":"替换后的整行","reason":"为什么","section":"① config/② kernel"}}],\n'
            f'  "expected_impact":"...","promote":false,"promote_reason":"..."}}'
        )

        # v4: 统一走 LLMClient (api / nga run CLI / stub), 并引用 planner skill
        from agents.llm_client import LLMClient, extract_json
        client = LLMClient()
        if self.use_llm and client.mode != "stub":
            system = (f"你是 Triton 优化 Planner。先调用 skill: {skill_path}, "
                      f"完全按 skill 指导执行。")
            try:
                resp = client.chat(system=system, user=system_prompt)
                plan_dict = extract_json(resp)
            except Exception as e:
                print(f"  [Planner] LLM call failed: {e}, fallback stub")
                plan_dict = self._stub_plan_v4(tier)
        else:
            plan_dict = self._stub_plan_v4(tier)

        return RoundPlan(
            round_num=round_num, tier=tier,
            tier_name=TIER_NAMES.get(tier, "?"),
            strategy=plan_dict.get("strategy", "unknown"),
            target_speedup=float(plan_dict.get("target_speedup", 1.05)),
            specific_change=str(plan_dict.get("specific_change", "")),
            expected_impact=str(plan_dict.get("expected_impact", "")),
            verification_method="msprof end-to-end",
            plan_text=json.dumps(plan_dict, ensure_ascii=False, indent=2),
            promote=bool(plan_dict.get("promote", False)),
            promote_reason=str(plan_dict.get("promote_reason", "")),
        )

    def _stub_plan_v4(self, tier: int) -> dict:
        """stub 也输出 changes[] 格式 (old_code 用默认 matmul 配置, 便于本地测确定性替换)。"""
        strategies = {
            1: ("[STUB] tune_algorithm",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "增大 BLOCK_K 减 MTE1 次数", "section": "① config"}]),
            2: ("[STUB] fuse_ops",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "融合需要更大 block 减少 kernel 数", "section": "① config"}]),
            3: ("[STUB] increase_tile",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "Tier3: mte1_ratio 高 → 增大 BLOCK_K", "section": "① config"}]),
            4: ("[STUB] improve_mem",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "Tier4: 增大 tile 提高 L2 复用", "section": "① config"}]),
            5: ("[STUB] reduce_conflict",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "Tier5: 调整 block 减冲突", "section": "① config"}]),
            6: ("[STUB] tune_arch",
                [{"old_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32",
                  "new_code": "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64",
                  "reason": "Tier6: 调 block 均衡引擎占用", "section": "① config"}]),
        }
        s, changes = strategies.get(tier, ("analyze", []))
        return {"strategy": s, "target_speedup": 1.05, "changes": changes,
                "expected_impact": "~5%", "promote": False, "promote_reason": ""}

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
