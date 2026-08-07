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
    promote: bool = False          # planner 决策: 是否换层 (晋升/回退)
    promote_to: int = 0            # ★目标层: 0=本层, <当前=回退前层, >当前=晋升后层
    promote_reason: str = ""
    promote_evidence: str = ""     # ★P1: 晋升/回退的数据依据 (planner 必须给出, 调度器据此严格晋升)
    handoff: Optional[dict] = None  # ★P1: 跳转手递 (给目标 tier 的瓶颈分析+优化方向)


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
    # 主 Playbook (只取前 3500 — 关键约束表都提到文件最前, 保证进 prompt; 避免 18KB 全文拖慢 nga)
    if fname:
        fpath = playbook_dir / fname
        if fpath.exists():
            parts.append(fpath.read_text(encoding="utf-8")[:3500])
    return "\n\n".join(parts) if parts else "(no playbook for this tier)"


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
    """将 history 格式化为 ★每层全量 梗概 (不只最近5轮):
    每层: 轮数 / best / 最近2轮试了啥→结果(+报错) → planner 能判断该层是否已探索完、避免重复和回退死循环.
    ★REVERT 标「↩回退」; ★err 带报错 (coder old_code 没匹配等)."""
    if not history:
        return "(no history)"
    by_tier = {}
    for r in history:
        by_tier.setdefault(r.get("tier") if r.get("tier") is not None else 0, []).append(r)
    lines = ["## 历史梗概 (★每层试过什么→结果, 判断该层还有没有空间/是否已探索完)"]
    for t in sorted(by_tier):   # ★tier=None 已归 0, sorted 不崩
        rs = by_tier[t]
        best = max((r.get("speedup") or 0) for r in rs)
        last = rs[-2:]   # 最近2轮
        detail = "; ".join(
            (f"{r.get('change') or r.get('strategy','?')}→{r.get('speedup')}x[{r.get('result','')}]"
             # ★D2: 预期 vs 实际 — planner 记得自己上轮预期多少、实际多少 → 避免反复提同款无效策略
             + (f"(预期{r.get('expected_impact','?')} vs 实{r.get('speedup')}x)"
                if r.get("expected_impact") else "")
             + ("↩回退" if r.get("decision") == "REVERT" else "")
             + (f"(err:{r.get('error','')[:30]})" if r.get("error") else ""))
            for r in last)
        lines.append(f"- T{t}: {len(rs)}轮, best {best:.2f}x | 最近试: {detail}")
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
        context_path: Optional[Path] = None,      # ★P2: 完整数据上下文 JSON
        trajectory_path: Optional[Path] = None,   # ★P1: 全量轨迹 (每层试过什么)
        handoff: Optional[dict] = None,           # ★P1: 跳转来的瓶颈分析+方向
    ) -> RoundPlan:
        """v4 Planner: 输入 = 小指令 + 读文件路径 + 内联小字段 → plan + 晋升决策。

        ★不嵌入 playbook/kernel 全文, 让 nga run 自己读文件 (避免 prompt 超限/卡顿)。
        ★P1: 每轮都读 optimization_trajectory.json (了解各层进度/晋升策略) + 完整数据上下文 JSON
            (每 kernel 全量 task/deep + 占比 + Top耗时) + 跳转手递 (上个 planner 的瓶颈分析+方向)。
        额外输出 promote/promote_to/promote_reason/promote_evidence/handoff。"""
        history_text = _format_history(history)
        config_text = _extract_config_constants(kernel_code)   # ★从单文件 config 区提取
        fusion_text = (json.dumps(fusion_analysis, ensure_ascii=False)[:600]
                       if fusion_analysis else "(无融合分析)")
        handoff_text = (json.dumps(handoff, ensure_ascii=False)[:600]
                        if handoff else "(无跳转手递)")
        # ★优秀案例: 本 Tier 历史大加速比轮次, planner 作参考学习 (读不到/报错 → 空, 不阻断)
        try:
            from memory.excellent_cases import format_for_planner as _fmt_ec
            ec_text = _fmt_ec(tier)
        except Exception:
            ec_text = "(本层优秀案例读取失败, 忽略)"

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
            f"4. 读诊断字段文件: {d7_path or '(未给, 用下面内联字段)'}\n"
            f"5. ★读完整数据上下文 JSON (每 kernel 全量 task/deep + 耗时占比 + Top耗时 + 轨迹state + 历史 + 手递): {context_path or '(未给)'}  (分析瓶颈必读, 比 07 字段更全)\n"
            f"6. ★读优化轨迹: {trajectory_path or '(未给)'}  (看全部轮次的 tier/策略/加速比/结果 → 判断各层是否已榨干, 防重复/防误跳/防死循环回退)\n"
            f"7. (若有) ★读跳转手递: {handoff_text}  (上个 tier planner 分析出的瓶颈+优化方向, 作为部分参考; 仍要结合其他数据独立判断)\n"
            f"8. ★读本层优秀案例 (历史大加速比轮次, 参考学习, 别重复发明):\n{ec_text}\n\n"
            f"## 内联诊断字段 (当前 tier 筛好的, 若第4步文件读不到就用这些):\n{extracted[:1600]}\n\n"
            f"## config 常量:\n{config_text or '(无)'}\n"
            f"## 历史 (最近几轮):\n{history_text}\n"
            f"## 融合分析 (Tier2 才有):\n{fusion_text}\n\n"
            f"## 任务:\n"
            f"1. ★先做前层优先检查: 判断瓶颈是否在前层(算法/融合)还有优化空间?\n"
            f"   - 算力利用率低/cube没吃满 → 算法可能非最优 (Tier1 有空间)\n"
            f"   - 多kernel串行+launch开销大 → 有融合空间 (Tier2 有空间)\n"
            f"   - 有 → promote=true, promote_to=<前层>, 允许回退; ★绝不在本层硬调\n"
            f"   - 用 trajectory 查该前层是否已试过 → 若已充分探索且无果, 别反复回跳同一层\n"
            f"2. 然后判断: 当前瓶颈是否属于 Tier{tier}({TIER_NAMES.get(tier,'?')}) 的优化范畴?\n"
            f"   - 属于 → promote=false, promote_to=0, 给出 changes[]\n"
            f"   - 不属于且前层无空间 → promote=true, promote_to=<下一层>\n"
            f"3. ★changes[] 只允许对应当前 tier 的策略 (Tier3 只改 BLOCK_*, 禁止改 DTYPE/算法/融合; 各 tier 归属见 SKILL 铁律表)\n"
            f"4. changes[] 的 old_code 必须与【当前读到的 kernel_op.py(已含之前所有轮次 coder 的修改累积)】某段【逐字符】相同\n"
            f"   (★从读到的文件逐字复制, 绝不用示例/记忆里的旧值; 拿不准就不改, 宁缺勿错)\n"
            f"5. 只改单文件 kernel_op.py ①config/②kernel, 不碰其他文件; 不引入 num_warps/num_stages 到 @triton.jit()\n"
            f"6. 目标加速比 1.05~1.5x\n"
            f"7. ★严格晋升: 若 promote=true, 必须填 promote_evidence (用完整数据上下文/trajectory 里的数据+历史证明当前层已无优化空间), 否则调度器会拒绝晋升\n"
            f"8. ★跳转手递: 若 promote=true 且跳转到别的 tier, 填 handoff: {{\"to_tier\":N,\"bottleneck\":\"你分析出的瓶颈\",\"optimization_direction\":\"目标层应该做什么\"}} → 调度器写 10_tier_handoff.json 给目标层 planner 作参考\n\n"
            f"## 输出 JSON only (★先看真实示例再写):\n"
            f"例: 想把 BLOCK_K 增大 → ★old_code 必须逐字复制自【你读到的当前 kernel】, 不是示例里的数值\n"
            f'{{"strategy":"增大BLOCK_K减MTE1次数","target_speedup":1.1,\n'
            f'  "changes":[{{"old_code":"<从你读到的kernel里逐字复制要替换的那一行>","new_code":"<替换后的整行>","reason":"为什么","section":"① config/② kernel","tier":当前tier}}],\n'
            f'  "expected_impact":"...","promote":false,"promote_to":0,"promote_reason":"","promote_evidence":"",\n'
            f'  "handoff":{{"to_tier":0,"bottleneck":"","optimization_direction":""}}}}\n'
            f"格式 (★old_code 必须逐字符来自【当前读到的 kernel_op.py = 源 + 之前所有轮次修改累积】; 别用示例/记忆里的旧值; 拿不准就不改; tier 必须=当前层):\n"
            f'{{"strategy":"...","target_speedup":1.1,\n'
            f'  "changes":[{{"old_code":"被替换的整行(逐字复制自当前kernel)","new_code":"替换后的整行","reason":"为什么","section":"① config/② kernel","tier":N}}],\n'
            f'  "expected_impact":"...","promote":false,"promote_to":0,"promote_reason":"...","promote_evidence":"...",\n'
            f'  "handoff":{{"to_tier":N,"bottleneck":"...","optimization_direction":"..."}}}}'
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
                plan_dict = self._stub_plan_v4(tier, kernel_code)
        else:
            plan_dict = self._stub_plan_v4(tier, kernel_code)

        # ★防御性解析: LLM 可能返回 "1.1x"/"tier 2" 等带后缀/描述 → 提取数字, 失败用默认
        def _safe_float(v, dflt):
            try:
                return float(v)
            except (TypeError, ValueError):
                import re as _re
                m = _re.search(r"[-+]?\d*\.?\d+", str(v or ""))
                return float(m.group()) if m else dflt

        def _safe_int(v, dflt):
            try:
                return int(v)
            except (TypeError, ValueError):
                import re as _re
                m = _re.search(r"\d+", str(v or ""))
                return int(m.group()) if m else dflt

        return RoundPlan(
            round_num=round_num, tier=tier,
            tier_name=TIER_NAMES.get(tier, "?"),
            strategy=plan_dict.get("strategy", "unknown"),
            target_speedup=_safe_float(plan_dict.get("target_speedup", 1.05), 1.05),
            specific_change=str(plan_dict.get("specific_change", "")),
            expected_impact=str(plan_dict.get("expected_impact", "")),
            verification_method="msprof end-to-end",
            plan_text=json.dumps(plan_dict, ensure_ascii=False, indent=2),
            promote=bool(plan_dict.get("promote", False)),
            promote_to=_safe_int(plan_dict.get("promote_to", 0) or 0, 0),   # 目标层 (0=本层)
            promote_reason=str(plan_dict.get("promote_reason", "")),
            promote_evidence=str(plan_dict.get("promote_evidence", "")),
            handoff=(plan_dict.get("handoff")
                     if isinstance(plan_dict.get("handoff"), dict) else None),
        )

    def _stub_plan_v4(self, tier: int, kernel_code: str = "") -> dict:
        """stub 也输出 changes[] 格式.
        ★bug 修复: old_code 硬编码 "BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32" 只匹配 input/matmul,
          其余 op (attention_mlp=64,64,64 / conv2d=BLOCK_K,BLOCK_OW / rms_norm 无 BLOCK 行) 全部
          coder 匹配失败 → --stub 一轮都跑不通. 改为从当前 kernel config 提取真实 BLOCK 行."""
        import re as _re
        m = _re.search(r"BLOCK_M\s*,\s*BLOCK_N\s*,\s*BLOCK_K\s*=\s*[\d, ]+", kernel_code or "")
        if m:
            old = m.group(0).strip()
            vals = [int(x) for x in _re.split(r"[\d,]+", old) if x.strip().isdigit()]
            new = _re.sub(r"=\s*[\d, ]+", f"= {', '.join(str(v * 2) if v < 512 else v for v in vals)}", old)
            change = {"old_code": old, "new_code": new,
                      "reason": "STUB 增大分块", "section": "① config"}
        else:
            # conv2d 用分离行 (BLOCK_K = 32 独立一行, 非组合) → 逐行匹配
            m2 = _re.search(r"^\s*BLOCK_K\s*=\s*\d+", kernel_code or "", _re.M)
            if m2:
                old = m2.group(0).strip()
                val = int(_re.search(r"=\s*(\d+)", old).group(1))
                new = _re.sub(r"=\s*\d+", f"= {val * 2 if val < 512 else val}", old)
                change = {"old_code": old, "new_code": new,
                          "reason": "STUB 增大 BLOCK_K", "section": "① config"}
            else:
                change = None
        strategies = {
            1: "[STUB] tune_algorithm", 2: "[STUB] fuse_ops", 3: "[STUB] increase_tile",
            4: "[STUB] improve_mem", 5: "[STUB] reduce_conflict", 6: "[STUB] tune_arch",
        }
        changes = [change] if change else []
        s = strategies.get(tier, "[STUB] analyze")
        return {"strategy": s, "target_speedup": 1.05, "changes": changes,
                "expected_impact": "~5%", "promote": False, "promote_reason": "",
                "promote_evidence": "", "handoff": None}
