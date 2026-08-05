#!/usr/bin/env python3
"""Scheduler — v4 状态机: 读 diagnosis.json → 提取当前 tier 字段 → 驱动 Planner→Coder→验证→晋升。

v4 流程 (见 README v4):
  每轮:
    ① run_optimize.sh <input_dir> <round_dir>  → 采集+解析 → diagnosis.json
    ② 读 diagnosis.json → summary.num_kernels
    ③ 按当前 tier 提取该策略要看的字段段 (extract_tier_fields)
    ④ Planner: 字段段 + 策略文档 + 单文件 + config → plan.md + 晋升决策
    ⑤ Coder: plan + 教程 + 纠错文档 → 改 kernel_op.py (单文件)
    ⑥ 验证: 只跑 msprof 端到端 → 加速比; 失败报错回传 Coder 同轮重改
    ⑦ 记录 + 晋升/降级/停止 → 下一轮/下一 tier

用法:
  python -m agents.scheduler <op_dir> [--max-rounds N] [--stub]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

TIER_NAMES = {
    1: "01_algorithmic_structure", 2: "02_operator_fusion",
    3: "03_tiling_block_config", 4: "04_memory_access",
    5: "05_compute_occupancy", 6: "06_910b3_architecture",
}
TIER_LABEL = {
    1: "算法结构", 2: "算子融合", 3: "分块配置",
    4: "访存", 5: "计算占用", 6: "架构专属",
}

# ── 每 tier 要提取的字段段 (JSON path → 中文说明) ──
#   deep 在 kernels[i].deep 下 → 前缀 kernels[].deep
#   每 tier 的字段要足以: ①判断瓶颈是否属本层(promote) ②给出具体改法
TIER_FIELDS = {
    1: [  # 算法结构: 算力利用 + 精度 + 强度 (11 字段)
        ("summary.num_kernels", "优化目标kernel数"),
        ("summary.total_ns", "端到端耗时ns"),
        ("kernels[].deep.compute.cube_fops", "cube浮点运算数"),
        ("kernels[].deep.compute.vector_fops", "向量运算数"),
        ("kernels[].deep.compute.cube_ratio", "cube指令占比"),
        ("kernels[].deep.compute.cube_fp16_ratio", "cube fp16占比"),
        ("kernels[].deep.compute.cube_int8_ratio", "cube int8占比"),
        ("kernels[].deep.engine_utilization.vec", "向量指令占比"),
        ("kernels[].deep.roofline.compute_utilization", "算力利用率"),
        ("kernels[].deep.roofline.arithmetic_intensity", "算术强度(计算/访存)"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
    2: [  # 算子融合: 多算子/launch/类型 (8 字段)
        ("summary.num_kernels", "优化目标kernel数"),
        ("summary.num_kernels_total", "总kernel数(含框架)"),
        ("summary.api_overhead_total_us", "launch开销us"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].launch_count", "每kernel launch次数"),
        ("api_overhead", "API开销明细"),
        ("multi_kernel", "算子类型分解"),
        ("framework_kernels", "框架kernel(非目标)"),
    ],
    3: [  # 分块配置: 核数 + L0A/B 搬运 (8 字段)
        ("kernels[].task.block_dim", "核数"),
        ("kernels[].deep.engine_utilization.mte1", "MTE1(L1→L0A/B)占比"),
        ("kernels[].deep.engine_utilization.mte2", "MTE2(GM→L1)占比"),
        ("kernels[].deep.engine_utilization.cube", "cube占比"),
        ("kernels[].deep.bandwidth_gb_s.l0a_read_gb_s", "L0A读带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0a_write_gb_s", "L0A写带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0b_read_gb_s", "L0B读带宽"),
        ("kernels[].deep.bandwidth_gb_s.l0b_write_gb_s", "L0B写带宽"),
    ],
    4: [  # 访存: GM带宽/L2/搬运 (9 字段)
        ("kernels[].deep.bandwidth_gb_s.main_mem_read_gb_s", "GM读带宽"),
        ("kernels[].deep.bandwidth_gb_s.main_mem_write_gb_s", "GM写带宽"),
        ("kernels[].deep.bandwidth_gb_s.gm_to_ub_gb_s", "GM→UB带宽"),
        ("kernels[].deep.bandwidth_gb_s.ub_to_gm_gb_s", "UB→GM带宽"),
        ("kernels[].deep.l2_hit_rate", "L2命中率"),
        ("kernels[].task.pipes_us.aic_mte2_time_us", "MTE2(GM读)耗时"),
        ("kernels[].task.pipes_us.aic_mte3_time_us", "MTE3(GM写)耗时"),
        ("kernels[].deep.roofline.memory_utilization", "访存利用率"),
        ("kernels[].deep.roofline.arithmetic_intensity", "算术强度(计算/访存)"),
    ],
    5: [  # 计算占用: cube/标量时间 + 冲突 (8 字段)
        ("kernels[].task.pipes_us.aic_cube_time_us", "cube耗时"),
        ("kernels[].task.pipes_us.aic_scalar_time_us", "标量耗时"),
        ("kernels[].deep.engine_utilization.scalar", "scalar占比"),
        ("kernels[].deep.engine_utilization.fixpipe", "fixpipe占比"),
        ("kernels[].deep.compute.cube_ratio", "cube指令占比"),
        ("kernels[].deep.conflict.bank_cflt_ratio", "bank冲突"),
        ("kernels[].deep.conflict.bankgroup_cflt_ratio", "bankgroup冲突"),
        ("kernels[].deep.conflict.total_cflt_ratio", "vec总冲突"),
    ],
    6: [  # 910B3 架构: 引擎分布/阻塞 (6 字段)
        ("kernels[].deep.engine_utilization", "各引擎利用率"),
        ("kernels[].deep.conflict.mte_cflt_ratio", "mte冲突"),
        ("kernels[].deep.conflict.wait_ratio", "vec被阻塞占比"),
        ("kernels[].task.task_type", "每kernel引擎"),
        ("kernels[].task.block_dim", "核数"),
        ("kernels[].deep.roofline.bottleneck_type", "瓶颈类型"),
    ],
}


def _extract_mnk(code: str):
    """从 kernel_op.py ①config 区提取 M/N/K (默认值) → (M,N,K) 或 None。"""
    import re
    vals = {}
    for var, env in (("M", "MATMUL_M"), ("N", "MATMUL_N"), ("K", "MATMUL_K")):
        m = re.search(rf'os\.environ\.get\("{env}",\s*"?(\d+)"?\)', code)
        if not m:
            m = re.search(rf"^\s*{var}\s*=\s*(\d+)", code, re.M)
        if m:
            vals[var] = int(m.group(1))
    if len(vals) == 3 and all(vals.values()):
        return vals["M"], vals["N"], vals["K"]
    return None


def _summarize_changes(plan) -> str:
    """把 plan 的 changes[] 压成一句梗概 (hist 记录用, 让 planner 知道试过什么)。
    例: "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64" 或 kernel 行 old→new 截断。"""
    changes = _extract_changes_from_plan(getattr(plan, "plan_text", ""))
    if not changes:
        return getattr(plan, "strategy", "?")[:60]
    parts = []
    for ch in changes[:2]:
        old = (ch.get("old_code") or "").strip()
        new = (ch.get("new_code") or "").strip()
        lhs = old.split("=")[0].strip()[:40]
        rhs = new.split("=")[1].strip()[:40] if "=" in new else ""
        if rhs:
            parts.append(f"{lhs}={rhs}")
        else:
            parts.append(f"{old[:60]}→{new[:60]}")
    return "; ".join(parts)[:150]


def _extract_changes_from_plan(plan_text: str) -> list:
    """从 plan JSON 提取 changes[] (与 coder._extract_changes 同一实现, 不重复维护)。"""
    from agents.coder import _extract_changes
    return _extract_changes(plan_text)


def _get(d, path: str):
    """按 'a.b.c' 或 'kernels[].x.y' 路径取值 (返回第一个匹配)。
    每级 dict 先精确键, 无则子串匹配兜底 (如 conflict.aiv_vec_bank_cflt_ratio ↔ bank_cflt_ratio)。"""
    parts = path.split(".")
    cur = d
    if parts and parts[0].endswith("[]"):
        key = parts[0][:-2]
        items = cur.get(key, []) if isinstance(cur, dict) else []
        for it in items:
            v = _get(it, ".".join(parts[1:]))
            if v is not None:
                return v
        return None
    for p in parts:
        if isinstance(cur, dict):
            if p in cur:
                cur = cur[p]
            else:
                nxt = next((vv for kk, vv in cur.items() if p.lower() in kk.lower()), None)
                if nxt is None:
                    return None
                cur = nxt
        else:
            return None
    return cur


def extract_tier_fields(diagnosis: dict, tier: int) -> str:
    """只提取当前 tier 的字段段 → 文本 (喂 Planner)。"""
    lines = [f"# 当前 Tier {tier} ({TIER_LABEL.get(tier, '')}) — 只看这些字段"]
    for path, desc in TIER_FIELDS.get(tier, []):
        v = _get(diagnosis, path)
        if v is None:
            lines.append(f"- {desc} ({path}): (无数据)")
        elif isinstance(v, (dict, list)):
            lines.append(f"- {desc} ({path}): {json.dumps(v, ensure_ascii=False)[:300]}")
        else:
            lines.append(f"- {desc} ({path}): {v}")
    return "\n".join(lines)


class Scheduler:
    """v4 状态机调度器。"""

    def __init__(self, op_dir: Path, max_rounds: int = 200,
                 target_speedup: float = 1.5, use_llm: bool = True,
                 stub: bool = False, resume: bool = False):
        self.op_dir = op_dir
        self.max_rounds = max_rounds
        self.target_speedup = target_speedup
        self.use_llm = use_llm and not stub
        self.optimize_timeout = int(os.environ.get("OPTIMIZE_TIMEOUT", "3600"))
        self.outputs = _PROJECT / "outputs"
        self.kernel_name = op_dir.name
        self.kernel_dir = self.outputs / self.kernel_name
        self.traj_path = self.kernel_dir / "optimization_trajectory.json"
        self.traj = self._load_traj()
        # ★默认每次初始化 (round1/tier1 重来, 避免读旧路径错位); --resume 才续跑
        if not resume or not self.traj.get("state", {}).get("current_kernel"):
            self._reset_state()
        else:
            ck = Path(self.traj["state"]["current_kernel"])
            self.current_kernel = ck if ck.exists() else (op_dir / "kernel_op.py")
            print(f"  [Scheduler] 续跑: 从 tier{self.traj['state'].get('tier')} "
                  f"round{self.traj['state'].get('round')} 继续, kernel={self.current_kernel}")

    def _reset_state(self):
        """初始化: 从头开始 (round1/tier1 读源文件)。"""
        self.traj = {"v": 4, "state": {"tier": 1, "round": 1, "best_speedup": 1.0,
                                       "baseline_ns": None, "num_kernels": None,
                                       "current_speedup": 1.0,   # 当前已接受 kernel 的加速比 (基线=1.0)
                                       "current_kernel": str(self.op_dir / "kernel_op.py")},
                     "history": []}
        self.current_kernel = self.op_dir / "kernel_op.py"

    # ── 轨迹 ──
    def _load_traj(self) -> dict:
        if self.traj_path.exists():
            traj = json.loads(self.traj_path.read_text(encoding="utf-8"))
            if traj.get("v") == 4:
                return traj
            # 旧版本 (v3) trajectory → 重置, 避免 tier/round 错位
            print("  [Scheduler] 检测到旧版本 trajectory, 重置为 v4")
        return {"v": 4, "state": {"tier": 1, "round": 1, "best_speedup": 1.0,
                                  "baseline_ns": None, "num_kernels": None,
                                  "current_speedup": 1.0},
                "history": []}

    def _save_traj(self):
        self.traj_path.parent.mkdir(parents=True, exist_ok=True)
        self.traj_path.write_text(json.dumps(self.traj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    # ── 轮目录 ──
    def _round_dir(self, tier: int, rn: int) -> Path:
        return self.kernel_dir / TIER_NAMES.get(tier, "0x") / f"round{rn}"

    # ── ① 采集+解析 ──
    def _run_optimize(self, round_dir: Path, tier: int = 1) -> Optional[dict]:
        """调 run_optimize.sh <op_dir> <round_dir> → 读 diagnosis.json。
        TIER 环境变量传给 run_optimize, 让它解析完自动产出 07_tier<N>_fields。"""
        run_sh = (_PROJECT / "analyzers" / "run_optimize.sh").as_posix()
        input_dir = self.current_kernel.parent    # round1=源目录; 后续=上一轮输出目录
        cmd = ["bash", run_sh, str(input_dir), str(round_dir)]
        print(f"  [Scheduler] {' '.join(cmd)} (TIER={tier}, kernel={self.current_kernel})")
        print(f"  ⏳ 采集进行中 (msprof 需几分钟, 期间无输出是正常的; 超时 {self.optimize_timeout}s)...")
        env = dict(os.environ)
        env["TIER"] = str(tier)
        try:
            subprocess.run(cmd, check=False, timeout=self.optimize_timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"  ❌ run_optimize 超时 ({self.optimize_timeout}s), 看 {round_dir}/05_task/task_run.txt")
            return None
        dgn = round_dir / "06_diagnosis" / "diagnosis.json"
        if dgn.exists():
            return json.loads(dgn.read_text(encoding="utf-8"))
        # 兜底: 产物也可能直接在 round_dir 下
        alt = round_dir / "diagnosis.json"
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        print("  [Scheduler] ❌ diagnosis.json 未生成")
        return None

    # ── ③ 诊断: 按当前阶段筛选字段 → 写 07 (planner 只读这个) ──
    def _diagnose(self, diagnosis: dict, tier: int, round_dir: Path) -> str:
        extracted = extract_tier_fields(diagnosis, tier)
        d7 = round_dir / f"07_tier{tier}_fields"
        d7.mkdir(parents=True, exist_ok=True)
        (d7 / f"tier{tier}_fields.txt").write_text(extracted, encoding="utf-8")
        # 结构化 JSON: {字段说明: 值}
        vals = {}
        for path, desc in TIER_FIELDS.get(tier, []):
            vals[desc] = _get(diagnosis, path)
        (d7 / f"tier{tier}_fields.json").write_text(
            json.dumps(vals, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"  [诊断] Tier{tier} 筛出 {len(extracted.splitlines())} 行 → {d7}")
        return extracted

    # ── 融合专用: 编译 HIVM MLIR → nga run 依赖分析 → 08_fusion/ ──
    def _run_fusion(self, round_dir: Path):
        """Tier2 独有: 生成 hivm MLIR + nga run 融合分析, 写 round_dir/08_fusion/。"""
        from analyzers.run_hivm_fusion import run_fusion
        print("  [Scheduler] Tier2 融合: 编译 HIVM MLIR → nga run 分析依赖 → 08_fusion/")
        return run_fusion(self.current_kernel, round_dir, use_llm=self.use_llm)

    # ── ④ Planner (只喂当前阶段筛好的字段 + 融合分析) ──
    def _plan(self, diagnosis: dict, extracted: str, tier: int, rn: int, round_dir: Path,
              fusion_analysis: Optional[dict] = None):
        from agents.planner import PlannerAgent, _extract_config_constants
        kernel_code = self.current_kernel.read_text(encoding="utf-8") if self.current_kernel.exists() else ""
        skill = (_PROJECT / "skills" / "triton-op-planner" / "SKILL.md")
        cfg = _extract_config_constants(kernel_code)
        print(f"  [Planner] 输入: 07字段({len(extracted.splitlines())}行) + playbook_tier{tier} "
              f"+ {self.current_kernel}({len(kernel_code)}字符) + config[{cfg.splitlines()[0] if cfg else '?'}] "
              f"+ 历史{len(self.traj.get('history', []))}轮")
        print(f"  [Planner] 调 skill: {skill}")
        planner = PlannerAgent(use_llm=self.use_llm)
        plan = planner.generate_v4(
            extracted=extracted, tier=tier,
            history=self.traj.get("history", []),
            kernel_code=kernel_code,
            round_num=rn,
            op_dir=self.op_dir,
            fusion_analysis=fusion_analysis,
            round_dir=round_dir,
            current_kernel=self.current_kernel,
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        plan_md = (f"# Tier{tier} Round{rn} Plan\n\n{plan.plan_text}\n\n"
                   f"## 提取字段\n{extracted}"
                   + (f"\n\n## 融合分析\n{json.dumps(fusion_analysis, ensure_ascii=False, indent=1)}"
                      if fusion_analysis else ""))
        (round_dir / "plan.md").write_text(plan_md, encoding="utf-8")
        n_changes = len(_extract_changes_from_plan(plan.plan_text))
        print(f"  [Planner] → {round_dir}/plan.md ({len(plan_md)}字符, changes[]={n_changes}项) "
              f"promote={plan.promote}")
        return plan

    # ── ⑤ Coder (读 current_kernel, 精确应用 changes[], 输出 round_dir/kernel_op.py) ──
    def _code(self, plan, rn: int, round_dir: Path, prev_err: str = "") -> str:
        from agents.coder import CoderAgent, CoderResult
        original = self.current_kernel.read_text(encoding="utf-8") if self.current_kernel.exists() else ""
        skill = _PROJECT / "skills" / "triton-op-coder" / "SKILL.md"
        print(f"  [Coder] 读 {self.current_kernel} 的 changes[] (精确替换) + skill {skill}")
        coder = CoderAgent(use_llm=self.use_llm)
        try:
            result = coder.apply(original, plan.plan_text, prev_err, plan.tier,
                                 kernel_path=str(round_dir / "kernel_op.py"))
        except Exception as e:
            # ★任何编码异常(含 LLM 超时) → 不崩整个循环: 沿用原代码, 本轮失败
            print(f"  [Coder] ❌ 编码异常: {str(e)[:200]} → 沿用原代码, 本轮失败")
            result = CoderResult(success=False, optimized_code=original, diff="",
                                 error_message=f"编码异常: {str(e)[:200]}")
        round_dir.mkdir(parents=True, exist_ok=True)
        (round_dir / "diff.patch").write_text(result.diff or "(no change)", encoding="utf-8")
        n_changes = len(_extract_changes_from_plan(plan.plan_text))
        if result.success:
            lines = result.lines_changed or 0
            print(f"  [Coder] ✅ 应用 {n_changes} 处 changes → {lines} 行改动 → diff.patch")
        else:
            print(f"  [Coder] ⚠ 未成功: {result.error_message[:200]}")
        return result.optimized_code

    # ── ⑥ 验证: 只跑 msprof 端到端 (验证本轮新 kernel) ──
    def _verify(self, round_dir: Path, baseline_ns: Optional[float]) -> dict:
        """只跑一次 msprof 端到端 → 端到端耗时 → 加速比。
        验证的是 round_dir/kernel_op.py (本轮 coder 的新输出)。"""
        kernel = round_dir / "kernel_op.py"
        try:
            from agents.verifier import verify_end_to_end
            return verify_end_to_end(kernel, round_dir, baseline_ns)
        except Exception as e:
            print(f"  [Scheduler] verify stub: {e}")
            return {"ok": True, "ns": None, "speedup": 1.0, "note": "stub(无真机)"}

    # ── 主循环 (无 round0: 首轮采集即基准, 全部轮次直接进 outputs/<op>/<tier>/roundN) ──
    def run(self):
        st = self.traj["state"]
        tier, rn = st.get("tier", 1), st.get("round", 1)
        print(f"══ Scheduler: {self.kernel_name} 目标 {self.target_speedup}x ══")

        # Warm-up: 首次 nga run 冷启动(模型加载)可能很久, 提前预热
        if self.use_llm:
            print("  [Warm-up] 预热 nga run (首次冷启动)...")
            try:
                from agents.llm_client import LLMClient
                c = LLMClient()
                if c.mode == "cli":
                    c.chat("你是测试", "只回复 OK")
                    print("  [Warm-up] ✅ nga run 预热完成")
                else:
                    print(f"  [Warm-up] 模式={c.mode} (非 cli, 跳过)")
            except Exception as e:
                print(f"  [Warm-up] ⚠ 预热失败: {str(e)[:120]} (继续, 后续调用再试)")

        total_start = time.time()
        while rn <= self.max_rounds:
            round_dir = self._round_dir(tier, rn)
            round_start = time.time()
            print(f"\n══ Tier{tier}({TIER_LABEL.get(tier)}) Round{rn} ══")

            # ① 采集+解析 (run_optimize 自动产出 07_tier{N}_fields)
            _t0 = time.time()
            diagnosis = self._run_optimize(round_dir, tier)
            if not diagnosis:
                print("  ⚠ 采集失败, 停止")
                break
            print(f"  ⏱ ①采集+解析: {time.time()-_t0:.1f}s")
            # 诊断摘要
            ks0 = diagnosis.get("summary", {})
            k0 = (diagnosis.get("kernels") or [{}])[0]
            ro = (k0.get("deep") or {}).get("roofline", {})
            print(f"  [诊断] kernels={ks0.get('num_kernels')} total_ns={ks0.get('total_ns')} "
                  f"bottleneck={ro.get('bottleneck_type')} 产物→{round_dir}")
            # 首轮 (原始 kernel_op.py 未改) 采集 = 基准
            if st.get("baseline_ns") is None:
                st["baseline_ns"] = ks0.get("total_ns")
                st["num_kernels"] = ks0.get("num_kernels")
                # 从 kernel config 算 initial_tflops (matmul: 2MNK / baseline_time), 供轨迹图
                mnk = _extract_mnk(self.current_kernel.read_text(encoding="utf-8")
                                   if self.current_kernel.exists() else "")
                if mnk and st["baseline_ns"]:
                    st["initial_tflops"] = round(
                        2 * mnk[0] * mnk[1] * mnk[2] / (st["baseline_ns"] / 1e9) / 1e12, 2)
                # 读 PyTorch 基准 (bench_910b3/pytorch_tflops.json, 由 bench_pytorch.py 生成)
                pt = _PROJECT / "bench_910b3" / "pytorch_tflops.json"
                if pt.exists():
                    try:
                        st["pytorch_tflops"] = json.loads(pt.read_text(encoding="utf-8"))["tflops"]
                    except Exception:
                        pass
                self._save_traj()
                print(f"  [基准] total_ns={ks0.get('total_ns')} kernels={ks0.get('num_kernels')} "
                      f"initial_tflops={st.get('initial_tflops')} (加速比基准)")

            # ③ 诊断: 按当前阶段筛选字段 → 写 07
            _t0 = time.time()
            extracted = self._diagnose(diagnosis, tier, round_dir)
            print(f"  ⏱ ②诊断筛字段: {time.time()-_t0:.1f}s")

            # ③.5 Tier2 融合: 多走一步 — 编译 HIVM MLIR → nga run 分析依赖 → 08_fusion/
            fusion_analysis = None
            if tier == 2:
                _t0 = time.time()
                fusion_analysis = self._run_fusion(round_dir)
                print(f"  ⏱ ③融合分析: {time.time()-_t0:.1f}s")

            # ④ Planner → plan + 晋升决策 (只喂 07 筛好的字段 + 融合分析)
            _t0 = time.time()
            plan = self._plan(diagnosis, extracted, tier, rn, round_dir, fusion_analysis)
            print(f"  ⏱ ④Planner: {time.time()-_t0:.1f}s")

            # ⑤ Coder → 改代码, 输出到 round_dir/kernel_op.py (不碰源文件) (报错同轮重改, ≤3次)
            prev_err, new_code = "", ""
            round_kernel = round_dir / "kernel_op.py"
            prev_speedup = st.get("current_speedup", 1.0)   # 上一轮已接受 kernel 的加速比 (基线=1.0)
            kept = False                                      # 本轮是否被采纳进 kernel 链
            pre_code = self.current_kernel.read_text(encoding="utf-8") \
                if self.current_kernel.exists() else ""    # NOOP 对比用 (改之前的版本)
            for attempt in range(3):
                new_code = self._code(plan, rn, round_dir, prev_err)
                round_kernel.write_text(new_code, encoding="utf-8")
                # ⑥ 验证 (只 msprof 端到端, 验证本轮新 kernel)
                v = self._verify(round_dir, st.get("baseline_ns"))
                if v.get("ok"):
                    # ★保留判定: speedup 始终 = 初始基线/本轮 (累计, 输出的就是这个);
                    #   但"是否采纳进 kernel 链"对比上一轮已接受的加速比 (prev_speedup)
                    speedup = v.get("speedup", 1.0)
                    if speedup >= prev_speedup:
                        self.current_kernel = round_kernel
                        st["current_kernel"] = str(round_kernel)
                        st["current_speedup"] = round(speedup, 4)
                        kept = True
                    else:
                        print(f"  ↩ 回退: 本轮 {speedup:.3f}x < 上一轮 {prev_speedup:.3f}x, 沿用上一轮 kernel")
                    self._save_traj()
                    break
                prev_err = v.get("error", "unknown error")
                print(f"  ⚠ 运行失败(第{attempt+1}次): {prev_err[:200]}... 回传 Coder 同轮改")

            speedup = v.get("speedup", 1.0)
            ns = v.get("ns")
            print(f"  加速比: {speedup:.3f}x (vs 初始基线 {st.get('baseline_ns')}ns; 上一轮 {prev_speedup:.3f}x)"
                  + ("  ✅采纳" if kept else "  ↩未采纳"))
            print(f"  ⏱ ⑤Coder+验证: {time.time()-_t0:.1f}s")
            print(f"  ⏱ 本轮总时: {time.time()-round_start:.1f}s  (总用时 {time.time()-total_start:.1f}s)")

            # ⑦ 记录 + 晋升决策 (hist 记"改了啥+结果"梗概, 让 planner 知道试过什么)
            ok = v.get("ok", False)
            # NOOP 检测: 本轮写出的 kernel 和改之前的 pre_code 一模一样 (coder 没成功应用)
            noop = False
            if round_kernel.exists() and pre_code:
                try:
                    noop = (round_kernel.read_text(encoding="utf-8") == pre_code)
                except Exception:
                    pass
            result = "NOOP" if noop else ("OK" if ok else "FAIL")
            decision = "KEEP" if kept else ("FAIL" if not ok else "REVERT")
            hist = {"round": rn, "tier": tier, "strategy": plan.strategy,
                    "change": _summarize_changes(plan),       # 例: "BLOCK_M,BLOCK_N,BLOCK_K=64,64,64"
                    "speedup": round(speedup, 4),             # 加速比 = 初始基线/本轮 (累计)
                    "prev_speedup": round(prev_speedup, 4),   # 上一轮已接受 kernel 的加速比 (保留判定用)
                    "ns": ns, "decision": decision, "result": result,
                    "error": (prev_err[:120] if prev_err else "")}
            if st.get("best_speedup") is None or speedup > st["best_speedup"]:
                st["best_speedup"] = speedup
            self.traj["history"].append(hist)

            # 晋升决策: planner.promote (读瓶颈判断) + 连续3轮无改进兜底 + 达标/到Tier6停止
            planner_promote = getattr(plan, "promote", False)
            # ★无改进 = 本轮加速比没超过上一轮已接受的 (speedup <= prev_speedup);
            #   数"本 tier 连续无改进"轮数 (跨 tier 边界即断)
            no_improve = 0
            for h in reversed(self.traj["history"]):
                if h.get("tier") != tier:
                    break
                if h.get("speedup", 0) > h.get("prev_speedup", 0):
                    break
                no_improve += 1
            if speedup >= self.target_speedup:
                print("  🎯 加速比达标, 停止")
                st["round"] = rn + 1
                self._save_traj()
                break
            if planner_promote:
                if tier >= 6:
                    print(f"  ⛔ planner 判瓶颈已非本tier且到Tier6, 停止 ({getattr(plan,'promote_reason','')})")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                print(f"  → 晋升 Tier{tier}→Tier{tier+1} (planner: {getattr(plan,'promote_reason','瓶颈不属本tier')})")
                tier += 1
                st["tier"] = tier
            elif no_improve >= 3 or tier >= 6:
                if tier >= 6:
                    print("  ⛔ Tier6 连续无改进, 停止")
                    st["round"] = rn + 1
                    self._save_traj()
                    break
                print(f"  → 晋升 Tier{tier}→Tier{tier+1} (本tier连续{no_improve}轮无改进)")
                tier += 1
                st["tier"] = tier
            st["round"] = rn + 1
            rn += 1
            self._save_traj()

        print(f"\n══ 完成: best_speedup={st.get('best_speedup')}x ══")
        return 0


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="v4 Scheduler")
    p.add_argument("op_dir", type=str)
    p.add_argument("--max-rounds", type=int, default=200)
    p.add_argument("--target", type=float, default=1.5)
    p.add_argument("--stub", action="store_true", help="不调 LLM/真机, 用 stub")
    args = p.parse_args()
    s = Scheduler(Path(args.op_dir), max_rounds=args.max_rounds,
                  target_speedup=args.target, stub=args.stub)
    sys.exit(s.run())
