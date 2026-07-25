"""loop-controller：把停止决策完全放进确定性代码（T3，全量实现）。

设计依据：架构文档 §3.5。控制器只读机器可读的 SimResult / Verdict，不靠 LLM 散文判断。
停止条件（满足任一即停，PASS 轮按下列优先级判定）：
  1. IRREDUCIBLE   —— 显式信号：瓶颈不可约。
  2. EPSILON       —— 本轮是新的 best-so-far，但相对改善 < ε（边际收益递减）。
  3. OSCILLATION   —— kernel 变体来回振荡（A,B,A 三轮窗口）。
  4. NO_PROGRESS   —— 连续 2 个 PASS 轮未刷新 best（含回退/持平）。
  5. MAX_ROUNDS    —— 轮数达 N（兜底护栏）。
另有 NUMERICAL_FAIL：显式 regen_failed=True（FAIL 且重生成仍不过），见 §3.5 #5。

两条计算口径（§3.5）：
  - 改善只在 correct=True 的轮次上计算；FAIL 轮的 cycles 无意义，不参与。
  - 但 FAIL 轮仍计入轮数预算（否则反复算错会永远不达上限）。
  - 因此「连续 FAIL」本身不提前停（交给 MAX_ROUNDS）；只有显式 regen_failed 才走 NUMERICAL_FAIL。

best-so-far 底线：全程保留历史最优变体；停止/回退时返回实测 cycles 最优的那一版，
而非最后一轮那一版。

ε 与 N 为可配置参数（构造参数 > 环境变量 LOOP_EPSILON / LOOP_MAX_ROUNDS > 默认值），
不硬编码——保密环境可按仿真成本调整。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .contracts import SimResult, Verdict

DEFAULT_EPSILON = 0.05
DEFAULT_MAX_ROUNDS = 6


class StopReason:
    NONE = "continue"
    EPSILON = "epsilon_converged"
    MAX_ROUNDS = "max_rounds"
    IRREDUCIBLE = "irreducible"
    OSCILLATION = "oscillation"
    NO_PROGRESS = "no_progress"
    NUMERICAL_FAIL = "numerical_fail"


@dataclass
class BestVariant:
    """历史最优变体。停止时返回这个，而不是最后一轮。"""

    variant_id: str
    cycles: int
    round_no: int


@dataclass
class Decision:
    should_stop: bool
    reason: str
    best: Optional[BestVariant]
    rolled_back: bool = False       # 本轮 PASS 但比 best 差 -> 回退保护触发
    improved_best: bool = False     # 本轮刷新了 best
    rel_improvement: float = 0.0    # 本轮相对 best-so-far 的改善（新 best 时才有意义）

    @property
    def converged(self) -> bool:
        return self.should_stop and self.reason != StopReason.NONE


def _resolve(arg, default, env_name, cast):
    if arg is not None:
        return cast(arg)
    env = os.environ.get(env_name)
    if env is not None and env != "":
        return cast(env)
    return default


class LoopController:
    def __init__(
        self,
        epsilon: Optional[float] = None,
        max_rounds: Optional[int] = None,
    ):
        self.epsilon: float = _resolve(epsilon, DEFAULT_EPSILON, "LOOP_EPSILON", float)
        self.max_rounds: int = _resolve(max_rounds, DEFAULT_MAX_ROUNDS, "LOOP_MAX_ROUNDS", int)
        self._best: Optional[BestVariant] = None
        self._round = 0
        self._variant_seq: list[str] = []
        self._pass_bottlenecks: list[Optional[str]] = []
        self._non_improving_streak = 0   # 连续未刷新 best 的 PASS 轮数

    # ---- read-only state ----
    @property
    def best(self) -> Optional[BestVariant]:
        return self._best

    @property
    def round_count(self) -> int:
        return self._round

    # ---- main entry ----
    def update(
        self,
        variant_id: str,
        sim: SimResult,
        verdict: Optional[Verdict] = None,
        *,
        irreducible: bool = False,
        regen_failed: bool = False,
    ) -> Decision:
        self._round += 1
        self._variant_seq.append(variant_id)

        if not sim.correct:
            return self._fail_round(regen_failed)

        if sim.cycles is None:
            raise ValueError("correct=True round must carry cycles to assess improvement")

        return self._pass_round(variant_id, sim, verdict, irreducible)

    # ---- FAIL branch ----
    def _fail_round(self, regen_failed: bool) -> Decision:
        # FAIL 轮：不参与改善计算；best 不变；但计入轮数预算。
        if regen_failed:
            return Decision(True, StopReason.NUMERICAL_FAIL, self._best)
        if self._round >= self.max_rounds:
            return Decision(True, StopReason.MAX_ROUNDS, self._best)
        return Decision(False, StopReason.NONE, self._best)

    # ---- PASS branch ----
    def _pass_round(self, variant_id, sim, verdict, irreducible) -> Decision:
        cycles = sim.cycles
        had_prior = self._best is not None
        prev = self._best.cycles if had_prior else None

        decision = Decision(False, StopReason.NONE, self._best)

        if not had_prior or cycles < self._best.cycles:
            # 刷新 best
            rel = (prev - cycles) / prev if had_prior else 0.0
            self._best = BestVariant(variant_id, cycles, self._round)
            self._non_improving_streak = 0
            decision.improved_best = True
            decision.rel_improvement = rel
        else:
            # 未刷新 best（持平或回退）
            decision.rolled_back = cycles > self._best.cycles
            self._non_improving_streak += 1
        decision.best = self._best

        bn = verdict.bottleneck if verdict else None
        self._pass_bottlenecks.append(bn)

        # ---- 停止判定（优先级）----
        if irreducible:
            decision.should_stop, decision.reason = True, StopReason.IRREDUCIBLE
        elif had_prior and decision.improved_best and decision.rel_improvement < self.epsilon:
            decision.should_stop, decision.reason = True, StopReason.EPSILON
        elif self._is_pingpong():
            decision.should_stop, decision.reason = True, StopReason.OSCILLATION
        elif self._non_improving_streak >= 2:
            decision.should_stop, decision.reason = True, StopReason.NO_PROGRESS
        elif self._round >= self.max_rounds:
            decision.should_stop, decision.reason = True, StopReason.MAX_ROUNDS

        return decision

    def _is_pingpong(self) -> bool:
        seq = self._variant_seq
        # A,B,A：本轮变体与两轮前相同，且与上一轮不同
        return (
            len(seq) >= 3
            and seq[-1] == seq[-3]
            and seq[-1] != seq[-2]
        )


# ---------------- single-round replay (E2, read-only) ----------------

def main(argv: Optional[list] = None) -> int:
    """命令行重放入口（只读）：replay <history.json>。

    history.json = 一组轮次记录 [{variant, correct, cycles, compiled?, bottleneck?, lever?, expected_gain?}]，
    逐个喂给全新的 LoopController，打印每轮决策。只读，不写 outputs。
    """
    import argparse
    import json
    import sys
    from pathlib import Path
    from .contracts import SimResult, Verdict

    ap = argparse.ArgumentParser(prog="control.loop_controller")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("replay", help="feed a saved round history, print stop decisions")
    rp.add_argument("history_json")
    args = ap.parse_args(argv)
    try:
        hist = json.loads(Path(args.history_json).read_text(encoding="utf-8"))
        ctrl = LoopController()
        last = None
        for i, entry in enumerate(hist, 1):
            sim = SimResult(
                correct=entry["correct"], max_abs_err=0.0,
                cycles=entry.get("cycles"), pipeline={},
                compiled=entry.get("compiled", True),
            )
            verdict = None
            if entry.get("bottleneck"):
                verdict = Verdict(
                    bottleneck=entry["bottleneck"], lever=entry.get("lever", ""),
                    cycles=entry.get("cycles") or 0,
                    expected_gain=entry.get("expected_gain", 0.1),
                )
            last = ctrl.update(entry.get("variant", f"v{i}"), sim, verdict)
            print(f"round {i}: should_stop={last.should_stop} reason={last.reason} "
                  f"rolled_back={last.rolled_back} "
                  f"best_cycles={(last.best.cycles if last.best else None)}")
        print(f"FINAL reason={(last.reason if last else 'n/a')} "
              f"best_cycles={(ctrl.best.cycles if ctrl.best else None)}")
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
