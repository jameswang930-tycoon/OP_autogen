"""确定性编排器（T11）：一条命令跑完全流程，全程无自然语言交互。

设计三原则（T10–T12 手册 §0）：
  ① 流程正确性归代码：编排器决定每一步调什么；LLM 只回答被问到的问题。
  ② LLM 每个输出过确定性闸门：解析闸门 → pre-sim gate → 正确性 → 回退 → 循环边界。
  ③ 每次 LLM 调用无状态：历史在文件（runlog / best-so-far / report）。

LLM 调用点全流程仅两个：
  1. llm_generate（triton-gen 模板注入）—— 每轮生成 kernel
  2. llm_choose_lever（sim-analyze 模板注入）—— 仅当某瓶颈类别有多个候选原语时
判停、归类、重试计数全部在确定性代码里，不委托给 LLM。

所有外部依赖（LLM / launcher / parse_raw / adapt）均可注入，故全部离线可测。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml

from . import placeholders, vocabulary
from .contracts import SimResult, Verdict
from .feedback_adapter import adapt as _default_adapt
from .feedback_adapter import parse_raw as _default_parse_raw
from .job_spec import Budget, NormalizedJob
from .launch_template import build_sim_result, launch as _default_launch
from .loop_controller import LoopController, StopReason
from .presim_gate import check as presim_check

SKILLS_DIR = Path(__file__).resolve().parent.parent / ".claude" / "skills"
EXT_REFS = SKILLS_DIR / "extension-guide" / "references"


# ---------------- exceptions / budgets ----------------

class ParseError(Exception):
    """LLM 输出不满足解析闸门。"""


class SimInfraError(Exception):
    """仿真设施故障（超时/断连）——可退避重试。"""


class BudgetExhausted(Exception):
    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(f"retry budget exhausted: {kind}")


class UnknownBottleneck(Exception):
    def __init__(self, bottleneck: str):
        self.bottleneck = bottleneck
        super().__init__(f"unknown bottleneck category: {bottleneck!r}")


# ---------------- LLM provider ----------------

class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
    def choose_lever(self, prompt: str) -> str: ...


class NoLLMBackend:
    """默认无后端：生产环境由 OpenCode/GLM 4.7 提供真实实现。"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("no LLM backend configured (inject one for real runs)")

    def choose_lever(self, prompt: str) -> str:
        raise NotImplementedError("no LLM backend configured (inject one for real runs)")


# ---------------- template rendering ----------------

def _skill_body(skill: str) -> str:
    md = (SKILLS_DIR / skill / "SKILL.md").read_text(encoding="utf-8")
    m = re.match(r"^---\n.*?\n---\n(.*)$", md, re.DOTALL)
    return m.group(1)


def _render(skill: str, values: dict) -> str:
    body = _skill_body(skill)
    out = body
    for key, val in values.items():
        out = out.replace("{{" + key + "}}", str(val))
    return out


def build_gen_prompt(
    job: NormalizedJob, *, baseline_src: Optional[str], verdict_json: Optional[str],
    feedback_summary: Optional[str], retrieved_experience: Optional[str],
    extension_index: str,
) -> str:
    values = {
        "OP": job.op,
        "SHAPES": job.shapes,
        "DTYPE": job.dtype,
        "BASELINE_SRC": baseline_src or "(none)",
        "VERDICT_JSON": verdict_json or "(none)",
        "FEEDBACK_SUMMARY": feedback_summary or "(none)",
        "RETRIEVED_EXPERIENCE": retrieved_experience or "(none)",
        "EXTENSION_INDEX": extension_index,
    }
    assert set(values) == set(placeholders.TRITON_GEN_PLACEHOLDERS), (
        f"gen prompt keys {set(values)} != template {set(placeholders.TRITON_GEN_PLACEHOLDERS)}"
    )
    return _render("triton-gen", values)


def build_analyze_prompt(verdict_json: str, feedback_summary: str, candidate_levers) -> str:
    values = {
        "VERDICT_JSON": verdict_json,
        "FEEDBACK_SUMMARY": feedback_summary,
        "CANDIDATE_LEVERS": candidate_levers,
    }
    assert set(values) == set(placeholders.SIM_ANALYZE_PLACEHOLDERS)
    return _render("sim-analyze", values)


# ---------------- parse gates ----------------

_PY = re.compile(r"```python\n(.*?)```", re.DOTALL)
_JS = re.compile(r"```json\n(.*?)```", re.DOTALL)


def parse_generate_response(resp: str, allowed_extensions: set[str]) -> tuple[str, dict]:
    py = _PY.findall(resp)
    js = _JS.findall(resp)
    if len(py) != 1:
        raise ParseError(f"expected exactly 1 python block, got {len(py)}")
    if len(js) != 1:
        raise ParseError(f"expected exactly 1 json block, got {len(js)}")
    try:
        meta = json.loads(js[0])
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"json block invalid: {exc}") from None
    for k in ("lever", "extension_used", "notes"):
        if k not in meta:
            raise ParseError(f"meta missing key {k!r}")
    ext = meta["extension_used"]
    if ext is not None and ext not in allowed_extensions:
        raise ParseError(f"extension_used {ext!r} not in cheatsheet")
    return py[0], meta


def parse_lever_response(resp: str, candidates: list[str]) -> str:
    js = _JS.findall(resp)
    if len(js) != 1:
        raise ParseError(f"expected exactly 1 json block, got {len(js)}")
    try:
        d = json.loads(js[0])
    except Exception as exc:  # noqa: BLE001
        raise ParseError(f"json block invalid: {exc}") from None
    lever = d.get("lever")
    if lever not in candidates:
        raise ParseError(f"chosen lever {lever!r} not in candidates {candidates}")
    return lever


def _primitives_by_category() -> dict[str, list[str]]:
    """从 extension 速查表读取 {category: [primitive names]}。无条目则空。"""
    out: dict[str, list[str]] = {}
    if not EXT_REFS.is_dir():
        return out
    for f in sorted([*EXT_REFS.glob("*.yaml"), *EXT_REFS.glob("*.yml")]):
        try:
            entry = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 速查表格式问题由 check_extension_cheatsheet 报
            continue
        if isinstance(entry, dict) and "category" in entry and "name" in entry:
            out.setdefault(entry["category"], []).append(entry["name"])
    return out


def extension_index_text() -> str:
    """速查表短索引文本（注入 triton-gen 的 EXTENSION_INDEX）。"""
    by_cat = _primitives_by_category()
    if not by_cat:
        return "(extension cheatsheet empty — fill references in the confidential env)"
    lines = []
    for cat in sorted(by_cat):
        lines.append(f"{cat}: {by_cat[cat]}")
    return "\n".join(lines)


def _allowed_extensions() -> set[str]:
    names = set()
    for prims in _primitives_by_category().values():
        names.update(prims)
    return names


# ---------------- records / report ----------------

@dataclass
class RoundRecord:
    n: int
    correct: bool
    cycles: Optional[int]
    bottleneck: Optional[str]
    lever: Optional[str]
    extension_used: Optional[str]
    rolled_back: bool
    kernel_path: str

    def to_dict(self, baseline_cycles: Optional[int]) -> dict:
        d = {
            "n": self.n, "correct": self.correct, "cycles": self.cycles,
            "bottleneck": self.bottleneck, "lever": self.lever,
            "extension_used": self.extension_used, "rolled_back": self.rolled_back,
            "kernel": self.kernel_path,
        }
        if baseline_cycles and self.cycles:
            d["speedup_vs_baseline"] = round(baseline_cycles / self.cycles, 4)
        return d


def _speedup(baseline_cycles: Optional[int], cycles: Optional[int]) -> Optional[float]:
    if baseline_cycles and cycles:
        return round(baseline_cycles / cycles, 4)
    return None


# ---------------- orchestrator ----------------

class Orchestrator:
    def __init__(
        self,
        job: NormalizedJob,
        *,
        llm: Optional[LLMProvider] = None,
        launcher: Optional[Callable[[str], dict]] = None,
        parse_raw_fn: Optional[Callable[[dict], list]] = None,
        adapt_fn: Optional[Callable] = None,
        store=None,
        log=None,
        output_dir: Optional[Path] = None,
    ):
        self.job = job
        self.llm = llm or NoLLMBackend()
        self.launcher = launcher or _default_launch
        self.parse_raw_fn = parse_raw_fn or _default_parse_raw
        self.adapt_fn = adapt_fn or _default_adapt
        self.store = store
        self.log = log
        self.output_dir = Path(output_dir) if output_dir else self._default_output_dir()
        self.controller = LoopController(
            epsilon=job.budget.epsilon, max_rounds=job.budget.max_rounds,
        )
        self._log_dir = self.output_dir / "log"
        self._best: Optional[tuple[int, int, str]] = None   # (round, cycles, kernel_src)
        self._final: Optional[tuple[int, Optional[int], str]] = None
        self._baseline_cycles: Optional[int] = None

    @staticmethod
    def _default_output_dir() -> Path:
        return Path("outputs") / f"run_{int(time.time())}"

    # ---- public ----
    def run(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._maybe_seed_baseline()

        rounds: list[RoundRecord] = []
        stop_reason, stop_detail = None, None
        try:
            while True:
                rec, decision = self._run_one_round(len(rounds) + 1)
                rounds.append(rec)
                if decision is not None and decision.should_stop:
                    stop_reason = decision.reason
                    break
        except BudgetExhausted as exc:
            stop_reason = f"BUDGET_{exc.kind.upper()}"
        except UnknownBottleneck as exc:
            stop_reason, stop_detail = "UNKNOWN_BOTTLENECK", exc.bottleneck

        report = self._build_report(rounds, stop_reason, stop_detail)
        self._write_outputs(report)
        return report

    # ---- baseline ----
    def _maybe_seed_baseline(self) -> None:
        if not self.job.has_baseline:
            return
        raw = self._launch_with_retries(self.job.baseline_src, "baseline")
        sim = build_sim_result(raw)
        if sim.correct and sim.cycles is not None:
            self._best = (0, sim.cycles, self.job.baseline_src)
            self._baseline_cycles = sim.cycles
        self._write_artifact("baseline_raw.txt", json.dumps(raw, ensure_ascii=False, indent=2))

    # ---- one optimization round ----
    def _run_one_round(self, n: int) -> tuple[RoundRecord, Optional[Any]]:
        verdict_json = None
        feedback_summary = None
        prior_verdict: Optional[Verdict] = getattr(self, "_last_verdict", None)

        retrieved = self._retrieve_experience()
        lever, ext_hint = self._resolve_lever(prior_verdict)

        # steps 3-4: produce a kernel that parses AND passes presim (separate budgets)
        kernel_src, meta = self._produce_kernel(
            verdict_json=_verdict_json(prior_verdict),
            feedback_summary=feedback_summary,
            retrieved_experience=retrieved,
            round_n=n,
        )

        # step 5: launch (infra retry budget)
        raw = self._launch_with_retries(kernel_src, f"round_{n}")
        sim = build_sim_result(raw)
        self._write_artifact(f"round_{n}_raw_sim.txt", json.dumps(raw, ensure_ascii=False, indent=2))

        # steps 6-7: parse + adapt
        verdict: Optional[Verdict] = None
        bottleneck = None
        if sim.correct:
            events = self.parse_raw_fn(raw)
            out = self.adapt_fn(events)
            # §5 词表闭包：未知 bottleneck 立即停（不许猜）
            if out.verdict.bottleneck not in vocabulary.all_ids():
                raise UnknownBottleneck(out.verdict.bottleneck)
            verdict = out.verdict
            bottleneck = verdict.bottleneck
            self._last_verdict = verdict
            feedback_summary = out.summary
            self._write_artifact(f"round_{n}_summary.txt", out.summary)

        # step 8: record (memory)
        self._record_attempt(n, sim, verdict, meta)

        # step 9: controller (stop decision)
        decision = self.controller.update(
            variant_id=f"r{n}", sim=sim, verdict=verdict,
        )

        # track best-so-far / final
        self._final = (n, sim.cycles, kernel_src)
        if sim.correct and sim.cycles is not None:
            if self._best is None or sim.cycles < self._best[1]:
                self._best = (n, sim.cycles, kernel_src)

        rec = RoundRecord(
            n=n, correct=sim.correct, cycles=sim.cycles,
            bottleneck=bottleneck, lever=lever,
            extension_used=meta.get("extension_used"), rolled_back=decision.rolled_back,
            kernel_path=f"log/round_{n}_response.txt",
        )
        return rec, decision

    # ---- lever resolution (step 2) ----
    def _resolve_lever(self, verdict: Optional[Verdict]) -> tuple[Optional[str], Optional[str]]:
        if verdict is None:
            return None, None
        cands = _primitives_by_category().get(verdict.bottleneck, [])
        if len(cands) <= 1:
            if cands:
                return cands[0], cands[0]
            return vocabulary.lever_for(verdict.bottleneck), None
        # multi-candidate -> LLM call point 2
        prompt = build_analyze_prompt(
            verdict_json=json.dumps(_verdict_dict(verdict), ensure_ascii=False),
            feedback_summary=getattr(self, "_last_summary", ""),
            candidate_levers=cands,
        )
        chosen = parse_lever_response(self.llm.choose_lever(prompt), cands)
        return chosen, chosen

    # ---- produce kernel (parse + presim budgets) ----
    def _produce_kernel(self, *, verdict_json, feedback_summary, retrieved_experience, round_n) -> tuple[str, dict]:
        allowed = _allowed_extensions()
        parse_fails = 0
        presim_fails = 0
        kernel_src: Optional[str] = None
        meta: dict = {}
        while True:
            prompt = build_gen_prompt(
                self.job,
                baseline_src=self.job.baseline_src,
                verdict_json=verdict_json,
                feedback_summary=feedback_summary,
                retrieved_experience=retrieved_experience,
                extension_index=extension_index_text(),
            )
            resp = self.llm.generate(prompt)
            try:
                kernel_src, meta = parse_generate_response(resp, allowed)
            except ParseError:
                parse_fails += 1
                if parse_fails >= self.job.budget.llm_retries:
                    raise BudgetExhausted("llm_retries")
                continue
            gate = presim_check(kernel_src)
            if gate.ok:
                self._write_artifact(f"round_{round_n}_prompt.txt", prompt)
                self._write_artifact(f"round_{round_n}_response.txt", kernel_src)
                return kernel_src, meta
            presim_fails += 1
            if presim_fails >= self.job.budget.presim_retries:
                raise BudgetExhausted("presim_retries")
            # regenerate (kernel_src will be overwritten next loop)

    # ---- launch with infra retry ----
    def _launch_with_retries(self, kernel_src: str, label: str) -> dict:
        kernel_path = self._write_artifact(f"{label}_kernel.py", kernel_src)
        last = None
        for _ in range(self.job.budget.sim_retries):
            try:
                return self.launcher(str(kernel_path))
            except SimInfraError as exc:
                last = exc
                continue
        raise BudgetExhausted("sim_retries")

    # ---- memory ----
    def _retrieve_experience(self) -> str:
        if self.store is None:
            return ""
        from memory.retrieve import retrieve, format_context
        from memory.schema import Fingerprint
        fp = Fingerprint(op_kind=self.job.op, bottleneck=None)
        hits = retrieve(self.store, fp, n=3)
        if hits:
            self._last_summary = ""
            return format_context(hits)
        return ""

    def _record_attempt(self, n: int, sim: SimResult, verdict: Optional[Verdict], meta: dict):
        if self.store is None or self.log is None:
            return
        from memory.schema import Fingerprint
        from memory.writeback import record_attempt
        fp = Fingerprint(op_kind=self.job.op, bottleneck=verdict.bottleneck if verdict else None)
        record_attempt(
            self.log, self.store, fp, retrieved_ids=[],
            passed=sim.correct, cycles=sim.cycles,
            extension_used=meta.get("extension_used"), stage=f"round_{n}",
        )

    # ---- outputs / report ----
    def _write_artifact(self, name: str, content: str) -> Path:
        p = self._log_dir / name
        p.write_text(content, encoding="utf-8")
        return p

    def _build_report(self, rounds: list[RoundRecord], stop_reason: Optional[str], stop_detail) -> dict:
        rec_cycles = self._best[1] if self._best else None
        rec_round = self._best[0] if self._best else None
        final_cycles = self._final[1] if self._final else None
        final_round = self._final[0] if self._final else None
        report = {
            "job": {"op": self.job.op, "shapes": self.job.shapes, "dtype": self.job.dtype,
                    "form": self.job.form, "has_baseline": self.job.has_baseline},
            "baseline": {"cycles": self._baseline_cycles, "present": self.job.has_baseline},
            "recommended": {
                "round": rec_round, "cycles": rec_cycles,
                "speedup_vs_baseline": _speedup(self._baseline_cycles, rec_cycles),
            },
            "final_round": {
                "round": final_round, "cycles": final_cycles,
                "speedup_vs_baseline": _speedup(self._baseline_cycles, final_cycles),
            },
            "stop": {"reason": stop_reason, "detail": stop_detail,
                     "rounds_used": len(rounds)},
            "rounds": [r.to_dict(self._baseline_cycles) for r in rounds],
        }
        return report

    def _write_outputs(self, report: dict) -> None:
        (self.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if self._best is not None:
            (self.output_dir / "recommended.py").write_text(self._best[2], encoding="utf-8")
        else:
            (self.output_dir / "recommended.py").write_text(
                "# no best-so-far (no baseline and never passed a round)\n", encoding="utf-8")
        if self._final is not None:
            (self.output_dir / "final_round.py").write_text(self._final[2], encoding="utf-8")


def _verdict_json(verdict: Optional[Verdict]) -> Optional[str]:
    if verdict is None:
        return None
    return json.dumps(_verdict_dict(verdict), ensure_ascii=False)


def _verdict_dict(verdict: Verdict) -> dict:
    return {"bottleneck": verdict.bottleneck, "lever": verdict.lever,
            "cycles": verdict.cycles, "expected_gain": verdict.expected_gain}


# ---------------- CLI ----------------

def main(argv: Optional[list] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="deterministic orchestrator")
    ap.add_argument("--job", required=True, help="path to job yaml")
    ap.add_argument("--output-dir", help="output directory (default outputs/<op>_<ts>)")
    args = ap.parse_args(argv)
    from .job_spec import load_job
    job = load_job(args.job, baseline_root=Path(args.job).resolve().parent)
    orch = Orchestrator(job, output_dir=Path(args.output_dir) if args.output_dir else None)
    report = orch.run()
    print(json.dumps({"stop": report["stop"], "recommended": report["recommended"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
