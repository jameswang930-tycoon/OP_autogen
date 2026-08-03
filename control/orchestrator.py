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
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import yaml

from . import placeholders, vocabulary
from .contracts import SimResult, Verdict
from .feedback_adapter import adapt as _default_adapt
from .feedback_adapter import parse_raw as _default_parse_raw
from .job_spec import Budget, NormalizedJob
from .llm_backend import LLMInvocationError, LLMTimeout
from .launch_template import (
    build_sim_result, launch as _default_launch,
    RemoteConnectionError, RemoteTimeout, RemoteScriptError, ResultNotFound, ResultMismatch,
)
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
    extension_index: str, compile_error: Optional[str] = None,
    optimization_hint: Optional[str] = None,
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
        "COMPILE_ERROR": compile_error or "",
        "OPTIMIZATION_HINT": optimization_hint or "",
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
    if ext is not None:
        # GLM52 P3：extension index 按模块归属渲染成全限定名 module.name，模型可能在
        # extension_used 里回写全限定名。校验与存档统一用裸名（canonical id），两种写法都接受。
        ext = ext.rsplit(".", 1)[-1] if "." in ext else ext
        if ext not in allowed_extensions:
            raise ParseError(f"extension_used {ext!r} not in cheatsheet")
        meta["extension_used"] = ext
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


def _load_extension_entries() -> list[dict]:
    """读 EXT_REFS 下全部原语条目的轻量字段（name/module/category/signature/applies_to）。

    读取**路径**与"index 形式"契约不变（仍是 EXT_REFS 下的 yaml、无占位符）；这里只是把
    每条的可选字段一并读出，供"模块归属 + 按场景检索"用（GLM52 优化指导第二部分）。缺字段
    按默认处理——必填字段与 category 校验仍由 check_extension_cheatsheet.py 按原契约负责。
    """
    out: list[dict] = []
    if not EXT_REFS.is_dir():
        return out
    for f in sorted([*EXT_REFS.glob("*.yaml"), *EXT_REFS.glob("*.yml")]):
        try:
            e = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 速查表格式问题由 check_extension_cheatsheet 报
            continue
        if isinstance(e, dict) and "category" in e and "name" in e:
            out.append({
                "name": e["name"],
                "module": e.get("module"),                # 可选：模块归属，拼全限定名用
                "category": e["category"],
                "signature": e.get("signature"),          # 可选：签名
                "applies_to": e.get("applies_to") or [],  # 可选：适用算子场景；缺则视为通用
            })
    return out


def _primitives_by_category() -> dict[str, list[str]]:
    """从 extension 速查表读取 {category: [primitive names]}。无条目则空。"""
    out: dict[str, list[str]] = {}
    for e in _load_extension_entries():
        out.setdefault(e["category"], []).append(e["name"])
    return out


def _allowed_extensions() -> set[str]:
    """全量合法原语名（parse 闸门用它校验 extension_used，不受场景检索影响）。"""
    return {e["name"] for e in _load_extension_entries()}


def _scene_applies(entry: dict, op_kind, *, strict: bool = False) -> bool:
    """applies_to 缺省视为通用；给出则要求 op_kind 命中。

    strict=True（choose_lever 候选池，V2-P2）：applies_to 缺省的原语**不进候选**——
    "要 LLM 选一个去用"的候选池宁可漏（漏标的选不到）也不能错（漏标专用原语被误选，
    如 img2col 归 memory 类却卷积专用、softmax 混进 reduce_sum）。index 展示仍宽松
    （strict=False，信息多点无害）。"""
    scenes = entry.get("applies_to") or []
    if scenes:
        return op_kind in scenes
    return not strict


def _relevant_entries(op_kind: Optional[str], bottleneck: Optional[str], *,
                      strict: bool = False) -> list[dict]:
    """按 (算子场景, 瓶颈) 检索相关原语子集（GLM52 第二部分"按场景检索"）。

    - bottleneck 给定：取该类别原语，再用 applies_to 过滤不适用的（如 img2col 不进 elementwise）。
    - bottleneck 缺省(round1)：取所有 applies_to 命中当前算子的原语（不限类别）。
    strict=True 时 applies_to 缺省的原语不进结果（choose_lever 候选池严格，见 _scene_applies）。
    读的还是 EXT_REFS 下同一份 index，契约不变；只是"读哪些"变智能。
    """
    entries = _load_extension_entries()
    if bottleneck is not None:
        return [e for e in entries
                if e["category"] == bottleneck and _scene_applies(e, op_kind, strict=strict)]
    return [e for e in entries if _scene_applies(e, op_kind, strict=strict)]


def _index_line(entry: dict) -> str:
    """索引行：模块全限定名 + 适用场景标注。有 module 则 `module.name`，杜绝模型猜模块。"""
    qname = f'{entry["module"]}.{entry["name"]}' if entry.get("module") else entry["name"]
    scenes = entry.get("applies_to") or []
    return f"{qname} [applies_to: {','.join(scenes)}]" if scenes else qname


def _extension_scene_hint(op_kind: Optional[str], bottleneck: Optional[str]) -> str:
    """gen prompt 的 EXTENSION_INDEX：精简场景提示（V2 backend 重构后**唯一路径**，无哑后端/双模式）。

    单轮命令行 run 本就触发 skill——不在此塞全量 index，只提示按算子/瓶颈触发对应 ext-* skill
    lazy-load 原语。原语具体内容靠环境侧填的 ext-* skill references/。"""
    bn = bottleneck or "(首轮未知)"
    return (f"(精简：不在此塞全量 index) 当前算子 {op_kind}、瓶颈 {bn}；"
            f"按需触发对应 ext-* skill（ext-reduction/activation/matmul/shape/quant）"
            f"lazy-load 该场景原语。")


# V2 预埋：optimization 知识按瓶颈指向 opt-* skill（references/ 环境侧填，空→降级）
_OPT_SKILL_BY_BOTTLENECK = {
    "compute_bound_at_peak": "opt-compute-bound",
    "memory_underfilled": "opt-memory-bound",
    "stall_dependency": "opt-stall-dependency",
}


def _optimization_skill_for(bottleneck: Optional[str]) -> Optional[str]:
    """瓶颈类别 → 对应的 optimization skill 名；无匹配返回 None。"""
    return _OPT_SKILL_BY_BOTTLENECK.get(bottleneck) if bottleneck else None


def _optimization_hint(op_kind: Optional[str], bottleneck: Optional[str]) -> str:
    """gen prompt 的 OPTIMIZATION_HINT：按瓶颈指向 optimization skill（V2 预埋）。

    references/ 空（公开分支）→ 只提示触发 skill；环境侧填内容后 agent 加载真实优化技巧。
    无瓶颈/无匹配 skill → 空串（降级，prompt 段为空，不影响运行）。"""
    skill = _optimization_skill_for(bottleneck)
    if not skill:
        return ""
    return (f"当前瓶颈 {bottleneck}；按需触发 {skill} skill 获取该类瓶颈的优化技巧/规范，"
            f"结合检索经验里的 opt_technique_ref（应用过则复用/调整）。")


def extension_index_text(op_kind: Optional[str] = None,
                         bottleneck: Optional[str] = None) -> str:
    """速查表短索引文本（注入 triton-gen 的 EXTENSION_INDEX）。

    GLM52 第二部分：按 (算子场景, 瓶颈) 只给相关子集 + 带模块归属，避免幻觉原语
    (tlext1.add) 与误归类污染 (img2col 进 elementwise)。检索为空则退回全量索引（永不空、
    永不因缺标注而隐藏原语）。两参缺省(向后兼容)时给全量索引。
    """
    entries = _load_extension_entries()
    if not entries:
        return "(extension cheatsheet empty — fill references in the confidential env)"
    relevant = _relevant_entries(op_kind, bottleneck) if (op_kind or bottleneck) else entries
    if not relevant:
        relevant = entries  # 保守退回：无相关标注时不隐藏任何原语
    by_cat: dict[str, list[str]] = {}
    for e in relevant:
        by_cat.setdefault(e["category"], []).append(_index_line(e))
    return "\n".join(f"{cat}: {by_cat[cat]}" for cat in sorted(by_cat))


def pick_lever(candidates: list[str], evidence: dict, exploration: str) -> str:
    """多候选杠杆的探索/利用选择（T13-5 渠道⑥）。

    - 单候选：直接返回，不引入随机性。
    - exploration == 'off'：利用——选历史证据最强的候选（确定性、可复现）。
    - exploration != 'off'（mild/aggressive）：探索——选证据最少的候选，加速经验覆盖。
    平局按候选出现顺序打破，保证确定。
    """
    if not candidates:
        raise ValueError("pick_lever: empty candidate list")
    if len(candidates) == 1:
        return candidates[0]
    idx = {c: i for i, c in enumerate(candidates)}
    if exploration == "off":
        return sorted(candidates, key=lambda c: (-evidence.get(c, 0), idx[c]))[0]
    return sorted(candidates, key=lambda c: (evidence.get(c, 0), idx[c]))[0]


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
    compiled: bool = True

    def to_dict(self, baseline_cycles: Optional[int]) -> dict:
        d = {
            "n": self.n, "correct": self.correct, "cycles": self.cycles,
            "compiled": self.compiled,
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


def _sim_result_dict(sim: SimResult) -> dict:
    return {
        "correct": sim.correct, "max_abs_err": sim.max_abs_err,
        "cycles": sim.cycles, "pipeline": sim.pipeline,
        "compiled": sim.compiled, "compile_log": sim.compile_log,
    }


def _event_dict(e) -> dict:
    return {"name": e.name, "start": e.start, "end": e.end, "duration": e.duration,
            "unit": e.unit, "stall_class": e.stall_class, "bytes": e.bytes}


def _decision_dict(d) -> dict:
    best = None
    if d.best is not None:
        best = {"variant_id": d.best.variant_id, "cycles": d.best.cycles, "round_no": d.best.round_no}
    return {"should_stop": d.should_stop, "reason": d.reason, "rolled_back": d.rolled_back,
            "improved_best": d.improved_best, "rel_improvement": d.rel_improvement, "best": best}


class _RoundTranscript:
    """每轮全量落盘（E1）：把该轮全部中间产物写到 log/round_N/，便于完整复盘。

    编号前缀让文件按流程顺序排列；失败轮也落盘已产生的部分，meta.txt 标注失败阶段。
    """

    def __init__(self, log_dir: Path, n: int):
        self.dir = log_dir / f"round_{n}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = n
        self.model: Optional[str] = None
        self.run_id: Optional[str] = None
        self.result_class: Optional[str] = None
        self.fail_stage: Optional[str] = None

    def write(self, name: str, content) -> None:
        p = self.dir / name
        if isinstance(content, (dict, list)):
            p.write_text(json.dumps(content, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
        else:
            p.write_text(str(content), encoding="utf-8")

    def note(self, *, model=None, run_id=None, result=None, fail_stage=None) -> None:
        if model is not None:
            self.model = model
        if run_id is not None:
            self.run_id = run_id
        if result is not None:
            self.result_class = result
        if fail_stage is not None:
            self.fail_stage = fail_stage

    def finalize(self) -> None:
        lines = [
            f"round: {self.n}",
            f"model: {self.model or '(unknown)'}",
            f"run_id: {self.run_id or '(none)'}",
            f"result: {self.result_class or '(running)'}",
        ]
        if self.fail_stage:
            lines.append(f"fail_stage: {self.fail_stage}")
        (self.dir / "meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


class _Progress:
    """旁路进度事件（E7）：stderr 人可读 + progress.jsonl 机器可读。不承载状态。

    quiet 仅出关键节点（baseline/result/stop）；normal 出各阶段；verbose 额外落盘路径。
    """

    KEY_STAGES = {"baseline_result", "result", "stop", "baseline_launching"}

    def __init__(self, mode: str, path, run_id: str):
        self.mode = mode
        self.run_id = run_id
        self._fh = open(path, "a", encoding="utf-8") if path else None

    def emit(self, stage: str, round_n: int = None, **fields):
        evt = {"run_id": self.run_id, "stage": stage}
        if round_n is not None:
            evt["round"] = round_n
        evt.update(fields)
        if self._fh is not None:
            self._fh.write(json.dumps(evt, ensure_ascii=False) + "\n")
            self._fh.flush()
        if self.mode == "quiet" and stage not in self.KEY_STAGES:
            return
        prefix = f"[round {round_n}]" if round_n is not None else f"[{self.run_id}]"
        details = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"{prefix} {stage} {details}".rstrip(), file=sys.stderr)

    def close(self):
        if self._fh is not None:
            self._fh.close()


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
        progress: str = "normal",
        progress_path: Optional[Path] = None,
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
        self._progress = _Progress(progress, progress_path, self.output_dir.name)

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
        except ResultMismatch as exc:
            stop_reason, stop_detail = "RESULT_MISMATCH", str(exc)

        best_desc = None
        if self._best is not None:
            best_desc = f"round{self._best[0]}({self._best[1]}cyc)"
        self._progress.emit("stop", reason=stop_reason, best=best_desc)
        self._progress.close()
        report = self._build_report(rounds, stop_reason, stop_detail)
        self._write_outputs(report)
        return report

    # ---- baseline ----
    def _maybe_seed_baseline(self) -> None:
        if not self.job.has_baseline:
            return
        self._progress.emit("baseline_launching")
        raw = self._launch_with_retries(self.job.baseline_src, "baseline")
        sim = build_sim_result(raw)
        if sim.correct and sim.cycles is not None:
            self._best = (0, sim.cycles, self.job.baseline_src)
            self._baseline_cycles = sim.cycles
            self._progress.emit("baseline_result", cycles=sim.cycles, best=True)
        else:
            self._progress.emit("baseline_result", correct=sim.correct)
        self._write_artifact("baseline_raw.txt", json.dumps(raw, ensure_ascii=False, indent=2))

    # ---- one optimization round ----
    def _run_one_round(self, n: int) -> tuple[RoundRecord, Optional[Any]]:
        t = _RoundTranscript(self._log_dir, n)
        try:
            feedback_summary = None
            prior_verdict: Optional[Verdict] = getattr(self, "_last_verdict", None)

            retrieved_text, retrieved_ids = self._retrieve_experience(
                prior_verdict.bottleneck if prior_verdict else None)
            self._progress.emit("retrieve", round_n=n, has_exp=bool(retrieved_text))
            lever, ext_hint = self._resolve_lever(prior_verdict, t)

            # steps 3-5: produce a kernel, launch, ensure it compiled (compile failures
            # do not count as a round; T13-3). Writes 01-07 + compile-fail logs.
            kernel_src, meta, raw, sim = self._produce_and_compile(
                verdict_json=_verdict_json(prior_verdict),
                feedback_summary=feedback_summary,
                retrieved_experience=retrieved_text,
                bottleneck=prior_verdict.bottleneck if prior_verdict else None,
                round_n=n, t=t,
            )
            self._progress.emit("result", round_n=n, correct=sim.correct,
                                cycles=sim.cycles)

            # steps 6-7: parse + adapt
            verdict: Optional[Verdict] = None
            bottleneck = None
            if sim.correct:
                events = self.parse_raw_fn(raw)
                t.write("08_events.json", [_event_dict(e) for e in events])
                out = self.adapt_fn(events)
                # §5 词表闭包：未知 bottleneck 立即停（不许猜）
                if out.verdict.bottleneck not in vocabulary.all_ids():
                    t.note(fail_stage="adapt:unknown_bottleneck")
                    raise UnknownBottleneck(out.verdict.bottleneck)
                verdict = out.verdict
                bottleneck = verdict.bottleneck
                self._last_verdict = verdict
                t.write("09_verdict.json", _verdict_dict(verdict))
                t.write("10_summary.txt", out.summary)

            # step 8: record (memory)
            self._record_attempt(n, sim, verdict, meta, retrieved_ids=retrieved_ids)

            # step 9: controller (stop decision)
            decision = self.controller.update(variant_id=f"r{n}", sim=sim, verdict=verdict)
            t.write("11_decision.json", _decision_dict(decision))

            # track best-so-far / final
            self._final = (n, sim.cycles, kernel_src)
            if sim.correct and sim.cycles is not None:
                if self._best is None or sim.cycles < self._best[1]:
                    self._best = (n, sim.cycles, kernel_src)
                    self._progress.emit("best", round_n=n, cycles=sim.cycles)

            t.note(model=getattr(self.llm, "last_model_echo", None),
                   result="pass" if sim.correct else "numerical_fail")
            rec = RoundRecord(
                n=n, correct=sim.correct, cycles=sim.cycles,
                bottleneck=bottleneck, lever=lever,
                extension_used=meta.get("extension_used"), rolled_back=decision.rolled_back,
                kernel_path=f"log/round_{n}/03_kernel.py", compiled=sim.compiled,
            )
            return rec, decision
        except BudgetExhausted as exc:
            t.note(fail_stage=f"budget:{exc.kind}", result=f"fail:{exc.kind}")
            raise
        except UnknownBottleneck:
            t.note(result="fail:unknown_bottleneck")
            raise
        finally:
            t.finalize()

    # ---- produce + compile (compile failures don't count as a round; T13-3) ----
    def _produce_and_compile(
        self, *, verdict_json, feedback_summary, retrieved_experience, bottleneck, round_n, t,
    ) -> tuple[str, dict, dict, SimResult]:
        compile_fails = 0
        compile_log = ""
        while True:
            kernel_src, meta = self._produce_kernel(
                verdict_json=verdict_json,
                feedback_summary=feedback_summary,
                retrieved_experience=retrieved_experience,
                bottleneck=bottleneck,
                round_n=round_n,
                compile_error=compile_log,
                t=t,
            )
            raw = self._launch_with_retries(kernel_src, f"round_{round_n}", t=t)
            sim = build_sim_result(raw)
            t.write("06_raw_sim.json", raw)
            t.write("07_sim_result.json", _sim_result_dict(sim))
            if sim.compiled:
                return kernel_src, meta, raw, sim
            # compile failure: feed the log back, regenerate (does not count as a round)
            compile_fails += 1
            compile_log = sim.compile_log or "(no compile log returned)"
            t.write(f"compile_fail_{compile_fails}.log", sim.compile_log or "")
            t.note(fail_stage=f"compile_fail_{compile_fails}")
            if compile_fails >= self.job.budget.compile_retries:
                t.note(result="fail:compile")
                raise BudgetExhausted("compile")

    # ---- lever resolution (step 2; exploration-based, T13-5 渠道⑥) ----
    def _resolve_lever(self, verdict: Optional[Verdict], t=None) -> tuple[Optional[str], Optional[str]]:
        if verdict is None:
            return None, None
        cands = [e["name"] for e in _relevant_entries(self.job.op, verdict.bottleneck, strict=True)]
        if not cands:
            # no primitive for this category: vanilla lever from the vocabulary
            return vocabulary.lever_for(verdict.bottleneck), None
        if len(cands) == 1:
            return cands[0], cands[0]
        evidence = self._evidence_by_primitive(cands)
        prompt = build_analyze_prompt(
            verdict_json=json.dumps(_verdict_dict(verdict), ensure_ascii=False),
            feedback_summary=getattr(self, "_last_summary", ""),
            candidate_levers=cands,
        )
        if t is not None:
            t.write("01b_prompt_lever.txt", prompt)
        # V2-P1：choose_lever 超时/失败按 lever_retries 重试（不计轮数）；耗尽则回退 vocabulary lever，
        # 不让一次 nga 子进程超时炸掉整个 run。
        resp = None
        for _ in range(max(1, self.job.budget.lever_retries)):
            try:
                resp = self.llm.choose_lever(prompt)
                break
            except (LLMTimeout, LLMInvocationError):
                continue
        if resp is None:
            if t is not None:
                t.write("02b_lever_fallback.txt",
                        "choose_lever retries exhausted -> vocabulary lever")
            return vocabulary.lever_for(verdict.bottleneck), None
        if t is not None:
            t.write("02b_response_lever.txt", resp)
        chosen = parse_lever_response(resp, cands)
        return chosen, chosen

    def _avoided_primitives(self) -> set[str]:
        """V2-P4 职责③：曾导致失败的 extension 原语黑名单。

        信号 (helped==0 and failed>0)：推荐该原语的经验在场时从未通过、且失败过——
        提示生成侧避开（schema 字段 helped/failed/extension_used 都已存在，无需扩 schema）。
        """
        if self.store is None:
            return set()
        avoid: set[str] = set()
        for exp in self.store.all():
            eu = getattr(exp, "extension_used", None)
            if eu and getattr(exp, "helped", 0) == 0 and getattr(exp, "failed", 0) > 0:
                avoid.add(eu)
        return avoid

    def _evidence_by_primitive(self, candidates: list[str]) -> dict:
        """每个候选原语的历史使用次数（来自经验库 extension_used）。无 store 则空。"""
        if self.store is None:
            return {c: 0 for c in candidates}
        counts = {c: 0 for c in candidates}
        for exp in self.store.all():
            eu = getattr(exp, "extension_used", None)
            if eu in counts:
                counts[eu] += getattr(exp, "used", 0)
        return counts

    # ---- produce kernel (parse + presim budgets) ----
    def _produce_kernel(
        self, *, verdict_json, feedback_summary, retrieved_experience, bottleneck, round_n,
        compile_error: Optional[str] = None, t=None,
    ) -> tuple[str, dict]:
        allowed = _allowed_extensions()
        parse_fails = 0
        presim_fails = 0
        kernel_src: Optional[str] = None
        meta: dict = {}
        while True:
            # V2-P4 职责②：迭代基准用 best-so-far kernel（有则用，无则原始 baseline）——不再每轮
            # 都从原始 baseline 重生成，浪费 best-so-far。
            iter_basis = self._best[2] if self._best is not None else self.job.baseline_src
            # V2-P4 职责③：避坑清单——历史失败过的 extension 原语，提示 LLM 避开。
            avoid = self._avoided_primitives()
            retrieved_aug = retrieved_experience or ""
            if avoid:
                retrieved_aug = (retrieved_aug + "\n" if retrieved_aug else "") + \
                    "避免以下曾导致失败的原语/用法（除非有确切正确用法）：" + ", ".join(sorted(avoid))
            prompt = build_gen_prompt(
                self.job,
                baseline_src=iter_basis,
                verdict_json=verdict_json,
                feedback_summary=feedback_summary,
                retrieved_experience=retrieved_aug or None,
                extension_index=_extension_scene_hint(self.job.op, bottleneck),
                optimization_hint=_optimization_hint(self.job.op, bottleneck),
                compile_error=compile_error,
            )
            resp = self.llm.generate(prompt)
            if t is not None:
                t.write("01_prompt_generate.txt", prompt)
                t.write("02_response_generate.txt", resp)
            self._progress.emit("generate", round_n=round_n)
            try:
                kernel_src, meta = parse_generate_response(resp, allowed)
            except ParseError:
                parse_fails += 1
                if t is not None:
                    t.note(fail_stage=f"parse_fail_{parse_fails}")
                if parse_fails >= self.job.budget.llm_retries:
                    if t is not None:
                        t.note(result="fail:llm_parse")
                    raise BudgetExhausted("llm_retries")
                continue
            gate = presim_check(kernel_src)
            if gate.ok:
                if t is not None:
                    t.write("03_kernel.py", kernel_src)
                    t.write("04_meta.json", meta)
                return kernel_src, meta
            presim_fails += 1
            if t is not None:
                t.note(fail_stage=f"presim_fail_{presim_fails}")
            if presim_fails >= self.job.budget.presim_retries:
                if t is not None:
                    t.note(result="fail:presim")
                raise BudgetExhausted("presim_retries")
            # regenerate (kernel_src will be overwritten next loop)

    # ---- launch with infra retry ----
    def _launch_with_retries(self, kernel_src: str, label: str, t=None) -> dict:
        if t is not None:
            t.write("05_launch_input.py", kernel_src)
            kernel_path = t.dir / "05_launch_input.py"
        else:
            kernel_path = self._write_artifact(f"{label}_kernel.py", kernel_src)
        # E4: 前四类 + SimInfraError 是基础设施问题 -> 退避重试、不计轮数；
        # ResultMismatch 是框架 bug -> 立即暴露、不重试。
        retryable = (RemoteConnectionError, RemoteTimeout, RemoteScriptError,
                     ResultNotFound, SimInfraError)
        for _ in range(self.job.budget.sim_retries):
            try:
                t0 = time.time()
                raw = self.launcher(str(kernel_path))
                self._progress.emit("launch", round_n=t.n if t is not None else None,
                                    elapsed_s=round(time.time() - t0, 2))
                return raw
            except retryable:
                continue
            except ResultMismatch as exc:
                if t is not None:
                    t.note(fail_stage=f"result_mismatch:{exc.expected_run_id}/{exc.actual_run_id}",
                           result="fail:result_mismatch")
                raise
        raise BudgetExhausted("sim_retries")

    # ---- memory ----
    def _warn_if_memory_disabled(self) -> None:
        """核心模块失效不许静默（GLM52 优化指导第五部分）。

        store/log 缺失时显式 warning——否则 memory 全程没工作（retrieve/record 静默 no-op）
        却无人发现。每实例只报一次，绑在真正用到 memory 的时机（非构造时）。
        """
        if getattr(self, "_memory_warned", False):
            return
        missing = [name for name, val in (("store", self.store), ("log", self.log))
                   if val is None]
        if missing:
            self._memory_warned = True
            warnings.warn(
                f"memory disabled ({'/'.join(missing)} is None): experience retrieve/record "
                "are silent no-ops; wire store + log if memory is intended to work.",
                stacklevel=3,
            )

    def _retrieve_experience(self, bottleneck: Optional[str]) -> tuple[str, list[str]]:
        """按"算子类型 + 已知瓶颈"检索 top-k 经验（GLM52 优化指导第三部分 Bug1 修复）。

        round2+ 时 bottleneck 取自上一轮 verdict——retrieve 必须用已知瓶颈构造
        fingerprint，否则 key 落到 "op|unknown"，与 record 存的 "op|<真实瓶颈>"
        错位、精确匹配永远 miss，只能靠 op_kind 回退、丢掉瓶颈区分度。
        返回 (注入文本, 命中经验ID)；后者回传 record_attempt 让分数真正迭代（Bug2）。
        """
        self._warn_if_memory_disabled()
        if self.store is None:
            return "", []
        from memory.retrieve import retrieve, format_context
        from memory.schema import Fingerprint
        fp = Fingerprint(op_kind=self.job.op, bottleneck=bottleneck)
        hits = retrieve(self.store, fp, n=3)
        if hits:
            self._last_summary = ""
            return format_context(hits), [e.id for e in hits]
        return "", []

    def _record_attempt(self, n: int, sim: SimResult, verdict: Optional[Verdict], meta: dict,
                        retrieved_ids: Optional[list[str]] = None):
        """写回：把 retrieve 实际命中的经验ID 传给 record_attempt（GLM52 Bug2 修复）。

        旧代码硬编码 retrieved_ids=[] → store.bump 永不更新 used/helped → 经验分数恒为
        初始值、好坏经验无法区分。改用本轮真实命中的 ID，让分数随成败真正迭代。
        """
        self._warn_if_memory_disabled()
        if self.store is None or self.log is None:
            return
        from memory.schema import Fingerprint
        from memory.writeback import record_attempt
        fp = Fingerprint(op_kind=self.job.op, bottleneck=verdict.bottleneck if verdict else None)
        record_attempt(
            self.log, self.store, fp, retrieved_ids=retrieved_ids or [],
            passed=sim.correct, cycles=sim.cycles, compiled=sim.compiled,
            extension_used=meta.get("extension_used"),
            opt_technique_ref=meta.get("opt_technique_ref"), stage=f"round_{n}",
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
    from memory.store import ExperienceStore
    from memory.log import RunLog
    job = load_job(args.job, baseline_root=Path(args.job).resolve().parent)

    # 先定 output_dir，保证 store/log 路径与 orchestrator 输出目录一致（否则经验存到与
    # 输出目录不一致的地方，has_exp 恒 False）。store/log 同目录自动管 best_cycles.json。
    out = Path(args.output_dir) if args.output_dir else (Path("outputs") / f"run_{int(time.time())}")
    mem_dir = out / "memory"
    store = ExperienceStore(mem_dir / "experience.json")
    log = RunLog(mem_dir / "runlog.jsonl")

    orch = Orchestrator(job, store=store, log=log, output_dir=out)
    report = orch.run()
    print(json.dumps({"stop": report["stop"], "recommended": report["recommended"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
