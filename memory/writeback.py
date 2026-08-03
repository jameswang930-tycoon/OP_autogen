"""写回:一次尝试结束后,记日志并更新经验统计。

这是闭合回路的关键动作——把校验结果沉淀回经验库。
"""

from __future__ import annotations

from typing import Optional

from .schema import AttemptRecord, Experience, Fingerprint
from .log import RunLog
from .store import ExperienceStore


def record_attempt(
    log: RunLog,
    store: ExperienceStore,
    fp: Fingerprint,
    retrieved_ids: list[str],
    passed: bool,
    cycles: Optional[int] = None,
    kernel_ref: Optional[str] = None,
    extension_used: Optional[str] = None,
    compiled: bool = True,
    stage: str = "drafting",
    opt_technique_ref: Optional[str] = None,
) -> AttemptRecord:
    """写回点:在仿真给出正确性结果之后调用。

    价值信号已从「正确性」转向「性能」（架构文档 §5.2 / §3.5）：
    - correct=True 且本轮刷新该 fingerprint 历史最优 cycles → helped+1（价值=性能改善）。
    - correct=True 未刷新最优 → used+1，helped 不动。
    - correct=False → 不碰 used/helped（score 不受影响），仅记 failed；仍写 runlog。
    FAIL 轮的 cycles 作废（§3.6），不写入记录。

    负面经验分类（T13-5）：compiled=False 为编译类失败（低价值，自纠）；
    compiled=True 且 correct=False 为语义误用（高价值负面，bump harmed）。
    """
    record_cycles = cycles if passed else None
    helped = bool(passed and record_cycles is not None
                  and store.update_best(fp.key(), record_cycles))
    if not compiled:
        failure_kind = "compile"
    elif not passed:
        failure_kind = "semantic"
    else:
        failure_kind = None
    record = AttemptRecord(
        fingerprint=fp.key(),
        retrieved=retrieved_ids,
        passed=passed,
        kernel_ref=kernel_ref,
        cycles=record_cycles,
        extension_used=extension_used,
        opt_technique_ref=opt_technique_ref,
        failure_kind=failure_kind,
        stage=stage,
    )
    log.append(record)
    store.bump(retrieved_ids, helped=helped, failed=not passed,
               semantic=(failure_kind == "semantic"))
    return record


def add_experience(
    store: ExperienceStore,
    fp: Fingerprint,
    text: str,
    source_run: Optional[str] = None,
    extension_used: Optional[str] = None,
    opt_technique_ref: Optional[str] = None,
) -> str:
    """新增经验(最小版:手工或用固定模板调用)。

    自动蒸馏是预留位:将来由离线蒸馏智能体读取日志、提炼、去重、消解矛盾后
    调用本函数入库。当前先手工让经验库有内容。
    """
    exp = Experience(
        text=text, applies_to=fp.key(),
        source_run=source_run, extension_used=extension_used,
        opt_technique_ref=opt_technique_ref,
    )
    return store.add(exp)
