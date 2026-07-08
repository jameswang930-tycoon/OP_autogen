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
    kernel_ref: Optional[str] = None,
    latency_us: Optional[float] = None,
    stage: str = "drafting",
) -> AttemptRecord:
    """写回点:在模拟器给出正确性结果之后调用。"""
    record = AttemptRecord(
        fingerprint=fp.key(),
        retrieved=retrieved_ids,
        passed=passed,
        kernel_ref=kernel_ref,
        latency_us=latency_us,
        stage=stage,
    )
    log.append(record)
    store.bump(retrieved_ids, passed=passed)
    return record


def add_experience(
    store: ExperienceStore,
    fp: Fingerprint,
    text: str,
    source_run: Optional[str] = None,
) -> str:
    """新增经验(最小版:手工或用固定模板调用)。

    自动蒸馏是预留位:将来由离线蒸馏智能体读取日志、提炼、去重、消解矛盾后
    调用本函数入库。当前先手工让经验库有内容。
    """
    exp = Experience(text=text, applies_to=fp.key(), source_run=source_run)
    return store.add(exp)
