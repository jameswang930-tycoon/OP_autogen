"""最小数据结构:算子特征、经验条目、日志记录。

字段刻意精简,后续需要时再加。架构说明见 docs/project_knowledge/memory_architecture.md。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class Fingerprint:
    """算子特征 = 检索与归类用的标识。

    主键只取「算子类型 + 瓶颈类别」。形状先不进主键,仅保留字段,
    留作将来细分的辅助信号(预留位)。
    """

    op_kind: str
    bottleneck: Optional[str] = None      # 由成本模型给出,例如 "compute_bound"
    shape_sig: Optional[str] = None       # 预留:形状特征,当前不参与主键

    def key(self) -> str:
        """主键:精确匹配用。"""
        return f"{self.op_kind}|{self.bottleneck or 'unknown'}"

    def loose_key(self) -> str:
        """放宽键:只按算子类型匹配。"""
        return self.op_kind


@dataclass
class Experience:
    """一条经验条目。"""

    text: str                              # 可读的经验/做法/坑
    applies_to: str                        # 适用的算子特征主键 (Fingerprint.key())
    source_run: Optional[str] = None       # 来源运行编号
    used: int = 0                          # 用过次数
    helped: int = 0                        # 帮上忙次数(在场且该次通过)
    failed: int = 0                        # 预留:在场且该次未通过(中性负向;bump 在 passed=False 时 +1)
    harmed: int = 0                        # 预留:被证实有害/误导次数(第一版恒 0,归因机制待引入)
    extension_used: Optional[str] = None   # 该经验推荐的 extension 原语(架构文档 §5.4)
    id: str = field(default_factory=lambda: _new_id("exp"))
    created_at: str = field(default_factory=_now)

    def score(self) -> float:
        """检索排序分数:帮上忙比例,做 Laplace 平滑,新条目中性起点约 0.5。

        这不是价值函数,只是可解释的计数比值。harmed 预留惩罚项:第一版恒 0
        不影响;引入归因后可改为 (helped - harmed + k)/(used + 2k) 之类。
        """
        return (self.helped + 1) / (self.used + 2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Experience":
        return cls(**d)


@dataclass
class AttemptRecord:
    """运行日志的一条记录:一次生成尝试。"""

    fingerprint: str                       # 算子特征主键
    retrieved: list[str]                   # 本次注入的经验编号
    passed: bool                           # 是否通过正确性校验
    kernel_ref: Optional[str] = None       # 生成产物的存放位置
    cycles: Optional[int] = None           # 实测 cycles（correct=False 时为空，性能作废，架构文档 §3.6）
    extension_used: Optional[str] = None   # 本轮用的 extension 原语（架构文档 §5.4）
    stage: str = "drafting"
    run_id: str = field(default_factory=lambda: _new_id("run"))
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return asdict(self)
