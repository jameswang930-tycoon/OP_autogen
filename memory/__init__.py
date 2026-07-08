"""最小记忆闭环:运行日志 + 经验库 + 检索 + 写回。

只暴露最小 API。回路演示见 memory_skeleton/demo.py,架构说明见
memory_skeleton/README.md,接入指南见 memory_skeleton/GLM_接入指南.md。
"""

from .schema import Fingerprint, Experience, AttemptRecord
from .fingerprint import compute_fingerprint, fingerprint_from_plan_json
from .log import RunLog
from .store import ExperienceStore
from .retrieve import retrieve, format_context
from .writeback import record_attempt, add_experience

__all__ = [
    "Fingerprint",
    "Experience",
    "AttemptRecord",
    "compute_fingerprint",
    "fingerprint_from_plan_json",
    "RunLog",
    "ExperienceStore",
    "retrieve",
    "format_context",
    "record_attempt",
    "add_experience",
]
