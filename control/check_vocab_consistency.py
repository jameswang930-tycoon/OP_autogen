"""词表一致性检查脚本。

词表是 single source of truth（架构文档 §5.6）。三方应引用同一份词表、无孤儿标签：
  - feedback_adapter 输出的 stall_class / bottleneck
  - memory 的 Fingerprint.bottleneck
  - extension-guide 速查表的索引

本脚本提供两样东西：
  1. main(path)  —— 校验词表本身格式合法、无重复 id（可作 CLI 跑，也可被测试调用）。
  2. check_labels_against_vocab(labels, source) —— 供 adapter / memory / extension
     在各自落地后，把己方用到的标签拿来逐一核对（出现词表外的标签即报错）。
     在 T2 阶段三方尚未落地，因此 main() 目前只校验词表自身；三方校验随各任务接入。
"""
from __future__ import annotations

from typing import Iterable, Optional

from . import vocabulary


def check_labels_against_vocab(
    labels: Iterable[str], source: str, path: Optional[str] = None
) -> None:
    """核对一组标签是否全部在词表内；遇到孤儿标签即抛 ValueError。"""
    allowed = vocabulary.all_ids(path)
    for label in labels:
        if label not in allowed:
            raise ValueError(
                f"orphan {source} label {label!r} not in vocabulary "
                f"(allowed: {sorted(allowed)})"
            )


def main(path: Optional[str] = None) -> int:
    """校验词表格式；合法返回 0，否则打印错误并返回 1。"""
    try:
        entries = vocabulary.load(path)
        vocabulary.validate_format(entries)
    except Exception as exc:  # noqa: BLE001 - 脚本要把任何校验失败转成非零退出
        print(f"FAIL: vocabulary check error: {exc}")
        return 1
    ids = sorted(e["id"] for e in entries)
    print(f"OK: {len(ids)} bottleneck categories: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
