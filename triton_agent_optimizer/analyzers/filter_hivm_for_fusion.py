#!/usr/bin/env python3
"""过滤 hivm_try.txt → 算子融合专用精简版 (不翻译, 删无关, 保留相关)。

原理: 字段名会变, 提取易失败 → 改为"删无关行, 保留相关行", 让 LLM 读保留部分。

保留 (融合需要):
  - func.func 签名 (GM 输入/输出)
  - memref.alloc (buffer 大小 → UB 容量判断)
  - hivm.hir.* 语义 op (load/store/mmatmul/vadd...)
  - hivm.* 同步 op (set_flag/wait_flag/pipe_barrier/sync_block*)
  - 末尾追加: HIVMIRAnalyzer 算出的 RAW/WAR/WAW 依赖清单 (可靠, 基于 buffer 名)

删除 (融合无关):
  - // 注释, 空行
  - arith./scf./memref.cast/tt./llvm./index./tensor. (标量/寻址/控制流)

用法: python filter_hivm_for_fusion.py <hivm_try.txt> [--out 输出.mlir]
"""
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivmir_analyzer import HIVMIRAnalyzer  # noqa: E402

KEEP_PREFIX = ("func.func", "memref.alloc", "hivm.hir.", "hivm.set_",
               "hivm.wait_", "hivm.pipe_", "hivm.sync_")
DROP_PREFIX = ("arith.", "scf.", "memref.cast", "memref.copy", "memref.reinterpret",
               "tt.", "llvm.", "index.", "tensor.", "bufferization.", "linalg.",
               "//", "#", "module")


def keep_line(line):
    s = line.strip()
    if not s:
        return False
    if s.startswith(DROP_PREFIX):
        return False
    if s.startswith(KEEP_PREFIX):
        return True
    if "memref.alloc()" in s:   # alloc 行 (buffer 大小 → UB 容量判断)
        return True
    return False


def filter_text(text):
    kept = []
    for line in text.splitlines():
        if keep_line(line):
            kept.append(line.rstrip())
    return "\n".join(kept)


def main(path, out_path):
    text = Path(path).read_text(encoding="utf-8")
    filtered = filter_text(text)

    # 末尾追加: 依赖清单 (可靠, 基于 buffer 名, 不靠字段名)
    ha = HIVMIRAnalyzer()
    report = ha.analyze_file(Path(path))
    dep_lines = ["", "# ── 依赖清单 (RAW/WAR/WAW, 基于 buffer 名, 可靠) ──"]
    for op in report.ops:
        for d in op.dependencies:
            dep_lines.append(
                f"#   op{d['from_op_id']} -> op{op.op_id}: {d['type']} on {d['buffer']}")
    if len(dep_lines) == 2:
        dep_lines.append("#   (无依赖)")

    # 末尾追加: 读取指引
    guide = ["", "# ── 算子融合怎么看 (给 LLM) ──",
             "# 1. 看 op 顺序: 谁先谁后 (load → 计算 → store)",
             "# 2. 看同步: set_flag/wait_flag/pipe_barrier 隔开哪两个阶段",
             "# 3. 看依赖: 上面的 RAW/WAR/WAW 清单 → 谁喂谁",
             "# 4. 找融合候选: RAW 链上相邻的 vadd/vmul/... 逐元素 op, 中间无 GM 写 → 可融合",
             "#    (消除中间 UB↔GM 读写)",
             "# 5. 找 WAR: 同一 buffer 被重写 → 分配独立 buffer 解锁并行",
             "# 6. 检查 UB 容量: 融合后 buffer 总大小 ≤ 192KB (910B3 UB)"]

    out = filtered + "\n".join(dep_lines) + "\n" + "\n".join(guide) + "\n"
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法: python filter_hivm_for_fusion.py <hivm_try.txt> [--out 输出.mlir]")
        sys.exit(1)
    src = args[0]
    out = "hivm_fusion_view.txt"
    if "--out" in args:
        out = args[args.index("--out") + 1]
    result = main(src, out)
    p = Path(out)
    p.write_text(result, encoding="utf-8")
    print(f"[filter] {src} → {out}  ({len(result.splitlines())} 行)")
