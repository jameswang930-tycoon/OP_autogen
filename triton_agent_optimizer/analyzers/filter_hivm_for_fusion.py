#!/usr/bin/env python3
"""过滤 hivm_try.txt → 算子融合专用精简版 (不翻译, 删无关, 保留相关)。

原理: 字段名会变, 提取易失败 → 改为"删无关行, 保留相关行", 让 LLM 读保留部分。

保留 (融合需要):
  - func.func 签名 (GM 输入/输出)
  - memref.alloc (buffer 大小 → UB 容量判断)
  - hivm.hir.* 语义 op (load/store/mmatmul/vadd...)
  - hivm.* 同步 op (set_flag/wait_flag/pipe_barrier/sync_block*)
  - 末尾追加: HIVMIRAnalyzer 算出的 RAW/WAR/WAW 依赖清单 (可靠, 基于 buffer 名)
  - 末尾追加: ★每 op 估算耗时占比 (msprof op PipeUtilization 实测 pipe 耗时映射, 融合优先级参考)

删除 (融合无关):
  - // 注释, 空行
  - arith./scf./memref.cast/tt./llvm./index./tensor. (标量/寻址/控制流)

用法:
  python filter_hivm_for_fusion.py <hivm_try.txt> [--out 输出.mlir] [--boards board_1.json,board_2.json]
  --boards: msprof op 解析出的 board.json (逗号分隔, 可多个) → 给每个 HIVM op 附实测 pipe 耗时
            (多 kernel 场景下同 pipe 取各 board 最大耗时, 表示该通路在任务中的总忙时间)
"""
import json
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivmir_analyzer import HIVMIRAnalyzer, OP_TO_ENGINE  # noqa: E402

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


# ── ★P3: 每 op 估算耗时 — 从 board.json 的 PipeUtilization.csv 提取 pipe 耗时, 映射到 HIVM op ──
#   op_type → pipe 名 (与 msprof op PipeUtilization 列名对应)
_OP_PIPE = {
    "gm_to_ub": "aiv_mte2_time_us", "ub_to_gm": "aiv_mte3_time_us",
    "gm_to_l1": "aic_mte2_time_us", "l1_to_l0": "aic_mte1_time_us",
    "l0_to_gm": "aic_mte3_time_us",
}
_VEC_ENGINE = "VecUnit"
_CUBE_OPS = {"matmul", "matrixmul", "mix_matmul", "mmadL1", "batchMmadL1"}


def _collect_pipe_us(board_paths):
    """收集所有 board 的每 pipe 耗时 (us): 同 pipe 取 max (多 kernel 时该通路总忙时间近似)."""
    pipe_us = {}
    for p in board_paths:
        if not Path(p).exists():
            continue
        try:
            bd = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = (bd.get("raw", {}).get("PipeUtilization", {}) or {}).get("rows", [])
        if not rows:
            continue
        for k, v in rows[0].items():
            kl = k.lower().strip()
            if kl.endswith("(us)") and any(pipe in kl for pipe in
                                           ("cube_time", "mac_time", "vec_time", "mte1_time",
                                            "mte2_time", "mte3_time", "fixp", "scalar_time")):
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                # 规范键: (us) → _us, 统一为 aic_/aiv_ 前缀名 (与 _OP_PIPE/_op_pipe_us 一致)
                key = kl.replace("(us)", "_us")
                if f > 0 and f > pipe_us.get(key, 0.0):
                    pipe_us[key] = f
    return pipe_us


def _op_pipe_us(op, pipe_us):
    """HIVM op → 估算耗时 (us): 按 op_type + buffer 区域映射 pipe; 无则 None.

    ★load/store 按 buffer 区域分流: outs 到 L1 → GM→L1 (aic_mte2, cube 通路);
      outs 到 UB → GM→UB (aiv_mte2); store 源在 L0C → L0C→GM (aic_mte3/fixpipe), 在 UB → UB→GM (aiv_mte3)."""
    op_type = op.op_type
    if op_type == "gm_to_ub":
        if op.memory_region == "L1":
            return pipe_us.get("aic_mte2_time_us")
        return pipe_us.get("aiv_mte2_time_us")
    if op_type == "ub_to_gm":
        if op.memory_region == "L0C":
            return pipe_us.get("aic_mte3_time_us") or pipe_us.get("aic_fixpipe_time_us")
        return pipe_us.get("aiv_mte3_time_us")
    if op_type in _OP_PIPE:
        return pipe_us.get(_OP_PIPE[op_type])
    if op_type in _CUBE_OPS:
        return pipe_us.get("aic_cube_time_us") or pipe_us.get("aic_mac_time_us")
    if OP_TO_ENGINE.get(op_type) == _VEC_ENGINE:
        return pipe_us.get("aiv_vec_time_us")
    return None  # sync op (set_flag/wait_flag/pipe_barrier) 无独立耗时


def build_time_note(report, pipe_us):
    """生成『每 op 估算耗时占比』清单 (追加到 view 末尾, 给 Tier2 planner 排融合优先级)."""
    lines = ["", "# ── 每 op 估算耗时占比 (来源: msprof op PipeUtilization 实测 pipe 耗时, 融合优先级参考) ──"]
    if not pipe_us:
        lines.append("#   (未传 --boards 或 board 无 PipeUtilization 数据 → 无法标注耗时, 用 op 顺序+依赖判断)")
        return lines
    pipes = sorted(pipe_us.items(), key=lambda kv: -kv[1])
    pipe_desc = ", ".join(f"{k}={v:.0f}us" for k, v in pipes[:8])
    lines.append(f"#   pipe 实测耗时: {pipe_desc}")
    total = max(pipe_us.values())  # 关键路径近似 = 最忙 pipe
    if not total:
        lines.append("#   (无 pipe 耗时数据)")
        return lines
    for op in report.ops:
        us = _op_pipe_us(op, pipe_us)
        if us is None:
            continue
        pct = us / total * 100
        lines.append(f"#   op{op.op_id} {op.op_type:12s} ≈ {us:6.1f}us ({pct:4.1f}% of 最忙pipe)")
    lines.append("#   ★融合优先级: 优先消除 RAW 链上『耗时占比最高的逐元素 op』的 GM 往返 (中间写回 GM 最贵);")
    lines.append("#     matmul 占大头 → 融合收益在 epilogue (bias/gelu 并进 cube 累加后); vec 占大头 → 减 GM 往返")
    return lines


def main(path, out_path, board_paths=None):
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

    # 末尾追加: 每 op 估算耗时 (P3, 需 --boards)
    pipe_us = _collect_pipe_us(board_paths or [])
    time_lines = build_time_note(report, pipe_us)

    # 末尾追加: 读取指引
    guide = ["", "# ── 算子融合怎么看 (给 LLM) ──",
             "# 1. 看 op 顺序: 谁先谁后 (load → 计算 → store)",
             "# 2. 看同步: set_flag/wait_flag/pipe_barrier 隔开哪两个阶段",
             "# 3. 看依赖: 上面的 RAW/WAR/WAW 清单 → 谁喂谁",
             "# 4. 找融合候选: RAW 链上相邻的 vadd/vmul/... 逐元素 op, 中间无 GM 写 → 可融合",
             "#    (消除中间 UB↔GM 读写)",
             "# 5. 找 WAR: 同一 buffer 被重写 → 分配独立 buffer 解锁并行",
             "# 6. 检查 UB 容量: 融合后 buffer 总大小 ≤ 192KB (910B3 UB)",
             "# 7. ★结合上面的『每 op 估算耗时占比』排优先级: 先融合耗时占比最高的链"]

    out = filtered + "\n".join(dep_lines) + "\n".join(time_lines) + "\n".join(guide) + "\n"
    Path(out_path).write_text(out, encoding="utf-8")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("用法: python filter_hivm_for_fusion.py <hivm_try.txt> [--out 输出.mlir] [--boards b1.json,b2.json]")
        sys.exit(1)
    src = args[0]
    out = "hivm_fusion_view.txt"
    boards = []
    i = 1
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
        elif args[i] == "--boards" and i + 1 < len(args):
            boards = [b.strip() for b in args[i + 1].split(",") if b.strip()]
            i += 2
        else:
            i += 1
    result = main(src, out, boards)
    p = Path(out)
    p.write_text(result, encoding="utf-8")
    if boards:
        print(f"[fusion] {out}: op+依赖+每op耗时占比 (boards={len(boards)})")
    else:
        print(f"[fusion] {out}: op+依赖 (无 --boards → 无耗时标注)")
    print(f"[filter] {src} → {out}  ({len(result.splitlines())} 行)")
