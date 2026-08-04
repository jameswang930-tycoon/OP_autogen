#!/usr/bin/env python3
"""HIVM 完整翻译 — hivm_try.txt → 人话。

把每个 HIVM op 翻译成: 引擎 / 从哪搬到哪 / 执行什么操作 / 数据块大小 / dtype / 依赖 / 所有属性。
不删任何内容: 每个 op 都给「翻译 + 原文」。翻译不了的标 [待补] 并保留原文。

════════════════════════════════════════════════════════════════════
★ 输入/输出路径 (由你改动下面两个变量即可, 不用传命令行参数):
    INPUT  = 输入 hivm_try.txt 的路径
    OUTPUT = 输出翻译文本的路径 (None = 直接打印到屏幕)
  例:
    INPUT  = "input/matmul/e2e_run/02_hivm/hivm_try.txt"
    OUTPUT = "input/matmul/e2e_run/06_diagnosis/hivm_translation.txt"
  也可命令行: python translate_hivm.py <hivm_try.txt> [--raw]
════════════════════════════════════════════════════════════════════
"""
import os
import re
import sys
from pathlib import Path

# ── ★ 由你改动: 输入文件 + 输出文件 ──
INPUT = "input/matmul/e2e_run/02_hivm/hivm_try.txt"   # ← 改成你的 hivm_try.txt 路径
OUTPUT = None                                          # ← 改输出路径; None=打印屏幕

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hivmir_analyzer import HIVMIRAnalyzer  # noqa: E402

# ── op_type → (引擎, 操作语义)  [依据 AscendNPU-IR HIVMDialect 官方文档] ──
OP_SEM = {
    "gm_to_ub": ("GM→UB", "数据搬运: 全局内存(GM) → 片上 UB/L1 (MTE2)"),
    "ub_to_gm": ("UB→GM", "数据搬运: 片上 UB → 全局内存(GM) (MTE3)"),
    "gm_to_l1": ("GM→L1", "数据搬运: 全局内存(GM) → L1 (MTE2)"),
    "l1_to_l0": ("L1→L0", "数据搬运: L1 → L0A/L0B (喂 Cube, MTE1)"),
    "l0_to_gm": ("L0→GM", "数据搬运: L0C → 全局内存(GM)"),
    "matmul": ("CubeUnit", "矩阵乘法 (全局内存输入, Cube)"),
    "mix_matmul": ("CubeUnit", "矩阵乘法 + 后向量函数 tile 级融合 (CV 混合)"),
    "mmadL1": ("CubeUnit", "矩阵乘加: L1 → L0C (Cube 核心 MMA, 我们的主计算)"),
    "batchMmadL1": ("CubeUnit", "批量矩阵乘加 (带 batch 维, L1→L0C)"),
    "mix_group_matmul": ("CubeUnit", "分组矩阵乘法 (MoE 每专家, 可融合后向量)"),
    "fixpipe": ("FixPipe", "L0C→UB 数据搬移 + 量化/激活/布局变换融合 (FIXP)"),
    "vadd": ("VecUnit", "向量加法 (UB 内)"),
    "vmul": ("VecUnit", "向量乘法 (UB 内)"),
    "vsub": ("VecUnit", "向量减法 (UB 内)"),
    "vdiv": ("VecUnit", "向量除法 (UB 内)"),
    "vmax": ("VecUnit", "向量取最大 (UB 内)"),
    "vmin": ("VecUnit", "向量取最小 (UB 内)"),
    "vexp": ("VecUnit", "向量指数 exp (UB 内)"),
    "vlog": ("VecUnit", "向量对数 log (UB 内)"),
    "vln": ("VecUnit", "向量自然对数 ln (UB 内)"),
    "vabs": ("VecUnit", "向量取绝对值 (UB 内)"),
    "vrelu": ("VecUnit", "向量 ReLU 激活 (UB 内)"),
    "vsqrt": ("VecUnit", "向量开方 sqrt (UB 内)"),
    "vrsqrt": ("VecUnit", "向量倒数开方 rsqrt (UB 内)"),
    "vtanh": ("VecUnit", "向量双曲正切 tanh (UB 内)"),
    "vneg": ("VecUnit", "向量取负 (UB 内)"),
    "vbrc": ("VecUnit", "向量广播 broadcast (UB 内)"),
    "vcvt": ("VecUnit", "向量类型转换 cvt (UB 内)"),
    "vmov": ("VecUnit", "向量搬运/拷贝 mov (UB 内)"),
    "vsel": ("VecUnit", "向量选择/条件选值 sel (UB 内)"),
    "vcmp": ("VecUnit", "向量比较 cmp (UB 内)"),
    "vdul": ("VecUnit", "向量解交织 dup (UB 内)"),
    "vdup": ("VecUnit", "向量重复 dup (UB 内)"),
    "vbitsel": ("VecUnit", "向量按位选择 (UB 内)"),
    "vconv": ("VecUnit", "向量格式转换 conv (UB 内)"),
    "set_flag": ("Sync", "同步-置标志位: 前序指令完成后置位通知 (set_pipe→wait_pipe)"),
    "wait_flag": ("Sync", "同步-等待标志位: 阻塞直到标志位就绪 (跨 pipe 依赖)"),
    "pipe_barrier": ("Sync", "同步-管道屏障: 同 pipe 队列内顺序约束"),
    "sync_block": ("Sync", "同步-块同步: 跨 kernel/block 栅栏"),
    "atomic_cas": ("Atomic", "原子比较并交换 CAS"),
    "get_block_idx": ("Scalar", "取 block 索引 (并行调度)"),
    "get_block_num": ("Scalar", "取 block 总数 (并行调度)"),
    "get_sub_block_idx": ("Scalar", "取子 block 索引"),
    "get_sub_block_num": ("Scalar", "取子 block 总数"),
    "get_sys_cnt": ("Scalar", "取系统计数器"),
    "load_scalar": ("Scalar", "标量加载 (从 LLVM 指针)"),
}

# address_space 值 → 中文区域
REGION_CN = {"ub": "UB(统一缓冲)", "cbuf": "L1", "ca": "L0A", "cb": "L0B",
             "cc": "L0C", "gm": "GM(全局内存)", "l1": "L1"}


def _region(type_str):
    m = re.search(r"#hivm\.address_space<(\w+)>", type_str or "")
    if not m:
        return "?"
    return REGION_CN.get(m.group(1), m.group(1))


def _extract_types(ins_str, outs_str):
    """从 ins/outs 提取 (操作数名, 区域) 列表"""
    ins_types = re.findall(r"(%\w+)\s*:\s*([^)]+)", ins_str)
    outs_types = re.findall(r"(%\w+)\s*:\s*([^)]+)", outs_str)
    ins = [(n, _region(t)) for n, t in ins_types]
    outs = [(n, _region(t)) for n, t in outs_types]
    return ins, outs


def _arg_regions(path):
    """从 func.func 签名解析函数参数 (memref<..., #hivm.address_space<gm>>) → 区域"""
    text = Path(path).read_text(encoding="utf-8")
    regions = {}
    for m in re.finditer(r"%(\w+)\s*:\s*memref<[^>]*#hivm\.address_space<(\w+)>", text):
        regions["%" + m.group(1)] = REGION_CN.get(m.group(2), m.group(2))
    return regions


def translate_file(path, show_raw=True):
    ha = HIVMIRAnalyzer()
    report = ha.analyze_file(Path(path))
    d = ha.to_dict(report)
    arg_regions = _arg_regions(path)

    print("=" * 70)
    print(f"HIVM 完整翻译: {path}   ({report.num_ops} 个操作, {len(report.buffers)} 个 buffer)")
    print("=" * 70)

    for i, op in enumerate(report.ops):
        eng, sem = OP_SEM.get(op.op_type, ("?", "未知操作 [待补]"))
        print()
        print(f"── op{op.op_id}  {op.op_type}  ──")
        print(f"   引擎   : {eng}")
        print(f"   语义   : {sem}")
        # 数据流 (从哪搬到哪): 每操作数查 buffer 区域
        if op.src or op.dst:
            def _r(name):
                b = d.get("buffers", {}).get(name)
                if b and b.get("region") and b["region"] not in ("unknown", "?"):
                    return b["region"]
                if name in arg_regions:
                    return arg_regions[name]
                return "?"
            flow = []
            if op.src:
                flow.append(f"源 {op.src}({_r(op.src)})")
            if op.src2:
                flow.append(f"源2 {op.src2}({_r(op.src2)})")
            if op.dst:
                flow.append(f"目标 {op.dst}({_r(op.dst)})")
            print(f"   数据流 : {' → '.join(flow)}")
        # 数据块大小 + dtype
        if op.size_kb:
            print(f"   数据块 : {op.size_kb:.1f} KB   (dtype={op.dtype or '?'})")
        elif op.dtype:
            print(f"   dtype  : {op.dtype}")
        # 属性
        if op.attrs:
            print(f"   属性   : {op.attrs}")
        # 依赖
        if op.dependencies:
            deps = ", ".join(f"op{d['from_op_id']}({d['type']} on {d['buffer']})"
                             for d in op.dependencies)
            print(f"   依赖   : {deps}")
        # 同步 op 的 pipe 信息 (从 instruction 提取 #hivm.pipe<>)
        if op.op_type in ("set_flag", "wait_flag", "pipe_barrier", "sync_block"):
            pipes = re.findall(r"#hivm\.pipe<(\w+)>", op.instruction)
            if pipes:
                print(f"   管道   : {pipes}")
        if show_raw and op.instruction:
            print(f"   原文   : {op.instruction}")

    # buffer 汇总 (翻译每个 buffer 的 region/大小/生产者消费者)
    print()
    print("─" * 70)
    print("Buffer 清单 (名称 / 区域 / 大小 / 生产者 / 消费者):")
    for name, b in d.get("buffers", {}).items():
        print(f"   {name:12s} 区域={b['region']:12s} 大小={b['size_kb']:.1f}KB  "
              f"producers={b['producers']} consumers={b['consumers']}")

    print()
    print("─" * 70)
    print("依赖汇总: RAW x{} WAR x{} WAW x{}  (总 {})".format(
        len(report.raw_deps), len(report.war_deps), len(report.waw_deps),
        len(report.raw_deps) + len(report.war_deps) + len(report.waw_deps)))
    return report


if __name__ == "__main__":
    import io
    raw = "--raw" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    repo = Path(__file__).resolve().parent.parent   # 仓库根 (analyzers/..)
    # 输入: 命令行参数 > 脚本顶部 INPUT 配置
    if args:
        path = args[0]
    elif INPUT:
        path = str(repo / INPUT) if not Path(INPUT).is_absolute() else INPUT
    else:
        print("用法: python translate_hivm.py <hivm_try.txt> [--raw]  或改脚本顶部 INPUT")
        sys.exit(1)
    if not Path(path).exists():
        sys.exit(f"❌ 找不到 {path}\n  改脚本顶部 INPUT 变量 或 传命令行参数")
    # 输出: 脚本顶部 OUTPUT 配置 > 屏幕
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    translate_file(path, show_raw=raw)
    sys.stdout = old
    text = buf.getvalue()
    if OUTPUT:
        out = Path(OUTPUT) if Path(OUTPUT).is_absolute() else repo / OUTPUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[translate_hivm] 已写入: {out}")
    else:
        print(text, end="")
