#!/usr/bin/env python3
"""
Triton Agent Optimizer v4.2 — 主入口（单文件驱动 + 真机双源 + 6 层轮次化 + sweep + 手递 + 优秀案例）

v4 流程:
  1. 单文件 kernel_op.py（①config + ②kernel + ③test 一体, coder 只改它）
  2. Scheduler 循环（每轮）:
     run_optimize.sh <op_dir> <round_dir> → diagnosis.json
       → 07_tier{N}_fields (筛选字段) → Planner(nga run) → plan.md (含 changes[])
       → Coder(nga run / 确定性替换 changes[]) → 改 kernel_op.py
       → (Tier2 多走一步) 08_fusion: 编译 HIVM MLIR → nga run fusion skill → 依赖分析
       → 只跑 msprof 端到端 → 加速比 → 报错就地回传 Coder → 晋升/降级/停止

LLM 调用（服务器无 Claude API, 用本地 codeagent）:
  export LLM_CLI_COMMAND="nga run"   # echo "<prompt>" | nga run
  三个 skill: skills/triton-op-planner / triton-op-coder / triton-op-fusion

════════════════════ 服务器运行步骤 ════════════════════
  0. 环境:
     conda activate triton-npu
     source /usr/local/Ascend/ascend-toolkit/set_env.sh
     cd triton_agent_optimizer
     echo "测试, 调用 skill triton-op-planner" | nga run   # 确认 nga 通
  1. 单文件能跑:
     python3 input/matmul/kernel_op.py
     # 预期: [info] kernel launched & synced OK
  2. 完整优化循环 (一键; input/ 下任一算子目录, 都是标准 kernel_op.py 单文件):
     # ── 多 kernel / matmul 族 (融合 + 分块空间) ──
     LLM_CLI_COMMAND="nga run" python3 main.py input/matmul --fresh --max-rounds 30            # 两层 MLP (3 kernel, Tier2 融合)
     LLM_CLI_COMMAND="nga run" python3 main.py input/attention_mlp --fresh --max-rounds 30     # 自注意力+MLP (5 kernel, Tier1/2)
     LLM_CLI_COMMAND="nga run" python3 main.py input/matmul_relu --fresh --max-rounds 30       # matmul+relu 2kernel (Tier2 epilogue 融合)
     LLM_CLI_COMMAND="nga run" python3 main.py input/matmul_transpose --fresh --max-rounds 30  # 转置 matmul (Tier4 strided→预转置)
     # ── attention 族 (算法 + 融合) ──
     LLM_CLI_COMMAND="nga run" python3 main.py input/flash_attention --fresh --max-rounds 30   # 多头因果 flash (Tier1 因果剪枝)
     # ── 归约 / 逐元素 (访存 + 计算优化) ──
     LLM_CLI_COMMAND="nga run" python3 main.py input/rms_norm --fresh --max-rounds 30          # RMSNorm 归约 (Tier4/5)
     LLM_CLI_COMMAND="nga run" python3 main.py input/rms_norm_residual --fresh --max-rounds 30  # RMSNorm+残差 (Tier2 残差并入/Tier4)
     LLM_CLI_COMMAND="nga run" python3 main.py input/layernorm --fresh --max-rounds 30         # LayerNorm 两遍扫 (Tier1 单遍合并)
     LLM_CLI_COMMAND="nga run" python3 main.py input/softmax --fresh --max-rounds 30           # 行级 softmax 归约 (Tier4/5)
     LLM_CLI_COMMAND="nga run" python3 main.py input/sigmoid --fresh --max-rounds 30           # 纯逐元素 sigmoid (Tier5 原生指令)
     LLM_CLI_COMMAND="nga run" python3 main.py input/vector_add --fresh --max-rounds 30        # 逐元素加法 (Tier4 连续化/Tier5)
     LLM_CLI_COMMAND="nga run" python3 main.py input/fused_add_mul --fresh --max-rounds 30     # (x+z)*w 逐元素链 (Tier2 融合/Tier5)
      # ── 卷积族 (算法 im2col + 融合) ──
      LLM_CLI_COMMAND="nga run" python3 main.py input/conv2d --fresh --max-rounds 30            # 卷积 (Tier1 im2col 走 cube)
      LLM_CLI_COMMAND="nga run" python3 main.py input/conv_bias_relu --fresh --max-rounds 30    # conv+bias+relu 3kernel (Tier2 融合)
      # ── ★新算子 (2026-08-13, 复杂算子补充) ──
      LLM_CLI_COMMAND="nga run" python3 main.py input/batchnorm2d --fresh --max-rounds 30       # BatchNorm2d 推理 (通道归约, Tier1 rsqrt 乘链/Tier2 融合)
      LLM_CLI_COMMAND="nga run" python3 main.py input/maxpool2d --fresh --max-rounds 30         # MaxPool2d 窗口访存 (Tier4 连续化/Tier5 max 链)
      LLM_CLI_COMMAND="nga run" python3 main.py input/conv1d --fresh --max-rounds 30            # Conv1d (Tier1 im2col 走 cube, 同 conv2d 教学)
      # ── ★工业界经典长链 (2026-08-14, 与 bench_910b3 工业级基准一一对应) ──
      LLM_CLI_COMMAND="nga run" python3 main.py input/batched_matmul --fresh --max-rounds 30    # BMM 多 batch GEMM (Tier1/3/4)
      LLM_CLI_COMMAND="nga run" python3 main.py input/swiglu_mlp --fresh --max-rounds 30         # LLaMA FFN: up/gate 2matmul+silu 门控+down (Tier2 门控并入/Tier3)
      LLM_CLI_COMMAND="nga run" python3 main.py input/resnet_block --fresh --max-rounds 30       # ResNet 残差块: conv+BN+relu+残差 (Tier2 BN/relu 并入 conv epilogue)
      LLM_CLI_COMMAND="nga run" python3 main.py input/vit_block --fresh --max-rounds 30          # ViT encoder: LN→QKV→MHA→GELU MLP (Tier1/2 融合)
      LLM_CLI_COMMAND="nga run" python3 main.py input/bert_block --fresh --max-rounds 30         # BERT encoder post-LN (Tier1/2)
      LLM_CLI_COMMAND="nga run" python3 main.py input/gqa_attention --fresh --max-rounds 30      # GQA+RoPE: QKV→RoPE→KV组复制→MHA (Tier1 RoPE 并入/Tier2)
      LLM_CLI_COMMAND="nga run" python3 main.py input/transformer_decoder_block --fresh --max-rounds 30  # LLaMA decoder 层 (17 kernel, Tier1/2 大融合)
      LLM_CLI_COMMAND="nga run" python3 main.py input/mamba_block --fresh --max-rounds 30        # Mamba SSM 块 (Tier1 扫描优化/Tier2)
      LLM_CLI_COMMAND="nga run" python3 main.py input/mixture_of_experts --fresh --max-rounds 30 # MoE topk: router+8 expert SwiGLU (Tier1 稀疏路由/Tier2)
      # 每轮: 采集→07字段→planner→coder→正确性校验→msprof端到端→加速比
      # 从头开始(清 outputs/<op> + 重置): 加 --fresh
      # 续跑(从上次 round 继续, 不清旧产物): 不加 --fresh
      # 轮数 30 = 每个算子跑满 30 轮看最优 (默认 max_rounds=200, 这里用 30 省时间)
  3. 只采集+解析 (不跑优化):
     bash analyzers/run_optimize.sh input/matmul input/matmul/e2e_run
  4. 只看各 tier 筛字段 (解析完 07 自动产出):
     cat input/matmul/e2e_run/07_tier1_fields/tier1_fields.txt
  5. 只测 Tier2 融合分析:
     python3 analyzers/run_hivm_fusion.py input/matmul/kernel_op.py /tmp/fusion_test
  6. 看诊断报告:
     cat <round_dir>/06_diagnosis/diagnosis.json
  ══════════════════════════════════════════════════════════

Usage:
  python main.py input/matmul --fresh --max-rounds 30 [--target 1.5] [--stub]
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from analyzers.merge_single_file import merge  # noqa: E402
from config import config  # noqa: E402,F401 — 触发 _load_dotenv() 加载 .env (LLM_CLI_COMMAND 等)


class _Tee:
    """把 stdout/stderr 同时写终端 + 运行日志文件 (outputs/<op>/optimization.log)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()   # 保证 log 实时可读
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        # 全部底层流都无 fileno (如管道/无终端环境) → 抛标准异常, 不让调用方拿到坏值
        raise io.UnsupportedOperation("fileno")

    def reconfigure(self, *args, **kwargs):
        """agent 模块会调 sys.stdout.reconfigure(encoding='utf-8') — 传播给所有流."""
        for s in self.streams:
            try:
                s.reconfigure(*args, **kwargs)
            except Exception:
                pass


def validate_kernel_op(kernel_op: Path) -> list:
    """★#5 新算子启动校验 — 防 A1/A3 隐性耦合, 返回警告列表.
    检查: KERNEL_LOOP / MATMUL_VERIFY / __main__ / 同名 kernel 多次调用(A1 聚合风险)."""
    import re
    code = kernel_op.read_text(encoding="utf-8") if kernel_op.exists() else ""
    warns = []
    if not code:
        return [f"❌ 空文件: {kernel_op}"]
    if "KERNEL_LOOP" not in code:
        warns.append("⚠ 缺 KERNEL_LOOP → verify 无法 ÷N 取单次端到端 (加速比会错)")
    if "MATMUL_VERIFY" not in code:
        warns.append("⚠ 缺 MATMUL_VERIFY 正确性校验块 → verify 每轮会 FAIL (正确性未通过)")
    if 'if __name__ == "__main__":' not in code:
        warns.append("⚠ 缺 __main__ 入口 → 无法直接运行")
    if "@triton.jit" not in code:
        warns.append("⚠ 无 @triton.jit kernel")
    # A1: 同名 kernel 被多次调用 → msprof op 按名聚合 → deep 画像混合 (不同形状时)
    jit_fns = set(re.findall(r"@triton\.jit\s*\ndef\s+(\w+)\s*\(", code))
    for fn in jit_fns:
        calls = len(re.findall(rf"\b{fn}\s*\[", code))
        if calls > 1:
            warns.append(f"⚠ kernel '{fn}' 被调用 {calls} 次 — 若形状/角色不同会被 msprof 同名聚合, "
                         f"deep 画像混合; 建议拆独立函数名 (如 matmul_kernel2)")
    return warns


def main():
    p = argparse.ArgumentParser(description="Triton Agent Optimizer v4")
    p.add_argument("op_dir", type=str, help="input/<op> 目录 (含 triton_kernel.py + config.json + test)")
    p.add_argument("--max-rounds", type=int, default=config.optim.max_rounds,
                   help="最大优化轮数 (默认从 config/.env 的 TRITON_AGENT_MAX_ROUNDS 读, 缺省 200)")
    p.add_argument("--target", type=float, default=0.0,
                   help="目标加速比, 达标即停; 0 或省略 = 不设目标, 跑满 --max-rounds 看最优")
    p.add_argument("--stub", action="store_true", help="不调 LLM/真机, 用 stub (本地测试)")
    p.add_argument("--remerge", action="store_true", help="强制重新合并单文件 (会覆盖 coder 改动)")
    p.add_argument("--fresh", action="store_true",
                   help="清空 outputs/<op>/ 旧产物 + 重置 trajectory, 从头开始")
    p.add_argument("--resume", action="store_true",
                   help="从上次 trajectory 续跑 (默认每次都初始化, 从 round1 重来)")
    p.add_argument("--sweep-blocks", action="store_true",
                   help="★D1: 优化前先扫 BLOCK 候选 (msprof 取最快写回 config), 固定好起点再进主循环")
    p.add_argument("--sweep-quick", action="store_true",
                   help="sweep 只扫前 4 个候选 (省时间)")
    args = p.parse_args()

    op_dir = Path(args.op_dir)
    if not op_dir.exists():
        print(f"[ERROR] Not found: {op_dir}")
        return 1

    # ①.5 --fresh: 清旧产物 + 重置 trajectory (从头开始)
    if args.fresh:
        import shutil
        out_dir = _PROJECT / "outputs" / op_dir.name
        if out_dir.exists():
            shutil.rmtree(out_dir)
            print(f"[main] --fresh: 已清空 {out_dir}")

    # ① 单文件 kernel_op.py (★源文件, coder 直接改它, 不覆盖)
    #    若缺 (旧式三文件 op) → 用 merge_single_file.py 生成一次
    kernel_op = op_dir / "kernel_op.py"
    if not kernel_op.exists():
        try:
            merge(op_dir, kernel_op)
        except Exception as e:
            # ★修复: 旧式三文件目录缺 test 驱动/文件结构不完整时 merge 会抛异常 (如 FileNotFoundError),
            #   不再让整个 main 崩溃出 traceback — 给清晰错误退出.
            print(f"[ERROR] 合并旧式三文件失败: {e}")
            print(f"  {op_dir} 不是标准的旧式三文件目录 (缺 test_matmul.py/test*.py/config.json), "
                  f"或文件内容无法合并成 kernel_op.py")
            return 1
    if not kernel_op.exists():
        print(f"[ERROR] 单文件不存在: {kernel_op}")
        return 1
    print(f"[main] 单文件: {kernel_op}")

    # ★#5 启动前校验 kernel_op.py 结构 (防 A1/A3 隐性耦合, 只警告不阻塞)
    for w in validate_kernel_op(kernel_op):
        print(f"[校验] {w}")

    # ★运行日志: 全部终端输出同时写入 outputs/<op>/optimization.log (每算子一个)
    #   (放在 --fresh 清理之后, 避免清掉 log; 追加模式保留历史运行)
    from datetime import datetime as _dt
    out_dir = _PROJECT / "outputs" / op_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "optimization.log"
    _log_f = open(log_path, "a", encoding="utf-8")
    _log_f.write(f"\n{'=' * 60}\n运行开始 {_dt.now().isoformat()}\n{'=' * 60}\n")
    _log_f.flush()
    sys.stdout = _Tee(sys.__stdout__, _log_f)
    sys.stderr = _Tee(sys.__stderr__, _log_f)
    print(f"[main] 运行日志 → {log_path}")

    # ② Scheduler 循环 (默认每次初始化; --resume 续跑)
    # ★D1: 前置 BLOCK 扫描 — 先固定好块再进主循环 (分块是乘性地基, 块差会让后面所有层诊断失真)
    if args.sweep_blocks:
        try:
            from analyzers.sweep_blocks import sweep
            print("\n[main] ══ 前置 BLOCK 扫描 (D1) ══")
            r = sweep(op_dir, quick=args.sweep_quick)
            if r.get("error"):
                print(f"[main] sweep 失败: {r['error']} → 用当前 BLOCK 继续")
        except Exception as e:
            print(f"[main] sweep 异常: {str(e)[:200]} → 继续主循环 (不阻断)")
    from agents.scheduler import Scheduler
    s = Scheduler(op_dir, max_rounds=args.max_rounds,
                  target_speedup=args.target, stub=args.stub, resume=args.resume)
    return s.run()




if __name__ == "__main__":
    sys.exit(main())
