#!/usr/bin/env python3
"""PyTorch 基准统一 msprof 测量包装器 — 让 PyTorch 侧与我们的算子走同一种测量 (msprof).

═══ 用法 ═══
  python3 pt_msprof.py <bench_script.py> [script args...]
  例:
    python3 pt_msprof.py bench_pytorch.py                     # torch.matmul
    python3 pt_msprof.py bench_pytorch_mlp.py                 # torch MLP
  调度器 AUTO_RUN_PT_BENCH 会用它代替直接跑 bench 脚本.

═══ 流程 ═══
  1. 一次 msprof 启动, 跑 <bench_script.py> (脚本内部 warmup + measure 次 forward,
     Event 计时并写自己的 json)
  2. 从 op_summary 同时算两种口径 (÷measure, 跳过热身行):
       端到端   = Σ全部 kernel 行 (含 aclnn 框架)
       纯kernel = Σ非 aclnn 行
  3. 覆盖 json: time_us = msprof 端到端, 追加 kernel_time_us = msprof 纯kernel,
     tflops 用 msprof 端到端重算 (flops 从脚本 Event 结果反推, 与 msprof 同口径)
  → 输出 json 与 verify 同口径 → 两端可比 (我们的 msprof E2E vs torch msprof E2E).

═══ 口径一致性 ═══
  与 agents/verifier._read_durations 完全一致 (端到端=Σ全部行, 纯kernel=Σ非aclnn行).
  msprof 是 device 端到端 (不含 host 侧 launch/gap); 两端同法, 故可比.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BENCH_DIR = Path(__file__).resolve().parent


def _script_json(script_name: str) -> Path:
    """bench_pytorch_X.py → pytorch_X_tflops.json (脚本固定写出的 json)."""
    stem = Path(script_name).stem
    rest = stem[len("bench_pytorch"):] if stem.startswith("bench_pytorch") else ""
    return _BENCH_DIR / "outputs" / f"pytorch{rest}_tflops.json"


def main():
    if len(sys.argv) < 2:
        print("用法: python3 pt_msprof.py <bench_script.py> [args...]")
        sys.exit(1)
    script = sys.argv[1]
    script_path = _BENCH_DIR / script if not Path(script).is_absolute() else Path(script)
    if not script_path.exists():
        print(f"❌ 脚本不存在: {script_path}")
        sys.exit(1)
    args = sys.argv[2:]

    out_json = _script_json(script)
    app_cmd = f"python3 {script_path} {' '.join(args)}".strip()

    # 1. 一次 msprof 跑脚本 (脚本自己 Event 计时 + 写 json)
    try:
        import shutil as _sh
        msprof_out = _BENCH_DIR / "outputs" / "_pt_msprof_tmp"
        _sh.rmtree(msprof_out, ignore_errors=True)
        msprof_out.mkdir(parents=True, exist_ok=True)
        cmd = ["msprof", f"--output={msprof_out}", f"--application={app_cmd}", "--ai-core=on"]
        print(f"[pt_msprof] 一次 msprof: {app_cmd}", flush=True)
        subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except Exception as e:
        print(f"❌ [pt_msprof] msprof 启动失败: {e} (脚本 Event 结果保留)")
        sys.exit(1)

    # 2. 从脚本 Event json 反推 flops (固定量, 不受计时方法影响)
    from bench_common import measure_pytorch_msprof
    flops = None
    try:
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        _tu = prev.get("time_us")
        _tf = prev.get("tflops")
        if _tu and _tf:
            flops = _tf * 1e12 * (_tu / 1e6)      # flops = tflops × 1e12 × event_s
        # warmup/measure: 脚本 json 有 measure; warmup 默认取脚本 --warmup 或 3
        _measure = prev.get("measure") or int(os.environ.get("BENCH_PT_MEASURE", "30"))
    except Exception:
        _measure = 30

    # 3. 统一 msprof 测量 → 覆盖 json (time_us=端到端, kernel_time_us=纯kernel)
    if out_json.exists():
        out_json.unlink()   # 防止 measure_pytorch_msprof 写前读到旧
    result = measure_pytorch_msprof(app_cmd, out_json, flops,
                                    measure=_measure, warmup=5)
    if result is None:
        print("⚠ [pt_msprof] msprof 测量失败 → 保留脚本 Event 结果")
        sys.exit(1)
    print(f"[pt_msprof] → {out_json}: time_us(端到端)={result['time_us']}us, "
          f"kernel_time_us(纯kernel)={result.get('kernel_time_us')}us")


if __name__ == "__main__":
    main()
