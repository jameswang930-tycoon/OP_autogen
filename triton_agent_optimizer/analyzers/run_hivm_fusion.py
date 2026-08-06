#!/usr/bin/env python3
"""Tier2 融合专用: 编译单文件 → HIVM MLIR → txt → nga run 融合分析 → 产物写 round_dir/08_fusion/。

流程 (算子融合阶段多走一步):
  ① 编译单文件 (rm ~/.triton, 跑 kernel_op.py) → ~/.triton/dump → kernel.ttadapter.mlir
  ② bishengir-compile → HIVM MLIR 文本 (不截断)
  ③ 存产物: 08_fusion/kernel.hivm.mlir + kernel.hivm.txt
  ④ nga run 调 skills/triton-op-fusion/SKILL.md 分析 RAW/WAR/WAW + 融合候选
  ⑤ 写 08_fusion/fusion_analysis.json + fusion_analysis.md

用法: python3 analyzers/run_hivm_fusion.py <single_file> <round_dir> [--stub]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def _compile_hivm(single_file: Path, work_dir: Path):
    """跑一遍单文件 → ~/.triton dump → 拷 ttadapter → bishengir → HIVM MLIR 文本。"""
    env = dict(os.environ, TRITON_DEBUG="1", TRITON_DISABLE_CACHE="1")
    subprocess.run("rm -rf ~/.triton", shell=True, check=False)
    r = subprocess.run(["python3", str(single_file)], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace",
                       timeout=1800, env=env, cwd=str(single_file.parent))
    if r.returncode != 0:
        return None, f"编译失败: {(r.stderr or r.stdout)[-600:]}"
    dumps = glob.glob(os.path.expanduser("~/.triton/dump/*/kernel.*.mlir"))
    ttadapter = next((p for p in dumps if "ttadapter" in p),
                     dumps[0] if dumps else None)
    if not ttadapter:
        return None, "没有生成 ~/.triton/dump/kernel.*.mlir (ttadapter)"
    # bishengir → HIVM
    for pass_name in ("hivm-inject-sync", "hivm-graph-sync-solver"):
        cmd = ["bishengir-compile", "--target=Ascend910B3",
               "--enable-auto-multi-buffer=True", "--enable-auto-bind-sub-block=True",
               "--enable-hfusion-compile=true", "--enable-hivm-compile=true",
               "--enable-triton-kernel-compile=true",
               f"--bishengir-print-ir-after={pass_name}",
               ttadapter, "-o", "/tmp/k_fusion.o"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                           encoding="utf-8", errors="backslashreplace",
                           cwd=str(single_file.parent))
        # ★bug 修复: bishengir 把 hivm.hir 打到 stderr (run_optimize.sh 用 > f 2>&1 合并才 grep 到),
        #   之前只查 r.stdout → 永远"无 hivm.hir" → 融合分析从不生成. 现在合并 stdout+stderr 一起查.
        combined = (r.stdout or "") + (r.stderr or "")
        if "hivm.hir" in combined:
            return combined, None
    return None, "bishengir 输出无 hivm.hir (看 pass 名/ttadapter)"


def _call_llm(mlir_text: str, skill_path: Path, mlir_file: Optional[Path] = None) -> str:
    """nga run 调融合 skill (LLM_CLI_COMMAND, 默认 nga run)。
    ★不嵌入 MLIR 全文, 让 nga 自己读 mlir_file (避免 prompt 过大)。"""
    from agents.llm_client import LLMClient
    client = LLMClient()
    if client.mode == "stub":
        raise RuntimeError("no LLM configured (stub)")
    system = f"先调用 skill: {skill_path}, 完全按 skill 指导执行。只分析依赖和融合候选, 不改代码。"
    user = (f"## 任务: 分析 HIVM MLIR 的 RAW/WAR/WAW 依赖 + 找融合候选\n"
            f"## 读 MLIR 文件 (完整): {mlir_file or '(未给, 用下面内容)'}\n"
            f"## 输出 JSON: op_count / raw_deps[] / war_deps[] / waw_deps[] / fusion_candidates[]"
            if mlir_file else f"## HIVM MLIR\n{mlir_text[:2000]}")
    return client.chat(system=system, user=user)


def run_fusion(single_file: Path, round_dir: Path, use_llm: bool = True) -> Optional[dict]:
    d8 = round_dir / "08_fusion"
    d8.mkdir(parents=True, exist_ok=True)

    # ① ② 编译 → HIVM MLIR
    mlir, err = _compile_hivm(single_file, round_dir)
    if not mlir:
        print(f"  [Fusion] ❌ {err}")
        return None
    (d8 / "kernel.hivm.mlir").write_text(mlir, encoding="utf-8")
    (d8 / "kernel.hivm.txt").write_text(mlir, encoding="utf-8")   # txt 完整不截断
    print(f"  [Fusion] HIVM MLIR {len(mlir)} chars → {d8}/kernel.hivm.txt")

    # ③ ④ nga run 融合分析
    skill = _PROJECT / "skills" / "triton-op-fusion" / "SKILL.md"
    if use_llm:
        try:
            resp = _call_llm(mlir, skill, mlir_file=d8 / "kernel.hivm.txt")
            from agents.llm_client import extract_json
            analysis = extract_json(resp)
        except Exception as e:
            print(f"  [Fusion] nga run 失败: {e}")
            analysis = {"error": str(e), "fusion_candidates": []}
    else:
        analysis = {"stub": True, "op_count": mlir.count("hivm.hir."),
                    "fusion_candidates": []}

    # ⑤ 写产物
    (d8 / "fusion_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8")
    (d8 / "fusion_analysis.md").write_text(
        f"# Tier2 融合分析\n\n```json\n{json.dumps(analysis, ensure_ascii=False, indent=2)}\n```",
        encoding="utf-8")
    n_cand = len(analysis.get("fusion_candidates", []))
    print(f"  [Fusion] ✅ fusion_analysis: {n_cand} 个融合候选 → {d8}")
    return analysis


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tier2 融合: HIVM MLIR + nga run 依赖分析")
    p.add_argument("single_file", type=str)
    p.add_argument("round_dir", type=str)
    p.add_argument("--stub", action="store_true")
    args = p.parse_args()
    r = run_fusion(Path(args.single_file), Path(args.round_dir), use_llm=not args.stub)
    sys.exit(0 if r else 1)
