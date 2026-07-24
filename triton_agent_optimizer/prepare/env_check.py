#!/usr/bin/env python3
"""
环境准备与检查脚本。

═══════════════════════════════════════════════════════════════════════════════
  检查项
═══════════════════════════════════════════════════════════════════════════════

  1. 项目结构 — 所有关键目录和文件
  2. Python 环境 — 版本 + 必要包
  3. Emulator — emulators/common/__init__.py
  4. Cost Simulator — costModel/cost_emulator/simulator.py
  5. CANN/Ascend — ASCEND_HOME, msprof, 编译器, npu-smi (仅910B3)
  6. Triton 环境 — triton, torch (仅910B3)
  7. Agent Optimizer — triton_agent_optimizer/ 内部结构

═══════════════════════════════════════════════════════════════════════════════
  使用
═══════════════════════════════════════════════════════════════════════════════

  python prepare/env_check.py
  python prepare/env_check.py --json  > env_report.json  # JSON 输出
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT = Path(__file__).resolve().parent.parent.parent
_AGENT = _PROJECT / "triton_agent_optimizer"


def check(label: str, condition: bool, detail: str = "") -> dict:
    return {"label": label, "status": "PASS" if condition else "FAIL",
            "detail": detail}


def run(cmd: list, timeout: int = 10) -> Tuple[bool, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout.strip()[:200]
    except Exception as e:
        return False, str(e)[:100]


# ═══════════════════════════════════════════════════════════════════════════════
#  检查函数
# ═══════════════════════════════════════════════════════════════════════════════

def check_project_structure() -> List[dict]:
    """检查项目关键目录/文件。"""
    paths = [
        ("emulators/common/", _PROJECT / "emulators" / "common" / "__init__.py"),
        ("cost_emulator/simulator.py", _PROJECT / "costModel" / "cost_emulator" / "simulator.py"),
        ("memory/ 模块", _PROJECT / "memory" / "__init__.py"),
        ("triton_agent_optimizer/", _AGENT / "config.py"),
        ("config.py", _AGENT / "config.py"),
        ("analyzers/", _AGENT / "analyzers" / "__init__.py"),
        ("agents/", _AGENT / "agents" / "__init__.py"),
        ("execution/", _AGENT / "execution" / "__init__.py"),
        ("feedback/", _AGENT / "feedback" / "__init__.py"),
        ("memory/ (agent)", _AGENT / "memory" / "__init__.py"),
        ("playbooks/", _AGENT / "docx" / "playbook_tier1_algorithm.md"),
        ("prepare/", _AGENT / "prepare" / "env_check.py"),
    ]
    results = []
    for name, p in paths:
        exists = p.exists()
        results.append(check(f"Project: {name}", exists,
                             str(p) if exists else f"MISSING: {p}"))
    return results


def check_python() -> List[dict]:
    """检查 Python 环境。"""
    results = []
    results.append(check("Python version", True,
                         f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"))

    # 必要包
    for pkg in ["numpy", "json", "pathlib", "subprocess", "importlib"]:
        try:
            __import__(pkg.replace(".", "").replace("-", "_"))
            results.append(check(f"Package: {pkg}", True))
        except ImportError:
            results.append(check(f"Package: {pkg}", False, "not installed"))

    # 可选包
    for pkg, purpose in [("matplotlib", "trajectory chart"),
                          ("networkx", "simulator --nx mode")]:
        try:
            __import__(pkg)
            results.append(check(f"Package(opt): {pkg}", True, purpose))
        except ImportError:
            results.append(check(f"Package(opt): {pkg}", True,
                                 f"not installed ({purpose}) — optional"))
    return results


def check_emulator() -> List[dict]:
    """检查 CPU Emulator 是否可用。"""
    results = []
    emu_path = _PROJECT / "emulators" / "common" / "__init__.py"
    if not emu_path.exists():
        results.append(check("Emulator: common", False, str(emu_path)))
        return results
    try:
        sys.path.insert(0, str(_PROJECT / "emulators"))
        from common import tl
        results.append(check("Emulator: tl class", True))
        results.append(check("Emulator: tl.load", hasattr(tl, "load")))
        results.append(check("Emulator: tl.store", hasattr(tl, "store")))
        results.append(check("Emulator: tl.program_id", hasattr(tl, "program_id")))
    except Exception as e:
        results.append(check("Emulator: import", False, str(e)[:100]))
    return results


def check_simulator() -> List[dict]:
    """检查 Cost Simulator。"""
    sim = _PROJECT / "costModel" / "cost_emulator" / "simulator.py"
    results = []
    if not sim.exists():
        results.append(check("Simulator: path", False, str(sim)))
        return results
    results.append(check("Simulator: path", True, str(sim)))
    ok, out = run([sys.executable, str(sim), "--llm",
                    "alloc(gm_1, 1KB) alloc(ub_1, 1KB) gm_to_ub(ub_1, gm_1)"])
    results.append(check("Simulator: --llm mode", ok, out[:100] if ok else out[:100]))
    return results


def check_ascend() -> List[dict]:
    """检查 CANN/Ascend 环境 (仅 910B3 服务器)。"""
    results = []
    ascend = os.environ.get("ASCEND_HOME", os.environ.get("ASCEND_HOME_PATH", ""))
    toolkit = os.environ.get("ASCEND_TOOLKIT_HOME", "")

    # 尝试标准路径
    standard_paths = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/cann",
        "/usr/local/Ascend",
    ]
    found_path = ascend or toolkit
    if not found_path:
        for sp in standard_paths:
            if Path(sp).exists():
                found_path = sp
                break

    is_ascend = bool(found_path)
    results.append(check("Ascend: CANN installed", is_ascend,
                         found_path or "not found (this is normal for local dev)"))

    if not is_ascend:
        results.append(check("Ascend: msprof", True, "skipped (local dev — no NPU needed)"))
        results.append(check("Ascend: npu-smi", True, "skipped (local dev)"))
        results.append(check("Ascend: set_env.sh", True, "skipped (local dev)"))
        return results

    # msprof
    msprof_paths = [
        shutil.which("msprof"),
        Path(found_path) / "tools" / "profiler" / "bin" / "msprof",
        Path(found_path) / "cann" / "tools" / "profiler" / "bin" / "msprof",
    ]
    msprof_found = any(p and Path(str(p)).exists() for p in msprof_paths if p)
    results.append(check("Ascend: msprof", msprof_found,
                         next((str(p) for p in msprof_paths if p and Path(str(p)).exists()), "not found")))

    # npu-smi
    npu = shutil.which("npu-smi") or (Path(found_path) / "bin" / "npu-smi")
    npu_ok = bool(npu and Path(str(npu)).exists())
    results.append(check("Ascend: npu-smi", npu_ok, str(npu) if npu_ok else "not found"))

    # 编译器
    compiler = (Path(found_path) / "compiler" / "bin")
    comp_ok = compiler.exists()
    results.append(check("Ascend: compiler", comp_ok, str(compiler) if comp_ok else "not found"))

    # set_env.sh
    set_env_paths = [
        Path(found_path) / "set_env.sh",
        Path(found_path) / "ascend-toolkit" / "set_env.sh",
        Path(found_path) / "cann" / "set_env.sh",
    ]
    set_env = next((p for p in set_env_paths if p.exists()), None)
    results.append(check("Ascend: set_env.sh", set_env is not None,
                         str(set_env) if set_env else "not found"))

    # NPU 设备
    if npu_ok:
        ok, out = run([str(npu), "info"], timeout=15)
        has_910b = "910B" in out
        results.append(check("Ascend: NPU 910B3", has_910b, out[:100] if ok else "npu-smi failed"))

    return results


def check_triton_env() -> List[dict]:
    """检查 Triton 环境 (仅 910B3 服务器)。"""
    results = []
    try:
        import triton
        results.append(check("Triton: installed", True, f"version={getattr(triton, '__version__', '?')}"))
    except ImportError:
        results.append(check("Triton: installed", True, "not installed (normal for local dev)"))
        return results

    try:
        import torch
        results.append(check("PyTorch: installed", True, torch.__version__))
        if hasattr(torch, "npu") and torch.npu.is_available():
            results.append(check("torch_npu: available", True))
        else:
            results.append(check("torch_npu: available", True, "not detected (check CANN env)"))
    except ImportError:
        results.append(check("PyTorch: installed", True, "not installed (normal for local dev)"))

    return results


def check_agent_optimizer() -> List[dict]:
    """检查 Agent Optimizer 内部结构。"""
    results = []
    for name, subpath in [
        ("playbooks (6 files)", "docx"),
        ("outputs/", "outputs"),
    ]:
        p = _AGENT / subpath
        results.append(check(f"Agent: {name}", p.exists(), str(p)))

    # 检查经验库
    exp_dir = _AGENT / "memory" / "experiences"
    results.append(check("Agent: memory/experiences/", exp_dir.exists(), str(exp_dir)))
    tier_files = list(exp_dir.glob("tier*.json")) if exp_dir.exists() else []
    results.append(check("Agent: experience files", len(tier_files) == 6,
                         f"{len(tier_files)}/6 tier files" if tier_files else "MISSING"))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════════

def main(json_output: bool = False):
    all_checks = []
    sections = [
        ("PROJECT STRUCTURE", check_project_structure),
        ("PYTHON ENVIRONMENT", check_python),
        ("CPU EMULATOR", check_emulator),
        ("COST SIMULATOR", check_simulator),
        ("ASCEND/CANN (910B3)", check_ascend),
        ("TRITON (910B3)", check_triton_env),
        ("AGENT OPTIMIZER", check_agent_optimizer),
    ]

    if json_output:
        report = {}
        for title, fn in sections:
            report[title] = fn()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    # 文本输出
    total_pass = 0
    total_fail = 0
    print("=" * 65)
    print("  Triton Agent Optimizer — Environment Check")
    print(f"  Project: {_PROJECT}")
    print(f"  Python:  {sys.version}")
    print(f"  Platform: {sys.platform}")
    print("=" * 65)

    for title, fn in sections:
        checks = fn()
        all_checks.extend(checks)
        passes = sum(1 for c in checks if c["status"] == "PASS")
        fails = sum(1 for c in checks if c["status"] == "FAIL")
        total_pass += passes
        total_fail += fails

        print(f"\n── {title} ({passes}/{len(checks)} OK) ──")
        for c in checks:
            icon = "[OK]" if c["status"] == "PASS" else "[FAIL]"
            detail = f" — {c['detail'][:120]}" if c.get("detail") else ""
            print(f"  {icon} {c['label']}{detail}")

    print(f"\n{'=' * 65}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed, "
          f"{total_pass + total_fail} checks")
    if total_fail == 0:
        print("  ALL CHECKS PASSED — environment ready.")
    else:
        print(f"  {total_fail} issue(s) found. See details above.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    json_mode = "--json" in sys.argv
    main(json_output=json_mode)
