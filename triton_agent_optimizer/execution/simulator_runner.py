#!/usr/bin/env python3
"""Cost Simulator 运行器 — subprocess 调用 cost_emulator/simulator.py。Stage 2 验证。"""

from __future__ import annotations
import json, subprocess, sys, os, tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
_SIMULATOR = _PROJECT_DIR / "costModel" / "cost_emulator" / "simulator.py"


@dataclass
class SimulatorCompare:
    """优化前后对比。"""
    total_ns_before: float; total_ns_after: float
    estimated_speedup: float
    bottleneck_before: str; bottleneck_after: str
    engine_util_before: dict; engine_util_after: dict


class SimulatorRunner:
    """调用 cost_emulator/simulator.py --llm --critical-path 进行性能预估。"""

    def __init__(self, python_exe: Optional[str] = None, timeout: int = 30):
        self.simulator = _SIMULATOR
        self.python = python_exe or sys.executable
        self.timeout = timeout

    def run(self, dsl_program: str) -> dict:
        """运行 simulator --llm --critical-path, 返回解析后的 dict。"""
        raw = self._invoke(["--llm", "--critical-path", dsl_program])
        return self._parse_summary(raw)

    def compare(self, dsl_before: str, dsl_after: str) -> SimulatorCompare:
        """优化前后对比。"""
        before = self.run(dsl_before)
        after = self.run(dsl_after)
        speedup = before["total_ns"] / after["total_ns"] if after["total_ns"] > 0 else 1.0
        return SimulatorCompare(
            total_ns_before=before["total_ns"], total_ns_after=after["total_ns"],
            estimated_speedup=round(speedup, 4),
            bottleneck_before=before.get("bottleneck", "?"),
            bottleneck_after=after.get("bottleneck", "?"),
            engine_util_before=before.get("engine_util", {}),
            engine_util_after=after.get("engine_util", {}),
        )

    def _invoke(self, args: list) -> str:
        cmd = [self.python, str(self.simulator)] + args
        r = subprocess.run(cmd, capture_output=True, timeout=self.timeout,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"Simulator failed: {r.stderr[:500]}")
        return r.stdout

    def _parse_summary(self, raw: str) -> dict:
        import re
        total_ns = 0.0; num_ops = 0
        m = re.search(r'total_ns:\s*([\d.]+)', raw)
        if m: total_ns = float(m.group(1))
        m = re.search(r'num_ops:\s*(\d+)', raw)
        if m: num_ops = int(m.group(1))
        # 找瓶颈 op (time_ratio 最大)
        max_ratio = 0.0; bottleneck = ""
        for m in re.finditer(r'op(\d+):.*?time_ratio=([\d.]+)%', raw):
            ratio = float(m.group(2))
            if ratio > max_ratio: max_ratio = ratio; bottleneck = f"op{m.group(1)}"
        engine_util = {}
        for m in re.finditer(r'([\w→]+):\s*busy=.*?utilization=([\d.]+)%', raw):
            engine_util[m.group(1)] = float(m.group(2)) / 100
        return {"total_ns": total_ns, "num_ops": num_ops,
                "bottleneck": bottleneck, "engine_util": engine_util}
