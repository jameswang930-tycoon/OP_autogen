#!/usr/bin/env python3
"""910B3 真机运行器 — benchmark + msprof 采集。仅在 910B3 环境可用。"""

from __future__ import annotations
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareResult:
    success: bool
    latency_ms: float = 0.0
    throughput_gb_s: float = 0.0
    speedup_vs_baseline: float = 1.0
    msprof_dir: str = ""
    hivmir_text: str = ""
    error_message: str = ""


class HardwareRunner:
    """910B3 真机运行器。本地环境不可用。"""

    def __init__(self, warmup: int = 30, repeat: int = 200):
        self.msprof_bin = shutil.which("msprof")
        self.available = self.msprof_bin is not None
        self.warmup = warmup; self.repeat = repeat

    def benchmark(self, binary_path: Path, baseline_latency_ms: float = 0.0) -> HardwareResult:
        if not self.available:
            return HardwareResult(success=False,
                error_message="msprof not found. Run on 910B3 server with CANN installed.")
        # TODO: 真机 benchmark (参考 perf_test/910B3/vecadd/bench_910b3_paths.py)
        return HardwareResult(success=False, error_message="Not yet implemented on 910B3")
