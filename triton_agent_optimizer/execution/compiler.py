#!/usr/bin/env python3
"""Ascend 编译器接口 — 编译 Triton kernel + 提取 HIVMIR。仅在 910B3 环境可用。"""

from __future__ import annotations
import shutil, subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompileResult:
    success: bool; binary_path: str = ""
    hivmir_text: str = ""; error_message: str = ""


class CompilerInterface:
    """Ascend 编译器接口。本地环境不可用, 返回明确错误。"""

    def __init__(self):
        self.ascend_compiler = shutil.which("ascendc")
        self.available = self.ascend_compiler is not None

    def compile(self, kernel_code: str, output_dir: Path) -> CompileResult:
        if not self.available:
            return CompileResult(success=False,
                error_message="Ascend compiler not found. Run on 910B3 server with CANN installed.")
        # TODO: 真实编译
        return CompileResult(success=False, error_message="Not yet implemented on 910B3")

    def extract_hivmir(self, kernel_code: str) -> str:
        if not self.available:
            return ""
        # TODO: 从编译器输出提取 HIVMIR
        return ""
