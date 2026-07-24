#!/usr/bin/env python3
"""
Ascend 编译器接口 — 编译 + HIVMIR 提取。

═══════════════════════════════════════════════════════════════════════════════
  完整流程 (910B3 服务器)
═══════════════════════════════════════════════════════════════════════════════

  Triton kernel (.py)
    │
    ▼
  ① 编译 (bishengir-compile / ascendc)
     参数: --mlir-print-ir-after-all --run-mode=sim
     输出: .om 二进制 + IR dump
    │
    ▼
  ② 提取 HIVMIR (解析 IR dump)
     查找 hivm. 前缀的操作 → 保存为 .mlir
    │
    ▼
  ③ 返回 CompileResult {binary_path, hivmir_text}

  这个文件整合了原 fusion_pipeline/extract_hivmir_from_compiler.py 的功能

═══════════════════════════════════════════════════════════════════════════════
  本地开发 (无 NPU)
═══════════════════════════════════════════════════════════════════════════════

  返回 CompileResult(available=False), 后续链路会跳过硬件步骤。
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CompileResult:
    """编译结果。"""
    success: bool
    available: bool = True          # False = 本地环境, 编译器不可用
    binary_path: str = ""
    hivmir_text: str = ""           # HIVMIR .mlir 文本
    hivmir_path: str = ""           # HIVMIR .mlir 文件路径
    error_message: str = ""


class CompilerInterface:
    """Ascend 编译器接口 — 编译 Triton kernel + 提取 HIVMIR。

    Usage:
        compiler = CompilerInterface()
        result = compiler.compile(kernel_code, output_dir)
        if result.success:
            # result.binary_path → .om 二进制
            # result.hivmir_text → HIVMIR .mlir 文本
            hivmir_analyzer.parse(result.hivmir_text)
    """

    # 编译器搜索路径
    COMPILER_CANDIDATES = [
        "bishengir-compile",
        "ascendc",
        "bisheng",
    ]

    # HIVM dialect 特征 (从 IR dump 中识别)
    HIVM_PATTERNS = [
        r'hivm\.(?:alloc|gm_to_ub|ub_to_gm|gm_to_l1|l1_to_l0|l0_to_gm|'
        r'vadd|vsub|vmul|vdiv|vmax|vmin|vexp|vlog|matrixmul|hir\.\w+)',
    ]

    def __init__(self):
        # 查找编译器
        self.compiler_bin = self._find_compiler()
        self.available = self.compiler_bin is not None

        # 查找 ASCEND 环境
        self.ascend_home = self._find_ascend_home()
        self.set_env_script = self._find_set_env()

    # ═══════════════════════════════════════════════════════════════════════════
    #  主入口: 编译 + HIVMIR 提取
    # ═══════════════════════════════════════════════════════════════════════════

    def compile(
        self,
        kernel_code: str,
        output_dir: Path,
        kernel_name: str = "kernel",
    ) -> CompileResult:
        """编译 Triton kernel + 提取 HIVMIR。

        Args:
            kernel_code: Triton kernel 源码
            output_dir: 输出目录 (如 roundN/compiler_output/)
            kernel_name: kernel 名称 (用于生成文件名)

        Returns:
            CompileResult
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 本地环境 → 返回不可用
        if not self.available:
            return CompileResult(
                success=False, available=False,
                error_message=(
                    "Ascend compiler not found. Run on 910B3 server.\n"
                    f"Searched: {self.COMPILER_CANDIDATES}\n"
                    f"ASCEND_HOME: {self.ascend_home or 'not set'}"
                ),
            )

        # 写入 kernel 文件
        kernel_file = output_dir / f"{kernel_name}.py"
        kernel_file.write_text(kernel_code, encoding="utf-8")

        # ── Step 1: 编译 (带 MLIR 插桩) ──
        print(f"  [compiler] Compiling: {kernel_file}")
        ir_dump_file = output_dir / f"{kernel_name}_ir_dump.txt"
        binary_file = output_dir / f"{kernel_name}.om"

        compile_ok, compile_output = self._run_compile(
            kernel_file, ir_dump_file, binary_file)

        if not compile_ok:
            return CompileResult(
                success=False,
                error_message=f"Compilation failed:\n{compile_output[:500]}",
            )

        # ── Step 2: 提取 HIVMIR ──
        print(f"  [compiler] Extracting HIVMIR from IR dump...")
        hivmir_text = self._extract_hivmir(ir_dump_file)

        if not hivmir_text:
            # 尝试从 stdout 提取
            hivmir_text = self._extract_hivmir_from_text(compile_output)

        if not hivmir_text:
            return CompileResult(
                success=False,
                error_message="HIVMIR extraction failed — no hivm.* ops found in IR dump",
            )

        # 保存 HIVMIR
        hivmir_file = output_dir / f"{kernel_name}_hivmir.mlir"
        hivmir_file.write_text(hivmir_text, encoding="utf-8")

        print(f"  [compiler] HIVMIR saved → {hivmir_file} "
              f"({len(hivmir_text)} chars)")

        return CompileResult(
            success=True,
            binary_path=str(binary_file),
            hivmir_text=hivmir_text,
            hivmir_path=str(hivmir_file),
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  编译执行
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_compile(
        self, kernel_file: Path, ir_dump: Path, binary: Path,
    ) -> tuple[bool, str]:
        """运行编译器, 捕获 IR dump。"""
        # 构建命令
        cmd = [
            self.compiler_bin,
            str(kernel_file),
            "-o", str(binary),
            "--run-mode=sim",                # 仿真模式 (不需要真实 NPU)
            "--mlir-print-ir-after-all",     # 每个 pass 后打印 IR
            f"--mlir-print-ir-tree-dir={ir_dump.parent}",
        ]

        env = {}
        if self.set_env_script:
            # source 环境脚本并在同一 shell 中运行编译
            # 用 bash -c 包装
            cmd = [
                "bash", "-c",
                f"source {self.set_env_script} && "
                f"{' '.join(str(c) for c in cmd)} "
                f"> {ir_dump} 2>&1"
            ]

        try:
            result = subprocess.run(
                cmd if isinstance(cmd[0], str) else cmd,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            output = result.stdout + "\n" + result.stderr

            # 同时读 IR dump 文件 (如果编译器写到文件)
            if ir_dump.exists():
                output += "\n" + ir_dump.read_text(encoding="utf-8",
                                                    errors="replace")

            return result.returncode == 0, output[:5000]

        except subprocess.TimeoutExpired:
            return False, "Compilation timed out (120s)"
        except Exception as e:
            return False, str(e)[:500]

    # ═══════════════════════════════════════════════════════════════════════════
    #  HIVMIR 提取
    # ═══════════════════════════════════════════════════════════════════════════

    def _extract_hivmir(self, ir_dump_file: Path) -> str:
        """从 IR dump 文件提取 HIVM dialect 部分。"""
        if not ir_dump_file.exists():
            return ""

        content = ir_dump_file.read_text(encoding="utf-8", errors="replace")

        return self._extract_hivmir_from_text(content)

    def _extract_hivmir_from_text(self, text: str) -> str:
        """从文本中提取 HIVMIR 内容。

        识别策略:
          1. 找连续的 hivm.* 操作块
          2. 提取 memref alloc + hivm ops
          3. 格式化为标准 HIVMIR 文本
        """
        # 按 pass 分割 IR dump
        passes = text.split('// -----// IR Dump')

        # 找包含 hivm 的 pass
        hivm_blocks = []
        for p in passes:
            if any(re.search(pat, p) for pat in self.HIVM_PATTERNS):
                hivm_blocks.append(p)

        if not hivm_blocks:
            # 直接在全文找 hivm.* 行
            lines = text.split('\n')
            hivm_lines = [
                l.strip() for l in lines
                if any(re.search(pat, l) for pat in self.HIVM_PATTERNS)
                or 'alloc' in l.lower() and 'memref' in l.lower()
            ]
            if hivm_lines:
                return '\n'.join(hivm_lines)

            return ""

        # 取最后一个 pass (最终 IR) 的 HIVM 部分
        last = hivm_blocks[-1]

        # 提取 hivm.* 和 alloc 行
        lines = last.split('\n')
        hivm_lines = []
        for line in lines:
            stripped = line.strip()
            # HIVM ops
            if any(re.search(pat, stripped) for pat in self.HIVM_PATTERNS):
                hivm_lines.append(stripped)
            # alloc 声明
            elif ('alloc' in stripped.lower()
                  and 'memref' in stripped.lower()):
                hivm_lines.append(stripped)

        if not hivm_lines:
            return ""

        # 格式化: 先 alloc, 再 ops
        alloc_lines = [l for l in hivm_lines if 'alloc' in l.lower()]
        op_lines = [l for l in hivm_lines if 'alloc' not in l.lower()]

        return '\n'.join(alloc_lines + op_lines)

    # ═══════════════════════════════════════════════════════════════════════════
    #  环境检测
    # ═══════════════════════════════════════════════════════════════════════════

    def _find_compiler(self) -> Optional[str]:
        """查找 Ascend 编译器。"""
        # 1. 环境变量
        for env_var in ["ASCEND_COMPILER", "BISHENG_COMPILER"]:
            import os
            val = os.environ.get(env_var)
            if val and Path(val).exists():
                return val

        # 2. PATH 搜索
        for name in self.COMPILER_CANDIDATES:
            found = shutil.which(name)
            if found:
                return found

        # 3. 标准路径
        standard_paths = [
            "/usr/local/Ascend/ascend-toolkit/latest/compiler/bin",
            "/usr/local/Ascend/cann/compiler/bin",
        ]
        for sp in standard_paths:
            spath = Path(sp)
            if spath.exists():
                for name in self.COMPILER_CANDIDATES:
                    candidate = spath / name
                    if candidate.exists():
                        return str(candidate)

        return None

    def _find_ascend_home(self) -> Optional[str]:
        import os
        for v in ["ASCEND_HOME", "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"]:
            val = os.environ.get(v)
            if val:
                return val
        for sp in ["/usr/local/Ascend", "/usr/local/Ascend/ascend-toolkit/latest"]:
            if Path(sp).exists():
                return sp
        return None

    def _find_set_env(self) -> Optional[str]:
        candidates = []
        if self.ascend_home:
            ah = Path(self.ascend_home)
            candidates = [
                ah / "set_env.sh",
                ah / "ascend-toolkit" / "set_env.sh",
                ah / "cann" / "set_env.sh",
            ]
        for sp in [
            "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
            "/usr/local/Ascend/cann/set_env.sh",
        ]:
            candidates.append(Path(sp))
        for c in candidates:
            if c.exists():
                return str(c)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  自测
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    compiler = CompilerInterface()
    print(f"Compiler available: {compiler.available}")
    print(f"Compiler bin:      {compiler.compiler_bin}")
    print(f"ASCEND_HOME:       {compiler.ascend_home}")
    print(f"set_env.sh:        {compiler.set_env_script}")

    # 测试 HIVMIR 提取逻辑
    test_output = """
// -----// IR Dump After SomePass // -----
module {
  hivm.alloc %ub_1 : memref<128KB>
  hivm.alloc %ub_2 : memref<128KB>
  hivm.gm_to_ub %ub_1, %gm_1 : memref<128KB>
  hivm.vadd %ub_2, %ub_1, 2.0
  hivm.ub_to_gm %gm_2, %ub_2 : memref<128KB>
}
"""
    extracted = compiler._extract_hivmir_from_text(test_output)
    print(f"\nHIVMIR extraction test:")
    print(f"  Input:  {len(test_output)} chars")
    print(f"  Output: {len(extracted)} chars")
    if "vadd" in extracted:
        print("  Extraction OK — found vadd op")
    else:
        print("  Extraction FAILED")

    print("\n[Compiler] OK")


if __name__ == "__main__":
    _self_test()
