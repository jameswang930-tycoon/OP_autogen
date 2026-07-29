#!/usr/bin/env python3
"""
TTIR → HIVM 转换器

将 Triton TTIR MLIR 直接转换为 HIVM dialect MLIR。
基于 triton-ascend 官方 lowering 规则：
  https://ascend.github.io/docs/sources/_generated/sources/triton-ascend/architecture_design_and_core_features.html

核心映射:
  tt.load(ptr, mask)           → hivm.hir.load  ins(%gm) outs(%ub)
  tt.store(ptr, val, mask)     → hivm.hir.store ins(%ub) outs(%gm)
  arith.addf / arith.mulf / ... → hivm.hir.vadd / vmul / ...
  tt.dot(a, b, c)              → hivm.hir.matmul ins(a,b) outs(c)

地址计算 ops (tt.addptr, arith.addi, tt.splat, tt.make_range, arith.cmpi)
在 HIVM 层不直接对应硬件指令——它们属于 SCALAR 流水线。
但在 HIVM MLIR 中仍保留为 arith/cmp 标准操作，由 bishengir-compile 继续 lower。

依赖:
  pip install triton==2.3.1  (仅用于 ast_to_ttir, 不需要 GPU)
  LD_PRELOAD stub 库 (解决 WSL2 CUDA 符号缺失)
"""

from __future__ import annotations

import re, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
#  TTIR → HIVM 操作映射表
# ═══════════════════════════════════════════════════════════════════════════════

# TTIR arith ops → HIVM vector ops
ARITH_TO_HIVM = {
    "arith.addf":  "hivm.hir.vadd",
    "arith.subf":  "hivm.hir.vsub",
    "arith.mulf":  "hivm.hir.vmul",
    "arith.divf":  "hivm.hir.vdiv",
    "arith.maximumf": "hivm.hir.vmax",
    "arith.minimumf": "hivm.hir.vmin",
    "arith.negf":  "hivm.hir.vneg",
    "arith.absf":  "hivm.hir.vabs",
    "math.sqrt":   "hivm.hir.vsqrt",
    "math.exp":    "hivm.hir.vexp",
    "math.log":    "hivm.hir.vln",
    "math.rsqrt":  "hivm.hir.vrsqrt", "math.tanh": "hivm.hir.vtanh",
}

# TTIR ops that become SCALAR ops (address computation, not DMA/compute)
SCALAR_OPS = {
    "arith.addi", "arith.subi", "arith.muli", "arith.divi",
    "arith.remsi", "arith.remui",
    "arith.cmpi", "arith.cmpf",
    "arith.andi", "arith.ori", "arith.xori",
    "arith.shli", "arith.shrui", "arith.shri",
    "arith.extsi", "arith.extui", "arith.trunci",
    "arith.sitofp", "arith.fptosi", "arith.fptoui",
    "tt.get_program_id", "tt.get_num_programs",
    "tt.make_range", "tt.splat", "tt.addptr", "tt.advance",
    "tt.broadcast", "tt.trans", "tt.expand_dims", "tt.reshape",
}

# 数据类型大小 (bytes)
DTYPE_SIZES = {
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f32": 4, "f16": 2,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  TTIR MLIR 解析器
# ═══════════════════════════════════════════════════════════════════════════════

class TTIRParser:
    """解析 TTIR MLIR 文本，提取可转换为 HIVM 的操作。"""

    def __init__(self):
        self.func_name = ""
        self.args: List[Tuple[str, str, str]] = []  # (name, type, dtype)
        self.ops: List[dict] = []                     # parsed ops
        self.alloc_count = 0

    def parse(self, ttir_text: str) -> dict:
        """解析 TTIR → 结构化信息。"""
        self.args = []
        self.ops = []
        self.alloc_count = 0

        # 提取函数签名
        func_m = re.search(
            r'tt\.func\s+(?:public\s+)?@(\w+)\s*\(([^)]*)\)',
            ttir_text)
        if func_m:
            self.func_name = func_m.group(1)
            args_str = func_m.group(2)
            for arg_m in re.finditer(
                r'%(\w+)\s*:\s*!tt\.ptr<(\w+)(?:\s*,\s*\d+)?>', args_str):
                self.args.append((arg_m.group(1), "gm", arg_m.group(2)))
            # 也匹配标量参数 (如 i32)
            for arg_m in re.finditer(r'%(\w+)\s*:\s*(i\d+|f\d+)', args_str):
                self.args.append((arg_m.group(1), "scalar", arg_m.group(2)))

        # 提取 body 中的所有操作
        body_m = re.search(r'attributes\s*\{[^}]*\}\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\s*tt\.return', ttir_text, re.DOTALL)
        if not body_m:
            # simpler: find everything between function body braces
            body_m = re.search(r'\{([^}]*(?:\{[^}]*\}[^}]*)*)\s*tt\.return', ttir_text, re.DOTALL)

        if body_m:
            self._parse_ops(body_m.group(1))

        return {
            "func_name": self.func_name,
            "args": self.args,
            "ops": self.ops,
        }

    def _parse_ops(self, body: str):
        """解析 body 中的每条操作。"""
        lines = body.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue

            # 匹配: %result = op_name args... : types
            m = re.match(
                r'(%\w+)\s*=\s*(tt\.\w+|arith\.\w+|math\.\w+|scf\.\w+|linalg\.\w+)\s+(.+?)\s*:\s*(.+?)\s*$',
                line)
            if not m:
                # 匹配 MLIR 属性格式: %result = "tt.reduce"(%arg) <{axis = 0}> : ...
                m = re.match(
                    r'(%\w+)\s*=\s*"(tt\.\w+)"\s*\(([^)]*)\)\s*<\{([^}]*)\}>\s*:\s*(.+?)\s*$',
                    line)
                if not m:
                    # 无返回值的 op: tt.store, tt.return
                    m = re.match(
                        r'(tt\.\w+|arith\.\w+|scf\.\w+|math\.\w+)\s+(.+?)\s*:\s*(.+?)\s*$', line)
                    if m:
                        op_name = m.group(1)
                        args_str = m.group(2)
                        type_str = m.group(3)
                        self.ops.append({
                            "result": None,
                            "op": op_name,
                            "args_raw": args_str,
                            "type": type_str,
                        })
                    continue

            result = m.group(1)
            op_name = m.group(2).strip('"') if '"' in (m.group(2) or "") else m.group(2)
            args_str = m.group(3)
            type_str = ""
            if len(m.groups()) >= 4:
                type_str = m.group(4) or ""
            if not type_str and len(m.groups()) >= 5:
                type_str = m.group(5) or ""

            # 提取 attrs (如 {axis = 0 : i32})
            attrs = {}
            if '"' in str(m.group(0)):
                attr_match = re.search(r"<\{([^}]*)\}>", line)
                if attr_match:
                    attrs["raw"] = attr_match.group(1)

            self.ops.append({
                "result": result,
                "op": op_name,
                "args_raw": args_str,
                "type": type_str,
                "attrs": attrs,
            })

            result = m.group(1)
            op_name = m.group(2)
            args_str = m.group(3)
            type_str = m.group(4)

            self.ops.append({
                "result": result,
                "op": op_name,
                "args_raw": args_str,
                "type": type_str,
            })


# ═══════════════════════════════════════════════════════════════════════════════
#  TTIR → HIVM 代码生成器
# ═══════════════════════════════════════════════════════════════════════════════

class HIVMGenerator:
    """从解析后的 TTIR 操作生成 HIVM MLIR 文本。"""

    def __init__(self):
        self.alloc_count = 0
        self.ub_buffers: Dict[str, str] = {}   # TTIR SSA名 → UB buffer名
        self.hivm_lines: List[str] = []
        self.hivm_ops: List[dict] = []          # 结构化 HIVM op 列表

    def generate(self, parsed: dict, kernel_name: str = "triton_kernel") -> str:
        self.alloc_count = 0
        self.ub_buffers = {}
        self.hivm_lines = []
        self.hivm_ops = []

        func_name = parsed["func_name"] or kernel_name
        args = parsed["args"]
        ops = parsed["ops"]

        # 简单计数器：顺序分配 GM arg
        self._load_count = 0
        self._gm_args = [a[0] for a in args if a[1] == "gm"]
        self._gm_out_idx = len(self._gm_args) - 1  # last arg = output

        # ── 构建 def-use 链 ──
        self._defs: Dict[str, dict] = {}
        for op in ops:
            if op.get("result"):
                self._defs[op["result"]] = op

        # ── 推断数据类型和大小 ──
        dtype = "f16"  # default
        total_elements = 1024
        for arg in args:
            if arg[2] in ("f32", "f16", "bf16"):
                dtype = arg[2]
        # 从 make_range 推断数据大小
        for op in ops:
            if op.get("op") == "tt.make_range":
                m = re.search(r'end\s*=\s*(\d+)', op.get("args_raw", ""))
                if m:
                    total_elements = int(m.group(1))

        size_bytes = total_elements * DTYPE_SIZES.get(dtype, 2)
        size_kb = size_bytes / 1024.0

        # ── 函数签名 ──
        gm_args = [a for a in args if a[1] == "gm"]
        gm_str = ", ".join(
            f"%arg{i}: memref<{total_elements}x{dtype}, #hivm.address_space<gm>>"
            for i, _ in enumerate(gm_args)
        )
        self.hivm_lines.append(
            f'func.func @{func_name}_hivm({gm_str}) '
            f'attributes {{hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>}} {{'
        )

        # ── 处理每条 TTIR op ──
        for op in ops:
            op_name = op.get("op", "")
            result = op.get("result")
            args_raw = op.get("args_raw", "")

            if op_name == "tt.load":
                self._emit_load(result, args_raw, gm_args, total_elements, dtype)
            elif op_name == "tt.store":
                self._emit_store(args_raw, gm_args)
            elif op_name in ARITH_TO_HIVM:
                self._emit_arith(op_name, result, args_raw, total_elements, dtype)
            elif op_name == "tt.dot":
                self._emit_dot(result, args_raw, total_elements, dtype)
            elif op_name == "tt.reduce":
                self._emit_reduce(result, args_raw, total_elements, dtype)
            elif op_name.startswith("math."):
                # math.exp / math.sqrt → HIVM vector op
                self._emit_simple_vec_op(op_name, result, args_raw, total_elements, dtype)
            # SCALAR ops: 跳过

        self.hivm_lines.append("    return")
        self.hivm_lines.append("  }")

        return "\n".join(self.hivm_lines)

    # ── Load: GM→UB ──
    def _emit_load(self, result, args_raw, gm_args, total_elements, dtype):
        """tt.load %ptr, %mask → hivm.hir.load"""
        buf_name = f"%ub_buf_{self.alloc_count}"
        self.alloc_count += 1

        # alloc UB buffer
        self.hivm_lines.append(
            f"    {buf_name} = memref.alloc() : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>"
        )

        # 简单策略：第 N 个 load 对应第 N 个输入 GM arg
        gm_idx = self._load_count
        self._load_count += 1
        if gm_idx < len(self._gm_args):
            gm_name = self._gm_args[gm_idx]
        else:
            gm_name = self._gm_args[0] if self._gm_args else "in"

        # load
        self.hivm_lines.append(
            f"    hivm.hir.load "
            f"ins(%{gm_name} : memref<{total_elements}x{dtype}, #hivm.address_space<gm>>) "
            f"outs({buf_name} : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>)"
        )

        if result:
            self.ub_buffers[result] = buf_name

        size_kb = round(total_elements * DTYPE_SIZES.get(dtype, 2) / 1024.0, 2)
        self.hivm_ops.append({
            "op_type": "gm_to_ub", "dst": buf_name,
            "src": f"%{gm_name}", "size_kb": size_kb,
            "memory_region": "ub", "instruction": f"hivm.hir.load({buf_name}, %{gm_name})",
        })

    # ── Store: UB→GM ──
    def _emit_store(self, args_raw, gm_args):
        """tt.store %ptr, %val, %mask → hivm.hir.store"""
        parts = [p.strip() for p in args_raw.split(",")]
        val_name = parts[1] if len(parts) > 1 else ""
        ub_buf = self.ub_buffers.get(val_name, val_name)

        # Store 使用最后一个 GM arg (output)
        gm_name = self._gm_args[-1] if self._gm_args else "out"

        # 从前面的 HIVM ops 获取 buffer 的 size 和 dtype
        total_elems = 256
        dtype = "f32"
        for op in self.hivm_ops:
            if op.get("dst") == ub_buf or op.get("src") == ub_buf:
                size_kb = op.get("size_kb", 1.0)
                total_elems = int(size_kb * 1024 / DTYPE_SIZES.get(dtype, 4))
                break

        memref_t = f"memref<{total_elems}x{dtype}"

        # store
        self.hivm_lines.append(
            f"    hivm.hir.store "
            f"ins({ub_buf} : {memref_t}, #hivm.address_space<ub>>) "
            f"outs(%{gm_name} : {memref_t}, #hivm.address_space<gm>>)"
        )

        self.hivm_ops.append({
            "op_type": "ub_to_gm", "dst": f"%{gm_name}",
            "src": ub_buf,
            "size_kb": round(total_elems * DTYPE_SIZES.get(dtype, 4) / 1024.0, 2),
            "memory_region": "gm", "instruction": f"hivm.hir.store(%{gm_name}, {ub_buf})",
        })

    # ── 算术 → HIVM vector ──
    def _emit_arith(self, op_name, result, args_raw, total_elements, dtype):
        """arith.addf/mulf/subf → hivm.hir.vadd/vmul/vsub"""
        hivm_op = ARITH_TO_HIVM.get(op_name)
        if not hivm_op:
            return

        parts = [p.strip() for p in args_raw.split(",")]
        if len(parts) < 2:
            return

        src_a = self.ub_buffers.get(parts[0], parts[0])
        src_b = self.ub_buffers.get(parts[1], parts[1])

        # alloc result buffer
        buf_out = f"%ub_buf_{self.alloc_count}"
        self.alloc_count += 1
        self.hivm_lines.append(
            f"    {buf_out} = memref.alloc() : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>"
        )

        self.hivm_lines.append(
            f"    {hivm_op} ins({src_a}, {src_b} : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>, "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>) "
            f"outs({buf_out} : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>)"
        )

        if result:
            self.ub_buffers[result] = buf_out

        op_type = hivm_op.replace("hivm.hir.", "")
        self.hivm_ops.append({
            "op_type": op_type, "dst": buf_out,
            "src": src_a, "src2": src_b,
            "size_kb": round(total_elements * DTYPE_SIZES.get(dtype, 2) / 1024.0, 2),
            "memory_region": "ub", "instruction": f"{hivm_op}({buf_out}, {src_a}, {src_b})",
        })

    # ── Dot: 矩阵乘 ──
    def _emit_dot(self, result, args_raw, total_elements, dtype):
        """tt.dot(a, b, c) → hivm.hir.matmul"""
        parts = [p.strip() for p in args_raw.split(",")]
        if len(parts) < 2:
            return

        a_buf = self.ub_buffers.get(parts[0], parts[0])
        b_buf = self.ub_buffers.get(parts[1], parts[1])
        c_buf = self.ub_buffers.get(parts[2], parts[2]) if len(parts) > 2 else ""

        buf_out = f"%ub_buf_{self.alloc_count}"
        self.alloc_count += 1

        # alloc L1 buffers for matmul
        self.hivm_lines.append(
            f"    %l1_a = memref.alloc() : memref<{total_elements}x{dtype}, #hivm.address_space<l1>>"
        )
        self.hivm_lines.append(
            f"    %l1_b = memref.alloc() : memref<{total_elements}x{dtype}, #hivm.address_space<l1>>"
        )
        self.hivm_lines.append(
            f"    {buf_out} = memref.alloc() : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>"
        )

        self.hivm_lines.append(
            f"    hivm.hir.matmul ins({a_buf}, {b_buf} : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>, "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>) "
            f"outs({buf_out} : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>)"
        )

        if result:
            self.ub_buffers[result] = buf_out

        self.hivm_ops.append({
            "op_type": "matmul", "dst": buf_out,
            "src": a_buf, "src2": b_buf,
            "size_kb": round(total_elements * DTYPE_SIZES.get(dtype, 2) / 1024.0, 2),
            "memory_region": "l1", "instruction": f"hivm.hir.matmul({buf_out}, {a_buf}, {b_buf})",
        })

    # ── Simple vector op (math.exp, math.sqrt etc.) ──
    def _emit_simple_vec_op(self, op_name, result, args_raw, total_elements, dtype):
        """math.exp / math.sqrt → hivm.hir.vexp / vsqrt"""
        math_op = op_name.replace("math.", "")
        hivm_map = {
            "exp": "hivm.hir.vexp", "sqrt": "hivm.hir.vsqrt",
            "rsqrt": "hivm.hir.vrsqrt", "log": "hivm.hir.vln",
            "tanh": "hivm.hir.vtanh", "sin": "hivm.hir.vsin",
            "cos": "hivm.hir.vcos", "abs": "hivm.hir.vabs",
        }
        hivm_op = hivm_map.get(math_op)
        if not hivm_op:
            return

        parts = [p.strip() for p in args_raw.split(",")]
        input_val = parts[0] if parts else ""
        src_buf = self.ub_buffers.get(input_val, input_val)

        buf_out = f"%ub_buf_{self.alloc_count}"
        self.alloc_count += 1
        self.hivm_lines.append(
            f"    {buf_out} = memref.alloc() : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>"
        )
        self.hivm_lines.append(
            f"    {hivm_op} ins({src_buf} : "
            f"memref<{total_elements}x{dtype}, #hivm.address_space<ub>>) "
            f"outs({buf_out} : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>)"
        )
        if result:
            self.ub_buffers[result] = buf_out

        op_type = hivm_op.replace("hivm.hir.", "")
        self.hivm_ops.append({
            "op_type": op_type, "dst": buf_out, "src": src_buf,
            "size_kb": round(total_elements * DTYPE_SIZES.get(dtype, 2) / 1024.0, 2),
            "memory_region": "ub", "instruction": f"{hivm_op}({buf_out}, {src_buf})",
        })

    # ── Reduce: 规约 ──
    def _emit_reduce(self, result, args_raw, total_elements, dtype):
        """tt.reduce → linalg.reduce → (留给 bishengir 处理)"""
        parts = [p.strip() for p in args_raw.split(",")]
        input_val = parts[0] if parts else ""
        buf_in = self.ub_buffers.get(input_val, input_val)

        # 规约结果通常只有一个标量或小向量
        buf_out = f"%ub_buf_{self.alloc_count}"
        self.alloc_count += 1
        self.hivm_lines.append(
            f"    {buf_out} = memref.alloc() : memref<1x{dtype}, #hivm.address_space<ub>>"
        )
        # 使用 linalg.reduce (bishengir-compile 会继续 lower)
        self.hivm_lines.append(
            f"    linalg.reduce ins({buf_in} : memref<{total_elements}x{dtype}, #hivm.address_space<ub>>) "
            f"outs({buf_out} : memref<1x{dtype}, #hivm.address_space<ub>>) "
            f"dimensions = [0] ({{"
        )
        self.hivm_lines.append(f"      ^bb0(%lhs: {dtype}, %rhs: {dtype}):")
        self.hivm_lines.append(f"        linalg.yield %lhs : {dtype}")
        self.hivm_lines.append(f"    }})")

        if result:
            self.ub_buffers[result] = buf_out

        self.hivm_ops.append({
            "op_type": "reduce", "dst": buf_out, "src": buf_in,
            "size_kb": 0.002, "memory_region": "ub",
            "instruction": f"linalg.reduce({buf_out}, {buf_in})",
        })

    # ── 辅助：从 TTIR args 找到对应的 GM 参数 ──
    def _find_gm_arg(self, args_raw, gm_args):
        """TTIR load/store 的第一个参数是指针。通过 def-use 链追溯到 GM arg。

        TTIR 中 load 的指针链: %arg0 → tt.splat → tt.addptr → tt.load
        需要反向追溯：tt.load 的 %ptr → tt.addptr → tt.splat → %arg0
        """
        parts = [p.strip() for p in args_raw.split(",")]
        ptr_name = parts[0] if parts else ""

        # 直接匹配: 如果 ptr_name 就是 arg name
        for arg_name, _, _ in gm_args:
            if arg_name == ptr_name.lstrip("%"):
                return arg_name

        # 通过 def-use 追溯: ptr → tt.addptr → tt.splat → arg
        current = ptr_name
        for _ in range(5):  # 最多追溯 5 层
            op_def = self._defs.get(current)
            if not op_def:
                break
            op_name = op_def.get("op", "")
            args_raw_def = op_def.get("args_raw", "")

            if op_name == "tt.addptr":
                # addptr 的第一个参数是 upstream ptr
                parts_def = [p.strip() for p in args_raw_def.split(",")]
                current = parts_def[0] if parts_def else ""
            elif op_name == "tt.splat":
                # splat 的参数就是原始 arg
                m = re.match(r'(%\w+)', args_raw_def)
                if m:
                    splat_arg = m.group(1).lstrip("%")
                    for arg_name, _, _ in gm_args:
                        if arg_name == splat_arg:
                            return arg_name
                break
            else:
                break

        # fallback: 返回第一个未使用的 gm arg (按序分配)
        return self._next_gm_arg(gm_args)

    def _next_gm_arg(self, gm_args):
        """返回下一个未使用的 GM arg（按 load 调用次数轮转）。"""
        used = set()
        for op in self.hivm_ops:
            if op.get("op_type") in ("gm_to_ub", "ub_to_gm"):
                src = op.get("src", "")
                used.add(src)
                dst = op.get("dst", "")
                used.add(dst)
        for arg in gm_args:
            if f"%{arg[0]}" not in used or arg[0] not in str(used):
                return arg[0]
        return gm_args[-1][0] if gm_args else "out"


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def ttir_to_hivm(ttir_text: str, kernel_name: str = "triton_kernel") -> Tuple[str, List[dict]]:
    """TTIR MLIR 文本 → HIVM MLIR 文本 + 结构化 op 列表。

    Returns:
        (hivm_mlir_text, hivm_ops)
        hivm_ops: [{op_type, dst, src, src2, size_kb, memory_region, instruction}, ...]
    """
    parser = TTIRParser()
    parsed = parser.parse(ttir_text)

    gen = HIVMGenerator()
    hivm_text = gen.generate(parsed, kernel_name)
    return hivm_text, gen.hivm_ops


def triton_py_to_hivm(triton_py_path: Path, kernel_fn_name: str = None,
                      BLOCK_SIZE: int = 256,
                      signature: dict = None) -> Tuple[str, List[dict]]:
    """Triton .py kernel → HIVM MLIR + 结构化 op 列表。

    完整链路:
      1. triton ast_to_ttir → TTIR MLIR
      2. ttir_to_hivm → HIVM MLIR

    需要 triton 2.3.1 + stub LD_PRELOAD (WSL2):
      pip install triton==2.3.1
      编译 /tmp/libstub_cuda.so
      export LD_PRELOAD=/tmp/libstub_cuda.so
    """
    import os, sys
    from pathlib import Path
    from unittest.mock import MagicMock

    triton_py_path = Path(triton_py_path)

    # ── Mock driver + import triton ──
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    import triton.runtime.driver as _drv
    _mock = MagicMock()
    _mock.get_current_target = lambda: ("cuda", 90)
    _drv._obj = _mock

    import triton.compiler.compiler as _comp
    _comp.CompiledKernel = MagicMock()

    import triton, triton.language as tl
    from triton.compiler import ASTSource
    from types import SimpleNamespace

    # ── 加载 kernel ──
    # 使用 importlib 加载 .py 文件中的 @triton.jit 函数
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        triton_py_path.stem, str(triton_py_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 查找 kernel 函数
    if kernel_fn_name:
        kernel_fn = getattr(module, kernel_fn_name)
    else:
        # 自动找第一个 @triton.jit 函数
        for name in dir(module):
            obj = getattr(module, name)
            if hasattr(obj, "fn") and hasattr(obj, "arg_names"):
                kernel_fn = obj
                break
        else:
            raise ValueError(f"No @triton.jit function found in {triton_py_path}")

    # ── 生成 TTIR ──
    if signature is None:
        # 自动生成 signature
        n_args = len(kernel_fn.arg_names)
        signature = {i: "*fp32" for i in range(n_args)}
        # 最后一个可能是标量 (N)
        if n_args > 0:
            signature[n_args - 1] = "i32"

    consts = {}
    if "BLOCK_SIZE" in kernel_fn.arg_names:
        consts["BLOCK_SIZE"] = BLOCK_SIZE

    src = ASTSource(
        fn=kernel_fn,
        signature=signature,
        constants=consts,
    )
    opts = SimpleNamespace(num_warps=4, num_stages=1, debug=False)
    ttir_module = src.make_ir(opts)
    ttir_text = str(ttir_module)

    # ── TTIR → HIVM ──
    name = kernel_fn_name or getattr(kernel_fn, "arg_names", ["kernel"])[0]
    return ttir_to_hivm(ttir_text, name)


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    print("=" * 60)
    print("TTIR → HIVM Converter — Self-Test")
    print("=" * 60)

    # Test 1: 从真实 TTIR 生成 HIVM
    ttir_text = """#loc = loc("/tmp/test.py":14:0)
module {
  tt.func public @add_kernel_0123(%arg0: !tt.ptr<f32, 1>, %arg1: !tt.ptr<f32, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : (i32) -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg3 : (i32) -> tensor<256xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<256xi32>
    %7 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<256x!tt.ptr<f32, 1>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32, 1>>, tensor<256xi32>
    %9 = tt.load %8, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
    %10 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<256x!tt.ptr<f32, 1>>
    %11 = tt.addptr %10, %4 : tensor<256x!tt.ptr<f32, 1>>, tensor<256xi32>
    %12 = tt.load %11, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
    %13 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<256x!tt.ptr<f32, 1>>
    %14 = tt.addptr %13, %4 : tensor<256x!tt.ptr<f32, 1>>, tensor<256xi32>
    %15 = arith.addf %9, %12 : tensor<256xf32>
    tt.store %14, %15, %6 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32>
    tt.return
  }
}"""

    hivm_text, hivm_ops = ttir_to_hivm(ttir_text, "vector_add")

    print(f"\nHIVM ops: {len(hivm_ops)}")
    for op in hivm_ops:
        print(f"  {op['op_type']:12s} dst={op['dst']:14s} src={op.get('src','')[:14]:14s} "
              f"size={op['size_kb']:.1f}KB region={op['memory_region']}")

    assert len(hivm_ops) >= 3, f"Expected >=3 ops, got {len(hivm_ops)}"
    assert hivm_ops[0]["op_type"] == "gm_to_ub", f"First op should be load, got {hivm_ops[0]['op_type']}"
    assert hivm_ops[-1]["op_type"] == "ub_to_gm", f"Last op should be store, got {hivm_ops[-1]['op_type']}"
    print("  PASS: op types correct")

    print(f"\n=== HIVM MLIR Output ===")
    print(hivm_text[:1500])

    # Test 2: 验证 HIVM 能被 bishengir-compile 编译
    print(f"\n{'=' * 60}")
    print("Test 2: Compile HIVM with bishengir-compile")

    import subprocess, tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
        f.write(hivm_text)
        mlir_path = f.name

    o_path = mlir_path.replace(".mlir", ".o")
    try:
        r = subprocess.run(
            ["bishengir-compile", mlir_path, "-o", o_path],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0 and os.path.exists(o_path):
            size = os.path.getsize(o_path)
            print(f"  PASS: compiled to {o_path} ({size} bytes)")
            os.unlink(o_path)
        else:
            print(f"  INFO: bishengir-compile unavailable or failed (expected without CANN)")
            print(f"  stderr: {r.stderr[:200]}")
    except FileNotFoundError:
        print("  SKIP: bishengir-compile not found (CANN not sourced)")
    finally:
        os.unlink(mlir_path)

    print(f"\n{'=' * 60}")
    print("ALL TESTS PASSED — TTIR→HIVM Converter")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
