#!/usr/bin/env python3
"""
HIVM MLIR → Ascend C kernel 代码生成器 (v1.0)

将 HIVM dialect MLIR 转换为可编译的 Ascend C (.asc) kernel。
生成的文件可以用 CMake+bisheng 编译 → msprof op simulator 采集 trace。

映射规则:
  hivm.hir.load  → DataCopy (GM→UB, MTE2)
  hivm.hir.store → DataCopy (UB→GM, MTE3)
  hivm.hir.vadd  → Add (VecUnit)
  hivm.hir.vsub  → Sub
  hivm.hir.vmul  → Mul
  hivm.hir.vdiv  → Div
  hivm.hir.vexp  → Exp
  hivm.hir.vabs  → Abs
  hivm.hir.vmax  → Maximum
  hivm.hir.vsqrt → Sqrt
  hivm.hir.matmul → MatMul
  linalg.reduce   → ReduceMax / ReduceSum (by operands)
"""

import re
from pathlib import Path

OP_TO_ASCENDC = {
    "gm_to_ub": ("DataCopy", "MTE2", 2),   # (kernel_fn, arg_count_for_sig)
    "ub_to_gm": ("DataCopy", "MTE3", 2),
    "vadd":     ("Add",     "VECTOR", 3),
    "vsub":     ("Sub",     "VECTOR", 3),
    "vmul":     ("Mul",     "VECTOR", 3),
    "vdiv":     ("Div",     "VECTOR", 3),
    "vexp":     ("Exp",     "VECTOR", 2),
    "vabs":     ("Abs",     "VECTOR", 2),
    "vmax":     ("Maximum", "VECTOR", 3),
    "vsqrt":    ("Sqrt",    "VECTOR", 2),
    "matmul":   ("MatMul",  "CUBE",   3),
    "reduce":   ("ReduceMax", "VECTOR", 2),
}

DTYPE_MAP = {"f16": "half", "f32": "float", "f64": "double",
             "i32": "int", "i8": "char"}


def generate(kernel_name: str, hivm_ops: list, total_elems: int = 256,
             dtype: str = "f32") -> str:
    """从 HIVM ops 列表生成 Ascend C kernel 源码。

    Args:
        kernel_name: kernel 函数名
        hivm_ops: [{"op_type": "gm_to_ub", "dst": "%ub0", "src": "%arg0",
                     "size_kb": 1.0, "memory_region": "ub"}, ...]
        total_elems: 每个 tile 的元素数
        dtype: 数据类型 (f16/f32)

    Returns:
        Ascend C .asc 源码字符串
    """
    ctype = DTYPE_MAP.get(dtype, "float")

    lines = []
    lines.append('// Auto-generated Ascend C kernel from HIVM MLIR')
    lines.append('// Pipeline: TTIR → HIVM → AscendC → msprof simulator')
    lines.append('')
    lines.append('#include "kernel_operator.h"')
    lines.append('#include "acl/acl.h"')
    lines.append('#include <vector>')
    lines.append('#include <iostream>')
    lines.append('using namespace std;')
    lines.append('')

    # ── Kernel ──
    lines.append(f'__global__ __vector__ void {kernel_name}(')
    load_srcs = sorted(set(
        op["src"].lstrip("%") for op in hivm_ops if op["op_type"] == "gm_to_ub"))
    store_dsts = sorted(set(
        op["dst"].lstrip("%") for op in hivm_ops if op["op_type"] == "ub_to_gm"))
    arg_names = load_srcs + [d for d in store_dsts if d not in load_srcs]
    for i, arg in enumerate(arg_names):
        comma = "," if i < len(arg_names) - 1 else ""
        lines.append(f"    __gm__ {ctype}* {arg}{comma}")
    lines.append(f')')
    lines.append(f'{{')
    lines.append(f'    AscendC::InitSocState();')
    lines.append(f'    constexpr uint32_t BLK = {total_elems};')
    lines.append(f'    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ub;')
    lines.append('')

    buf_count = 0
    buf_map = {}

    for op in hivm_ops:
        ot = op["op_type"]
        if ot not in OP_TO_ASCENDC:
            continue

        fn_name, pipe, n_inputs = OP_TO_ASCENDC[ot]
        buf_name = f"buf_{buf_count}"
        buf_count += 1
        buf_map[op.get("dst", f"%buf_{buf_count}")] = buf_name

    # 重新遍历发射 AscendC 调用
    buf_count = 0
    for op in hivm_ops:
        ot = op["op_type"]
        if ot not in OP_TO_ASCENDC:
            continue

        fn_name, pipe, n_inputs = OP_TO_ASCENDC[ot]
        buf_name = f"buf_{buf_count}"
        buf_count += 1

        if ot == "gm_to_ub":
            src = op.get("src", "").lstrip("%")
            lines.append(f'    AscendC::LocalTensor<{ctype}> {buf_name} = ub.template Alloc<{ctype}, BLK>();')
            lines.append(f'    AscendC::GlobalTensor<{ctype}> gm_{src};')
            lines.append(f'    gm_{src}.SetGlobalBuffer({src} + block_idx * BLK, BLK);')
            lines.append(f'    AscendC::DataCopy({buf_name}, gm_{src}, BLK);')

        elif ot == "ub_to_gm":
            src_buf = find_input_buffer(hivm_ops, op.get("src", ""), buf_map, buf_name)
            dst = op.get("dst", "").lstrip("%")
            lines.append(f'    AscendC::GlobalTensor<{ctype}> gm_{dst};')
            lines.append(f'    gm_{dst}.SetGlobalBuffer({dst} + block_idx * BLK, BLK);')
            lines.append(f'    AscendC::DataCopy(gm_{dst}, {src_buf}, BLK);')

        elif ot in ("vadd", "vsub", "vmul", "vdiv", "vmax", "vexp", "vabs", "vsqrt"):
            if n_inputs == 2:
                src = op.get("src", "")
                src_buf = resolve_buf(src, buf_map, buf_name)
                lines.append(f'    AscendC::LocalTensor<{ctype}> {buf_name} = ub.template Alloc<{ctype}, BLK>();')
                lines.append(f'    AscendC::{fn_name}({buf_name}, {src_buf}, BLK);')
            elif n_inputs == 3:
                src = op.get("src", "")
                src2 = op.get("src2", "")
                src_buf = resolve_buf(src, buf_map, buf_name)
                src2_buf = resolve_buf(src2, buf_map, buf_name)
                lines.append(f'    AscendC::LocalTensor<{ctype}> {buf_name} = ub.template Alloc<{ctype}, BLK>();')
                lines.append(f'    AscendC::{fn_name}({buf_name}, {src_buf}, {src2_buf}, BLK);')

        if ot != "ub_to_gm":
            lines.append(f'    AscendC::PipeBarrier<AscendC::PIPE_ALL>();')
        lines.append('')

    lines.append(f'}}')
    lines.append('')

    # ── Host ──
    lines.append('int main() {')
    lines.append(f'    constexpr uint32_t BLK = {total_elems};')
    lines.append(f'    constexpr uint32_t N = BLK * 8;')
    lines.append(f'    size_t bytes = N * sizeof({ctype});')
    lines.append('')
    lines.append('    aclInit(nullptr); aclrtSetDevice(0);')
    lines.append('')

    for arg in arg_names:
        lines.append(f'    vector<{ctype}> h_{arg}(N);')
        lines.append(f'    for (uint32_t i = 0; i < N; i++) h_{arg}[i] = ({ctype})(i * 0.1f);')

    lines.append('')
    for arg in arg_names:
        lines.append(f'    {ctype}* d_{arg};')
        lines.append(f'    aclrtMalloc((void**)&d_{arg}, bytes, ACL_MEM_MALLOC_HUGE_FIRST);')
        lines.append(f'    aclrtMemcpy(d_{arg}, bytes, h_{arg}.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);')

    args_call = ", ".join(f"d_{a}" for a in arg_names)
    lines.append(f'    {kernel_name}<<<8, 0>>>({args_call});')
    lines.append(f'    aclrtSynchronizeDevice();')
    lines.append('')

    for arg in arg_names:
        lines.append(f'    aclrtMemcpy(h_{arg}.data(), bytes, d_{arg}, bytes, ACL_MEMCPY_DEVICE_TO_HOST);')

    lines.append(f'    cout << "DONE" << endl;')
    for arg in arg_names:
        lines.append(f'    aclrtFree(d_{arg});')
    lines.append(f'    aclrtResetDevice(0); aclFinalize();')
    lines.append(f'    return 0;')
    lines.append(f'}}')

    return "\n".join(lines)


def find_input_buffer(ops, ssa_name, buf_map, fallback):
    """找到 SSA 值对应的 buffer 名。"""
    if ssa_name in buf_map:
        return buf_map[ssa_name]
    return fallback


def resolve_buf(ssa_name, buf_map, fallback):
    """解析 SSA 名为对应的 AscendC buffer 名。"""
    stripped = ssa_name.lstrip("%")
    if stripped in buf_map:
        return buf_map[stripped]
    # 尝试从 HIVM 格式解析: %ub_buf_N
    m = re.match(r"ub_buf_(\d+)", stripped)
    if m:
        return f"buf_{m.group(1)}"
    return fallback


def generate_and_build(kernel_name: str, hivm_ops: list,
                       output_dir: Path, total_elems: int = 256,
                       dtype: str = "f32") -> Path:
    """生成 AscendC 源码 → bisheng 手动编译 → 可执行文件。

    使用官方文档推荐的手动编译方式 (不需要 CMake):
      ① bisheng -c kernel.asc -o kernel.o --npu-arch=dav-2201
      ② bisheng kernel.o -o app -L simulator/lib -lruntime_camodel -lnpu_drv_camodel -lm -lstdc++
      ③ msprof op simulator --soc-version=Ascend910B3 ./app

    Returns:
        可执行文件路径, 或 None (编译失败)
    """
    import subprocess, os, shutil

    work_dir = output_dir / "ascendc_build"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成 .asc 源码 (包含 kernel + host main)
    asc_code = generate(kernel_name, hivm_ops, total_elems, dtype)
    asc_path = work_dir / f"{kernel_name}.asc"
    asc_path.write_text(asc_code, encoding="utf-8")

    # 2. 找 CANN 安装路径和模拟器库
    ascend_base = os.environ.get("ASCEND_HOME", "/usr/local/Ascend")
    # 尝试找 cann 安装路径
    for candidate in [
        f"{ascend_base}/cann-9.0.0",
        f"{ascend_base}/cann",
        f"{ascend_base}/ascend-toolkit/latest",
        "/usr/local/Ascend/cann-9.0.0",
        "/usr/local/Ascend/cann",
    ]:
        if os.path.isdir(candidate):
            cann_dir = candidate
            break
    else:
        cann_dir = ascend_base

    bisheng_bin = shutil.which("bisheng") or f"{cann_dir}/bin/bisheng"
    sim_lib = f"{cann_dir}/x86_64-linux/simulator/Ascend910B3/lib"

    if not os.path.isdir(sim_lib):
        # 尝试其他路径
        import glob
        sim_dirs = glob.glob(f"{cann_dir}/**/simulator/Ascend910B3/lib", recursive=True)
        sim_lib = sim_dirs[0] if sim_dirs else ""

    obj_path = work_dir / f"{kernel_name}.o"
    exe_path = work_dir / kernel_name

    # 3. Step ①: 编译 → .o
    try:
        r = subprocess.run(
            [bisheng_bin, "-c", str(asc_path), "-o", str(obj_path),
             "--npu-arch=dav-2201"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            print(f"  [HIVM→AscendC] Compile failed: {r.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  [HIVM→AscendC] Compile error: {e}")
        return None

    # 4. Step ②: 链接 → executable
    try:
        link_cmd = [
            bisheng_bin, str(obj_path), "-o", str(exe_path),
            f"-L{sim_lib}", f"-Wl,-rpath,{sim_lib}",
            "-lruntime_camodel", "-lnpu_drv_camodel",
            "-lm", "-lstdc++",
        ]
        r = subprocess.run(link_cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            print(f"  [HIVM→AscendC] Link failed: {r.stderr[:200]}")
            return None
    except Exception as e:
        print(f"  [HIVM→AscendC] Link error: {e}")
        return None

    if exe_path.exists():
        return exe_path
    return None


def run_msprof(executable: Path, output_dir: Path,
               soc_version: str = "Ascend910B3") -> Path:
    """运行 msprof op simulator 采集 trace。

    Returns:
        OPPROF 目录路径
    """
    import subprocess, shutil

    msprof_bin = shutil.which("msprof")
    if not msprof_bin:
        print("  [msprof] msprof binary not found")
        return None

    msprof_out = output_dir / "msprof"
    msprof_out.mkdir(parents=True, exist_ok=True)

    try:
        r = subprocess.run(
            [msprof_bin, "op", "simulator",
             f"--soc-version={soc_version}",
             f"--output={msprof_out}",
             f"--timeout=5",
             str(executable)],
            capture_output=True, text=True, timeout=300,
        )
        opprof_dirs = sorted(msprof_out.glob("OPPROF_*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        return opprof_dirs[0] if opprof_dirs else None
    except Exception as e:
        print(f"  [msprof] Simulation failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

def _self_test():
    # 模拟 softmax 的 HIVM ops
    ops = [
        {"op_type": "gm_to_ub", "dst": "%ub0", "src": "%arg0",
         "size_kb": 1.0, "memory_region": "ub"},
        {"op_type": "gm_to_ub", "dst": "%ub1", "src": "%arg1",
         "size_kb": 1.0, "memory_region": "ub"},
        {"op_type": "vadd", "dst": "%ub2", "src": "%ub0", "src2": "%ub1",
         "size_kb": 1.0, "memory_region": "ub"},
        {"op_type": "ub_to_gm", "dst": "%arg2", "src": "%ub2",
         "size_kb": 1.0, "memory_region": "gm"},
    ]

    code = generate("test_add", ops, 256, "f32")
    print(f"Generated {len(code)} chars of Ascend C code")
    print(code[:1000])
    print("...")

    # 验证关键内容
    assert "DataCopy" in code
    assert "Add" in code
    assert "__global__ __vector__" in code
    assert "aclInit" in code
    print("PASS")


if __name__ == "__main__":
    _self_test()
