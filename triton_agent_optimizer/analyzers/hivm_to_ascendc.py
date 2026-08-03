#!/usr/bin/env python3
"""
HIVM ops → 单个 Ascend C .asc 文件 + CMake ASC 构建 → msprof op simulator

v4.0 (2026-07-31): 完全匹配 7月28日成功格式
  - 单个 .asc 文件 (kernel + ACL host + main 合一)
  - __global__ __vector__ (ASC构建系统要求, 不能用 __aicore__)
  - 模板参数: blockLength as template param
  - block_idx 内置变量分块
  - <<<numBlocks, 0>>> 2参数 launch (无 stream)
  - ACL runtime: aclInit → aclrtMalloc → aclrtMemcpy → launch → sync
  - CMakeLists.txt: find_package(ASC REQUIRED) + add_executable
  - target_compile_definitions: ASCENDC_TRACE_ON (启用msprof trace)

参考:
  - 7月28日成功的 ~/msprof_demo/msprof.asc (产出真实 trace 数据)
  - CANN 9.0 官方样例 (gitee.com/ascend/samples)
  - ASC构建系统约束: 禁止 __aicore__, 必须用 __vector__/__cube__/__mix__
"""

import os, re, subprocess, shutil, json
from pathlib import Path
from typing import Optional, List, Dict

DTYPE_MAP = {"f16": "half", "f32": "float", "i32": "int"}

# HIVM op → AscendC API (无 AscendC:: 前缀, 不在此文件里用)
OP_API = {
    "gm_to_ub": ("DataCopy", 2), "ub_to_gm": ("DataCopy", 2),
    "vadd": ("Add", 3), "vsub": ("Sub", 3), "vmul": ("Mul", 3),
    "vdiv": ("Div", 3), "vexp": ("Exp", 2), "vabs": ("Abs", 2),
    "vmax": ("Max", 3), "vsqrt": ("Sqrt", 2), "vrelu": ("Relu", 2),
    # matmul: CubeUnit 太复杂(需要 Matmul 类+REGIST_MATMUL_OBJ+Init+SetTensor+IterateAll),
    # 跳过 AscendC 生成, 用 SATURATION_PARAMS 公式估算 CubeUnit timing
}


def generate(kernel_name: str, hivm_ops: list, total_elems: int = 256,
             dtype: str = "f32", num_blocks: int = 8) -> str:
    """生成单个 .asc 文件 (kernel + host + main)。

    格式: 匹配 7月28日成功产出 trace 的 msprof.asc
    - __global__ __vector__ 模板 kernel
    - AscendC::InitSocState() + LocalMemAllocator
    - block_idx 分块
    - ACL host: aclInit → aclrtMalloc → memcpy → <<<N,0>>> → sync → verify
    """
    ctype = DTYPE_MAP.get(dtype, "float")
    ctype_cpp = dtype  # "f32" → "float" for C++ side
    cpptype = "float" if dtype == "f32" else ("half" if dtype == "f16" else "int")

    # 收集 GM 参数
    load_srcs = sorted(set(
        op["src"].lstrip("%") for op in hivm_ops if op["op_type"] == "gm_to_ub"))
    store_dsts = sorted(set(
        op["dst"].lstrip("%") for op in hivm_ops if op["op_type"] == "ub_to_gm"))
    all_gm = load_srcs + [d for d in store_dsts if d not in load_srcs]

    block_size = total_elems
    total_n = block_size * num_blocks

    lines = []
    lines.append('/*')
    lines.append(f' * Auto-generated Ascend C kernel (v4.0 — msprof compatible)')
    lines.append(f' * Kernel: {kernel_name} | Ops: {len(hivm_ops)} | Tile: {block_size}')
    lines.append(f' * Build: CMake ASC (find_package(ASC REQUIRED))')
    lines.append(f' * Profile: msprof op simulator --soc-version=Ascend910B3 ./demo')
    lines.append(' */')
    lines.append('')
    lines.append('#include "acl/acl.h"')
    lines.append('#include "kernel_operator.h"')
    lines.append('#include <cstdint>')
    lines.append('#include <cstdio>')
    lines.append('#include <cstdlib>')
    lines.append('#include <iostream>')
    lines.append('#include <vector>')
    lines.append('')
    lines.append('using namespace AscendC;')
    lines.append('')

    # ═══════════════ KERNEL ═══════════════
    lines.append(f'// ========== Kernel ==========')
    gm_params = ", ".join(f"__gm__ {cpptype}* {a}" for a in all_gm)
    lines.append(f'template <typename T, uint32_t blockLength>')
    lines.append(f'__global__ __vector__ void {kernel_name}(')
    if gm_params:
        lines.append(f'    {gm_params})')
    else:
        lines.append(f'    )')
    lines.append(f'{{')
    lines.append(f'    InitSocState();')
    lines.append(f'')
    lines.append(f'    GlobalTensor<T> xGm;')
    # 每个 GM arg 一个 GlobalTensor
    for a in all_gm:
        lines.append(f'    GlobalTensor<T> gm_{a};')
    lines.append('')

    # 第一遍: SetGlobalBuffer
    for i, a in enumerate(all_gm):
        lines.append(f'    gm_{a}.SetGlobalBuffer({a} + block_idx * blockLength, blockLength);')
    lines.append('')

    # UB allocator
    lines.append(f'    LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;')
    lines.append('')

    # emit ops — 每个 op 对应一条 AscendC API 调用 + PipeBarrier
    buf_counter = 0
    buf_map: Dict[str, str] = {}

    def _resolve(ssa: str) -> str:
        s = ssa.lstrip("%")
        if s in buf_map: return buf_map[s]
        m = re.match(r"ub_buf_(\d+)", s)
        if m:
            cand = f"ub{m.group(1)}"
            if cand in buf_map.values(): return cand
        if buf_map: return next(iter(buf_map.values()))
        return "ub0"

    for op in hivm_ops:
        ot = op["op_type"]
        if ot not in OP_API:
            continue
        fn_name, n_inputs = OP_API[ot]
        buf_name = f"ub{buf_counter}"
        buf_counter += 1

        if ot == "gm_to_ub":
            src = op.get("src", "").lstrip("%")
            lines.append(f'    LocalTensor<T> {buf_name} = ubAllocator.Alloc<T, blockLength>();')
            lines.append(f'    DataCopy({buf_name}, gm_{src}, blockLength);')
            lines.append(f'    PipeBarrier<PIPE_ALL>();')
            dst_ssa = op.get("dst", "").lstrip("%")
            buf_map[dst_ssa] = buf_name

        elif ot == "ub_to_gm":
            src_buf = _resolve(op.get("src", ""))
            dst = op.get("dst", "").lstrip("%")
            lines.append(f'    DataCopy(gm_{dst}, {src_buf}, blockLength);')
            lines.append(f'    PipeBarrier<PIPE_ALL>();')

        elif n_inputs == 2:
            src_buf = _resolve(op.get("src", ""))
            lines.append(f'    LocalTensor<T> {buf_name} = ubAllocator.Alloc<T, blockLength>();')
            lines.append(f'    {fn_name}({buf_name}, {src_buf}, blockLength);')
            lines.append(f'    PipeBarrier<PIPE_ALL>();')
            dst_ssa = op.get("dst", "").lstrip("%")
            buf_map[dst_ssa] = buf_name

        elif n_inputs == 3:
            src_buf = _resolve(op.get("src", ""))
            src2_buf = _resolve(op.get("src2", ""))
            lines.append(f'    LocalTensor<T> {buf_name} = ubAllocator.Alloc<T, blockLength>();')
            lines.append(f'    {fn_name}({buf_name}, {src_buf}, {src2_buf}, blockLength);')
            lines.append(f'    PipeBarrier<PIPE_ALL>();')
            dst_ssa = op.get("dst", "").lstrip("%")
            buf_map[dst_ssa] = buf_name

        lines.append('')
    lines.append(f'}}')
    lines.append('')

    # ═══════════════ HOST ═══════════════
    lines.append(f'// ========== Host ==========')
    lines.append(f'template <typename T>')
    lines.append(f'std::vector<T> kernel_launch(std::vector<T>& h_input)')
    lines.append(f'{{')
    lines.append(f'    constexpr uint32_t numBlocks = {num_blocks};')
    lines.append(f'    constexpr uint32_t blockLength = {block_size};')
    lines.append(f'    uint32_t totalLength = h_input.size();')
    lines.append(f'    size_t totalByteSize = totalLength * sizeof(T);')
    lines.append(f'    int32_t deviceId = 0;')
    lines.append('')
    # Alloc device memory for each GM arg
    for a in all_gm:
        lines.append(f'    T* d_{a} = nullptr;')
    lines.append(f'    std::vector<T> h_output(totalLength);')
    lines.append('')
    lines.append(f'    aclInit(nullptr);')
    lines.append(f'    aclrtSetDevice(deviceId);')
    lines.append('')
    for a in all_gm:
        lines.append(f'    aclrtMalloc((void**)&d_{a}, totalByteSize, ACL_MEM_MALLOC_HUGE_FIRST);')
    lines.append('')
    # First arg = input (copy from host), rest = output (zero-init or copy)
    if all_gm:
        lines.append(f'    aclrtMemcpy(d_{all_gm[0]}, totalByteSize, h_input.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);')
    lines.append('')
    # Launch
    lines.append(f'    {kernel_name}<T, blockLength><<<numBlocks, 0>>>(')
    lines.append(f'        {", ".join("d_" + a for a in all_gm)});')
    lines.append(f'    aclrtSynchronizeDevice();')
    lines.append('')
    # Copy output back (last arg if store exists, otherwise first)
    out_arg = all_gm[-1] if all_gm else all_gm[0] if all_gm else ""
    if out_arg:
        lines.append(f'    aclrtMemcpy(h_output.data(), totalByteSize, d_{out_arg}, totalByteSize, ACL_MEMCPY_DEVICE_TO_HOST);')
    lines.append('')
    for a in all_gm:
        lines.append(f'    aclrtFree(d_{a});')
    lines.append(f'    aclrtResetDevice(deviceId);')
    lines.append(f'    aclFinalize();')
    lines.append('')
    lines.append(f'    return h_output;')
    lines.append(f'}}')
    lines.append('')

    # ═══════════════ MAIN ═══════════════
    lines.append(f'// ========== Main ==========')
    lines.append(f'int32_t main(int32_t argc, char* argv[])')
    lines.append(f'{{')
    lines.append(f'    using DataType = {cpptype};')
    lines.append(f'    constexpr uint32_t totalLength = {total_n};')
    lines.append(f'    std::vector<DataType> x(totalLength);')
    lines.append(f'    for (uint32_t i = 0; i < totalLength; ++i) {{')
    lines.append(f'        x[i] = static_cast<DataType>(i * 0.1f);')
    lines.append(f'    }}')
    lines.append('')
    lines.append(f'    std::vector<DataType> output = kernel_launch<DataType>(x);')
    lines.append('')
    lines.append(f'    std::cout << "{kernel_name} done (N=" << totalLength << ")" << std::endl;')
    lines.append(f'    return 0;')
    lines.append(f'}}')

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────
#  CMakeLists.txt 生成
# ──────────────────────────────────────────────────────────

def generate_cmake(kernel_name: str) -> str:
    """生成匹配 CANN 9.0 ASC 构建系统的 CMakeLists.txt"""
    return f"""cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture: dav-2201, dav-3510")

find_package(ASC REQUIRED)

project(kernel_samples LANGUAGES ASC CXX)

set(ASC_SOURCES
    {kernel_name}.asc
)

set(COMPILE_OPTIONS
    $<$<COMPILE_LANGUAGE:ASC>: --npu-arch=${{CMAKE_ASC_ARCHITECTURES}}>
)

add_executable(demo ${{ASC_SOURCES}})

target_compile_definitions(demo PRIVATE ASCENDC_TRACE_ON)
target_link_libraries(demo PRIVATE tiling_api platform)
target_compile_options(demo PRIVATE ${{COMPILE_OPTIONS}})
"""


# ──────────────────────────────────────────────────────────
#  构建 + msprof: 完整流水线
# ──────────────────────────────────────────────────────────

def generate_and_build(kernel_name: str, hivm_ops: list,
                       output_dir: Path, total_elems: int = 256,
                       dtype: str = "f32") -> Optional[Path]:
    """生成 .asc → CMake ASC 构建 → 可执行文件。

    关键: 必须从 CANN 安装目录 source set_env.sh (cd /usr/local/Ascend/cann && source ./set_env.sh)
    """
    work_dir = output_dir / "ascendc_build"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成单个 .asc
    asc_code = generate(kernel_name, hivm_ops, total_elems, dtype)
    asc_path = work_dir / f"{kernel_name}.asc"
    asc_path.write_text(asc_code, encoding="utf-8")

    # 2. 生成 CMakeLists.txt
    cmake_path = work_dir / "CMakeLists.txt"
    cmake_path.write_text(generate_cmake(kernel_name), encoding="utf-8")

    # 3. CMake ASC 构建
    # ★ 必须 cd 到 CANN 目录再 source ★
    cann_dir = "/usr/local/Ascend/cann"
    if not os.path.isdir(cann_dir):
        print(f"  [HIVM→AscendC] ERROR: CANN not found at {cann_dir}")
        return None

    build_dir = work_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # cmake
    r = subprocess.run(
        ["bash", "-c",
         f"cd {cann_dir} && source ./set_env.sh 2>/dev/null && "
         f"cd {build_dir} && "
         f"cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DCMAKE_ASC_RUN_MODE=sim 2>&1"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        errs = [l for l in (r.stdout + r.stderr).split("\n") if "Error" in l or "error:" in l]
        if errs:
            print(f"  [HIVM→AscendC] CMake failed:")
            for e in errs[:3]: print(f"    {e[:200]}")
            return None

    # make
    r = subprocess.run(
        ["bash", "-c",
         f"cd {cann_dir} && source ./set_env.sh 2>/dev/null && "
         f"cd {build_dir} && make -j1 2>&1"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        errs = [l for l in (r.stdout + r.stderr).split("\n") if "error:" in l]
        if errs:
            print(f"  [HIVM→AscendC] Make failed:")
            for e in errs[:3]: print(f"    {e[:200]}")
            return None

    exe = build_dir / "demo"
    if exe.exists():
        print(f"  [HIVM→AscendC] Build OK ({exe.stat().st_size} bytes)")
        return exe
    return None


def run_msprof(executable: Path, output_dir: Path,
               soc_version: str = "Ascend910B3") -> Optional[Path]:
    """msprof op simulator 采集 trace。

    关键: 输出必须到 Linux 原生路径 (/tmp/), 不能用 /mnt/d/ (Windows 文件系统)。
    cd 到 CANN 目录 source set_env.sh, simulator 路径在最前面。
    """
    cann_dir = "/usr/local/Ascend/cann"
    simlib = f"/usr/local/Ascend/cann-9.0.0/x86_64-linux/simulator/Ascend910B3/lib"
    # ★ 输出到 Linux 原生 /tmp, 避免 Windows 文件系统权限问题 ★
    native_out = f"/tmp/msprof_{os.getpid()}"
    prof_out = output_dir / "msprof"

    try:
        r = subprocess.run(
            ["bash", "-c",
             f"cd {cann_dir} && source ./set_env.sh 2>/dev/null && "
             f"export LD_LIBRARY_PATH={simlib}:$LD_LIBRARY_PATH && "
             f"rm -rf {native_out} && mkdir -p {native_out} && "
             f"msprof op simulator "
             f"--soc-version={soc_version} "
             f"--output={native_out} "
             f"--timeout=5 "
             f"{executable} 2>&1"],
            capture_output=True, text=True, timeout=300,
        )
        opprof_dirs = sorted(Path(native_out).glob("OPPROF_*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if opprof_dirs:
            csv_files = list(opprof_dirs[0].glob("**/*instr_exe.csv"))
            if csv_files:
                # 拷贝到目标目录
                if prof_out.exists():
                    shutil.rmtree(prof_out)
                prof_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(opprof_dirs[0]), str(prof_out))
                # 清理临时目录
                shutil.rmtree(native_out, ignore_errors=True)
                print(f"  [msprof] OK: {len(csv_files)} instr_exe.csv → {prof_out}")
                return prof_out
            print(f"  [msprof] WARN: no instr_exe.csv in {opprof_dirs[0]}")
            shutil.rmtree(native_out, ignore_errors=True)
            return None
        else:
            # 打印 msprof 输出帮助调试
            print(f"  [msprof] No OPPROF_* created. msprof output:")
            for line in (r.stdout + r.stderr).split("\n")[-10:]:
                if line.strip():
                    print(f"    {line.strip()[:200]}")
            shutil.rmtree(native_out, ignore_errors=True)
            return None
    except Exception as e:
        print(f"  [msprof] Error: {e}")
        return None


# ──────────────────────────────────────────────────────────
#  Self-test
# ──────────────────────────────────────────────────────────

def _self_test():
    ops = [
        {"op_type": "gm_to_ub", "dst": "%ub0", "src": "%x", "size_kb": 1.0},
        {"op_type": "gm_to_ub", "dst": "%ub1", "src": "%y", "size_kb": 1.0},
        {"op_type": "vadd", "dst": "%ub2", "src": "%ub0", "src2": "%ub1", "size_kb": 1.0},
        {"op_type": "ub_to_gm", "dst": "%z", "src": "%ub2", "size_kb": 1.0},
    ]
    asc = generate("test_kernel", ops, 256, "f32")
    assert '__global__ __vector__' in asc, "Missing __vector__"
    assert '#include "acl/acl.h"' in asc, "Missing ACL include"
    assert 'InitSocState()' in asc, "Missing InitSocState"
    assert 'LocalMemAllocator<Hardware::UB>' in asc, "Missing Allocator"
    assert '<<<numBlocks, 0>>>' in asc, "Missing launch syntax"
    assert 'aclInit' in asc, "Missing aclInit"
    assert 'aclFinalize' in asc, "Missing aclFinalize"
    assert 'main(' in asc, "Missing main"
    cm = generate_cmake("test_kernel")
    assert 'find_package(ASC REQUIRED)' in cm, "Missing find_package"
    assert 'ASCENDC_TRACE_ON' in cm, "Missing ASCENDC_TRACE_ON"
    print(f"[HIVM→AscendC v4.0] Self-test PASSED ({len(asc)} chars .asc, {len(cm)} chars CMake)")
    print(f"  Format: __global__ __vector__ + ACL host + <<<N,0>>> launch")
    print(f"  Build:  CMake ASC (find_package(ASC REQUIRED))")
    print(f"  Profile: msprof op simulator --soc-version=Ascend910B3 ./demo")


if __name__ == "__main__":
    _self_test()
