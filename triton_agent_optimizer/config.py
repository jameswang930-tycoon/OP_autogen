#!/usr/bin/env python3
"""
Triton Agent Optimizer — 全局配置中心
======================================

所有模块通过 `from config import ...` 获取配置参数。
不硬编码任何路径或阈值——全部集中在这里。

硬件参数从 costModel/cost_emulator/simulator.py 动态读取，
不在本文件中重复定义。

使用方式:
    from config import Config
    cfg = Config()                          # 自动检测环境 (本地/服务器)
    cfg.simulator_path                      # simulator.py 的路径
    cfg.emulator_root                       # emulators/ 的路径
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import platform
import subprocess
from pathlib import Path as _Path

# ═════════════════════════════════════════════════════════════════════════════════
#  .env 文件加载 (无需 python-dotenv 依赖)
# ═════════════════════════════════════════════════════════════════════════════════

def _load_dotenv():
    """手动解析 .env 文件，加载到 os.environ。"""
    _dir = _Path(__file__).resolve().parent
    for env_file in (_dir / ".env", _dir.parent / ".env"):
        if env_file.exists():
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip()
                    # 去掉引号
                    if (val.startswith('"') and val.endswith('"')) or \
                       (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    # 只设置未设置的变量 (环境变量优先级更高)
                    if key not in os.environ:
                        os.environ[key] = val

_load_dotenv()
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ═════════════════════════════════════════════════════════════════════════════════
#  环境类型枚举
# ═════════════════════════════════════════════════════════════════════════════════

class Environment:
    """运行环境类型"""
    LOCAL_WINDOWS = "local_windows"     # 本地 Windows 开发
    LOCAL_LINUX   = "local_linux"       # 本地 Linux 开发
    ASCEND_910B3  = "ascend_910b3"      # 华为昇腾 910B3 服务器


# ═════════════════════════════════════════════════════════════════════════════════
#  环境自动检测
# ═════════════════════════════════════════════════════════════════════════════════

def _detect_environment() -> str:
    """自动检测当前运行环境。

    检测逻辑:
      1. 检查 ASCEND_HOME 或 ASCEND_TOOLKIT_HOME 环境变量 → 910B3 服务器
      2. 否则检查操作系统 → Windows 或 Linux 本地
    """
    # 检查 Ascend 环境变量
    ascend_home = os.environ.get("ASCEND_HOME", "")
    ascend_toolkit = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    if ascend_home or ascend_toolkit:
        return Environment.ASCEND_910B3

    # 检查 CANN 安装目录是否存在
    cann_paths = [
        Path("/usr/local/Ascend/cann"),
        Path("/usr/local/Ascend/ascend-toolkit/latest"),
    ]
    for p in cann_paths:
        if p.exists():
            return Environment.ASCEND_910B3

    # 回退到操作系统判断
    if platform.system() == "Windows":
        return Environment.LOCAL_WINDOWS
    return Environment.LOCAL_LINUX


def _find_repo_root() -> Path:
    """从当前文件向上查找仓库根目录 (包含 .git 的目录)。"""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    # 回退: config.py 的上上级目录
    return Path(__file__).resolve().parent.parent


# ═════════════════════════════════════════════════════════════════════════════════
#  平台路径配置 (所有和文件系统相关的路径)
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class PlatformPaths:
    """平台相关路径——根据运行环境自动确定。

    所有路径都是 pathlib.Path 对象, 支持 / 运算符拼接。
    不存在的目录在首次访问时会自动创建 (通过 _ensure_dir)。
    """

    env: str = field(default_factory=_detect_environment)

    # ── 仓库内部路径 (所有环境通用) ──────────────────────────────────────────
    repo_root: Path = field(default_factory=_find_repo_root)

    @property
    def triton_agent_root(self) -> Path:
        """triton_agent_optimizer/ 目录"""
        return self.repo_root / "triton_agent_optimizer"

    @property
    def cost_model_root(self) -> Path:
        """(deprecated) costModel/ 目录"""
        return self.repo_root / "costModel" / "cost_emulator"

    @property
    def simulator_path(self) -> Path:
        """(deprecated) simulator.py 路径"""
        return self.cost_model_root / "simulator.py"

    @property
    def emulator_root(self) -> Path:
        """emulators/ 目录 (CPU Triton 模拟层)"""
        return self.repo_root / "emulators"

    @property
    def emulator_common_path(self) -> Path:
        """emulators/common/__init__.py (tl 类 + launch_kernel + verify)"""
        return self.emulator_root / "common" / "__init__.py"

    @property
    def compiler_path(self) -> Path:
        """Ascend 编译器接口 (编译 + HIVMIR 提取)"""
        return self.triton_agent_root / "execution" / "compiler.py"

    @property
    def memory_root(self) -> Path:
        """memory/ 目录 (经验检索模块)"""
        return self.repo_root / "memory"

    @property
    def docs_root(self) -> Path:
        """docs/ 目录 (项目知识文档)"""
        return self.repo_root / "docs"

    # ── Agent 优化器内部路径 ──────────────────────────────────────────────────
    @property
    def output_dir(self) -> Path:
        """输出根目录 (所有优化产出物)"""
        d = self.triton_agent_root / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def rounds_dir(self) -> Path:
        """每轮优化产物目录 (plan.md + diff.patch + msprof + hivmir)"""
        d = self.output_dir / "rounds"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def journal_path(self) -> Path:
        """优化日志 JSONL 文件"""
        return self.output_dir / "optimization_journal.jsonl"

    @property
    def playbooks_dir(self) -> Path:
        """优化指导手册目录"""
        return self.triton_agent_root / "playbooks"

    @property
    def cases_dir(self) -> Path:
        """优秀案例库目录"""
        return self.triton_agent_root / "cases"

    @property
    def example_output_dir(self) -> Path:
        """Simulator 示例输出目录"""
        return self.triton_agent_root / "example_output"

    # ── Python 环境 ───────────────────────────────────────────────────────────
    @property
    def python_executable(self) -> str:
        """Python 可执行文件路径。

        优先使用 .venv 中的 Python, 回退到系统 Python。
        不同环境的查找顺序:
          Windows 本地: .venv/Scripts/python.exe → 系统 python
          Linux 本地:   .venv/bin/python → 系统 python3 → python
          910B3 服务器:  conda 环境 OP_autogen_hjkc 中的 python
        """
        if self.env == Environment.LOCAL_WINDOWS:
            venv_python = self.repo_root / ".venv" / "Scripts" / "python.exe"
            if venv_python.exists():
                return str(venv_python)
            # 尝试 conda
            conda_python = shutil.which("python")
            if conda_python:
                return conda_python
            return "python"

        elif self.env == Environment.LOCAL_LINUX:
            venv_python = self.repo_root / ".venv" / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)
            # 尝试 conda
            conda_python = shutil.which("python3")
            if conda_python:
                return conda_python
            return "python3"

        else:  # ASCEND_910B3
            # 在服务器上, 优先使用 conda 环境
            conda_base = os.environ.get("CONDA_PREFIX", "")
            if conda_base:
                conda_python = Path(conda_base) / "bin" / "python"
                if conda_python.exists():
                    return str(conda_python)
            # 回退: CANN 自带的 Python 或系统 Python
            for candidate in ["python3", "python"]:
                p = shutil.which(candidate)
                if p:
                    return p
            return "python3"

    # ── Ascend 910B3 服务器路径 (仅在服务器环境有效) ──────────────────────────

    @property
    def ascend_home(self) -> Optional[Path]:
        """Ascend 安装根目录。

        标准路径:
          - 社区版:   /usr/local/Ascend/cann
          - 商用版:   /usr/local/Ascend/ascend-toolkit/latest
          - 自定义:   从 ASCEND_HOME 环境变量读取
        """
        # 1. 环境变量优先
        env_val = os.environ.get("ASCEND_HOME")
        if env_val:
            return Path(env_val)

        # 2. 检查标准安装路径
        candidates = [
            Path("/usr/local/Ascend"),
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/cann"),
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    @property
    def ascend_toolkit_home(self) -> Optional[Path]:
        """Ascend Toolkit 目录。

        标准路径:
          - 商用版: /usr/local/Ascend/ascend-toolkit/latest
          - 社区版: /usr/local/Ascend/cann
        """
        env_val = os.environ.get("ASCEND_TOOLKIT_HOME")
        if env_val:
            return Path(env_val)
        if self.ascend_home:
            candidates = [
                self.ascend_home / "ascend-toolkit" / "latest",
                self.ascend_home / "cann",
                self.ascend_home,
            ]
            for p in candidates:
                if p.exists():
                    return p
        return None

    @property
    def msprof_path(self) -> Optional[Path]:
        """msprof 性能分析工具路径。

        标准路径 (按优先级):
          1. PATH 中的 msprof
          2. ${ASCEND_TOOLKIT_HOME}/tools/profiler/bin/msprof
          3. ${ASCEND_HOME}/cann/tools/profiler/bin/msprof
        """
        # 1. 检查 PATH
        p = shutil.which("msprof")
        if p:
            return Path(p)

        # 2. 检查 toolkit 路径
        if self.ascend_toolkit_home:
            candidate = self.ascend_toolkit_home / "tools" / "profiler" / "bin" / "msprof"
            if candidate.exists():
                return candidate

        # 3. 检查 cann 子路径
        if self.ascend_home:
            candidate = self.ascend_home / "cann" / "tools" / "profiler" / "bin" / "msprof"
            if candidate.exists():
                return candidate

        return None

    @property
    def ascend_compiler_bin(self) -> Optional[Path]:
        """Ascend 编译器 (ATC/Ascend C Compiler) 路径。

        标准路径:
          - ${ASCEND_TOOLKIT_HOME}/compiler/bin
          - ${ASCEND_HOME}/ascend-toolkit/latest/compiler/bin

        用途: 编译 Triton kernel → NPU 二进制, 提取 HIVMIR
        """
        if self.ascend_toolkit_home:
            candidate = self.ascend_toolkit_home / "compiler" / "bin"
            if candidate.exists():
                return candidate
        if self.ascend_home:
            candidate = self.ascend_home / "ascend-toolkit" / "latest" / "compiler" / "bin"
            if candidate.exists():
                return candidate
        return None

    @property
    def ascend_dmi_path(self) -> Optional[Path]:
        """ascend-dmi 工具路径 (设备管理 + 带宽测试)。

        标准路径:
          - ${ASCEND_HOME}/ascend-toolkit/latest/bin/ascend-dmi
        """
        p = shutil.which("ascend-dmi")
        if p:
            return Path(p)
        if self.ascend_home:
            candidate = self.ascend_home / "ascend-toolkit" / "latest" / "bin" / "ascend-dmi"
            if candidate.exists():
                return candidate
        return None

    @property
    def ascend_set_env_script(self) -> Optional[Path]:
        """CANN 环境设置脚本。

        标准路径:
          - 商用版: ${ASCEND_HOME}/ascend-toolkit/set_env.sh
          - 社区版: ${ASCEND_HOME}/cann/set_env.sh
        """
        candidates = []
        if self.ascend_home:
            candidates.extend([
                self.ascend_home / "ascend-toolkit" / "set_env.sh",
                self.ascend_home / "cann" / "set_env.sh",
            ])
        for p in candidates:
            if p.exists():
                return p
        return None

    @property
    def ascend_lib64_path(self) -> Optional[Path]:
        """Ascend 运行时库路径 (LD_LIBRARY_PATH 需要)。

        标准路径: ${ASCEND_TOOLKIT_HOME}/lib64
        """
        if self.ascend_toolkit_home:
            candidate = self.ascend_toolkit_home / "lib64"
            if candidate.exists():
                return candidate
        return None

    @property
    def opp_path(self) -> Optional[Path]:
        """Ascend OPP (算子包) 路径。

        环境变量: ASCEND_OPP_PATH
        用途: 算子编译时查找算子定义
        """
        env_val = os.environ.get("ASCEND_OPP_PATH")
        if env_val:
            return Path(env_val)
        return None

    @property
    def npu_smi_path(self) -> Optional[Path]:
        """npu-smi 工具路径 (NPU 设备管理, 类似 nvidia-smi)。

        标准路径: ${ASCEND_TOOLKIT_HOME}/bin/npu-smi
        """
        candidates = [
            shutil.which("npu-smi"),
        ]
        if self.ascend_toolkit_home:
            candidates.append(
                str(self.ascend_toolkit_home / "bin" / "npu-smi")
            )
        for c in candidates:
            if c and Path(c).exists():
                return Path(c)
        return None

    # ── 环境摘要 ──────────────────────────────────────────────────────────────
    def check_ascend_env(self) -> Dict[str, bool]:
        """检查 Ascend 环境是否完整可用。

        返回:
          {
            'cann_installed': bool,      # CANN toolkit 是否安装
            'msprof_available': bool,    # msprof 是否可用
            'compiler_available': bool,  # 编译器是否可用
            'npu_smi_available': bool,   # npu-smi 是否可用
            'npu_accessible': bool,      # NPU 设备是否可访问
            'is_910b3_server': bool,     # 是否在 910B3 服务器上
          }
        """
        result = {
            "cann_installed": self.ascend_home is not None,
            "msprof_available": self.msprof_path is not None,
            "compiler_available": self.ascend_compiler_bin is not None,
            "npu_smi_available": self.npu_smi_path is not None,
            "npu_accessible": False,
            "is_910b3_server": self.env == Environment.ASCEND_910B3,
        }

        # 检查 NPU 是否可访问 (尝试运行 npu-smi)
        if self.npu_smi_path:
            try:
                import subprocess
                result["npu_accessible"] = subprocess.call(
                    [str(self.npu_smi_path), "info"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10
                ) == 0
            except Exception:
                pass

        return result


# ═════════════════════════════════════════════════════════════════════════════════
#  硬件参数 (从 simulator.py 动态读取, 不硬编码)
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareParams:
    """910B3 硬件参数——从 simulator.py 的 SATURATION_PARAMS + MEMORY_CAPACITY_KB
    动态读取, 确保与 cost model 保持同步。

    不在此文件中硬编码任何数值——单一数据源原则:
      costModel/cost_emulator/simulator.py 是唯一权威的硬件参数来源。
    """

    saturation_params: Dict[int, Dict[str, float]] = field(default_factory=dict)
    memory_capacity_kb: Dict[str, Optional[float]] = field(default_factory=dict)
    engine_names: Dict[int, str] = field(default_factory=dict)
    engine_for: Dict[str, int] = field(default_factory=dict)

    # 从 bench_910b3_paths.py 实测的额外参数 (simulator 里没有的)
    n_ai_cores: int = 20          # AI Core 数 (transfer)
    n_vec_cores: int = 40         # Vec Core 数 (compute)
    freq_ghz: float = 1.8         # 核心频率
    ub_kb_per_core: int = 192     # UB per core (来自 bench, 比 simulator 的 512KB 更可信)
    l2_mb_shared: int = 192       # L2 共享缓存

    _simulator_module: Any = None

    def load_from_simulator(self, simulator_path: Path):
        """从 simulator.py 动态导入并读取硬件参数。

        读取:
          - SATURATION_PARAMS: 7 个引擎的饱和度曲线 (vpeak/k0/peak_clamp)
          - MEMORY_CAPACITY_KB: UB/L1/L0/GM 容量
          - ENGINE_FOR: 操作名 → 引擎编号
          - ENG_NAME: 引擎编号 → 引擎名

        注意: simulator.py 不在 sys.path 上, 需要通过 importlib 加载。
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "simulator", str(simulator_path)
        )
        if spec is None or spec.loader is None:
            raise FileNotFoundError(
                f"Cannot load simulator from {simulator_path}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules["simulator"] = module
        spec.loader.exec_module(module)
        self._simulator_module = module

        self.saturation_params = module.SATURATION_PARAMS
        self.memory_capacity_kb = module.MEMORY_CAPACITY_KB
        self.engine_names = module.ENG_NAME
        self.engine_for = module.ENGINE_FOR

    def peak_bandwidth_gb_s(self, engine: int) -> float:
        """引擎的峰值带宽 (GB/s/核), 单核。"""
        return self.saturation_params[engine]["peak_clamp"]

    def aggregate_bandwidth_gb_s(self, engine: int) -> float:
        """引擎的聚合带宽 (GB/s), 多核并行。

        传输引擎用 AI Core 数 (20), VecUnit 用 Vec Core 数 (40)。
        CubeUnit 的并行核数待实测确认, 暂用 20。
        """
        if engine == 2:  # VecUnit: 40 Vec Cores
            return self.peak_bandwidth_gb_s(2) * self.n_vec_cores
        # transfer engines 0/1/3/4/6 和 cube engine 5: 20 AI Cores
        return self.peak_bandwidth_gb_s(engine) * self.n_ai_cores

    def get_engine_status(self, engine: int) -> str:
        """引擎校准状态: 'MEASURED' | 'PLACEHOLDER'"""
        p = self.saturation_params[engine]
        if engine in (0, 1, 2):
            return "MEASURED"
        return "PLACEHOLDER"

    def is_reliable(self, engine: int) -> bool:
        """该引擎的参数是否可靠 (可用于精确优化决策)。"""
        return self.get_engine_status(engine) == "MEASURED"


# ═════════════════════════════════════════════════════════════════════════════════
#  优化控制参数
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationParams:
    """优化循环控制参数——控制 Agent 的行为。

    所有阈值都可以通过环境变量覆盖:
      TRITON_AGENT_MAX_ROUNDS=300 python main.py ...
    """

    # ── 迭代控制 ──────────────────────────────────────────────────────────────
    max_rounds: int = 200
    """单 kernel 最大优化轮次, 防止无限循环。"""

    max_time_hours: float = 6.0
    """单 kernel 最大优化时间 (小时), 超时即停止。"""

    max_consecutive_reverts: int = 5
    """连续 Revert 次数超过此值 → 停止 (无改进空间)。"""

    min_speedup_threshold: float = 1.01
    """加速比低于此值 → Revert. 设为 >1.0 以避免噪音引起的假 Keep。"""

    target_speedup: float = 1.5
    """目标加速比, 达到即停止 (用户可通过 CLI --target-speedup 覆盖)。"""

    peak_threshold: float = 0.90
    """性能达到理论峰值的 90% → 停止 (接近硬件极限, 再优化收益小)。"""

    plateau_rounds: int = 10
    """连续 N 轮加速比波动小于 plateau_variance → 平台期, 停止。"""

    plateau_variance: float = 0.02
    """平台期检测的波动阈值 (2%)。"""

    # ── 策略晋升 ──────────────────────────────────────────────────────────────
    tier_promotion_threshold: int = 3
    """同一策略层级连续 N 轮无改进 → 晋升到下一层级。"""

    strategy_tiers: Dict[int, str] = field(default_factory=lambda: {
        1: "Algorithmic Structure",
        2: "Operator Fusion",
        3: "Tiling & Block Config",
        4: "Memory Access & Coalescing",
        5: "Compute & Occupancy",
        6: "910B3 Architecture-Specific",
    })
    """6 层优化策略定义 (从结构影响最大到最小)。详见 docx/OPTIMIZATION_METHODOLOGY.md"""

    # ── Emulator 重试 ─────────────────────────────────────────────────────────
    emulator_retry_max: int = 3
    """CPU Emulator 验证失败后最大重试次数 (每次重试把错误信息喂回 Coder)。"""

    # ── 瓶颈诊断 ──────────────────────────────────────────────────────────────
    bottleneck_time_ratio_threshold: float = 5.0
    """time_ratio < 此值的 op 不视为瓶颈 (占比太小, 不值得优化)。"""

    budget_exhausted_tier_threshold: int = 6
    """当所有 Tier 到达此值且无改进 → 策略耗尽, 停止。"""

    def __post_init__(self):
        """从环境变量覆盖默认值。"""
        overrides = {
            "TRITON_AGENT_MAX_ROUNDS": ("max_rounds", int),
            "TRITON_AGENT_MAX_TIME_HOURS": ("max_time_hours", float),
            "TRITON_AGENT_MAX_CONSECUTIVE_REVERTS": ("max_consecutive_reverts", int),
            "TRITON_AGENT_MIN_SPEEDUP": ("min_speedup_threshold", float),
            "TRITON_AGENT_TARGET_SPEEDUP": ("target_speedup", float),
            "TRITON_AGENT_PEAK_THRESHOLD": ("peak_threshold", float),
            "TRITON_AGENT_PLATEAU_ROUNDS": ("plateau_rounds", int),
            "TRITON_AGENT_PLATEAU_VARIANCE": ("plateau_variance", float),
            "TRITON_AGENT_TIER_PROMOTION": ("tier_promotion_threshold", int),
            "TRITON_AGENT_EMULATOR_RETRY": ("emulator_retry_max", int),
        }
        for env_var, (attr_name, converter) in overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(self, attr_name, converter(val))
                except (ValueError, TypeError):
                    pass  # 静默忽略无效值, 使用默认值


# ═════════════════════════════════════════════════════════════════════════════════
#  验证参数
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationParams:
    """验证三阶段参数 (CPU Emulator → Cost Simulator → 910B3 Hardware)。"""

    # ── Stage 1: CPU Emulator ─────────────────────────────────────────────────
    emulator_shapes: List[int] = field(default_factory=lambda: [
        1,         # 标量边界
        3,         # 小非对齐
        7,         # 小素数 (测试 mask 逻辑)
        64,        # 小典型值
        256,       # 中典型值
        512,       # 中典型值
        1024,      # 大典型值
        1025,      # 大非对齐 (1024+1)
        2049,      # 大非对齐 (2048+1)
        4096,      # 大值
        8192,      # 压力值
        65536,     # 极大值 (对应 128KB fp16)
    ])
    """多 shape 测试列表。借鉴 "The Correctness Illusion" 论文的
    op-schema-aware fuzzing 方法——单 shape 测试有盲区。"""

    emulator_dtypes: List[str] = field(default_factory=lambda: [
        "fp16",     # 主要精度
        "fp32",     # 高精度 (用于对照)
    ])
    """测试的 dtype 列表。bf16 暂不支持 (910B3 原生不支持 bf16)。"""

    emulator_tolerance: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "fp16": {"rtol": 1e-2, "atol": 1e-2},
        "fp32": {"rtol": 1e-5, "atol": 1e-5},
    })
    """数值验证容差。fp16 宽松 (3 位有效数字), fp32 严格。"""

    # ── Stage 2: Cost Simulator ───────────────────────────────────────────────
    simulator_timeout_seconds: int = 30
    """Simulator 调用超时 (秒)。复杂 DSL 程序的展开可能耗时。"""

    # ── Stage 3: 910B3 Hardware ────────────────────────────────────────────────
    hardware_warmup: int = 30
    """基准测试预热次数。"""

    hardware_repeat: int = 200
    """基准测试重复次数 (取中位数/均值)。"""

    hardware_timeout_minutes: int = 10
    """单次硬件基准测试超时 (分钟)。"""

    # ── 验证流程控制 ──────────────────────────────────────────────────────────
    skip_hardware_on_local: bool = True
    """本地环境 (无 NPU) 是否跳过 Stage 3。设为 False 会报错。"""

    require_simulator_before_hardware: bool = True
    """上板前是否必须先跑 simulator 预估。推荐 True (节省硬件时间)。"""

    def __post_init__(self):
        """从环境变量覆盖。"""
        overrides = {
            "TRITON_AGENT_HW_WARMUP": ("hardware_warmup", int),
            "TRITON_AGENT_HW_REPEAT": ("hardware_repeat", int),
            "TRITON_AGENT_HW_TIMEOUT_MIN": ("hardware_timeout_minutes", int),
        }
        for env_var, (attr_name, converter) in overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(self, attr_name, converter(val))
                except (ValueError, TypeError):
                    pass


# ═════════════════════════════════════════════════════════════════════════════════
#  上下文管理参数
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class ContextParams:
    """上下文窗口管理参数。

    三层上下文模型:
      Hot (热):  最近 N 轮完整保留 (代码 + diff + 结果 + 决策)
      Warm (温): N+1 ~ M 轮摘要保留 (策略 + 加速比 + 瓶颈类型)
      Cold (冷): M+1 轮以上仅关键数据点 (加速比, 瓶颈类型)
    """

    hot_window_size: int = 5
    """热窗口: 保留完整上下文的最近轮次数。"""

    warm_window_size: int = 15
    """温窗口: 保留摘要的轮次范围上限。超过此值进入冷层。"""

    max_context_tokens: int = 800_000
    """LLM 上下文窗口最大 token 数 (1M 窗口留 20% 余量)。
    估算公式: tokens ≈ len(text) / 2 (英文) 或 len(text) / 1.5 (中英混合)
    """

    max_cases_to_retrieve: int = 3
    """从经验库检索的最大案例数。"""

    max_playbook_sections_per_round: int = 2
    """每轮最多注入的 Playbook 章节数 (防止上下文膨胀)。"""

    def __post_init__(self):
        """从环境变量覆盖。"""
        overrides = {
            "TRITON_AGENT_HOT_WINDOW": ("hot_window_size", int),
            "TRITON_AGENT_WARM_WINDOW": ("warm_window_size", int),
            "TRITON_AGENT_MAX_CONTEXT_TOKENS": ("max_context_tokens", int),
        }
        for env_var, (attr_name, converter) in overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(self, attr_name, converter(val))
                except (ValueError, TypeError):
                    pass


# ═════════════════════════════════════════════════════════════════════════════════
#  日志与输出参数
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class OutputParams:
    """日志和输出控制参数。"""

    log_level: str = "INFO"
    """日志级别: DEBUG | INFO | WARNING | ERROR"""

    save_gantt_per_round: bool = False
    """是否每轮都保存 Gantt 流水图。默认 False (太大, 只在最终报告/debug 时生成)。"""

    save_all_rounds_dir: bool = True
    """是否保存每轮的 plan.md + diff.patch 到 rounds/ 目录。推荐 True。"""

    trajectory_chart_dpi: int = 150
    """优化轨迹图 DPI。"""

    trajectory_chart_format: str = "png"
    """优化轨迹图格式: png | svg | pdf"""

    def __post_init__(self):
        overrides = {
            "TRITON_AGENT_LOG_LEVEL": ("log_level", str),
            "TRITON_AGENT_SAVE_GANTT_PER_ROUND": ("save_gantt_per_round",
                                                    lambda v: v.lower() in ("true", "1", "yes")),
        }
        for env_var, (attr_name, converter) in overrides.items():
            val = os.environ.get(env_var)
            if val is not None:
                try:
                    setattr(self, attr_name, converter(val))
                except (ValueError, TypeError):
                    pass


# ═════════════════════════════════════════════════════════════════════════════════
#  总配置类 (所有模块通过此类访问配置)
# ═════════════════════════════════════════════════════════════════════════════════

class Config:
    """全局配置单例——聚合所有参数。

    用法:
        from config import config
        print(config.paths.simulator_path)
        print(config.optim.max_rounds)

    首次实例化时会:
      1. 自动检测运行环境 (Windows / Linux / 910B3 服务器)
      2. 从 simulator.py 动态加载硬件参数
      3. 创建必要的输出目录

    本地环境 (无 NPU) 也能正常实例化——硬件相关路径为 None,
    Stage 3 验证会自动跳过。
    """

    def __init__(self):
        # ── 平台路径 ──────────────────────────────────────────────────────────
        self.paths = PlatformPaths()

        # ── 硬件参数 ──────────────────────────────────────────────────────────
        self.hardware = HardwareParams()
        self._load_hardware_params()

        # ── 优化控制 ──────────────────────────────────────────────────────────
        self.optim = OptimizationParams()

        # ── 验证参数 ──────────────────────────────────────────────────────────
        self.verify = VerificationParams()

        # ── 上下文管理 ────────────────────────────────────────────────────────
        self.context = ContextParams()

        # ── 日志输出 ──────────────────────────────────────────────────────────
        self.output = OutputParams()

        # ── 环境摘要 ──────────────────────────────────────────────────────────
        self.env_info = self._build_env_info()

        # ── msprof simulator / 真机切换 ───────────────────────────────────────
        self._detect_msprof()

    def _detect_msprof(self):
        """检测 msprof 工具状态，自动决定用 simulator 还是真机。

        优先级:
          1. 环境变量 TRITON_AGENT_MSPROF_MODE = "simulator" | "hardware"
          2. 自动检测: 有 npu-smi → hardware, 有 msprof → simulator
        """
        import shutil

        # 用户强制指定
        mode = os.environ.get("TRITON_AGENT_MSPROF_MODE", "")
        if mode in ("simulator", "hardware"):
            self.msprof_mode = mode
            self.msprof_available = True
            return

        # 自动检测
        self.msprof_bin = shutil.which("msprof") or ""
        self.msprof_available = bool(self.msprof_bin)
        self.npu_available = False

        # 检测 NPU (npu-smi)
        npu_smi = shutil.which("npu-smi") or ""
        if npu_smi:
            try:
                r = subprocess.run([npu_smi, "info"], capture_output=True, timeout=10)
                self.npu_available = r.returncode == 0
            except Exception:
                pass

        if self.npu_available:
            self.msprof_mode = "hardware"
        elif self.msprof_available:
            self.msprof_mode = "simulator"
        else:
            self.msprof_mode = "none"

    def _load_hardware_params(self):
        """加载 910B3 硬件参数 (来自华为官方文档 + msprof 仿真验证)。"""
        # 直接使用硬编码的经过验证的参数
        self.hardware.saturation_params = {
            0: {"vpeak": 121.08, "k0": 6.65, "peak_clamp": 80.83},   # GM→UB
            1: {"vpeak": 190.19, "k0": 10.72, "peak_clamp": 76.67},  # UB→GM
            2: {"vpeak": 461.0,  "k0": 4.50, "peak_clamp": 404.0},   # VecUnit
            3: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # GM→L1
            4: {"vpeak": 100.0,  "k0": 6.65, "peak_clamp": 100.0},   # L1→L0
            5: {"vpeak": 150.0,  "k0": 0,    "peak_clamp": 150.0},   # CubeUnit
            6: {"vpeak": 37.5,   "k0": 6.65, "peak_clamp": 37.5},    # L0→GM
        }
        self.hardware.engine_names = {
            0: "GM→UB", 1: "UB→GM", 2: "VecUnit",
            3: "GM→L1", 4: "L1→L0", 5: "CubeUnit", 6: "L0→GM",
        }
        self.hardware.memory_capacity_kb = {
            "UB": 192.0, "L1": 2048.0, "L0": 1024.0, "GM": None,
        }

    def _build_env_info(self) -> Dict[str, Any]:
        """构建环境信息摘要 (打印或日志用)。"""
        return {
            "environment": self.paths.env,
            "python": self.paths.python_executable,
            "repo_root": str(self.paths.repo_root),
            "emulator_available": self.paths.emulator_common_path.exists(),
            "ascend_env": self.paths.check_ascend_env(),
            "hardware_params_loaded": len(self.hardware.saturation_params) > 0,
        }

    def print_info(self):
        """打印当前配置摘要 (用于确认环境是否正确)。"""
        info = self.env_info
        print(f"Environment:       {info['environment']}")
        print(f"Python:            {info['python']}")
        print(f"Repo root:         {info['repo_root']}")
        print(f"Simulator:         {'[OK]' if info['simulator_available'] else '[MISSING]'}")
        print(f"Emulator:          {'[OK]' if info['emulator_available'] else '[MISSING]'}")
        print(f"Hardware params:   {'[OK]' if info['hardware_params_loaded'] else '[FAIL]'}")

        asc = info["ascend_env"]
        if asc["is_910b3_server"]:
            print(f"Ascend CANN:       {'[OK]' if asc['cann_installed'] else '[MISSING]'}")
            print(f"msprof:            {'[OK]' if asc['msprof_available'] else '[MISSING]'}")
            print(f"Compiler:          {'[OK]' if asc['compiler_available'] else '[MISSING]'}")
            print(f"NPU accessible:    {'[OK]' if asc['npu_accessible'] else '[NO]'}")
        else:
            print("Ascend server:     (not detected — Stage 3 will be skipped)")

        print(f"\nOptimization config:")
        print(f"  max_rounds={self.optim.max_rounds}, "
              f"target_speedup={self.optim.target_speedup}x, "
              f"max_time={self.optim.max_time_hours}h")
        print(f"  context: hot={self.context.hot_window_size}, "
              f"warm={self.context.warm_window_size}, "
              f"max_tokens={self.context.max_context_tokens:,}")

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典 (用于 JSON 输出或日志记录)。"""
        return {
            "environment": self.paths.env,
            "paths": {
                "repo_root": str(self.paths.repo_root),
                "simulator": str(self.paths.simulator_path),
                "emulator": str(self.paths.emulator_root),
                "output_dir": str(self.paths.output_dir),
            },
            "hardware_params_loaded": len(self.hardware.saturation_params) > 0,
            "optimization": {
                "max_rounds": self.optim.max_rounds,
                "target_speedup": self.optim.target_speedup,
                "max_time_hours": self.optim.max_time_hours,
                "strategy_tiers": self.optim.strategy_tiers,
            },
            "context": {
                "hot_window": self.context.hot_window_size,
                "warm_window": self.context.warm_window_size,
                "max_tokens": self.context.max_context_tokens,
            },
            "ascend_env": self.paths.check_ascend_env(),
        }


# ═════════════════════════════════════════════════════════════════════════════════
#  全局单例 (import 即用)
# ═════════════════════════════════════════════════════════════════════════════════

config = Config()


# ═════════════════════════════════════════════════════════════════════════════════
#  便捷导出 (向后兼容 from config import XXX 的写法)
# ═════════════════════════════════════════════════════════════════════════════════

# 路径
paths   = config.paths
simulator_path = config.paths.simulator_path
emulator_root  = config.paths.emulator_root
output_dir     = config.paths.output_dir
rounds_dir     = config.paths.rounds_dir

# 硬件
hardware = config.hardware

# 优化参数
optim = config.optim

# 验证参数
verify = config.verify

# 上下文参数
context = config.context

# 输出参数
output = config.output


# ═════════════════════════════════════════════════════════════════════════════════
#  自检入口
# ═════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config.print_info()
    print(f"\nFull config dict available via: config.to_dict()")
