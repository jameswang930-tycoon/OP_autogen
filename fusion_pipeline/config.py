"""
配置文件 - 融合分析流水线
"""

# 华为昇腾 910B3 配置
NPU_CONFIG = {
    'name': 'Ascend 910B3',
    'architecture': 'arm64',
    'ai_cores': 20,  # AI Core 数量
    'vec_cores': 40,  # Vec Core 数量
    'frequency_ghz': 1.8,  # 频率

    # 内存配置
    'memory': {
        'ub_size_kb': 192,  # Unified Buffer 大小
        'l2_size_mb': 192,  # L2 Cache 大小
        'hbm_size_gb': 32,  # HBM 大小
    },

    # 带宽配置（字节/周期/核心）
    'bandwidth': {
        'ddr_read': 32,
        'ddr_write': 32,
        'l2_read': 110,
        'ub_to_l2': 64,
    },
}

# Simulator 引擎配置
ENGINE_CONFIG = {
    'engines': {
        0: {'name': 'GM→UB', 'peak_gb_s': 1500},
        1: {'name': 'UB→GM', 'peak_gb_s': 750},
        2: {'name': 'VecUnit', 'peak_gb_s': 16000},
        3: {'name': 'GM→L1', 'peak_gb_s': 750},
        4: {'name': 'L1→L0', 'peak_gb_s': 2000},
        5: {'name': 'CubeUnit', 'peak_gb_s': 3000},
        6: {'name': 'L0→GM', 'peak_gb_s': 750},
    },
}

# HIVMIR 解析配置
HIVMIR_CONFIG = {
    # 操作类型到引擎的映射
    'op_to_engine': {
        'gm_to_ub': 'GM→UB',
        'ub_to_gm': 'UB→GM',
        'gm_to_l1': 'GM→L1',
        'l1_to_l0': 'L1→L0',
        'l0_to_gm': 'L0→GM',
        'vadd': 'VecUnit',
        'vsub': 'VecUnit',
        'vmul': 'VecUnit',
        'vdiv': 'VecUnit',
        'matrixmul': 'CubeUnit',
        'batchmatmul': 'CubeUnit',
    },

    # 缓冲区前缀到内存区域的映射
    'buffer_region': {
        'gm': 'Global Memory (HBM)',
        'ub': 'Unified Buffer',
        'l1': 'L1 SRAM',
        'l0': 'L0 Register File',
    },
}

# 报告生成配置
REPORT_CONFIG = {
    'output_dir': './fusion_analysis_output',
    'text_report': {
        'filename': 'complete_fusion_report.txt',
        'encoding': 'utf-8',
    },
    'visualization': {
        'filename': 'complete_fusion_analysis.png',
        'dpi': 300,
        'figsize': (18, 12),
        'top_n': 10,  # 饼图和时序图显示的 Top N 操作
    },
    'json_output': {
        'enabled': True,
        'filename': 'operations.json',
    },
}

# msprof 配置（在 910B3 服务器上使用）
MSPROF_CONFIG = {
    'msprof_path': '/usr/local/Ascend/ascend-toolkit/latest/toolkit/python/site-packages/msprof',
    'output_dir': './prof_data',

    # msprof 命令选项
    'options': {
        'op-mode': True,  # 算子级性能分析
        'timeline': True,  # 时序数据
        'memory': True,  # 内存分析
    },
}

# 编译器配置（用于提取 HIVMIR）
COMPILER_CONFIG = {
    'bishengir_compile': 'bishengir-compile',

    # MLIR 调试选项
    'mlir_options': {
        'print_ir_before_all': False,
        'print_ir_after_all': True,
        'print_ir_after_change': True,
        'print_ir_after_failure': True,
    },

    # 输出选项
    'output_ir': True,
    'ir_format': 'mlir',  # 或 'llvm-ir'
}

# 性能基准配置
BENCHMARK_CONFIG = {
    'warmup_iterations': 30,
    'repeat_iterations': 200,
    'data_size_mb': 256,  # 测试数据大小

    # 支持的数据类型
    'supported_dtypes': ['fp16', 'fp32', 'bf16'],

    # Tile 大小扫描范围
    'tile_sizes': [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768],
}