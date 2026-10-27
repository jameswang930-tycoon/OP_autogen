"""分析层 — v4 采集链 (仅保留活跃模块).

各子模块职责:
  - integrate.py           task.json + board_*.json → diagnosis.json (roofline 诊断中枢)
  - merge_single_file.py   旧式三文件 op 目录 → 单文件 kernel_op.py (旧 op 一次性合并)
  - pipeline_parse_task.py 通用 msprof → task.json (骨架: 每kernel耗时/launch/api)
  - pipeline_parse_board.py  msprof op → board_*.json (深层: 带宽/引擎/算力/conflict)
  - pipeline_schema.py     公共 schema + json 读写工具
  - check_fields.py        字段完整性校验
  - filter_hivm_for_fusion.py HIVM 文本过滤
  - run_hivm_fusion.py     HIVM 融合分析入口
  - sweep_blocks.py        Tier3 分块扫描 (L0 合法 BLOCK 枚举 + Event 实测)
  - hivmir_analyzer.py     HIVM MLIR 解析 (类在内部使用，外部不通过本 __init__ 调用)
  - run_optimize.sh        采集驱动脚本 (被 Scheduler._run_optimize 调用)

各模块在 __init__ 不导出，外部统一通过包路径 import（如
`analyzers.integrate.integrate`、`analyzers.sweep_blocks.sweep`）。
HIVMIRAnalyzer 等类仅在包内被 hivmir_analyzer 的同层 import 用，不通过本 __init__。
"""
