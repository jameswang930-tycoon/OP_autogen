"""合成夹具（T5）：符合 Event 契约的假仿真事件，覆盖三种典型瓶颈形态。

作用：让 feedback_adapter 的全部逻辑在**无保密信息**的开发机上就能跑通、测过。
保密环境接手时只需实现 parse_raw() 把真实仿真输出转成同样的 Event 列表，
adapter 其余部分零改动。

三份夹具：
  COMPUTE_BOUND        —— 算力达峰，计算单元串行占满（bottleneck=compute_bound_at_peak）
  MEMORY_UNDERFILLED   —— 访存主导且带宽未填满（bottleneck=memory_underfilled）
  STALL_DEPENDENCY     —— 长串行依赖链，每段等上一段（bottleneck=stall_dependency）

字段语义见 control/contracts.py 的 Event。start/end/duration 单位为仿真 cycles。
"""
from ..contracts import Event

# 1) compute-bound：ALU 三段串行占满 200 cyc，访存小幅重叠在开头
COMPUTE_BOUND = [
    Event("alu_mac_0", 0, 60, 60, "ALU", "compute_bound_at_peak"),
    Event("alu_mac_1", 60, 120, 60, "ALU", "compute_bound_at_peak"),
    Event("alu_mac_2", 120, 200, 80, "ALU", "compute_bound_at_peak"),
    Event("mte_load", 0, 30, 30, "MTE", "memory_underfilled", bytes=4096),
]

# 2) memory-underfilled：两段访存串行占满 160 cyc（带宽未饱和），计算小幅重叠
MEMORY_UNDERFILLED = [
    Event("mte_gather_0", 0, 80, 80, "MTE", "memory_underfilled", bytes=2048),
    Event("mte_gather_1", 80, 160, 80, "MTE", "memory_underfilled", bytes=2048),
    Event("alu_fma", 0, 40, 40, "ALU", "compute_bound_at_peak"),
]

# 3) stall-dependency：load->compute->load->compute->store 长串行链，每段依赖上一段
STALL_DEPENDENCY = [
    Event("load_a", 0, 50, 50, "MTE", "stall_dependency", bytes=8192),
    Event("compute_a", 50, 90, 40, "ALU", "stall_dependency"),
    Event("load_b", 90, 140, 50, "MTE", "stall_dependency", bytes=8192),
    Event("compute_b", 140, 180, 40, "ALU", "stall_dependency"),
    Event("store_c", 180, 210, 30, "MTE", "stall_dependency", bytes=8192),
]

ALL_FIXTURES = {
    "compute_bound_at_peak": COMPUTE_BOUND,
    "memory_underfilled": MEMORY_UNDERFILLED,
    "stall_dependency": STALL_DEPENDENCY,
}
