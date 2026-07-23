#!/usr/bin/env python3
"""
Triton Agent Optimizer - 主框架
基于 KernelAgent 和 AutoKernel 的设计思路
集成 DSL 流水线和 msprof 分析数据
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

# 添加 fusion_pipeline 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "fusion_pipeline"))

from complete_data_merge import (
    run_simulator_and_parse,
    HIVMIRParser,
    DataMerger,
    CompleteReportGenerator,
    CombinedOp,
)


@dataclass
class OptimizationResult:
    """优化结果"""
    iteration: int
    strategy: str
    speedup: float
    correctness: bool
    decision: str  # 'keep' / 'revert'
    bottleneck_before: Optional[str] = None
    bottleneck_after: Optional[str] = None


class PlannerAgent:
    """规划智能体 - 分析瓶颈并制定策略"""

    def analyze_bottleneck(self, combined_ops: List[CombinedOp]) -> Dict:
        """分析瓶颈操作"""
        if not combined_ops:
            return {'bottleneck_op': None, 'bottleneck_type': None, 'strategies': []}

        # 找到时间占比最大的操作
        max_op = max(combined_ops, key=lambda x: x.time_ratio)

        # 分类瓶颈类型
        bottleneck_type = self._classify_bottleneck(max_op)

        # 生成优化策略
        strategies = self._generate_strategies(max_op, bottleneck_type)

        return {
            'bottleneck_op': max_op,
            'bottleneck_type': bottleneck_type,
            'strategies': strategies
        }

    def _classify_bottleneck(self, op: CombinedOp) -> str:
        """分类瓶颈类型"""
        # 内存传输瓶颈
        if op.op_type in ['gm_to_ub', 'ub_to_gm', 'gm_to_l1', 'l1_to_l0', 'l0_to_gm']:
            if op.bw_utilization < 0.5:
                return 'memory_latency'  # 带宽未饱和
            else:
                return 'memory_bandwidth'  # 带宽饱和

        # 计算瓶颈
        elif op.op_type in ['vadd', 'vsub', 'vmul']:
            return 'compute_vec'

        elif op.op_type == 'matrixmul':
            return 'compute_cube'

        # 依赖瓶颈
        elif len(op.dependencies) > 2:
            return 'dependency'

        return 'unknown'

    def _generate_strategies(self, op: CombinedOp, bottleneck_type: str) -> List[str]:
        """生成优化策略"""
        strategies = []

        if bottleneck_type == 'memory_latency':
            # 带宽未饱和，可以增加 tile size 或合并传输
            strategies.extend([
                'increase_tile_size',
                'merge_transfers',
                'prefetch_memory'
            ])

        elif bottleneck_type == 'memory_bandwidth':
            # 带宽饱和，需要减少数据传输或使用更快引擎
            strategies.extend([
                'fusion_eliminate_intermediate',
                'use_faster_memory_level',
                'optimize_data_layout'
            ])

        elif bottleneck_type == 'compute_vec':
            strategies.extend([
                'optimize_vectorization',
                'increase_parallelism'
            ])

        elif bottleneck_type == 'compute_cube':
            strategies.extend([
                'optimize_matmul_tiling',
                'use_tensor_cores'
            ])

        elif bottleneck_type == 'dependency':
            # 依赖过多，考虑融合或并行化
            strategies.extend([
                'fusion_break_dependency',
                'parallelize_independent_ops'
            ])

        return strategies

    def select_strategy(self, bottleneck_info: Dict, iteration: int) -> str:
        """选择当前迭代的策略"""
        strategies = bottleneck_info.get('strategies', [])

        if not strategies:
            return 'none'

        # 根据迭代次数选择不同策略
        # 前期优先基础优化，后期尝试高级策略
        if iteration < 10:
            # 前期：基础优化
            priority_order = [
                'increase_tile_size',
                'optimize_vectorization',
                'merge_transfers'
            ]
        elif iteration < 30:
            # 中期：融合优化
            priority_order = [
                'fusion_eliminate_intermediate',
                'fusion_break_dependency',
                'optimize_matmul_tiling'
            ]
        else:
            # 后期：高级优化
            priority_order = [
                'use_faster_memory_level',
                'use_tensor_cores',
                'optimize_data_layout'
            ]

        # 选择优先级最高的可用策略
        for strategy in priority_order:
            if strategy in strategies:
                return strategy

        # 如果优先策略都不可用，返回第一个可用策略
        return strategies[0] if strategies else 'none'


class CoderAgent:
    """编码智能体 - 应用优化策略"""

    def __init__(self):
        self.strategy_handlers = {
            'increase_tile_size': self._optimize_tile_size,
            'merge_transfers': self._merge_transfers,
            'fusion_eliminate_intermediate': self._apply_fusion,
            'optimize_vectorization': self._optimize_vectorization,
            'optimize_matmul_tiling': self._optimize_matmul_tiling,
            # 可以添加更多策略处理器
        }

    def apply_optimization(self, kernel_code: str, strategy: str,
                          bottleneck_info: Dict) -> str:
        """应用优化策略"""
        handler = self.strategy_handlers.get(strategy)

        if handler:
            return handler(kernel_code, bottleneck_info)
        else:
            # 默认：不修改
            return kernel_code

    def _optimize_tile_size(self, kernel_code: str, bottleneck_info: Dict) -> str:
        """优化 tile size"""
        bottleneck_op = bottleneck_info.get('bottleneck_op')

        if not bottleneck_op:
            return kernel_code

        # 找到当前的 BLOCK_SIZE 并增加
        import re

        # 查找 BLOCK_SIZE 定义
        pattern = r'BLOCK_SIZE\s*[:=]\s*(\d+)'
        match = re.search(pattern, kernel_code)

        if match:
            current_size = int(match.group(1))
            # 尝试增加到 2x 或 1.5x
            new_size = int(current_size * 1.5)

            # 替换
            optimized_code = re.sub(
                pattern,
                f'BLOCK_SIZE = {new_size}',
                kernel_code
            )

            return optimized_code

        return kernel_code

    def _merge_transfers(self, kernel_code: str, bottleneck_info: Dict) -> str:
        """合并相邻的内存传输"""
        # 这是一个简化的实现
        # 实际需要分析代码结构并进行合并
        return kernel_code

    def _apply_fusion(self, kernel_code: str, bottleneck_info: Dict) -> str:
        """应用算子融合"""
        # 这是一个简化的实现
        # 实际需要识别可以融合的算子并合并代码
        return kernel_code

    def _optimize_vectorization(self, kernel_code: str, bottleneck_info: Dict) -> str:
        """优化向量化"""
        return kernel_code

    def _optimize_matmul_tiling(self, kernel_code: str, bottleneck_info: Dict) -> str:
        """优化矩阵乘法 tiling"""
        return kernel_code


class VerifierAgent:
    """验证智能体 - 验证正确性和性能"""

    def __init__(self, test_cases: List[Dict] = None):
        self.test_cases = test_cases or []

    def verify(self, original_kernel: str, optimized_kernel: str) -> Dict:
        """验证优化结果"""
        # 1. 编译检查
        compile_success = self._compile_check(optimized_kernel)

        if not compile_success:
            return {
                'correctness': False,
                'speedup': 0.0,
                'decision': 'revert',
                'error': 'Compilation failed'
            }

        # 2. 正确性验证（简化版）
        correctness = self._verify_correctness(optimized_kernel)

        # 3. 性能测试（简化版）
        speedup = self._benchmark(optimized_kernel)

        # 4. 决策
        decision = 'keep' if correctness and speedup > 1.01 else 'revert'

        return {
            'correctness': correctness,
            'speedup': speedup,
            'decision': decision
        }

    def _compile_check(self, kernel_code: str) -> bool:
        """编译检查"""
        # 简化实现：检查语法错误
        try:
            compile(kernel_code, '<string>', 'exec')
            return True
        except SyntaxError:
            return False

    def _verify_correctness(self, kernel_code: str) -> bool:
        """正确性验证"""
        # 简化实现：假设通过
        # 实际应该运行测试用例并对比结果
        return True

    def _benchmark(self, kernel_code: str) -> float:
        """性能基准测试"""
        # 简化实现：返回固定加速比
        # 实际应该运行 kernel 并测量时间
        import random
        return random.uniform(0.9, 1.5)


class Orchestrator:
    """调度器 - 协调智能体协作"""

    def __init__(self,
                 kernel_file: str,
                 hivmir_file: Optional[str] = None,
                 max_iterations: int = 50):

        self.kernel_file = Path(kernel_file)
        self.hivmir_file = Path(hivmir_file) if hivmir_file else None
        self.max_iterations = max_iterations

        # 初始化智能体
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.verifier = VerifierAgent()

        # 加载初始 kernel
        self.original_kernel = self._load_kernel()
        self.current_kernel = self.original_kernel

        # 历史记录
        self.history: List[OptimizationResult] = []

    def run(self) -> Dict:
        """运行完整优化流程"""
        print("\n" + "=" * 80)
        print("Triton Agent Optimizer - 开始优化")
        print("=" * 80)

        for iteration in range(self.max_iterations):
            print(f"\n[Iteration {iteration + 1}/{self.max_iterations}]")

            # Step 1: 分析 DSL 和 msprof 数据
            print("  Step 1: 分析 DSL 流水线和性能数据...")
            dsl_pipeline = self._analyze_current_kernel()

            if not dsl_pipeline:
                print("  ✗ 分析失败")
                break

            # Step 2: 诊断瓶颈
            print("  Step 2: 诊断性能瓶颈...")
            bottleneck_info = self.planner.analyze_bottleneck(dsl_pipeline)

            if not bottleneck_info['bottleneck_op']:
                print("  ✓ 未发现瓶颈，优化完成")
                break

            print(f"     瓶颈: Op{bottleneck_info['bottleneck_op'].op_id} "
                  f"({bottleneck_info['bottleneck_op'].op_type})")
            print(f"     类型: {bottleneck_info['bottleneck_type']}")
            print(f"     占比: {bottleneck_info['bottleneck_op'].time_ratio:.2f}%")

            # Step 3: 选择优化策略
            print("  Step 3: 选择优化策略...")
            strategy = self.planner.select_strategy(bottleneck_info, iteration)
            print(f"     策略: {strategy}")

            # Step 4: 应用优化
            print("  Step 4: 应用优化...")
            optimized_kernel = self.coder.apply_optimization(
                self.current_kernel,
                strategy,
                bottleneck_info
            )

            # Step 5: 验证
            print("  Step 5: 验证优化结果...")
            result = self.verifier.verify(self.current_kernel, optimized_kernel)

            print(f"     正确性: {'✓' if result['correctness'] else '✗'}")
            print(f"     加速比: {result['speedup']:.2f}x")
            print(f"     决策: {result['decision']}")

            # 记录结果
            optimization_result = OptimizationResult(
                iteration=iteration + 1,
                strategy=strategy,
                speedup=result['speedup'],
                correctness=result['correctness'],
                decision=result['decision'],
                bottleneck_before=f"Op{bottleneck_info['bottleneck_op'].op_id}"
            )
            self.history.append(optimization_result)

            # Step 6: 更新状态
            if result['decision'] == 'keep':
                self.current_kernel = optimized_kernel
                print("  ✓ 保留优化")
            else:
                print("  ✗ 回退优化")

            # Step 7: 检查终止条件
            if self._should_stop(bottleneck_info, iteration):
                print("\n  终止条件满足，停止优化")
                break

        # 生成最终报告
        final_report = self._generate_final_report()

        print("\n" + "=" * 80)
        print("优化完成")
        print("=" * 80)

        return final_report

    def _load_kernel(self) -> str:
        """加载 kernel 代码"""
        with open(self.kernel_file, 'r', encoding='utf-8') as f:
            return f.read()

    def _analyze_current_kernel(self) -> Optional[List[CombinedOp]]:
        """分析当前 kernel"""
        # 运行 simulator
        dsl_program = self._extract_dsl_from_kernel(self.current_kernel)
        sim_ops, _, total_ns = run_simulator_and_parse(dsl_program)

        if not sim_ops:
            return None

        # 如果有 HIVMIR，解析它
        hivmir_ops = None
        if self.hivmir_file and self.hivmir_file.exists():
            parser = HIVMIRParser()
            hivmir_text = self.hivmir_file.read_text(encoding='utf-8')
            hivmir_ops = parser.parse(hivmir_text)

        # 合并数据
        merger = DataMerger()
        return merger.merge(sim_ops, hivmir_ops or [])

    def _extract_dsl_from_kernel(self, kernel_code: str) -> str:
        """从 kernel 代码提取 DSL"""
        # 简化实现：生成示例 DSL
        # 实际应该解析 kernel 代码并转换为 DSL
        return """
        alloc(gm_1, 128KB)
        alloc(ub_1, 128KB)
        alloc(ub_2, 128KB)
        gm_to_ub(ub_1, gm_1)
        vadd(ub_2, ub_1, 1.0)
        ub_to_gm(gm_2, ub_2)
        """

    def _should_stop(self, bottleneck_info: Dict, iteration: int) -> bool:
        """检查是否应该停止"""
        # 条件 1: 达到最大迭代次数
        if iteration >= self.max_iterations - 1:
            return True

        # 条件 2: 瓶颈占比很小（< 5%）
        bottleneck_op = bottleneck_info.get('bottleneck_op')
        if bottleneck_op and bottleneck_op.time_ratio < 5.0:
            return True

        # 条件 3: 连续多次无改进（检查历史）
        if len(self.history) >= 5:
            recent_results = self.history[-5:]
            if all(r.decision == 'revert' for r in recent_results):
                return True

        return False

    def _generate_final_report(self) -> Dict:
        """生成最终报告"""
        # 计算总体加速比
        keep_results = [r for r in self.history if r.decision == 'keep']

        total_speedup = 1.0
        for r in keep_results:
            total_speedup *= r.speedup

        return {
            'total_iterations': len(self.history),
            'successful_optimizations': len(keep_results),
            'total_speedup': total_speedup,
            'history': [
                {
                    'iteration': r.iteration,
                    'strategy': r.strategy,
                    'speedup': r.speedup,
                    'decision': r.decision
                }
                for r in self.history
            ]
        }


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Triton Agent Optimizer')
    parser.add_argument('--kernel', type=str, required=True, help='Triton kernel 文件')
    parser.add_argument('--hivmir', type=str, help='HIVMIR 文件（可选）')
    parser.add_argument('--max-iterations', type=int, default=50, help='最大迭代次数')
    parser.add_argument('--output', type=str, default='./optimization_result.json', help='输出文件')

    args = parser.parse_args()

    # 运行优化
    optimizer = Orchestrator(
        kernel_file=args.kernel,
        hivmir_file=args.hivmir,
        max_iterations=args.max_iterations
    )

    result = optimizer.run()

    # 保存结果
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()