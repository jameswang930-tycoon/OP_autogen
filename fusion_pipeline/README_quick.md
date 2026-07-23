# Fusion Pipeline

完整的算子融合分析流水线，整合 HIVMIR 和 msprof op simulator 数据。

## 快速开始

```bash
# 运行示例
python fusion_pipeline/run_example.py

# 或使用自定义参数
python fusion_pipeline/complete_data_merge.py --help
```

## 输出示例

运行后会生成：

1. **详细文本报告** (`example_report.txt`)
2. **可视化图表** (`example_analysis.png`)

包含：
- 每个操作的详细信息
- 时间占比分析
- 依赖关系图
- 引擎利用率统计

## 完整文档

参见 [README.md](README.md)