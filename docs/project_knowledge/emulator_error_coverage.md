# Emulator Error Coverage

## 评分

| 维度 | 评分 |
|------|------|
| 行号定位 | ★★★★☆ traceback 精确可靠，缺 column 信息 |
| 运行时错误检测 | ★★★★★ load/store/dot 核心 API 全覆盖 |
| 数值异常感知 | ★★★★☆ NaN/Inf/Zero flag 实用，无 root cause |
| 多 program 去重 | ★★★★★ |
| 竞争/并发语义 | ★☆☆☆☆ CPU 串行无法暴露 store race |
| Triton API 完整度 | ★★★☆☆ 常用够用，2D block ptr/scan/sort 缺失 |
| LLM 可读性 | ★★★★★ |

## 高风险盲区（P0）

1. **store 写重叠** — 多 pid 写同一位置，CPU 串行完全静默
2. **grid 覆盖率不足** — emulator 不检查 grid 是否覆盖全量数据
3. **静默数值错误**（~30-40%） — stride 互换/pid 解码/索引公式错误

## 常见语义差异（P1）

- constexpr power-of-2 不校验
- bfloat16 映射为 float32 精度被提升
- atomic_add CPU 串行无法暴露竞争

## 完全不覆盖的 API

tl.make_block_ptr/advance、2D offsets、scan、histogram/sort、extern_elementwise、Pipeline、warp-level primitives。
