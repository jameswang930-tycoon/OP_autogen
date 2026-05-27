# Emulator Error Coverage

## Scoring

| Dimension | Score |
|-----------|-------|
| Line number localization | 4/5 — traceback is reliable, lacks column info |
| Runtime error detection | 5/5 — full coverage of load/store/dot core APIs |
| Numeric anomaly awareness | 4/5 — NaN/Inf/Zero flags practical, no root cause |
| Multi-program dedup | 5/5 |
| Race / concurrency semantics | 1/5 — CPU serial execution cannot expose store races |
| Triton API completeness | 3/5 — common APIs sufficient, 2D block ptr/scan/sort missing |
| LLM readability | 5/5 |

## High-risk blind spots (P0)

1. **Store write overlap** — multiple pids writing to the same address is completely silent under serial execution
2. **Grid coverage gaps** — emulator does not check whether grid covers all data elements
3. **Silent numerical errors** (~30-40%) — stride swap, pid decode errors, wrong index formulas

## Common semantic differences (P1)

- constexpr power-of-2 not validated
- bfloat16 mapped to float32 (precision promoted)
- atomic_add serial execution cannot expose races

## Completely uncovered APIs

tl.make_block_ptr/advance, 2D offsets, scan, histogram/sort, extern_elementwise, Pipeline, warp-level primitives.
