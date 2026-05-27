# Emulator API Gaps

Triton Language features not covered by the CPU emulator.
See also: docs/emulator_error_coverage.md for detailed scoring.

## Completely unimplemented APIs

| API | Priority | Use case |
|-----|----------|----------|
| tl.make_block_ptr / tl.advance | P0 | 2D tiled access (common in matmul, conv) |
| 2D offset patterns | P0 | Multi-dimensional block pointer arithmetic |
| tl.scan (prefix sum) | P1 | Sorted array, cumsum operations |
| tl.sort / tl.histogram | P2 | Rare in inference kernels |
| tl.extern_elementwise | P2 | Custom external functions |

## Semantic differences from real Triton

| Aspect | Emulator | Real Triton | Impact |
|--------|----------|-------------|--------|
| tl.sum/max/min | keepdims=True | keepdims=False | Kernel code needs .reshape() workaround |
| tl.float16 | Maps to float32 | True fp16 | All fp16 precision issues hidden |
| atomic_add | Serial (no races) | Parallel (race conditions possible) | Store races completely silent |
| Grid coverage | Not checked | Must cover all elements | Missing elements undetected |
| constexpr power-of-2 | Not validated | Required for some ops | Invalid BLOCK_SIZE passes silently |

## Recommended API additions

1. **tl.make_block_ptr** — enables 2D block pointer patterns used in real Triton kernels
2. **Grid coverage check** — warn when grid * BLOCK_SIZE < total_elements
3. **Store overlap detection** — flag when multiple programs write to same address
