# Implementation Patterns

Practical patterns and gotchas discovered while writing emulator-verified kernels.

## Scalar index load trick

`tl.load` requires array offsets, not scalars. To load a single element by index `pid`:

```python
# WRONG: pid is an int, tl.load expects array offsets
val = tl.load(ptr, pid)

# CORRECT: construct shape-(1,) offset via arange
val = tl.load(ptr, pid + tl.arange(0, 1))  # returns shape-(1,) xarray
scalar_val = int(val)  # extract scalar for use in range() or arithmetic
```

First encountered: 2026-05-27, gcn_spmm kernel. Needed to load row_start/row_end by node index.

## CSR preprocessing for sparse/graph operators

Sparse operators (SpMM, scatter-add) must pre-sort edges into CSR format **outside the kernel**.
The kernel only handles the dense inner loop over features.

```python
# Outside kernel (in emulate wrapper):
sorted_idx = np.argsort(edge_dst)
sorted_src = edge_src[sorted_idx]
row_start, row_end = build_csr(sorted_dst, N)

# Inside kernel:
for e in range(row_start, row_end):  # iterate edges for this node
    src = tl.load(edge_src_ptr, e + tl.arange(0, 1))
    feats = tl.load(node_feat_ptr, int(src) * F + f_off, mask=f_mask)
    acc += w * feats
```

Reason: kernel code must use only `tl.*` APIs and flat array access. CSR conversion requires
argsort/indexing that only numpy can do.

## Matmul interface: 2D arrays, not flat

`emulate_matmul(a, b)` expects 2D numpy arrays `[M, K]` and `[K, N]`, not flat arrays.
Do NOT flatten before calling.

```python
# WRONG:
h = emulate_matmul(x.ravel(), w.ravel(), M=N, K=F_in, N_dim=F_out)

# CORRECT:
h = emulate_matmul(x, weight)  # both are 2D
```

## Import path convention

Operators in `emulators/test/<op>/__init__.py` must set up two sys.path entries:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # for common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))          # for sibling ops
```

Importing a sibling operator: `from test.matmul import emulate_matmul` (not `from matmul import ...`).
The `test/` prefix is required because the second sys.path points to `emulators/`, and matmul lives at `emulators/test/matmul/`.

## BLOCK_F alignment for non-power-of-2 features

When features don't align to a power of 2, round BLOCK_F up:

```python
BLOCK_F = ((F + 15) // 16) * 16  # round up to multiple of 16
```

Then mask with `f_off < F` to avoid out-of-bounds access.
