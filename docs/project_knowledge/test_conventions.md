# Test Conventions

## Directory Strategy

- `emulators/common/` — Triton API emulator infrastructure (tl, launch_kernel, verify, etc.)
- `emulators/test/` — all operators (basic + integration), each as `<op>/`

All operator development goes into `emulators/test/<op>/`.

## Import Paths

```python
# Standard header in emulators/test/<op>/__init__.py:
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # -> emulators/ (loads common)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))          # -> emulators/test/ (loads sibling ops)

# Load common:
from common import tl, launch_kernel_1d, verify, ...

# Load sibling operator:
from test.matmul import emulate_matmul
from test.relu import emulate_relu
```

## Running Tests

```bash
# Single operator:
cd emulators && python3 -c "from test.<op> import test; test()"

# All tests:
cd emulators && python3 test/run_all_tests.py

# With torch reference (needs .venv):
../.venv/bin/python3 -c "from test.<op> import test; test()"
```

## Weight Policy

Kernel correctness tests use **random weights**, not pretrained weights.

- Triton kernel output vs PyTorch reference output must use the **same weight dict within one test run**.
- No cross-environment weight reuse, no pretrained model download, no torchvision dependency.
- Random weights cover all computation paths equally well. The goal is verifying operator correctness, not end-to-end inference accuracy.

## Operator Registration

After an operator passes all tests, add its import and call to `emulators/test/run_all_tests.py`.
