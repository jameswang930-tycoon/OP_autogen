# Test Conventions

## 目录策略

- `emulators/` 根目录 — 基础算子（add、matmul 等），作为 API 稳定性参考
- `emulators/test/` — 新开发的复杂场景算子（ResNet 集成、GCN 等）

新算子开发统一使用 `emulators/test/<op>/`。

## Import 路径

```python
# emulators/test/<op>/__init__.py 的标准头部：
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # → emulators/（加载 common）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))          # → emulators/test/（加载兄弟算子）

# 加载 common:
from common import tl, launch_kernel_1d, verify, ...

# 加载兄弟算子:
from test.matmul import emulate_matmul
from test.relu import emulate_relu
```

## 测试运行

```bash
# 单个算子:
cd emulators && python3 -c "from test.<op> import test; test()"

# 全量:
cd emulators && python3 test/run_all_tests.py

# 需要 torch 的 reference:
../.venv/bin/python3 -c "from test.<op> import test; test()"
```

## 算子注册

新算子通过后在 `emulators/test/run_all_tests.py` 中添加 import 和调用。
