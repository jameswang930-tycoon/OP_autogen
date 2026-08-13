# -*- coding: utf-8 -*-
"""V6: feedback/remeasure_best.py — mock 全流程 (假 outputs + 假 kernel + 假 subprocess)."""
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

tmp = Path(tempfile.mkdtemp())
# 假 outputs: 两个算子 (一个慢一个快)
out_root = tmp / "outputs"
for name, us in (("matmul", 500.0), ("vector_add", 80.0)):
    kd = out_root / name
    kd.mkdir(parents=True)
    (kd / "best_kernel.py").write_text(
        "import os, torch, triton, triton.language as tl\n"
        "N = 1024\n"
        "BLOCK = 256\n"
        "@triton.jit\n"
        "def k(x, y, o, BLOCK: tl.constexpr):\n"
        "    pass\n"
        "def main():\n"
        f"    x = torch.randn(N, device='npu')\n"
        f"    y = torch.randn(N, device='npu')\n"
        f"    o = torch.empty(N, device='npu')\n"
        "    for _ in range(LOOP):\n"
        "        k[(1,)](x, y, o, BLOCK=BLOCK)\n"
        "    torch.npu.synchronize()\n"
        "if __name__ == '__main__':\n"
        "    main()\n", encoding="utf-8")

def check(name, cond, detail=""):
    print(f"  {'OK ' if cond else 'FAIL'} {name} {detail}")
    assert cond, f"{name}: {detail}"

import feedback.remeasure_best as RB

def _fake_run(cmd, *a, **kw):
    from types import SimpleNamespace
    # 根据注入文件名判断 冷/热
    tag = "cold" if "cold" in str(cmd[1]) else "hot"
    us = 500.0 if tag == "cold" else 210.0    # matmul 冷 500, 热 210 (虚高 2.38x)
    if "vector_add" in str(cmd[1]):
        us = 80.0 if tag == "cold" else 25.0  # vector_add 冷 80, 热 25 (虚高 3.2x)
    return SimpleNamespace(stdout=f"EVENT_E2E_US:{us:.2f}\n", stderr="")

# 把输出目录指到假目录
with mock.patch("subprocess.run", new=_fake_run), \
     mock.patch.object(RB, "_PROJECT_DIR", tmp):
    rc = RB.main()

check("V6: 退出码 0", rc == 0)
rs = json.loads((out_root / "matmul" / "final_output" / "remeasure_best.json").read_text(encoding="utf-8"))
check("V6: matmul json 冷=500", rs["cold_l2_us_industrial"] == 500.0, rs)
check("V6: matmul json 热=210", rs["hot_l2_us_old_verify"] == 210.0)
check("V6: matmul 虚高 2.38x", abs(rs["l2_inflate_x"] - 2.38) < 0.01, rs["l2_inflate_x"])
rs2 = json.loads((out_root / "vector_add" / "final_output" / "remeasure_best.json").read_text(encoding="utf-8"))
check("V6: vector_add 虚高 3.2x", abs(rs2["l2_inflate_x"] - 3.2) < 0.01, rs2["l2_inflate_x"])
check("V6: json 含工业级方法说明", "do_bench" in rs["method"])

# --skip-existing: 第二次跑不重新测量
calls = []
def _fake_run2(cmd, *a, **kw):
    calls.append(cmd)
    from types import SimpleNamespace
    return SimpleNamespace(stdout="EVENT_E2E_US:1.00\n", stderr="")
with mock.patch("subprocess.run", new=_fake_run2), \
     mock.patch.object(RB, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["remeasure_best.py", "--skip-existing"]):
    RB.main()
check("V6: --skip-existing 不重跑", len(calls) == 0, len(calls))

# --op 过滤
calls2 = []
def _fake_run3(cmd, *a, **kw):
    calls2.append(cmd)
    from types import SimpleNamespace
    return SimpleNamespace(stdout="EVENT_E2E_US:1.00\n", stderr="")
with mock.patch("subprocess.run", new=_fake_run3), \
     mock.patch.object(RB, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["remeasure_best.py", "--op", "matmul"]):
    RB.main()
check("V6: --op 只测指定算子", len(calls2) == 2, len(calls2))  # cold+hot 各 1 次

# 无 outputs → 报错退出 1
with mock.patch.object(RB, "_PROJECT_DIR", tmp / "empty"):
    check("V6: 无 outputs 退出 1", RB.main() == 1)

print("\n═══ V6 (remeasure_best) 全部通过 ═══")
