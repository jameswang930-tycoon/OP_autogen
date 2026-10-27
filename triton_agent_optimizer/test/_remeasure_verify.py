# -*- coding: utf-8 -*-
"""V6: feedback/remeasure_best.py — mock 全流程 (假 outputs + 假 kernel + 假 subprocess)."""
import json, os, sys, tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# ★V6b: --l2 模式 — mock msprof op 产出 L2Cache.csv (冷 0.2 / 热 0.9) → json 含命中率
import csv as _csv
def _fake_run_l2(cmd, *a, **kw):
    from types import SimpleNamespace
    if cmd[0] == "python3":   # Event 计时
        tag = "cold" if "cold" in str(cmd[1]) else "hot"
        us = 500.0 if tag == "cold" else 210.0
        return SimpleNamespace(stdout=f"EVENT_E2E_US:{us:.2f}\n", stderr="")
    # msprof op: 写假 L2Cache.csv
    outdir = Path(cmd[cmd.index("--output=") + 1]) if "--output=" in cmd else None
    o = Path([c for c in cmd if c.startswith("--output=")][0].split("=", 1)[1])
    opprof = o / "OPPROF_1"
    opprof.mkdir(parents=True, exist_ok=True)
    hit = 0.2 if "cold" in str(cmd[-1]) else 0.9
    with open(opprof / "L2Cache.csv", "w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["block_id", "aic_total_hit_rate(%)"])
        w.writerow(["0", f"{hit * 100:.1f}"])
    return SimpleNamespace(stdout="ok", stderr="")

# 删掉之前 mock 生成的 json, 让 --l2 重新测
import shutil as _sh
_sh.rmtree(out_root / "matmul" / "final_output", ignore_errors=True)
with mock.patch("subprocess.run", new=_fake_run_l2), \
     mock.patch.object(RB, "_PROJECT_DIR", tmp), \
     mock.patch.object(sys, "argv", ["remeasure_best.py", "--op", "matmul", "--l2"]):
    rc = RB.main()
check("V6b: --l2 退出码 0", rc == 0)
rs3 = json.loads((out_root / "matmul" / "final_output" / "remeasure_best.json").read_text(encoding="utf-8"))
check("V6b: json 含冷命中率 0.2", rs3.get("l2_hit_rate_cold") == 0.2, rs3.get("l2_hit_rate_cold"))
check("V6b: json 含热命中率 0.9", rs3.get("l2_hit_rate_hot") == 0.9, rs3.get("l2_hit_rate_hot"))
check("V6b: 命中率列保留耗时/虚高", rs3.get("cold_l2_us_industrial") == 500.0)

print("\n═══ V6 (remeasure_best) 全部通过 ═══")
