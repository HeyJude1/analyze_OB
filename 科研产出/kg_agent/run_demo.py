"""一键 demo：build --offline -> validate -> retrieve -> export-viz，并打印可视化打开方式。

用法（任意目录）：
    python 科研产出/kg_agent/run_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run(args: list) -> int:
    command = [sys.executable, "-m", "kgagent.cli"] + args
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    print("\n$ python -m kgagent.cli " + " ".join(args))
    return subprocess.call(command, cwd=str(SRC), env=env)


def main() -> int:
    steps = [
        ["build", "--offline"],
        ["validate"],
        ["retrieve", "--op", "dgemm", "--profile", "x86_avx2"],
        ["retrieve", "--op", "dgemm", "--profile", "k1_rvv256"],
        ["retrieve", "--op", "dgemm", "--profile", "arm_neon"],
        ["stats", "--matrix"],
        ["export-viz"],
    ]
    for args in steps:
        code = run(args)
        if code != 0:
            print("\n[FAIL] 步骤失败：%s（退出码 %d）" % (" ".join(args), code))
            return code
    print("\n[OK] 全链路跑通（多架构：x86 AVX2/AVX512、ARM NEON/SVE、RISC-V RVV）。")
    print("     可视化：双击打开 %s" % (ROOT / "viz" / "graph.html"))
    print("     检索 trace 回放数据：%s" % (ROOT / "viz" / "traces.js"))
    print("     架构覆盖矩阵：%s" % (ROOT / "data" / "arch_coverage.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
