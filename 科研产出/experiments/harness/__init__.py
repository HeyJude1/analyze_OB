"""experiments/harness：实验执行基础设施（QEMU harness、KCS 静态检查等）。

用法（从 experiments/ 目录的任意脚本）：
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/
    from harness.qemu import compile_run
"""

from .qemu import compile_run, parse_driver_output, read_task_meta, win_to_wsl

__all__ = ["compile_run", "parse_driver_output", "read_task_meta", "win_to_wsl"]
