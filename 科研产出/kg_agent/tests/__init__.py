"""测试包初始化：把 kgagent 包（src/）加入 sys.path，并定位 kg_agent 根目录。

这样在 kg_agent/ 下直接 `python -m unittest discover` 即可，无需设置 PYTHONPATH。
"""

import sys
from pathlib import Path

KG_ROOT = Path(__file__).resolve().parents[1]
SRC = KG_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
