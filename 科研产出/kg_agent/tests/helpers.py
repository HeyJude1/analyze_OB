"""测试共用夹具：进程内只跑一次离线 build，输出到临时目录（不污染仓库 data/）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from kgagent import validate
from kgagent.config import load_config

KG_ROOT = Path(__file__).resolve().parents[1]

_CACHE = {}


def config():
    if "cfg" not in _CACHE:
        _CACHE["cfg"] = load_config(KG_ROOT / "config.yaml")
    return _CACHE["cfg"]


def built():
    """返回 (cfg, data_dir, report)；首次调用执行一次 build --offline。"""
    if "data_dir" not in _CACHE:
        cfg = config()
        data_dir = Path(tempfile.mkdtemp(prefix="kgagent_test_")) / "data"
        report = validate.build(cfg, mode="offline", data_dir=data_dir)
        _CACHE["data_dir"] = data_dir
        _CACHE["report"] = report
    return config(), _CACHE["data_dir"], _CACHE["report"]


def fresh_data_dir(name: str = "data") -> Path:
    return Path(tempfile.mkdtemp(prefix="kgagent_test_")) / name
