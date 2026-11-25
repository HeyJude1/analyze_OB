"""配置加载（config.yaml）与路径解析。

路径规则（避免把绝对路径写进仓库）：
- `paths.workspace_root` 相对 config.yaml 所在目录（即 kg_agent/）解析；
- `paths.analysis_results / operator_corpus / log_summary` 相对 workspace_root 解析；
- `paths.data_dir / viz_dir` 相对 config.yaml 所在目录（kg_agent/）解析。
环境变量 `KG_WORKSPACE_ROOT` 可覆盖 workspace_root，便于复现包换机器。
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from . import yamlmini

__all__ = ["Config", "load_config"]


class Config:
    """只读配置对象 + 解析后的绝对路径。"""

    def __init__(self, raw: dict, config_path: Path):
        self.raw = raw
        self.config_path = config_path
        self.root = config_path.parent  # kg_agent/
        paths = raw.get("paths", {}) or {}
        ws = os.environ.get("KG_WORKSPACE_ROOT") or paths.get("workspace_root", "../..")
        self.workspace_root = (self.root / ws).resolve()
        data_dir = os.environ.get("KG_DATA_DIR") or paths.get("data_dir", "data")
        self.data_dir = (self.root / data_dir).resolve()
        self.viz_dir = (self.root / paths.get("viz_dir", "viz")).resolve()
        self.analysis_dir = (self.workspace_root / paths["analysis_results"]).resolve()
        self.corpus_dir = (self.workspace_root / paths["operator_corpus"]).resolve()
        self.log_summary = (self.workspace_root / paths["log_summary"]).resolve()
        self.schema_dir = (self.root / "schema").resolve()
        self.schema_version = raw.get("schema_version", "kg-v1.5")
        self.extractor_version = raw.get("extractor_version", "kgagent-v1.5")
        self.prompt_version = raw.get("prompt_version", "s3-rules-v1")
        self.model_version = raw.get("model_version", "offline-rules-v1")

    # ---- 便捷访问 ----
    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def corpus(self) -> dict:
        return self.raw.get("corpus", {}) or {}

    @property
    def operators(self) -> dict:
        return self.corpus.get("operators", {}) or {}

    @property
    def analysis_operators(self) -> list:
        return list(self.corpus.get("analysis_operators", []) or [])

    @property
    def evidence_cfg(self) -> dict:
        return self.raw.get("evidence", {}) or {}

    def for_split(self, split: str, data_dir: Path) -> Config:
        """Filter sources before feature extraction and graph derivation."""
        definition = self.raw.get("splits", {}).get(split)
        if not definition:
            raise ValueError("未知构建划分：%s" % split)
        raw = deepcopy(self.raw)
        allow = set(definition["operators"])
        unknown = allow - set(self.all_operator_specs)
        if unknown:
            raise ValueError("划分 %s 引用了未登记的算子：%s" % (split, sorted(unknown)))
        raw["corpus"]["operators"] = {op: spec for op, spec in self.operators.items() if op in allow}
        raw["corpus"]["analysis_operators"] = []
        raw["evidence"]["enabled"] = False
        raw["multiarch"]["operators"] = {op: spec for op, spec in self.multiarch_operators.items() if op in allow}
        raw["multiarch"]["arch_sources"] = {
            arch: spec for arch, spec in raw["multiarch"]["arch_sources"].items()
            if arch in definition["architectures"]
        }
        for spec in raw["multiarch"]["arch_sources"].values():
            spec["files"] = {op: files for op, files in spec.get("files", {}).items() if op in allow}
        raw["multiarch"]["corpus_dir"] = str(data_dir.resolve() / "corpus_multiarch")
        return Config(raw, self.config_path)

    @property
    def multiarch_cfg(self) -> dict:
        return self.raw.get("multiarch", {}) or {}

    @property
    def multiarch_operators(self) -> dict:
        """多架构层新引入的算子（不来自 cx_operator_tool 语料，如 asum/axpby）。"""
        return self.multiarch_cfg.get("operators", {}) or {}

    @property
    def arch_sources(self) -> dict:
        return self.multiarch_cfg.get("arch_sources", {}) or {}

    @property
    def all_operator_specs(self) -> dict:
        """语料算子 + 多架构新增算子的合并视图（配置字段一致）。"""
        merged = dict(self.operators)
        merged.update(self.multiarch_operators)
        return merged

    @property
    def multiarch_repo(self) -> Path:
        repo = self.multiarch_cfg.get("repo", "../openblas_src")
        return (self.workspace_root / repo).resolve()

    @property
    def multiarch_corpus_dir(self) -> Path:
        rel = self.multiarch_cfg.get("corpus_dir", "data/corpus_multiarch")
        return (self.root / rel).resolve()

    @property
    def retrieval_cfg(self) -> dict:
        return self.raw.get("retrieval", {}) or {}

    @property
    def llm_cfg(self) -> dict:
        return self.raw.get("llm", {}) or {}

    @property
    def version_tuple(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "extractor_version": self.extractor_version,
            "prompt_version": self.prompt_version,
            "model_version": self.model_version,
        }

    def rel_workspace(self, path: Path) -> str:
        """转成 workspace 相对 POSIX 路径（provenance.source_file 用；语料在 workspace 外时带 ../）。"""
        resolved = Path(path).resolve()
        try:
            rel = resolved.relative_to(self.workspace_root)
        except ValueError:
            rel = Path(os.path.relpath(str(resolved), str(self.workspace_root)))
        return rel.as_posix()

    def rel_root(self, path: Path) -> str:
        resolved = Path(path).resolve()
        try:
            rel = resolved.relative_to(self.root)
        except ValueError:
            rel = Path(os.path.relpath(str(resolved), str(self.root)))
        return rel.as_posix()

    def ensure_dirs(self, data_dir: Path | None = None) -> None:
        target = Path(data_dir) if data_dir else self.data_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "traces").mkdir(parents=True, exist_ok=True)


def load_config(config_path=None) -> Config:
    """加载配置；默认取包所在 kg_agent/config.yaml。"""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config.yaml"
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError("找不到 config.yaml：%s" % p)
    return Config(yamlmini.load(p), p)
