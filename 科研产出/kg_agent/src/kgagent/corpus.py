"""S0 CorpusAgent：语料清单（确定性，无 LLM）。

产出 `data/corpus_manifest.jsonl`：每个源工件一条记录（分析 JSON、_command.txt、generic kernel、
RVV kernel、QEMU 日志、日志汇总 CSV），带 sha256 / 字节数 / 行数 / 架构 / -march / dtype。
sha256 用于 manifest.json 的"输入指纹"，保证 build 可复现。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

from . import yamlmini

__all__ = [
    "run",
    "write_jsonl",
    "sha256_file",
    "count_lines",
    "parse_command_file",
    "load_log_summary",
    "load_json",
]

_COMMIT = "OpenBLAS 0.3.27.dev"
_REPO = "OpenBLAS (rvv-OpenBLAS-develop / OpenBLAS-develop)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, records) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


_CP_RE = re.compile(r"\bcp\s+(\S+)\s+(?:\./)?(zzz_\S+\.c)\s*$")
_MARCH_RE = re.compile(r"-march=([^\s]+)")
_CP_ALIAS_RE = re.compile(r"-DASMNAME=([A-Za-z0-9_]+)")
_QEMU_RE = re.compile(r"qemu[^\n]*?-cpu[^\n]*", re.IGNORECASE)


def parse_command_file(path: Path) -> dict:
    """解析 _command.txt：-march / -mabi / qemu 参数 / 模板源文件映射。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    marches = _MARCH_RE.findall(text)
    rvv_march = next((m for m in marches if "rv64gcv" in m), marches[-1] if marches else None)
    mabis = re.findall(r"-mabi=([^\s]+)", text)
    aliases = sorted(set(_CP_ALIAS_RE.findall(text)))
    cp_map = {}
    for line in text.splitlines():
        line = line.strip()
        m = _CP_RE.search(line)
        if m:
            cp_map[Path(m.group(2)).name] = m.group(1)
    qemu = _QEMU_RE.search(text)
    return {
        "march": rvv_march,
        "mabi": mabis[0] if mabis else None,
        "aliases": aliases,
        "cp_map": cp_map,
        "qemu_cmd": qemu.group(0).strip() if qemu else None,
    }


def repo_path_for(artifact_kind: str, cp_map: dict, op: str, kernel_path: Path | None = None) -> str | None:
    """从 `cp <repo源> zzz_*_template.c` 反推仓库路径（generic / rvv 两个 kernel）。

    优先用 kernel 文件里的预处理标记 `# <n> "<zzz_..._template.c>"` 精确定位模板名，
    因为不同算子的 kernel 模板名并不都叫 `*_k_template.c`（如 dgemm 是 zzz_gemmkernel_2x2_template.c）。
    """
    target = None
    if kernel_path is not None and Path(kernel_path).exists():
        head = Path(kernel_path).read_text(encoding="utf-8", errors="replace")[:4000]
        m = re.search(r'#\s*\d+\s+"([^"]*_template\.c)"', head)
        if m:
            target = Path(m.group(1)).name
    if target is None and kernel_path is not None:
        suffix = "_k_rvv.c" if artifact_kind == "kernel_rvv" else "_k.c"
        stem = Path(kernel_path).name[: -len(suffix)]
        want = (
            [t for t in cp_map if t.endswith("_k_rvv_template.c")]
            if artifact_kind == "kernel_rvv"
            else [t for t in cp_map if t.endswith("_k_template.c")]
        )
        target = want[0] if want else None
    if target is None:
        return None
    src = cp_map.get(target)
    if not src:
        return None
    src = src.replace("\\", "/")
    idx = src.find("/kernel/")
    return src[idx + 1:] if idx >= 0 else None


def load_log_summary(path: Path) -> list:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _pick_log_file(corpus_dir: Path, dir_name: str, csv_rows: list, op: str):
    row_by_dir = {r["dir"]: r for r in csv_rows if r.get("dir")}
    candidates = []
    row = row_by_dir.get(dir_name)
    if row:
        candidates.append(corpus_dir / dir_name / row["log_file"])
    for p in sorted((corpus_dir / dir_name).glob("*_log.txt")):
        candidates.append(p)
    for p in candidates:
        if p.exists():
            return p, row
    return None, row


def run(cfg) -> list:
    """S0：生成语料清单（内存列表），由 cli/build 落盘。"""
    corpus_dir = cfg.corpus_dir
    if not corpus_dir.exists():
        raise FileNotFoundError("算子语料目录不存在：%s" % corpus_dir)
    import_logs = cfg.evidence_cfg.get("enabled", True)
    csv_rows = load_log_summary(cfg.log_summary) if import_logs else []
    out = []

    def add(artifact_id, op, kind, path: Path, extra=None):
        path = Path(path)
        rec = {
            "artifact_id": artifact_id,
            "op": op,
            "kind": kind,
            "path": cfg.rel_workspace(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "line_count": count_lines(path),
            "commit": _COMMIT,
            "repo": _REPO,
            "extractor_version": cfg.extractor_version,
        }
        if extra:
            rec.update(extra)
        out.append(rec)
        return rec

    # 4 个算子的三层分析 JSON
    for op in cfg.analysis_operators:
        p = cfg.analysis_dir / ("%s_analysis.json" % op)
        if not p.exists():
            raise FileNotFoundError("分析 JSON 不存在：%s" % p)
        add("art:%s:analysis" % op, op, "analysis", p, {"variant": "analysis", "arch": "generic"})

    # 12 个算子的语料：_command.txt / _k.c / _k_rvv.c / _log.txt
    for op, spec in cfg.operators.items():
        dir_name = spec["dir"]
        op_dir = corpus_dir / dir_name
        if not op_dir.is_dir():
            raise FileNotFoundError("算子目录不存在：%s" % op_dir)
        dtype = spec.get("dtype", "d")
        cmd_path = op_dir / "_command.txt"
        cmd = parse_command_file(cmd_path) if cmd_path.exists() else {}
        base_extra = {
            "dtype": dtype,
            "compile_march": cmd.get("march"),
            "compile_mabi": cmd.get("mabi"),
            "compiler": "riscv64-linux-gcc 14.3.0",
            "bench_dir": dir_name,
        }
        if import_logs and cmd_path.exists():
            add("art:%s:command" % op, op, "command", cmd_path, dict(base_extra, variant="command", arch="riscv64"))

        generic = sorted(op_dir.glob("*_k.c"))
        rvv = sorted(op_dir.glob("*_k_rvv.c"))
        if not generic or not rvv:
            raise FileNotFoundError("算子 %s 缺少 _k.c / _k_rvv.c 配对（%s）" % (op, op_dir))
        rec_g = add(
            "art:%s:kernel_generic" % op, op, "kernel_generic", generic[0],
            dict(base_extra, variant="generic", arch="generic",
                 repo_path=repo_path_for("kernel_generic", cmd.get("cp_map", {}), op, generic[0])),
        )
        rec_r = add(
            "art:%s:kernel_rvv" % op, op, "kernel_rvv", rvv[0],
            dict(base_extra, variant="rvv", arch="riscv64",
                 repo_path=repo_path_for("kernel_rvv", cmd.get("cp_map", {}), op, rvv[0])),
        )
        log_path, csv_row = _pick_log_file(corpus_dir, dir_name, csv_rows, op) if import_logs else (None, None)
        if log_path:
            extra = dict(base_extra, variant="log", arch="riscv64")
            if csv_row:
                extra["csv_geomean_ge32"] = float(csv_row["geomean_speedup_size_ge_32"])
                extra["csv_n_sizes"] = int(csv_row["n_sizes"])
                extra["csv_runner"] = csv_row["runner"]
            add("art:%s:log" % op, op, "log", log_path, extra)

    # 证据汇总 CSV
    if import_logs:
        add("art:evidence:riscv_log_summary", "all", "log_summary", cfg.log_summary,
            {"variant": "summary", "arch": "riscv64", "qemu_only": True})
    return sorted(out, key=lambda r: r["artifact_id"])


def artifact_index(artifacts: list) -> dict:
    return {a["artifact_id"]: a for a in artifacts}
