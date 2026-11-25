"""S5 EvidenceAgent：把 QEMU 基准日志变成可核查的 PerformanceEvidence（确定性，无 LLM）。

数据源：
1. `科研产出/data/riscv_log_summary.csv`（45 个算子 × 200 尺寸的汇总，已算好 size>=32 的几何平均）；
2. 12 个算子的原始 `*_log.txt`（用正则重算几何平均做交叉核对，证明 CSV 数字可溯源）。

硬规则（SPEC §2.5 不变量 12 / 方案 §3.5）：
- 所有证据 platform_id = hw:qemu-riscv64-vlen256，measure_env=qemu，confidence=0.35，rerun_needed=true；
- 证据只作 operator 级线索，**不参与策略级性能排序归因**（strategy 节点不存收益数字）；
- speedup < 0.9 的算子标 negative_example=true（负知识，G6）。
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from . import corpus

__all__ = ["run", "parse_log", "geomean"]

#: 方案 §4.6 的日志行正则（兼容 Migration 数值与单位间无空格）
LOG_LINE_RE = re.compile(
    r"^\s*(\d+):Ref:\s+([\d.]+)\s*MFlops\s+([\d.]+)\s*ms\s*\|\s*Migration:\s+([\d.]+)\s*MFlops\s+([\d.]+)\s*ms\s*\|\s*Speedup:\s*([\d.]+)x\s*=>\s*(\w+)"
)

_NEGATIVE_THRESHOLD = 0.9


def geomean(values: list) -> float:
    values = [v for v in values if v and v > 0]
    if not values:
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))


def parse_log(path) -> dict:
    """解析 <op>_log.txt，返回 size>=32 的 pass 记录统计（与 CSV 口径一致）。"""
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LOG_LINE_RE.match(line)
            if not m:
                continue
            size = int(m.group(1))
            speedup = float(m.group(6))
            result = m.group(7).lower()
            rows.append({"size": size, "speedup": speedup, "pass": result == "pass"})
    ok = [r for r in rows if r["pass"] and r["size"] >= 32]
    return {
        "n_lines": len(rows),
        "n_points": len(ok),
        "geomean": round(geomean([r["speedup"] for r in ok]), 4) if ok else None,
        "pass_rate": round(len(ok) / len(rows), 4) if rows else 0.0,
        "min": min((r["speedup"] for r in ok), default=None),
        "max": max((r["speedup"] for r in ok), default=None),
    }


def _canonical_op(op_name: str, known_ops) -> str:
    """CSV 里的 BLAS 名 -> 图谱 Operator 规范名（dgemm->gemm, chemv->hemv, ddot->dot）。"""
    name = op_name.strip().lower()
    if name in known_ops:
        return name
    if name and name[0] in "sdcz" and name[1:] in known_ops:
        return name[1:]
    return name


def _read_csv_with_lines(path: Path):
    # utf-8-sig：兼容 Windows 下 CSV 的 BOM（否则首列 key 会变成 '\ufeffop'）
    text = Path(path).read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    reader = csv.DictReader(lines)
    return [(idx + 2, row) for idx, row in enumerate(reader)]  # +2: 跳过表头且 1-based


def run(cfg, artifacts: list) -> dict:
    """S5 主入口：返回 {evidences, by_op}。"""
    if not cfg.evidence_cfg.get("enabled", True):
        return {"evidences": [], "by_op": {}, "csv_rows": [], "recomputed": {}}
    known_ops = set(cfg.all_operator_specs.keys())
    ev_cfg = cfg.evidence_cfg
    platform_id = ev_cfg.get("platform_id", "hw:qemu-riscv64-vlen256")
    protocol = ev_cfg.get("protocol", "qemu-user-tcg / 单次运行 / size 1..200")
    confidence = float(ev_cfg.get("confidence", 0.35))
    measured_at = ev_cfg.get("measured_at", "2025-11-14T15:37:00+08:00")
    csv_artifact = next((a for a in artifacts if a["kind"] == "log_summary"), None)
    csv_rel = csv_artifact["path"] if csv_artifact else cfg.rel_workspace(cfg.log_summary)
    log_index = {a["op"]: a for a in artifacts if a["kind"] == "log"}
    log_by_op_dir = {(a["op"], a.get("bench_dir")): a for a in artifacts if a["kind"] == "log"}

    evidences = []
    recomputed = {}
    for line_no, row in _read_csv_with_lines(cfg.log_summary):
        csv_op = row["op"]
        op = _canonical_op(csv_op, known_ops)
        geomean_value = float(row["geomean_speedup_size_ge_32"])
        pass_rate = float(row["pass_rate"])
        n_sizes = int(row["n_sizes"])
        dir_name = row.get("dir") or ""
        # 只有当日志工件与 CSV 行指向同一 bench 目录时才做"重算交叉核对"，避免混用不同实现的日志
        log_artifact = log_by_op_dir.get((op, dir_name))
        raw_ref = "%s/%s/%s" % (cfg.rel_workspace(cfg.corpus_dir), dir_name, row["log_file"])
        extra = {}
        if log_artifact is not None:
            raw_ref = log_artifact["path"]
            try:
                parsed = parse_log(cfg.workspace_root / log_artifact["path"])
                extra["log_recomputed_geomean"] = parsed["geomean"]
                extra["log_n_points"] = parsed["n_points"]
                recomputed[op] = parsed
            except OSError:  # pragma: no cover
                pass
        negative = geomean_value < _NEGATIVE_THRESHOLD
        base_prov = {
            "source_file": csv_rel,
            "line_start": line_no,
            "line_end": line_no,
            "line_confidence": "exact",
            "commit": None,
            "extractor_version": cfg.extractor_version,
        }
        common = {
            "aggregation": "geo_mean",
            "baseline": {"impl": "OpenBLAS reference (linked lib)", "version": "0.3.27.dev"},
            "platform_id": platform_id,
            "protocol": protocol,
            "confidence": confidence,
            "rerun_needed": True,
            "measured_at": measured_at,
            "evidence_status": "qemu_only",
            "measure_env": "qemu",
            "runner": row.get("runner"),
            "negative_example": negative,
            "caveat": "QEMU 计时不代表真实微架构；Ref 为同库参考实现而非策略消融，只能作 operator 级证据",
        }
        geomean_ev = {
            "id": "ev:%s:geomean" % op,
            "op": op,
            "csv_op": csv_op,
            "name": "%s QEMU speedup 几何平均（size>=32）" % csv_op,
            "metric": "speedup",
            "value": geomean_value,
            "workload": {
                "op": csv_op, "variant": "rvv_mig",
                "size_min": int(row["min_size"]), "size_max": int(row["max_size"]),
                "n_points": n_sizes, "size_filter": ">=32",
            },
            "raw_ref": raw_ref,
            "provenance": base_prov,
        }
        geomean_ev.update(common)
        geomean_ev.update(extra)
        pass_ev = {
            "id": "ev:%s:pass_rate" % op,
            "op": op,
            "csv_op": csv_op,
            "name": "%s QEMU pass_rate（全部尺寸）" % csv_op,
            "metric": "pass_rate",
            "value": pass_rate,
            "aggregation": "ratio",
            "workload": {
                "op": csv_op, "variant": "rvv_mig",
                "size_min": int(row["min_size"]), "size_max": int(row["max_size"]),
                "n_points": n_sizes, "size_filter": "all",
            },
            "raw_ref": raw_ref,
            "provenance": base_prov,
        }
        pass_ev.update(common)
        evidences.extend([geomean_ev, pass_ev])
    evidences.sort(key=lambda e: e["id"])
    return {"evidences": evidences, "recomputed": recomputed, "platform_id": platform_id}
