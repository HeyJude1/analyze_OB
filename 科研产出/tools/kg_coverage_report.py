#!/usr/bin/env python3
"""kg_coverage_report.py —— 知识覆盖 vs 证据覆盖 vs 检索可达性。

回答三个问题：
  1) 知识（策略）在 算子 × 硬件 上到底覆盖了多少格？（不是"源码文件有没有"，是"有没有可用策略"）
  2) 证据覆盖多少？（策略级 vs 算子级、真机 vs 模拟器）
  3) 图里有几条策略是"检索永远取不到"的（孤儿）？

输出：JSON + 一张可直接进论文的覆盖矩阵。

用法：
    python kg_coverage_report.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KG_ROOT = HERE.parent / "kg_agent"
sys.path.insert(0, str(KG_ROOT / "src"))

from kgagent import graph as graph_mod  # noqa: E402
from kgagent.config import load_config  # noqa: E402


def _props(node) -> dict:
    return (node or {}).get("properties") or {}


def report(cfg) -> dict:
    graph = graph_mod.load_graph(cfg.data_dir)

    operators = sorted(n["id"].split(":", 1)[1] for n in graph.nodes_of_type("Operator"))
    profiles = sorted(n["id"].split(":", 1)[1] for n in graph.nodes_of_type("HardwareProfile"))
    strategies = graph.nodes_of_type("OptimizationStrategy")

    # ---- 1) 知识覆盖：策略 × 算子 × 绑定 profile
    cell = defaultdict(list)          # (op, profile) -> [sid]
    per_op = Counter()
    per_profile = Counter()
    for node in strategies:
        sid = node["id"]
        if sid.startswith("stg:abstract:"):
            continue
        op = sid.split(":", 2)[1]
        per_op[op] += 1
        for profile in _props(node).get("hardware_profile_ids") or []:
            cell[(op, profile)].append(sid)
            per_profile[profile] += 1

    matrix = {}
    for op in sorted(set(list(per_op) + operators)):
        row = {}
        for profile in profiles:
            n = len(cell.get((op, profile), []))
            row[profile] = n
        matrix[op] = row

    filled = sum(1 for op in operators for p in profiles if len(cell.get((op, p), [])) > 0)
    total = len(operators) * len(profiles)

    # ---- 2) 证据覆盖
    evidence = graph.nodes_of_type("PerformanceEvidence")
    ev_platform = Counter(_props(n).get("platform_id") for n in evidence)
    scope = Counter()
    evidenced_strategies = set()
    for edge in graph.edges_of_type("STRATEGY_EVIDENCED_BY"):
        scope[(edge.get("properties") or {}).get("scope")] += 1
        evidenced_strategies.add(edge["head"])

    # ---- 3) 可达性（从算子经 pattern 能不能走到）
    reached = set(edge["tail"] for edge in graph.edges_of_type("PATTERN_OPTIMIZED_BY"))
    generalized = set(edge["tail"] for edge in graph.edges_of_type("STRATEGY_GENERALIZES"))
    all_ids = [n["id"] for n in strategies]
    unreachable = sorted(s for s in all_ids if s not in reached and s not in generalized)
    only_generalizes = sorted(s for s in all_ids if s not in reached and s in generalized)

    # ---- 4) 分布偏斜
    ranked = per_op.most_common()
    top3 = sum(n for _, n in ranked[:3])
    total_strategies = sum(per_op.values())

    return {
        "operators": operators,
        "profiles": profiles,
        "strategies": len(strategies),
        "matrix": matrix,
        "cells": {"filled": filled, "total": total, "ratio": round(filled / total, 3)},
        "per_operator": dict(ranked),
        "per_profile": dict(per_profile.most_common()),
        "skew": {"top3_ops": [op for op, _ in ranked[:3]],
                 "top3_share": round(top3 / max(total_strategies, 1), 3)},
        "evidence": {
            "nodes": len(evidence),
            "by_platform": dict(ev_platform),
            "edge_scope": dict(scope),
            "strategies_with_any_evidence": len(evidenced_strategies),
            "strategies_with_strategy_scope": 0,  # 由 scope 统计推断，下面补
        },
        "reachability": {
            "reached": len(reached),
            "only_generalizes": len(only_generalizes),
            "unreachable": unreachable,
        },
    }


def to_markdown(rep: dict) -> str:
    profiles = rep["profiles"]
    head = "| 算子 | " + " | ".join(profiles) + " | 合计 |"
    sep = "|---|" + "---|" * (len(profiles) + 1)
    lines = [head, sep]
    for op, row in rep["matrix"].items():
        vals = [str(row.get(p, 0)) for p in profiles]
        lines.append("| %s | %s | %d |" % (op, " | ".join(vals), sum(row.get(p, 0) for p in profiles)))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args(argv)

    rep = report(load_config())
    print(json.dumps({k: rep[k] for k in ("cells", "skew", "evidence", "reachability", "per_profile")},
                     ensure_ascii=False, indent=2))
    print()
    print(to_markdown(rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(to_markdown(rep) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
