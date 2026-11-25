#!/usr/bin/env python3
"""kg_consistency_audit.py —— 图内"跨字段一致性"审计（不依赖人工、不依赖 LLM）。

动机：AI/模板流水线建图时，最容易漏的不是"抽不到"，而是"字段之间互相矛盾"。
schema 校验（validate.py）只查类型和端点，查不出"这条策略要求 AVX512、却绑在 AVX2 profile 上"。

本脚本做 8 类跨字段检查：
  C1 ISA 矛盾         策略硬性要求某 ISA，但绑定的 profile 不支持
  C2 证据平台越界     证据的 platform_id 不在该策略的 hardware_profile_ids 里
  C3 参数越界         value_in_instance 不在参数声明的 range 里
  C4 冲突+组合并存    同一对策略既冲突又可组合
  C5 唯一键重复       (op, family, action, arch_scope, arch) 重复
  C6 命名歧义         ID 末位是 s/d/c/z（同时也是 dtype 码），但与条件 dtype 不符
  C7 硬条件不足       少于 2 条硬条件（项目自定验收线是 ≥2）
  C8 证据方向矛盾     role=support 但 speedup ≤ 1

用法：
    python kg_consistency_audit.py [--json out.json]
退出码：0 = 无 ERROR；1 = 有 ERROR（C5/C4）；警告不影响退出码。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KG_ROOT = HERE.parent / "kg_agent"
sys.path.insert(0, str(KG_ROOT / "src"))

from kgagent import graph as graph_mod  # noqa: E402
from kgagent.config import load_config  # noqa: E402

DTYPE_CODES = {"s", "d", "c", "z"}


def _props(node) -> dict:
    return (node or {}).get("properties") or {}


def _bound_profiles(props: dict) -> list:
    return list(props.get("hardware_profile_ids") or [])


def audit(cfg) -> dict:
    graph = graph_mod.load_graph(cfg.data_dir)
    strategies = graph.nodes_of_type("OptimizationStrategy")
    findings = {k: [] for k in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]}
    seen_keys = defaultdict(list)

    profile_isa = {}
    for node in graph.nodes_of_type("HardwareProfile"):
        profile_isa[node["id"]] = set(_props(node).get("isa") or [])

    compose_pairs = set()
    for edge in graph.edges_of_type("STRATEGY_COMPOSES_WITH"):
        compose_pairs.add(frozenset((edge["head"], edge["tail"])))

    for node in strategies:
        sid = node["id"]
        props = _props(node)
        bound = _bound_profiles(props)

        # ---- C1 ISA 矛盾
        for edge in graph.out_edges(sid, "STRATEGY_REQUIRES_ISA"):
            if not (edge.get("properties") or {}).get("hard"):
                continue
            isa_node = graph.nodes.get(edge["tail"]) or {}
            required = _props(isa_node).get("isa")
            if not required:
                continue
            for profile in bound:
                have = profile_isa.get(profile)
                if have is not None and required not in have:
                    findings["C1"].append({
                        "strategy": sid, "required_isa": required, "bound_profile": profile,
                        "profile_isa": sorted(have),
                    })

        # ---- C2 证据平台越界（scope=operator 的证据是算子级继承，属设计内，不报）
        for edge in graph.out_edges(sid, "STRATEGY_EVIDENCED_BY"):
            edge_props = edge.get("properties") or {}
            if edge_props.get("scope") == "operator":
                continue
            ev = _props(graph.nodes.get(edge["tail"]))
            platform = ev.get("platform_id")
            if platform:
                bare = platform.split(":", 1)[1] if platform.startswith("hw:") else platform
                if bare not in bound:
                    findings["C2"].append({
                        "strategy": sid, "evidence": edge["tail"],
                        "platform": platform, "bound": bound,
                    })

        # ---- C3 参数越界
        for edge in graph.out_edges(sid, "STRATEGY_HAS_PARAMETER"):
            value = (edge.get("properties") or {}).get("value_in_instance")
            rng = _props(graph.nodes.get(edge["tail"])).get("range")
            if value is None or not isinstance(rng, list) or not rng:
                continue
            if value not in rng:
                findings["C3"].append({
                    "strategy": sid, "parameter": edge["tail"], "value": value, "range": rng,
                })

        # ---- C4 冲突 + 组合并存
        for edge in graph.out_edges(sid, "STRATEGY_CONFLICTS_WITH"):
            if frozenset((sid, edge["tail"])) in compose_pairs:
                findings["C4"].append({"a": sid, "b": edge["tail"]})

        # ---- C5 唯一键（必须含算子与层级；否则同族不同层的策略会被误报）
        parts = sid.split(":")
        op_of_sid = parts[1] if len(parts) > 1 else None
        key = (op_of_sid, props.get("family"), props.get("action"), props.get("level"),
               props.get("arch_scope"), props.get("arch"))
        seen_keys[key].append(sid)

        # ---- C6 命名歧义
        m = re.search(r":([a-z_]+)_([sdcz])$", sid)
        if m:
            dtypes = set()
            for cond in props.get("conditions") or []:
                if cond.get("pred") == "dtype" and isinstance(cond.get("value"), list):
                    dtypes |= set(cond["value"])
            if dtypes and m.group(2) not in dtypes:
                findings["C6"].append({
                    "strategy": sid, "suffix": m.group(2), "condition_dtype": sorted(dtypes),
                    "note": "后缀与 dtype 不符；若 _s/_d 表示来源(normalize.py:259-262)，则应改名避免歧义",
                })

        # ---- C7 硬条件不足
        hard = sum(1 for c in props.get("conditions") or [] if c.get("hard"))
        if hard < 2:
            findings["C7"].append({"strategy": sid, "hard_conditions": hard,
                                   "condition_status": props.get("condition_status")})

        # ---- C8 证据方向矛盾
        for edge in graph.out_edges(sid, "STRATEGY_EVIDENCED_BY"):
            ev = _props(graph.nodes.get(edge["tail"]))
            role = (edge.get("properties") or {}).get("role")
            value = ev.get("value")
            if role == "supports" and ev.get("metric") == "speedup" and isinstance(value, (int, float)) and value <= 1.0:
                findings["C8"].append({"strategy": sid, "evidence": edge["tail"], "value": value})

    for key, ids in seen_keys.items():
        if len(ids) > 1:
            findings["C5"].append({"key": list(key), "strategies": ids})

    stats = {
        "strategies": len(strategies),
        "profiles": len(profile_isa),
        "errors": len(findings["C4"]) + len(findings["C5"]),
        "warnings": sum(len(findings[k]) for k in ("C1", "C2", "C3", "C6", "C7", "C8")),
        "by_check": {k: len(v) for k, v in findings.items()},
    }
    return {"stats": stats, "findings": findings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    cfg = load_config()
    report = audit(cfg)
    print(json.dumps(report["stats"], ensure_ascii=False, indent=2))
    for code in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"):
        items = report["findings"][code]
        if not items:
            continue
        print("\n== %s (%d) ==" % (code, len(items)))
        for item in items[:6]:
            print("  " + json.dumps(item, ensure_ascii=False))
        if len(items) > 6:
            print("  ... 另有 %d 条" % (len(items) - 6))

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if report["stats"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
