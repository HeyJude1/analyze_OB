#!/usr/bin/env python3
"""validate_kg.py —— 建图产物的契约检查（门 A–E）。

设计原则：只查"能机器判定"的东西；语义正确性交给 verifier 角色抽样人工判。
退出码：0 = 通过；1 = 有 ERROR（未通过门）；2 = 只有 WARNING。

用法：
  python validate_kg.py --kg out/kg.json
  python validate_kg.py --nodes data/nodes.jsonl --edges data/edges.jsonl --profile kgagent
  python validate_kg.py --kg out/kg.json --strict        # 只允许 7 实体 / 12 关系（ob1 口径）
  python validate_kg.py --kg out/kg.json --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ENTITY_TYPES = {
    "OptimizationPrinciple", "SourcePattern", "OptimizationStrategy", "TunableParameter",
    "CodeCharacteristic", "ArchitectureCapability", "CodeExample", "HardwareProfile",
    # kg_agent 口径
    "Operator", "ComputationPattern", "ImplementationTemplate", "ISACapability",
    "Microarchitecture", "PerformanceEvidence", "SourceArtifact", "FailureCase",
}
STRICT_ENTITY_TYPES = {
    "OptimizationPrinciple", "SourcePattern", "OptimizationStrategy", "TunableParameter",
    "CodeCharacteristic", "ArchitectureCapability", "CodeExample",
}
RELATION_TYPES = {
    "OPTIMIZES_PATTERN", "TARGETS", "APPLIES_WHEN", "REQUIRES", "HAS_PARAMETER",
    "IS_ILLUSTRATED_BY", "GENERALIZES", "COMPOSES_WITH", "INSTANCE_OF",
    "CONFLICTS_WITH", "ANALOGOUS_TO", "HAS_CHARACTERISTIC",
    # kg_agent 口径
    "OPERATOR_HAS_PATTERN", "PATTERN_HAS_CHARACTERISTIC", "PATTERN_OPTIMIZED_BY",
    "STRATEGY_APPLIES_WHEN", "STRATEGY_HAS_TEMPLATE", "STRATEGY_HAS_PARAMETER",
    "STRATEGY_REQUIRES_ISA", "STRATEGY_TARGETS_UARCH", "STRATEGY_EVIDENCED_BY",
    "STRATEGY_GENERALIZES", "STRATEGY_COMPOSES_WITH", "STRATEGY_CONFLICTS_WITH",
    "EVIDENCE_ON", "EVIDENCE_SOURCE", "STRATEGY_BOUND_TO_PROFILE", "STRATEGY_SOURCED_FROM",
}
STRICT_RELATION_TYPES = {
    "OPTIMIZES_PATTERN", "TARGETS", "APPLIES_WHEN", "REQUIRES", "HAS_PARAMETER",
    "IS_ILLUSTRATED_BY", "GENERALIZES", "COMPOSES_WITH", "INSTANCE_OF",
    "CONFLICTS_WITH", "ANALOGOUS_TO", "HAS_CHARACTERISTIC",
}
PREDICATES = {
    "dtype", "precision", "stride_x", "stride_y", "loop_depth", "size_regime",
    "aligned", "tail_divisible", "reduction", "structural", "isa",
    "vector_bits", "vector_regs",
}


def read_input(args):
    if args.kg:
        data = json.loads(Path(args.kg).read_text(encoding="utf-8"))
        return data.get("entities") or data.get("nodes") or [], data.get("relations") or data.get("edges") or []
    if args.nodes and args.edges:
        def rl(p):
            return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        return rl(args.nodes), rl(args.edges)
    raise SystemExit("需要 --kg 或 --nodes/--edges")


def norm_name(s: str) -> str:
    return re.sub(r"[\s_\-/]+", "", (s or "").lower())


def check(entities, relations, strict=False) -> dict:
    et = STRICT_ENTITY_TYPES if strict else ENTITY_TYPES
    rt = STRICT_RELATION_TYPES if strict else RELATION_TYPES
    errors, warnings = [], []

    # ---- 门 A：schema
    ids = set()
    for e in entities:
        eid = e.get("id") or e.get("uid")
        if not eid:
            errors.append({"gate": "A", "msg": "实体缺 id", "entity": e.get("name")})
            continue
        if eid in ids:
            errors.append({"gate": "A", "msg": "id 重复", "entity": eid})
        ids.add(eid)
        if e.get("type") not in et:
            errors.append({"gate": "A", "msg": "实体类型不在闭集", "entity": eid, "type": e.get("type")})
    for r in relations:
        if r.get("type") not in rt:
            errors.append({"gate": "A", "msg": "关系类型不在闭集", "relation": r.get("id"), "type": r.get("type")})
        h, t = r.get("head"), r.get("tail")
        h = h.get("id") if isinstance(h, dict) else h
        t = t.get("id") if isinstance(t, dict) else t
        if h not in ids or t not in ids:
            errors.append({"gate": "A", "msg": "关系端点不存在", "relation": r.get("id"), "head": h, "tail": t})

    # ---- 门 B：溯源（行号）
    # 配置型/证据型实体不来自源码行，豁免行号要求（但仍要求 source_file 或 evidence）
    no_prov = []
    exempt = {"HardwareProfile", "ISACapability", "Microarchitecture", "PerformanceEvidence",
              "SourceArtifact", "ArchitectureCapability"}
    for e in entities:
        p = e.get("provenance") or {}
        if e.get("status") == "abstract" or e.get("type") in exempt:
            continue
        if not p.get("source_file") or (p.get("line_start") is None and p.get("line") is None):
            no_prov.append(e.get("id") or e.get("name"))
    if no_prov:
        errors.append({"gate": "B", "msg": "实体缺行号溯源", "count": len(no_prov), "sample": no_prov[:10]})

    # ---- 门 C：策略条件可执行 + 参数在域
    bad_cond, bad_param = [], []
    for e in entities:
        if e.get("type") != "OptimizationStrategy":
            continue
        props = e.get("properties") or {}
        conds = props.get("conditions") or []
        hard = [c for c in conds if c.get("hard")]
        if len(hard) < 2:
            bad_cond.append({"entity": e.get("id"), "hard": len(hard)})
        for c in conds:
            if c.get("pred") not in PREDICATES:
                bad_cond.append({"entity": e.get("id"), "bad_predicate": c.get("pred")})
        for p in props.get("parameters") or []:
            rng = p.get("range")
            val = p.get("value")
            if isinstance(rng, list) and rng and val is not None and val not in rng:
                bad_param.append({"entity": e.get("id"), "param": p.get("name"), "value": val, "range": rng})
    if bad_cond:
        errors.append({"gate": "C", "msg": "策略条件不足或谓词不合法", "count": len(bad_cond), "sample": bad_cond[:10]})
    if bad_param:
        errors.append({"gate": "C", "msg": "参数值越界", "count": len(bad_param), "sample": bad_param[:10]})

    # ---- 门 D：同名/重复
    seen = defaultdict(list)
    for e in entities:
        seen[(e.get("type"), norm_name(e.get("name") or ""))].append(e.get("id"))
    dups = {k: v for k, v in seen.items() if len(v) > 1}
    if dups:
        warnings.append({"gate": "D", "msg": "同类型同名实体（需合并或标注别名）", "count": len(dups),
                         "sample": [{"type": k[0], "name": k[1], "ids": v} for k, v in list(dups.items())[:5]]})

    # ---- 门 E：性能声明必须有证据
    claims = []
    for e in entities:
        props = e.get("properties") or {}
        claim = props.get("performance_claim") or props.get("performance_impact")
        if not claim:
            continue
        ev = props.get("evidence_ids") or props.get("evidence") or []
        if not ev:
            claims.append({"entity": e.get("id"), "claim": str(claim)[:80]})
    if claims:
        warnings.append({"gate": "E", "msg": "有性能声明但无证据引用（应标 claim_status=unverified）",
                         "count": len(claims), "sample": claims[:5]})

    stats = {
        "entities": len(entities),
        "relations": len(relations),
        "entity_types": dict(Counter(e.get("type") for e in entities).most_common()),
        "relation_types": dict(Counter(r.get("type") for r in relations).most_common()),
        "errors": len(errors),
        "warnings": len(warnings),
    }
    return {"ok": not errors, "strict": strict, "stats": stats, "errors": errors, "warnings": warnings}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=None)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--strict", action="store_true", help="只允许 ob1 的 7 实体 / 12 关系")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    entities, relations = read_input(args)
    rep = check(entities, relations, strict=args.strict)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    if rep["errors"]:
        return 1
    return 2 if rep["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
