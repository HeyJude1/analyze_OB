"""S6 ValidatorAgent：组装 KG 快照 + schema/不变量校验 + manifest。

本模块是唯一的"写 data/"入口：
  build()  -> 跑 S0→S6，落盘 nodes.jsonl / edges.jsonl / manifest.json + 中间产物
  check()  -> 对任意 data 目录做不变量校验（CLI validate 复用），返回错误/统计

不变量（方案 §2.5）：
 1. 节点 id 唯一、边端存在、类型在闭集；2. 每条边有 provenance；
 3. 性能数字只出现在 PerformanceEvidence（策略节点禁止收益数字）；
 4. 条件谓词引用 condition_vocab，value 类型匹配；
 5. 边 (head, rel, tail) 唯一；6. CONFLICTS_WITH 对称；
 7. 软合并保留 status（本版不物理删除）；8. 模板有骨架+槽位；
 9. 策略条件 condition_status 非空；10. manifest 版本四元组；
 11. 验收：>=20 条策略具备 >=2 条件 + 模板 + file+line provenance + hardware 绑定。
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from . import corpus, diff, evidence as evidence_mod, extract, multiarch, normalize as normalize_mod, schema as schema_mod, static
from . import NODE_TYPES, RELATION_TYPES, EXTRACTOR_VERSION, SCHEMA_VERSION

__all__ = ["build", "assemble", "check", "write_data", "load_data", "BuildError", "PERF_KEY_RE", "PERF_TEXT_RE"]

#: 只允许出现在 PerformanceEvidence 的性能数值键（不变量 3）
PERF_KEY_RE = re.compile(
    r"^(speedup|throughput|latency|mflops|gflops|gflops_peak|stream_bw_gbps|bandwidth_gbps|"
    r"performance_impact|perf_impact|speedup_vs_\w+)$",
    re.IGNORECASE,
)
#: 策略文本里禁止出现的收益数字（"提升 20%" / "加速 3x" / "2 倍"）
PERF_TEXT_RE = re.compile(r"(加速|提升|提速|speedup)\s*[约~]?\s*\d+(\.\d+)?\s*(%|x|倍|MFlops|GFlops)", re.IGNORECASE)

_ERROR_LIMIT = 80


class BuildError(RuntimeError):
    """构建后校验失败。"""


# ---------------------------------------------------------------- 组装


def _node(node_id, node_type, name, properties, provenance, confidence=None):
    node = {
        "id": node_id,
        "type": node_type,
        "name": name,
        "properties": properties,
        "provenance": provenance,
    }
    if confidence is not None:
        node["confidence"] = confidence
    return node


def _prov(source_file, line_start=None, line_end=None, confidence="exact", commit=None, **extra):
    prov = {
        "source_file": source_file,
        "line_start": line_start,
        "line_end": line_end,
        "line_confidence": confidence,
        "commit": commit,
        "extractor_version": EXTRACTOR_VERSION,
    }
    prov.update(extra)
    return prov


def _artifact_prov(artifact: dict) -> dict:
    return _prov(
        artifact["path"], 1, artifact["line_count"], "exact",
        commit=artifact.get("commit"), sha256=artifact.get("sha256"),
        repo_path=artifact.get("repo_path"),
    )


def _char_id(cond: dict) -> str:
    value = cond.get("value")
    if isinstance(value, list):
        vslug = "_".join(str(v) for v in value)
    elif isinstance(value, bool):
        vslug = "true" if value else "false"
    else:
        vslug = str(value)
    vslug = re.sub(r"[^A-Za-z0-9_.-]+", "_", vslug).strip("_") or "na"
    return "ch:%s:%s_%s" % (cond["pred"], cond["op"], vslug)


def _char_name(cond: dict) -> str:
    value = cond.get("value")
    if isinstance(value, list):
        vtext = "[" + ", ".join(str(v) for v in value) + "]"
    else:
        vtext = str(value)
    return "%s %s %s" % (cond["pred"], cond["op"], vtext)


def _edge(rel, head, tail, properties, provenance, confidence=0.8):
    return {
        "id": "e:%s::%s::%s" % (head, rel, tail),
        "type": rel,
        "head": head,
        "tail": tail,
        "properties": properties or {},
        "provenance": provenance,
        "confidence": confidence,
    }


def assemble(cfg, bundle, artifacts, chars, diffs, extract_out, norm_out, evidence_out) -> tuple:
    """把各阶段产物组装成 nodes / edges。"""
    nodes = []
    edges = []
    artifact_by_id = {a["artifact_id"]: a for a in artifacts}
    evidence_by_id = {e["id"]: e for e in evidence_out["evidences"]}
    hardware_profiles = bundle.hardware_profiles()
    profile_ids = sorted(hardware_profiles.keys())

    # ---------- SourceArtifact ----------
    for art in sorted(artifacts, key=lambda a: a["artifact_id"]):
        props = {
            "repo": art.get("repo"),
            "commit_or_tag": art.get("commit"),
            "path": art.get("repo_path") or art["path"],
            "kind": art["kind"],
            "op": art.get("op"),
            "dtype": art.get("dtype"),
            "arch": art.get("arch"),
            "bytes": art["bytes"],
            "line_count": art["line_count"],
            "sha256": art["sha256"],
            "license": "BSD-3-Clause" if art.get("repo") else None,
            "retrieved_at": "2025-11-14",
            "extractor_version": art.get("extractor_version", EXTRACTOR_VERSION),
            "workspace_path": art["path"],
        }
        if art.get("snapshot_path"):
            props["snapshot_path"] = art["snapshot_path"]
        if art.get("compile_march"):
            props["compile_march"] = art["compile_march"]
        if art.get("qemu_only"):
            props["qemu_only"] = True
        if art.get("arch") and art["kind"] == "multiarch_impl":
            props["arch"] = art["arch"]
            props["arch_family"] = art.get("arch_family")
            props["isa"] = art.get("isa")
            props["isa_node"] = art.get("isa_node")
            props["profiles"] = art.get("profiles")
            props["variant_role"] = art.get("variant")
        nodes.append(_node(art["artifact_id"], "SourceArtifact", art["artifact_id"],
                           props, _artifact_prov(art), 0.95))

    # ---------- ISACapability ----------
    for isa_id, info in sorted(normalize_mod.ISA_CATALOG.items()):
        props = {k: v for k, v in info.items() if k != "name"}
        nodes.append(_node(isa_id, "ISACapability", info["name"], props,
                           _prov("科研产出/kg_agent/schema/hardware_profiles.yaml", None, None, "approx",
                                 note="ISA 能力手工 curation（方案 §3.4）；数值不在此节点"),
                           0.9))

    # ---------- Microarchitecture / HardwareProfile ----------
    for profile_id in profile_ids:
        prof = hardware_profiles[profile_id]
        uarch_id = "uarch:%s" % profile_id
        uarch_props = {
            "vendor": prof.get("vendor"),
            "uarch": prof.get("uarch"),
            "vlen_bits": prof.get("vector_bits"),
            "fp64_lanes": prof.get("fp64_lanes"),
            "lmul_max": prof.get("lmul_max"),
            "num_vector_regs": prof.get("vector_regs"),
            "l1d_kb": prof.get("l1d_kb"),
            "l2_kb": prof.get("l2_kb"),
            "l3_mb": prof.get("l3_mb"),
            "cacheline_b": prof.get("cacheline_b"),
            "source_ref": "科研产出/kg_agent/schema/hardware_profiles.yaml",
            "note": prof.get("note"),
        }
        nodes.append(_node(uarch_id, "Microarchitecture", str(prof.get("uarch") or profile_id),
                           uarch_props, _prov("科研产出/kg_agent/schema/hardware_profiles.yaml",
                                              None, None, "approx", note="硬件数值唯一来源"),
                           0.9))
        hw_props = {
            "profile_name": profile_id,
            "uarch_id": uarch_id,
            "isa": list(prof.get("isa") or []),
            "cores": prof.get("cores"),
            "freq_ghz": prof.get("freq_ghz"),
            "ram_gb": None,
            "compiler": prof.get("compiler"),
            "measure_env": prof.get("measure_env"),
            "note": prof.get("note"),
        }
        nodes.append(_node("hw:%s" % profile_id, "HardwareProfile", profile_id, hw_props,
                           _prov("科研产出/kg_agent/schema/hardware_profiles.yaml", None, None, "approx"),
                           0.9))

    # ---------- Operator ----------
    for op, meta in sorted(norm_out["operators"].items() if "operators" in norm_out else extract_out["operators"].items()):
        generic_art = artifact_by_id.get("art:%s:kernel_generic" % op)
        if generic_art is None:
            # 多架构新增算子（asum/axpby）没有 cx 语料 generic kernel，用第一个多架构实现做 provenance
            generic_art = next(
                (a for a in sorted(artifacts, key=lambda a: a["artifact_id"])
                 if a.get("op") == op and a.get("kind") == "multiarch_impl"), None)
        prov = _artifact_prov(generic_art) if generic_art else _prov("科研产出/kg_agent/config.yaml", None, None, "approx")
        props = {
            "blas_level": meta["blas_level"],
            "dtype_variants": meta["dtype_variants"],
            "op_class": meta["op_class"],
            "flops_formula": meta["flops_formula"],
            "bytes_formula": meta["bytes_formula"],
            "interface": meta["interface"],
            "structural": meta.get("structural", "none"),
            "reduction": bool(meta.get("reduction", False)),
        }
        nodes.append(_node("op:%s" % op, "Operator", op.upper(), props, prov, 0.95))

    # ---------- ComputationPattern ----------
    patterns = extract_out["patterns"]
    for pattern in patterns:
        props = {
            "stage": pattern["stage"],
            "category": pattern["category"],
            "description": pattern["description"],
            "operator": pattern["op"],
            "pattern_type": pattern["pattern_type"],
            "support_count": pattern["support_count"],
        }
        nodes.append(_node(pattern["id"], "ComputationPattern", pattern["name"], props,
                           pattern["provenance"], 0.85))
        edges.append(_edge(
            "OPERATOR_HAS_PATTERN", "op:%s" % pattern["op"], pattern["id"],
            {"file": Path(pattern["provenance"]["source_file"]).name, "stage": pattern["stage"]},
            pattern["provenance"], 0.9,
        ))

    # ---------- Strategy / Template / Parameter ----------
    strategy_by_id = {}
    for strategy in norm_out["strategies"]:
        provenance = strategy.get("provenance") or _prov("科研产出/kg_agent/config.yaml", None, None, "approx")
        conditions = strategy["conditions"]
        props = {
            "family": strategy["family"],
            "level": strategy["level"],
            "action": strategy["action"],
            "stage": strategy["stage"],
            "description": strategy["description"],
            "rationale": strategy["rationale"],
            "trade_offs": strategy.get("trade_offs"),
            "conditions": conditions,
            "condition_status": strategy["condition_status"],
            "hard_condition_count": strategy["hard_condition_count"],
            "hardware_profile_ids": strategy["hardware_profile_ids"],
            "parameter_derivation": {
                p["id"].split(":", 1)[1]: p.get("derived_from")
                for p in norm_out["parameters"]
                if p["id"] in strategy["parameter_ids"] and p.get("derived_from")
            },
            "template_id": strategy["template_id"],
            "status": strategy["status"],
            "support_count": strategy["support_count"],
            "abstraction_level": strategy["abstraction_level"],
            "source": strategy["source"],
            "arch_scope": strategy.get("arch_scope", "arch_agnostic"),
            "arch": strategy.get("arch"),
            "arch_family": strategy.get("arch_family"),
            "arch_evidence": strategy.get("arch_evidence") or [],
            "arch_parameters": strategy.get("arch_parameters") or {},
            "arches": strategy.get("arches") or [],
            "extraction_method": strategy.get("extraction_method", "static"),
            "evidence_status": (
                "qemu_operator_evidence"
                if "ev:%s:geomean" % strategy.get("op") in evidence_by_id
                and strategy.get("arch_scope") != "parameterized"
                and (strategy.get("arch_scope") != "arch_specific" or strategy.get("arch") == "rvv")
                else "source_only"
            ),
            "origin_files": [o.get("file_path") for o in strategy.get("origins", []) if o.get("file_path")][:5],
        }
        nodes.append(_node(strategy["id"], "OptimizationStrategy", strategy["name"], props,
                           provenance, strategy["confidence"]))
        strategy_by_id[strategy["id"]] = strategy

    templates_by_strategy = {t["strategy_id"]: t for t in norm_out["templates"]}
    for template in norm_out["templates"]:
        strategy = strategy_by_id.get(template["strategy_id"])
        provenance = template.get("provenance") or (strategy or {}).get("provenance") or _prov(
            "科研产出/kg_agent/config.yaml", None, None, "approx")
        props = {
            "strategy_id": template["strategy_id"],
            "language": template["language"],
            "skeleton": template["skeleton"],
            "slots": template["slots"],
            "includes": template["includes"],
            "preconditions": template["preconditions"],
            "check_hints": template["check_hints"],
            "isa_variant": template["isa_variant"],
        }
        nodes.append(_node(template["id"], "ImplementationTemplate", template["name"], props,
                           provenance, 0.8))

    for param in norm_out["parameters"]:
        provenance = _prov("科研产出/kg_agent/schema/vocab.yaml", None, None, "approx",
                           note="参数名规范化自 schema/vocab.yaml + S1/S2 抽取")
        # 用第一个引用它的策略的 provenance 更准确
        for strategy in norm_out["strategies"]:
            if param["id"] in strategy["parameter_ids"] and strategy.get("provenance"):
                provenance = strategy["provenance"]
                break
        props = {
            "name": param["name"],
            "kind": param["kind"],
            "unit": param["unit"],
            "value_in_source": param["value_in_source"],
            "range": param["range"],
            "derived_from": param["derived_from"],
            "effect_direction": param["effect_direction"],
            "used_by_strategies": param["used_by"],
        }
        nodes.append(_node(param["id"], "TunableParameter", param["name"], props, provenance, 0.8))

    # ---------- PerformanceEvidence ----------
    for ev in evidence_out["evidences"]:
        props = {k: v for k, v in ev.items() if k not in ("id", "name", "provenance", "op", "csv_op")}
        nodes.append(_node(ev["id"], "PerformanceEvidence", ev["name"], props, ev["provenance"], ev["confidence"]))
        edges.append(_edge("EVIDENCE_ON", ev["id"], ev["platform_id"], {}, ev["provenance"], 0.9))
        source_artifact = "art:evidence:riscv_log_summary"
        op_log = artifact_by_id.get("art:%s:log" % ev["op"])
        if op_log is not None:
            source_artifact = op_log["artifact_id"]
        edges.append(_edge("EVIDENCE_SOURCE", ev["id"], source_artifact, {}, ev["provenance"], 0.9))

    # ---------- CodeCharacteristic（条件落点） ----------
    char_nodes = {}
    for strategy in norm_out["strategies"]:
        for cond in strategy["conditions"]:
            cid = _char_id(cond)
            if cid in char_nodes:
                continue
            char_type = bundle.char_type_map.get(cond["pred"], "structural_property")
            props = {
                "char_type": char_type,
                "value": cond["value"],
                "predicate": cond["pred"],
                "unit": "loops" if cond["pred"] == "loop_depth" else None,
                "extraction_method": "static",
                "confidence": 0.85,
            }
            provenance = strategy.get("provenance") or _prov("科研产出/kg_agent/config.yaml", None, None, "approx")
            char_nodes[cid] = _node(cid, "CodeCharacteristic", _char_name(cond), props, provenance, 0.85)
    for cid in sorted(char_nodes):
        nodes.append(char_nodes[cid])

    # ---------- 策略相关边 ----------
    for strategy in norm_out["strategies"]:
        sid = strategy["id"]
        provenance = strategy.get("provenance") or _prov("科研产出/kg_agent/config.yaml", None, None, "approx")
        # pattern -> strategy
        for pattern_id in strategy["pattern_ids"]:
            edges.append(_edge(
                "PATTERN_OPTIMIZED_BY", pattern_id, sid,
                {"stage": strategy["stage"], "support": strategy["support_count"],
                 "is_primary": True, "analysis_id": strategy.get("source")},
                provenance, 0.8,
            ))
            # pattern -> characteristic
            for cond in strategy["conditions"]:
                role = "defining" if cond["pred"] in ("dtype", "loop_depth", "structural", "reduction") else "optional"
                edges.append(_edge(
                    "PATTERN_HAS_CHARACTERISTIC", pattern_id, _char_id(cond), {"role": role},
                    provenance, 0.7,
                ))
        # applies_when
        for cond in strategy["conditions"]:
            attrs = {"pred": cond["pred"], "op": cond["op"], "value": cond["value"],
                     "hard": cond["hard"], "weight": cond.get("weight", 1.0)}
            edges.append(_edge("STRATEGY_APPLIES_WHEN", sid, _char_id(cond), attrs, provenance, 0.85))
        # parameters
        for pid in strategy["parameter_ids"]:
            param = next((p for p in norm_out["parameters"] if p["id"] == pid), None)
            role = "tunable" if param and param.get("range") else "required"
            instance_value = (strategy.get("parameter_values") or {}).get(pid)
            edges.append(_edge(
                "STRATEGY_HAS_PARAMETER", sid, pid,
                {"role": role, "value_in_instance": instance_value},
                provenance, 0.8,
            ))
        # template
        template = templates_by_strategy.get(sid)
        if template is not None:
            edges.append(_edge("STRATEGY_HAS_TEMPLATE", sid, template["id"], {"is_default": True}, provenance, 0.85))
        # requires ISA
        for req in strategy["isa_requirements"]:
            min_width = None
            for pid in strategy["parameter_ids"]:
                if pid == "par:sew":
                    p = next((x for x in norm_out["parameters"] if x["id"] == pid), None)
                    min_width = p.get("value_in_source") if p else None
            edges.append(_edge(
                "STRATEGY_REQUIRES_ISA", sid, req["isa_node_id"],
                {"hard": req["hard"], "min_width": min_width, "note": req.get("note") or ""},
                provenance, 0.8,
            ))
        # targets uarch
        for profile_id in strategy["hardware_profile_ids"]:
            prof = hardware_profiles.get(profile_id, {})
            fit = "ideal" if prof.get("measure_env") == "native" and profile_id.startswith("k1") else (
                "good" if prof.get("measure_env") == "native" else "neutral")
            parameterization = {
                p["id"].split(":", 1)[1]: p.get("value_in_source")
                for p in norm_out["parameters"]
                if p["id"] in strategy["parameter_ids"] and p.get("value_in_source") is not None
            }
            edges.append(_edge(
                "STRATEGY_TARGETS_UARCH", sid, "uarch:%s" % profile_id,
                {"fit": fit, "parameterization": parameterization},
                provenance, 0.75,
            ))
        # evidence：QEMU 证据只来自 RISC-V 语料，因此只绑定 rvv / arch_agnostic 策略；
        # x86/ARM 的 arch_specific 策略与 parameterized 合并策略只保留源码证据（arch_evidence），
        # 避免把"QEMU RVV 上的 operator 级数字"错记到其它架构的策略上。
        if strategy["op"] and strategy.get("arch_scope") != "parameterized" and not (
                strategy.get("arch_scope") == "arch_specific" and strategy.get("arch") != "rvv"):
            for ev_id in ("ev:%s:geomean" % strategy["op"], "ev:%s:pass_rate" % strategy["op"]):
                ev = evidence_by_id.get(ev_id)
                if ev is None:
                    continue
                if ev["metric"] == "pass_rate":
                    role = "supports" if ev["value"] >= 0.99 else "contradicts"
                else:
                    role = "contradicts" if ev.get("negative_example") else "supports"
                edges.append(_edge(
                    "STRATEGY_EVIDENCED_BY", sid, ev_id,
                    {"role": role, "scope": "operator"}, provenance, 0.6,
                ))

    # ---------- 组合 / 冲突 / 泛化 ----------
    for item in norm_out["composes"]:
        head_prov = strategy_by_id.get(item["head"], {}).get("provenance") or _prov(
            "科研产出/kg_agent/config.yaml", None, None, "approx")
        edges.append(_edge("STRATEGY_COMPOSES_WITH", item["head"], item["tail"],
                           item["properties"], head_prov, 0.7))
    for item in norm_out["conflicts"]:
        head_prov = strategy_by_id.get(item["head"], {}).get("provenance") or _prov(
            "科研产出/kg_agent/config.yaml", None, None, "approx")
        edges.append(_edge("STRATEGY_CONFLICTS_WITH", item["head"], item["tail"],
                           item["properties"], head_prov, 0.7))
        # 对称存储
        edges.append(_edge("STRATEGY_CONFLICTS_WITH", item["tail"], item["head"],
                           dict(item["properties"]), head_prov, 0.7))
    for item in norm_out["generalizes"]:
        head_prov = strategy_by_id.get(item["head"], {}).get("provenance") or _prov(
            "科研产出/kg_agent/config.yaml", None, None, "approx")
        edges.append(_edge("STRATEGY_GENERALIZES", item["head"], item["tail"],
                           item["properties"], head_prov, 0.7))

    nodes.sort(key=lambda n: (n["type"], n["id"]))
    edges = _dedupe_edges(edges)
    edges.sort(key=lambda e: (e["type"], e["head"], e["tail"]))
    return nodes, edges


def _dedupe_edges(edges: list) -> list:
    """同一 (head, rel, tail) 只保留一条（幂等 upsert；不变量 9）。"""
    seen = {}
    for edge in edges:
        key = (edge["head"], edge["type"], edge["tail"])
        if key not in seen:
            seen[key] = edge
    return list(seen.values())


# ---------------------------------------------------------------- 写盘


def write_jsonl(path: Path, records) -> int:
    return corpus.write_jsonl(path, records)


def read_jsonl(path: Path) -> list:
    return corpus.read_jsonl(path)


def _sha256_file(path: Path) -> str:
    return corpus.sha256_file(path)


def write_data(cfg, data_dir: Path, nodes: list, edges: list, extras: dict) -> dict:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus.write_jsonl(data_dir / "nodes.jsonl", nodes)
    corpus.write_jsonl(data_dir / "edges.jsonl", edges)
    manifest = _make_manifest(cfg, data_dir, nodes, edges, extras)
    (data_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _make_manifest(cfg, data_dir: Path, nodes: list, edges: list, extras: dict) -> dict:
    input_paths = extras.get("input_paths") or []
    inputs = []
    for path in sorted(set(input_paths), key=lambda p: str(p)):
        p = Path(path)
        if not p.exists():
            continue
        inputs.append({
            "path": cfg.rel_workspace(p) if p.is_relative_to(cfg.workspace_root) else p.as_posix(),
            "sha256": _sha256_file(p),
            "bytes": p.stat().st_size,
        })
    nodes_by_type = {}
    for node in nodes:
        nodes_by_type[node["type"]] = nodes_by_type.get(node["type"], 0) + 1
    edges_by_type = {}
    for edge in edges:
        edges_by_type[edge["type"]] = edges_by_type.get(edge["type"], 0) + 1
    stats = extras.get("stats") or {}
    stats.update({
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_by_type": dict(sorted(nodes_by_type.items())),
        "edges_by_type": dict(sorted(edges_by_type.items())),
    })
    data_files = {}
    for name in ("nodes.jsonl", "edges.jsonl"):
        p = data_dir / name
        with open(p, "r", encoding="utf-8") as fh:
            line_total = sum(1 for _ in fh)
        data_files[name] = {"sha256": _sha256_file(p), "lines": line_total}
    return {
        "schema_version": cfg.schema_version,
        "extractor_version": cfg.extractor_version,
        "prompt_version": cfg.prompt_version,
        "model_version": cfg.model_version,
        "build_mode": extras.get("mode", "offline"),
        "snapshot_id": "snap-%s-%s" % (cfg.schema_version, extras.get("split") or "reference"),
        "split": extras.get("split"),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": inputs,
        "stats": stats,
        "data": data_files,
        "source_stats": extras.get("source_stats", {}),
    }


# ---------------------------------------------------------------- 校验


def _iter_values(obj, path="$"):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key, value, "%s.%s" % (path, key)
            yield from _iter_values(value, "%s.%s" % (path, key))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            yield from _iter_values(value, "%s[%d]" % (path, idx))


def _arch_stats(nodes: list, edges: list) -> dict:
    """多架构统计：arch × operator 覆盖矩阵、各 arch 策略数、跨架构共有策略表。"""
    impls = {}  # arch -> {op: 实现文件数}
    for node in nodes:
        if node["type"] != "SourceArtifact":
            continue
        props = node["properties"]
        kind = props.get("kind")
        op = props.get("op")
        if not op:
            continue
        if kind == "kernel_rvv":
            arch = "rvv"
        elif kind == "multiarch_impl":
            arch = props.get("arch")
        else:
            continue
        if not arch:
            continue
        impls.setdefault(arch, {})
        impls[arch][op] = impls[arch].get(op, 0) + 1
    ops_by_arch = {arch: sorted(ops) for arch, ops in sorted(impls.items())}
    op_arch_count = {}
    for arch, ops in impls.items():
        for op in ops:
            op_arch_count[op] = op_arch_count.get(op, 0) + 1
    ops_multi_arch = {op: n for op, n in sorted(op_arch_count.items()) if n >= 2}
    strategies = [n for n in nodes if n["type"] == "OptimizationStrategy"]
    by_scope = {}
    by_arch = {}
    with_scope = 0
    for strategy in strategies:
        props = strategy["properties"]
        scope = props.get("arch_scope")
        if scope:
            with_scope += 1
            by_scope[scope] = by_scope.get(scope, 0) + 1
        arch = props.get("arch")
        if arch:
            by_arch[arch] = by_arch.get(arch, 0) + 1
    cross = []
    for strategy in strategies:
        props = strategy["properties"]
        if props.get("arch_scope") != "parameterized":
            continue
        cross.append({
            "id": strategy["id"],
            "op": strategy.get("properties", {}).get("op") or strategy["id"].split(":")[1],
            "action": props.get("action"),
            "arches": list(props.get("arches") or []),
            "arch_parameters": props.get("arch_parameters") or {},
            "arch_evidence_count": len(props.get("arch_evidence") or []),
        })
    return {
        "arch_matrix": {arch: dict(sorted(ops.items())) for arch, ops in sorted(impls.items())},
        "ops_by_arch": ops_by_arch,
        "ops_with_multi_arch": len(ops_multi_arch),
        "ops_with_multi_arch_list": sorted(ops_multi_arch),
        "op_arch_count": dict(sorted(op_arch_count.items())),
        "implementation_files": sum(sum(ops.values()) for ops in impls.values()),
        "strategies_with_arch_scope": with_scope,
        "strategies_by_arch_scope": dict(sorted(by_scope.items())),
        "strategies_by_arch": dict(sorted(by_arch.items())),
        "cross_arch_shared_count": len(cross),
        "cross_arch_shared": sorted(cross, key=lambda c: c["id"]),
    }


def check(cfg, nodes: list, edges: list, bundle=None, manifest: dict = None, require_acceptance: bool = True) -> dict:
    """对节点/边做 schema + 不变量校验；返回 {errors, warnings, stats, acceptance}。"""
    bundle = bundle or schema_mod.load_schema(cfg.schema_dir)
    errors = []
    warnings = []
    nodes_by_id = {}
    node_ids = set()

    def err(msg):
        if len(errors) < _ERROR_LIMIT:
            errors.append(msg)

    # 1. 节点 schema + id 唯一
    for node in nodes:
        schema_errors = bundle.check_node(node)
        for e in schema_errors:
            err("节点 %s schema: %s" % (node.get("id", "?"), e))
        nid = node.get("id")
        if nid in node_ids:
            err("节点 id 重复：%s" % nid)
        node_ids.add(nid)
        nodes_by_id[nid] = node
        if node.get("type") not in NODE_TYPES:
            err("节点 %s 类型不在闭集：%r" % (nid, node.get("type")))
        prov = node.get("provenance") or {}
        if not prov.get("source_file"):
            err("节点 %s 缺少 provenance.source_file" % nid)
        for key, value, path in _iter_values(node.get("properties") or {}):
            if PERF_KEY_RE.match(key) and node.get("type") != "PerformanceEvidence":
                err("节点 %s 在非 PerformanceEvidence 上存性能数值键 %r（%s）" % (nid, key, path))

    # 2. 边 schema + 端点 + 类型/定义域
    edge_keys = set()
    relations = bundle.relations
    for edge in edges:
        schema_errors = bundle.check_edge(edge)
        for e in schema_errors:
            err("边 %s schema: %s" % (edge.get("id", "?"), e))
        head, tail, rel = edge.get("head"), edge.get("tail"), edge.get("type")
        if head not in nodes_by_id:
            err("边 %s 的 head 不存在：%s" % (edge.get("id"), head))
        if tail not in nodes_by_id:
            err("边 %s 的 tail 不存在：%s" % (edge.get("id"), tail))
        if rel not in relations:
            err("边 %s 类型不在闭集：%r" % (edge.get("id"), rel))
            continue
        spec = relations[rel]
        if head in nodes_by_id and nodes_by_id[head]["type"] != spec["head"]:
            err("边 %s 定义域错误：%s 是 %s，期望 %s" % (
                edge.get("id"), head, nodes_by_id[head]["type"], spec["head"]))
        if tail in nodes_by_id and nodes_by_id[tail]["type"] != spec["tail"]:
            err("边 %s 值域错误：%s 是 %s，期望 %s" % (
                edge.get("id"), tail, nodes_by_id[tail]["type"], spec["tail"]))
        key = (head, rel, tail)
        if key in edge_keys:
            err("边 (head,rel,tail) 重复：%s %s %s" % key)
        edge_keys.add(key)
        if not (edge.get("provenance") or {}).get("source_file"):
            err("边 %s 缺少 provenance.source_file" % edge.get("id"))

    # 3. 策略不变量
    strategy_edges = {}
    for edge in edges:
        strategy_edges.setdefault(edge["head"], []).append(edge)
        if edge["type"] == "STRATEGY_CONFLICTS_WITH":
            if (edge["tail"], "STRATEGY_CONFLICTS_WITH", edge["head"]) not in edge_keys:
                err("CONFLICTS_WITH 不对称：%s -> %s" % (edge["head"], edge["tail"]))
    acceptance = []
    for node in nodes:
        if node.get("type") != "OptimizationStrategy":
            continue
        props = node["properties"]
        status = props.get("condition_status")
        if not status:
            err("策略 %s 的 condition_status 为空" % node["id"])
        elif status not in ("executable", "partial", "text"):
            err("策略 %s 的 condition_status 非法：%r" % (node["id"], status))
        conditions = props.get("conditions") or []
        if status == "executable" and len(conditions) < 2:
            err("策略 %s 标记 executable 但条件不足 2 个" % node["id"])
        for cond in conditions:
            info = bundle.predicates.get(cond.get("pred"))
            if not info:
                err("策略 %s 条件谓词 %r 不在闭集" % (node["id"], cond.get("pred")))
                continue
            if cond.get("op") not in (info.get("ops") or ["eq"]):
                err("策略 %s 条件 op %r 非法" % (node["id"], cond.get("op")))
            kind = info.get("type")
            value = cond.get("value")
            if kind == "enum":
                values = info.get("values") or []
                if cond["op"] == "in":
                    if not isinstance(value, list) or not value or any(v not in values for v in value):
                        err("策略 %s 条件 value 非法：%r" % (node["id"], value))
                elif value not in values:
                    err("策略 %s 条件 value 非法：%r" % (node["id"], value))
            elif kind == "int" and not isinstance(value, int):
                err("策略 %s 条件 value 应为 int：%r" % (node["id"], value))
            elif kind == "bool" and not isinstance(value, bool):
                err("策略 %s 条件 value 应为 bool：%r" % (node["id"], value))
        for key in ("description", "rationale"):
            text = props.get(key) or ""
            if PERF_TEXT_RE.search(text):
                err("策略 %s 的 %s 中出现收益数字文本（应只放在 PerformanceEvidence）" % (node["id"], key))
        for profile_id in props.get("hardware_profile_ids") or []:
            if "hw:%s" % profile_id not in node_ids:
                err("策略 %s 绑定不存在的 hardware profile：%s" % (node["id"], profile_id))
        # 验收项
        template_edges = [e for e in strategy_edges.get(node["id"], []) if e["type"] == "STRATEGY_HAS_TEMPLATE"]
        prov = node.get("provenance") or {}
        has_line = isinstance(prov.get("line_start"), int) and isinstance(prov.get("line_end"), int)
        has_template = bool(template_edges) and template_edges[0]["tail"] in node_ids
        has_hw = bool(props.get("hardware_profile_ids")) and any(
            e["type"] in ("STRATEGY_TARGETS_UARCH", "STRATEGY_REQUIRES_ISA")
            for e in strategy_edges.get(node["id"], []))
        ok = (len(conditions) >= 2 and has_template and has_line and has_hw and bool(prov.get("source_file")))
        if ok:
            acceptance.append(node["id"])
        template_node = nodes_by_id.get(template_edges[0]["tail"]) if has_template else None
        if template_node is not None:
            tprops = template_node["properties"]
            if not tprops.get("skeleton") or "{{" not in tprops.get("skeleton", ""):
                err("模板 %s 骨架不含 {{slot}}" % template_node["id"])
            if not tprops.get("slots"):
                err("模板 %s 槽位为空" % template_node["id"])

    if require_acceptance and len(acceptance) < 20:
        err("通过 SPEC §6 验收的策略只有 %d 条（要求 >=20）" % len(acceptance))

    # ---- 多架构不变量（task-11） ----
    arch_scope_values = set(bundle.vocab_list("arch_scope") or
                            ["arch_agnostic", "parameterized", "arch_specific"])
    arch_names = set(bundle.vocab_list("arch_names") or [])
    for node in nodes:
        if node["type"] != "OptimizationStrategy":
            continue
        props = node["properties"]
        scope = props.get("arch_scope")
        if not scope:
            err("策略 %s 缺少 arch_scope 标注" % node["id"])
            continue
        if scope not in arch_scope_values:
            err("策略 %s 的 arch_scope 非法：%r" % (node["id"], scope))
        evidence = props.get("arch_evidence") or []
        if scope == "arch_specific":
            if not props.get("arch"):
                err("arch_specific 策略 %s 缺少 arch" % node["id"])
            elif arch_names and props["arch"] not in arch_names:
                err("策略 %s 的 arch 非法：%r" % (node["id"], props["arch"]))
            if not evidence:
                err("arch_specific 策略 %s 缺少 arch_evidence" % node["id"])
        if scope == "parameterized":
            arches = set(props.get("arches") or [])
            if len(arches) < 2:
                err("parameterized 策略 %s 的 arches 少于 2 个：%r" % (node["id"], arches))
            params = props.get("arch_parameters") or {}
            for arch_name in arches:
                if arch_name not in params:
                    err("parameterized 策略 %s 缺少 %s 的 arch_parameters" % (node["id"], arch_name))
            if len(evidence) < 2:
                err("parameterized 策略 %s 的 arch_evidence 少于 2 条" % node["id"])

    arch = _arch_stats(nodes, edges)
    split = (manifest or {}).get("split")
    if split:
        spec = cfg.raw.get("splits", {}).get(split)
        if not spec:
            err("manifest 中的划分未登记：%s" % split)
        else:
            allow = set(spec["operators"])
            actual = {n["id"].split(":", 1)[1] for n in nodes if n["type"] == "Operator"}
            if actual != allow:
                err("构图算子不符合划分：期望 %s，实际 %s" % (sorted(allow), sorted(actual)))
            for node in nodes:
                if node["type"] == "PerformanceEvidence":
                    err("隔离图谱不得载入历史性能证据：%s" % node["id"])
                if node["type"] != "SourceArtifact":
                    continue
                props = node["properties"]
                if props.get("op") not in allow or props.get("kind") not in (
                    "kernel_generic", "kernel_rvv", "multiarch_impl",
                ):
                    err("隔离图谱含不在白名单内的源码工件：%s" % node["id"])
    if require_acceptance:
        arch_required = len(spec["operators"]) if split and spec else 8
        if arch["ops_with_multi_arch"] < arch_required:
            err("有 >=2 个架构实现的算子只有 %d 个（要求 >=%d）" % (
                arch["ops_with_multi_arch"], arch_required))
        if arch["strategies_with_arch_scope"] < 20:
            err("有 arch_scope 标注的策略只有 %d 条（要求 >=20）" % arch["strategies_with_arch_scope"])
        if arch["cross_arch_shared_count"] < 1:
            err("跨架构共有策略为 0（应有 >=1 条 parameterized 策略）")
        for item in arch["cross_arch_shared"]:
            if len(item["arches"]) < 2:
                err("跨架构共有策略 %s 只覆盖 %r" % (item["id"], item["arches"]))

    # 4. 模板/参数/证据基本不变量
    for node in nodes:
        if node["type"] == "ImplementationTemplate":
            if not node["properties"].get("slots"):
                err("模板 %s 槽位为空" % node["id"])
        if node["type"] == "PerformanceEvidence":
            props = node["properties"]
            if not props.get("protocol") or not props.get("aggregation"):
                err("证据 %s 缺少 protocol/aggregation" % node["id"])
            if props.get("measure_env") == "qemu" and props.get("rerun_needed") is not True:
                err("证据 %s 的 rerun_needed 应为 true（QEMU 证据）" % node["id"])
            profile = nodes_by_id.get(props.get("platform_id"), {})
            if (profile.get("type") != "HardwareProfile"
                    or profile["properties"].get("measure_env") != props.get("measure_env")
                    or (node["id"], "EVIDENCE_ON", profile.get("id")) not in edge_keys):
                err("证据 %s 的 profile、测量环境与 EVIDENCE_ON 边不一致" % node["id"])
            if not isinstance(props.get("value"), (int, float)):
                err("证据 %s 的 value 不是数值" % node["id"])
        if node["type"] == "SourceArtifact":
            for key in ("repo", "path", "commit_or_tag", "sha256", "extractor_version"):
                if not node["properties"].get(key):
                    err("SourceArtifact %s 缺少 %s" % (node["id"], key))

    # 5. manifest 版本四元组
    if manifest is not None:
        for key in ("schema_version", "extractor_version", "prompt_version", "model_version"):
            if not manifest.get(key):
                err("manifest 缺少版本四元组字段 %r" % key)
        if manifest.get("stats", {}).get("nodes") != len(nodes):
            err("manifest.stats.nodes 与 nodes.jsonl 不一致")

    # 统计
    node_by_type = {}
    for node in nodes:
        node_by_type[node["type"]] = node_by_type.get(node["type"], 0) + 1
    edge_by_type = {}
    for edge in edges:
        edge_by_type[edge["type"]] = edge_by_type.get(edge["type"], 0) + 1
    strategies = [n for n in nodes if n["type"] == "OptimizationStrategy"]
    executable = [s for s in strategies if s["properties"].get("condition_status") == "executable"]
    with_template = [s for s in strategies if any(
        e["type"] == "STRATEGY_HAS_TEMPLATE" for e in strategy_edges.get(s["id"], []))]
    with_prov = [s for s in strategies if isinstance((s.get("provenance") or {}).get("line_start"), int)]
    with_hw = [s for s in strategies if s["properties"].get("hardware_profile_ids")]
    stats = {
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_by_type": dict(sorted(node_by_type.items())),
        "edges_by_type": dict(sorted(edge_by_type.items())),
        "strategies": len(strategies),
        "strategies_executable": len(executable),
        "strategies_with_template": len(with_template),
        "strategies_with_line_provenance": len(with_prov),
        "strategies_with_hardware_binding": len(with_hw),
        "acceptance_count": len(acceptance),
        "acceptance_required": 20,
        "condition_executable_rate": round(len(executable) / len(strategies), 4) if strategies else 0.0,
        "arch": arch,
    }
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
        "acceptance": sorted(acceptance),
        "arch": arch,
    }


# ---------------------------------------------------------------- build


def _resolve_input_path(cfg, rel_path: str) -> Path:
    """工件路径可能是 workspace 相对（cx 语料/分析 JSON）或 kg_agent 相对（data/corpus_multiarch 复制件）。"""
    path = Path(rel_path)
    if path.is_absolute():
        return path
    for base in (cfg.workspace_root, cfg.root):
        candidate = base / path
        if candidate.exists():
            return candidate
    return cfg.workspace_root / path


def _isolated_inputs(reference_cfg, split, artifacts, records):
    definition = reference_cfg.raw["splits"][split]
    allow = set(definition["operators"])
    near = set(definition.get("shared_source_cases", []))
    held_dirs = set(definition.get("additional_holdout_dirs", []))
    held_dirs.update(spec["dir"] for op, spec in reference_cfg.all_operator_specs.items()
                     if op not in allow | near)
    all_records = multiarch.discover(reference_cfg)
    held_hashes = {
        r["sha256"] for r in all_records if r["op"] not in allow | near
    }
    for directory in sorted(held_dirs):
        for path in sorted((reference_cfg.corpus_dir / directory).glob("*_k*.c")):
            held_hashes.add(corpus.sha256_file(path))
    core_overlap = [art["path"] for art in artifacts if art["sha256"] in held_hashes]
    if core_overlap:
        raise BuildError("构图主源码与留出源码相同，需先修正配对：%s" % core_overlap)
    excluded = [
        {"arch": r["arch"], "op": r["op"], "path": r["rel_repo_path"], "sha256": r["sha256"]}
        for r in records if r["sha256"] in held_hashes
    ]
    train_hashes = {a["sha256"] for a in artifacts} | {r["sha256"] for r in records}
    shared_sources = [
        {"op": r["op"], "path": r["rel_repo_path"], "sha256": r["sha256"]}
        for r in all_records if r["op"] in near and r["sha256"] in train_hashes
    ]
    for op in sorted(near):
        directory = reference_cfg.all_operator_specs[op]["dir"]
        for path in sorted((reference_cfg.corpus_dir / directory).glob("*_k*.c")):
            sha256 = corpus.sha256_file(path)
            if sha256 in train_hashes:
                shared_sources.append({"op": op, "path": "%s/%s" % (directory, path.name), "sha256": sha256})
    return [r for r in records if r["sha256"] not in held_hashes], {
        "operators": sorted(allow),
        "architectures": definition["architectures"],
        "holdout_source_hashes_checked": len(held_hashes),
        "holdout_source_hashes": sorted(held_hashes),
        "excluded_shared_sources": excluded,
        "shared_source_cases": sorted(near),
        "overlaps_with_shared_source_cases": shared_sources,
        "legacy_analysis_and_historical_logs": "excluded",
    }


def _snapshot_sources(cfg, data_dir, core_artifacts, multiarch_artifacts):
    """Freeze the exact bytes used for extraction beside the resulting graph."""
    for artifact in core_artifacts:
        origin = _resolve_input_path(cfg, artifact["path"])
        destination = data_dir / "corpus_core" / artifact["op"] / origin.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        artifact["snapshot_path"] = destination.relative_to(data_dir).as_posix()
    for artifact in multiarch_artifacts:
        path = _resolve_input_path(cfg, artifact["path"]).resolve()
        artifact["snapshot_path"] = path.relative_to(data_dir.resolve()).as_posix()
    license_path = cfg.multiarch_repo / "LICENSE"
    if license_path.is_file():
        shutil.copyfile(license_path, data_dir / "OPENBLAS_LICENSE")


def build(cfg, mode: str = "offline", data_dir=None, llm_client=None, verbose: bool = False,
          split: str | None = None) -> dict:
    """跑 S0→S6 并落盘（含多架构层）。返回 {manifest, report}。"""
    if mode not in ("offline", "llm"):
        raise ValueError("mode 只能是 offline/llm")
    if mode == "llm" and llm_client is None:
        mode = "offline"  # 没有可用 LLM 时降级
    data_dir = Path(data_dir) if data_dir else (
        cfg.data_dir.with_name("data_" + split) if split else cfg.data_dir)
    reference_cfg = cfg
    if split:
        if data_dir.resolve() == cfg.data_dir.resolve():
            raise ValueError("隔离图谱需要独立的 --data-dir")
        if mode != "offline":
            raise ValueError("隔离图谱目前仅支持离线构建")
        cfg = cfg.for_split(split, data_dir)
    cfg.ensure_dirs(data_dir)
    bundle = schema_mod.load_schema(cfg.schema_dir)

    # S0：单语料 + 多架构（自动发现 -> 复制 -> sha256）
    artifacts = corpus.run(cfg)
    split_audit = None
    if split:
        records, split_audit = _isolated_inputs(reference_cfg, split, artifacts, multiarch.discover(cfg))
        ma_artifacts = multiarch.materialize(cfg, records)
        ma_chars, ma_impls = multiarch.parse_impls(cfg, ma_artifacts)
        _snapshot_sources(cfg, data_dir, artifacts, ma_artifacts)
    else:
        ma_artifacts, ma_chars, ma_impls = multiarch.run(cfg, artifacts)
    all_artifacts = artifacts + ma_artifacts
    corpus.write_jsonl(data_dir / "corpus_manifest.jsonl", all_artifacts)
    corpus.write_jsonl(data_dir / "multiarch_manifest.jsonl", ma_artifacts)

    # S1：generic/RVV 静态特征 + 多架构（C/汇编）特征
    chars, impls = static.run(cfg, artifacts)
    corpus.write_jsonl(data_dir / "characteristics.jsonl", chars + ma_chars)

    # S2：generic vs RVV 差分 + 多架构差分（C 可差分 / 汇编特征存在）
    diffs = diff.run(cfg, impls)
    ma_diffs = multiarch.diffs(cfg, ma_impls, impls)
    corpus.write_jsonl(data_dir / "diffs.jsonl", diffs + ma_diffs)

    # S3/S4：策略抽取 + 多架构 + 跨架构合并 + 规范化
    extract_out = extract.run(
        cfg, artifacts, impls, diffs, mode=mode, llm_client=llm_client,
        ma_artifacts=ma_artifacts, ma_impls=ma_impls, ma_diffs=ma_diffs,
    )
    corpus.write_jsonl(data_dir / "strategies_raw.jsonl", extract_out["strategies"])

    norm_out = normalize_mod.run(cfg, extract_out)
    evidence_out = evidence_mod.run(cfg, all_artifacts)

    nodes, edges = assemble(cfg, bundle, all_artifacts, chars, diffs, extract_out, norm_out, evidence_out)
    arch = _arch_stats(nodes, edges)
    source_stats = {
        "corpus_artifacts": len(artifacts),
        "multiarch_artifacts": len(ma_artifacts),
        "multiarch_impls": len(ma_impls),
        "multiarch_diffs": len(ma_diffs),
        "characteristics": len(chars) + len(ma_chars),
        "diffs": len(diffs) + len(ma_diffs),
        "raw_strategies": len(extract_out["strategies"]),
        "normalized_strategies": len(norm_out["strategies"]),
        "templates": len(norm_out["templates"]),
        "parameters": len(norm_out["parameters"]),
        "evidences": len(evidence_out["evidences"]),
        "patterns": len(extract_out["patterns"]),
        "operators": len(extract_out["operators"]),
        "multiarch_repo": cfg.multiarch_cfg.get("repo"),
        "multiarch_commit": cfg.multiarch_cfg.get("commit"),
    }
    if split_audit:
        source_stats["split_audit"] = split_audit
    input_paths = [_resolve_input_path(cfg, a["path"]) for a in all_artifacts]
    extras = {
        "mode": mode,
        "split": split,
        "input_paths": input_paths + [cfg.config_path] + sorted(cfg.schema_dir.glob("*.*")),
        "source_stats": source_stats,
        "stats": {
            "ops_with_multi_arch": arch["ops_with_multi_arch"],
            "strategies_with_arch_scope": arch["strategies_with_arch_scope"],
            "cross_arch_shared_count": arch["cross_arch_shared_count"],
            "implementation_files": arch["implementation_files"],
            "arch_correction": "多架构实现 = cx_operator_tool RVV 语料 + openblas_src 复制件",
        },
    }
    manifest = write_data(cfg, data_dir, nodes, edges, extras)
    # 多架构覆盖矩阵落盘（可打印的统计产物）
    (data_dir / "arch_coverage.json").write_text(
        json.dumps({
            "schema_version": cfg.schema_version,
            "arch_matrix": arch["arch_matrix"],
            "ops_by_arch": arch["ops_by_arch"],
            "ops_with_multi_arch": arch["ops_with_multi_arch"],
            "ops_with_multi_arch_list": arch["ops_with_multi_arch_list"],
            "op_arch_count": arch["op_arch_count"],
            "implementation_files": arch["implementation_files"],
            "strategies_by_arch_scope": arch["strategies_by_arch_scope"],
            "strategies_by_arch": arch["strategies_by_arch"],
            "cross_arch_shared_count": arch["cross_arch_shared_count"],
            "cross_arch_shared": arch["cross_arch_shared"],
        }, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = check(cfg, nodes, edges, bundle=bundle, manifest=manifest)
    report["manifest"] = manifest
    report["source_stats"] = source_stats
    report["arch"] = arch
    if not report["ok"]:
        raise BuildError("构建校验失败：\n" + "\n".join(report["errors"][:20]))
    return report


def load_data(data_dir) -> tuple:
    data_dir = Path(data_dir)
    nodes = read_jsonl(data_dir / "nodes.jsonl")
    edges = read_jsonl(data_dir / "edges.jsonl")
    manifest = None
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return nodes, edges, manifest


def validate_data_dir(cfg, data_dir=None, require_acceptance: bool = True) -> dict:
    data_dir = Path(data_dir) if data_dir else cfg.data_dir
    if not (data_dir / "nodes.jsonl").exists():
        return {"ok": False, "errors": ["%s 不存在，请先运行 build" % (data_dir / "nodes.jsonl")],
                "warnings": [], "stats": {}, "acceptance": []}
    nodes, edges, manifest = load_data(data_dir)
    bundle = schema_mod.load_schema(cfg.schema_dir)
    report = check(cfg, nodes, edges, bundle=bundle, manifest=manifest, require_acceptance=require_acceptance)
    if manifest and manifest.get("split"):
        root = data_dir.resolve()
        audit = manifest.get("source_stats", {}).get("split_audit", {})
        holdout_hashes = set(audit.get("holdout_source_hashes", []))
        if not holdout_hashes or len(holdout_hashes) != audit.get("holdout_source_hashes_checked"):
            report["errors"].append("缺少可复核的留出源文件哈希清单")
        for name in ("nodes.jsonl", "edges.jsonl"):
            if _sha256_file(root / name) != manifest.get("data", {}).get(name, {}).get("sha256"):
                report["errors"].append("快照文件哈希不符：%s" % name)
        for node in nodes:
            if node["type"] != "SourceArtifact":
                continue
            props = node["properties"]
            path = (root / (props.get("snapshot_path") or "")).resolve()
            if props.get("sha256") in holdout_hashes:
                report["errors"].append("构图源码与留出源文件字节相同：%s" % node["id"])
            if not path.is_relative_to(root) or not path.is_file() or _sha256_file(path) != props.get("sha256"):
                report["errors"].append("源文件快照缺失或哈希不符：%s" % node["id"])
        report["ok"] = not report["errors"]
    return report
