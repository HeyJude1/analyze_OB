"""S4 AlignAgent：规范化 / 对齐 / ID 稳定化（半自动，无 LLM 也能跑）。

职责：
1. 受控词表校验（family/action/level/stage/status 必须在 schema/vocab.yaml 闭集内）；
2. 条件谓词校验（pred/op/value 类型必须匹配 condition_vocab.yaml）；
3. ID 稳定化：{prefix}:{slug}，同一 (op, family, action, source) 不重复；发布后不可变；
4. 参数合并为全局 TunableParameter 节点（par:<name>）；
5. 模板实例化（tpl:<op>:<slug>）；
6. 计算流程对齐（按 stage/category 把策略挂到 ComputationPattern）；
7. 抽象策略 GENERALIZES、组合 COMPOSES_WITH、冲突 CONFLICTS_WITH 的确定性规则。

输出结构供 S6 validate.py 组装 nodes/edges。
"""

from __future__ import annotations

import json

from .schema import PREDICATE_INFO

__all__ = ["run", "ISA_CATALOG", "slug_of", "ACTION_ABSTRACT"]

#: ISA 能力节点目录（硬件能力闭集；数值不在这里，数值只来自 hardware_profiles.yaml / Microarchitecture）
ISA_CATALOG = {
    "isa:avx2.fma.f64": {
        "isa": "AVX2", "feature": "fma", "element_widths": [64],
        "instr_examples": ["_mm256_fmadd_pd", "_mm256_loadu_pd", "_mm256_storeu_pd"],
        "requires_flags": ["-mavx2", "-mfma"], "name": "AVX2 + FMA (f64)",
        "description": "256-bit 向量浮点 + 融合乘加，x86 Haswell 及以上",
    },
    "isa:avx512.fma.f64": {
        "isa": "AVX512F", "feature": "fma", "element_widths": [64],
        "instr_examples": ["_mm512_fmadd_pd", "_mm512_loadu_pd"],
        "requires_flags": ["-mavx512f", "-mfma"], "name": "AVX512F + FMA (f64)",
        "description": "512-bit 向量浮点 + 掩码寄存器，Skylake-X/Ice Lake",
    },
    "isa:rvv1.0.vector": {
        "isa": "RVV 1.0", "feature": "vector_arith", "element_widths": [32, 64],
        "instr_examples": ["vsetvli", "vle64.v", "vse64.v", "vlse64.v"],
        "requires_flags": ["-march=rv64gcv", "-mabi=lp64d"], "name": "RVV 1.0 向量算术/访存",
        "description": "RISC-V Vector 1.0：可变 VLEN、vsetvl、单位/跨步向量访存",
    },
    "isa:rvv1.0.fma.f64": {
        "isa": "RVV 1.0", "feature": "fma", "element_widths": [64],
        "instr_examples": ["vfmacc.vv", "vfmacc.vf", "vfredosum.vs"],
        "requires_flags": ["-march=rv64gcv_zvl256b"], "name": "RVV 1.0 FMA (f64)",
        "description": "RVV 融合乘加（vfmacc），dgemm/向量算子向量化的核心指令",
    },
    "isa:neon.fma.f64": {
        "isa": "NEON", "feature": "fma", "element_widths": [64],
        "instr_examples": ["fmla v3.2d, v1.2d, v2.2d", "ld1 {v2.2d}", "vfmaq_f64"],
        "requires_flags": ["-march=armv8-a"], "name": "ARM NEON FMA (f64)",
        "description": "ARMv8-A ASIMD/NEON：128-bit 向量 + 融合乘加（fmla/vfmaq_f64）",
    },
    "isa:sve.fma.f64": {
        "isa": "SVE", "feature": "fma", "element_widths": [64],
        "instr_examples": ["fmla z0.d, p0/m, z1.d, z2.d", "whilelt p0.d", "prfm pldl1keep"],
        "requires_flags": ["-march=armv8.2-a+sve"], "name": "ARM SVE FMA (f64)",
        "description": "ARM SVE：可伸缩向量 + 谓词寄存器 + 融合乘加，VL 由实现决定（此处按 256-bit 记录）",
    },
}

#: action -> 抽象策略（GENERALIZES 目标）
ACTION_ABSTRACT = {
    "vectorize": "stg:abstract:vector_fma",
    "tile": "stg:abstract:register_blocking",
    "unroll": "stg:abstract:k_unroll",
    "tail_handle": "stg:abstract:tail_handling",
    "reorder": "stg:abstract:pointer_precompute",
}

#: 抽象策略节点（在 normalize 里补全，保证即使 S3 未产出也能被引用）
ABSTRACT_NODES = {
    "stg:abstract:vector_fma": ("向量化 + FMA（跨算子抽象）", "SIMD向量指令", "vectorize", "instruction"),
    "stg:abstract:register_blocking": ("寄存器分块（以寄存器复用换访存）", "分块Tiling", "tile", "algorithm"),
    "stg:abstract:k_unroll": ("循环展开（k 方向多步合并）", "循环展开", "unroll", "code"),
    "stg:abstract:tail_handling": ("尾部/余数处理", "尾部处理", "tail_handle", "code"),
    "stg:abstract:pointer_precompute": ("指针预计算", "指针/索引优化", "reorder", "code"),
    "stg:abstract:stride_dispatch": ("步长分支派发", "指针/索引优化", "specialize", "code"),
}


def slug_of(value: str) -> str:
    """ID slug：只保留 ASCII 字母数字与 _.-；中文名走 fallback（本流水线 id 都由规则构造）。"""
    out = []
    for ch in str(value):
        if ch.isascii() and (ch.isalnum() or ch in "_.-"):
            out.append(ch.lower())
        elif ch in ":/ ":
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "node"


def _unique(base: str, used: set) -> str:
    candidate = base
    idx = 2
    while candidate in used:
        candidate = "%s_%d" % (base, idx)
        idx += 1
    used.add(candidate)
    return candidate


def _normalize_conditions(conditions: list, bundle) -> tuple:
    """校验条件合法性；返回 (合法条件, 丢弃原因)。"""
    valid = []
    dropped = []
    vocab = PREDICATE_INFO or bundle.predicates
    for cond in conditions or []:
        pred = cond.get("pred")
        info = vocab.get(pred)
        if not info:
            dropped.append({"condition": cond, "reason": "谓词 %r 不在 condition_vocab 闭集" % pred})
            continue
        op = cond.get("op")
        allowed_ops = info.get("ops") or ["eq"]
        if op not in allowed_ops:
            dropped.append({"condition": cond, "reason": "op %r 不允许用于谓词 %r（允许 %r）" % (op, pred, allowed_ops)})
            continue
        value = cond.get("value")
        kind = info.get("type")
        ok = True
        if kind == "enum":
            values = info.get("values") or []
            if op == "in":
                ok = isinstance(value, list) and len(value) > 0 and all(v in values for v in value)
            else:
                ok = value in values
        elif kind == "int":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif kind == "bool":
            ok = isinstance(value, bool)
        if not ok:
            dropped.append({"condition": cond, "reason": "value %r 与谓词 %r 类型不符" % (value, pred)})
            continue
        item = {"pred": pred, "op": op, "value": value, "hard": bool(cond.get("hard", False))}
        if "weight" in cond:
            item["weight"] = cond["weight"]
        valid.append(item)
    # 去重且排序（确定性）
    dedup = {}
    for cond in valid:
        dedup[(cond["pred"], cond["op"], json.dumps(cond["value"], sort_keys=True), cond["hard"])] = cond
    valid = sorted(dedup.values(), key=lambda c: (not c["hard"], c["pred"], c["op"], json.dumps(c["value"], sort_keys=True)))
    return valid, dropped


def _merge_params(raw_params: list, used_param_ids: set) -> list:
    merged = {}
    for p in raw_params or []:
        name = slug_of(p.get("name") or "param")
        pid = "par:%s" % name
        item = merged.get(pid)
        if item is None:
            merged[pid] = {
                "id": pid,
                "name": p.get("label") or p.get("name") or name,
                "kind": p.get("kind") or "blocking",
                "unit": p.get("unit") or "elements",
                "value_in_source": p.get("value_in_source"),
                "range": list(p.get("range") or []),
                "derived_from": p.get("derived_from"),
                "effect_direction": p.get("effect_direction"),
                "used_by": 0,
            }
            used_param_ids.add(pid)
            item = merged[pid]
        else:
            if item.get("value_in_source") is None and p.get("value_in_source") is not None:
                item["value_in_source"] = p.get("value_in_source")
            for r in p.get("range") or []:
                if r not in item["range"]:
                    item["range"].append(r)
            if not item.get("derived_from") and p.get("derived_from"):
                item["derived_from"] = p.get("derived_from")
            if not item.get("effect_direction") and p.get("effect_direction"):
                item["effect_direction"] = p.get("effect_direction")
        item["used_by"] += 1
    for item in merged.values():
        item["range"] = sorted(set(item["range"]))
    return sorted(merged.values(), key=lambda p: p["id"])


def _pick_pattern(op: str, stage: str, family: str, patterns_by_op: dict) -> list:
    patterns = patterns_by_op.get(op) or []
    if not patterns:
        return []
    same = [p for p in patterns if p["stage"] == stage]
    if stage == "core":
        priority = {"mm_microkernel": 0, "reduction": 1, "elementwise_update": 2, "loop_invariant": 3}
    elif stage == "transform":
        priority = {"packing": 0, "loop_invariant": 1}
    elif stage == "prep":
        priority = {"parameter_check": 0, "index_init": 1, "loop_invariant": 2}
    elif stage == "post":
        priority = {"tail_case": 0, "scale_accumulation": 1, "branch_elimination": 2}
    else:
        priority = {}
    if same:
        same.sort(key=lambda p: (priority.get(p["category"], 9), p["id"]))
        return [same[0]["id"]]
    if stage == "core":
        core = [p for p in patterns if p["stage"] == "core"]
        if core:
            return [sorted(core, key=lambda p: p["id"])[0]["id"]]
    return [sorted(patterns, key=lambda p: p["id"])[0]["id"]]


def _isa_requirements(raw: list, bundle) -> list:
    out = []
    seen = set()
    for req in raw or []:
        node_id = req.get("isa")
        if node_id not in ISA_CATALOG:
            continue
        if node_id in seen:
            continue
        seen.add(node_id)
        out.append({"isa_node_id": node_id, "hard": bool(req.get("hard", False)), "note": req.get("note") or ""})
    return out


def run(cfg, extract_out: dict) -> dict:
    """S4 主入口：返回规范化后的 strategies/templates/parameters/isa/links。"""
    bundle = _get_bundle(cfg)
    patterns = extract_out["patterns"]
    patterns_by_op = {}
    for pattern in patterns:
        patterns_by_op.setdefault(pattern["op"], []).append(pattern)

    used_strategy_ids = set()
    used_template_ids = set()
    used_param_ids = set()
    strategies = []
    templates = []
    by_raw_key = {}

    for raw in extract_out["strategies"]:
        op = raw.get("op")
        source = raw.get("source")
        family = raw.get("family")
        action = raw.get("action")
        level = raw.get("level")
        stage = raw.get("stage")
        if family not in bundle.families:
            family = "指针/索引优化"
        if action not in bundle.actions:
            action = "reorder"
        if level not in bundle.levels:
            level = "code"
        if stage not in bundle.stages:
            stage = "core"

        if source == "analysis":
            base_id = "stg:%s:%s_%s_a" % (op, action, level)
        elif source == "static":
            base_id = "stg:%s:%s_s" % (op, action)
        elif source == "diff":
            base_id = "stg:%s:%s_d" % (op, action)
        elif source == "multiarch":
            base_id = "stg:%s:%s_%s" % (op, action, raw.get("arch") or "arch")
        elif source == "cross_arch":
            base_id = raw.get("cross_arch_id") or "stg:%s:%s_x" % (op, action)
        else:
            base_id = raw.get("abstract_target") or "stg:abstract:%s" % slug_of(raw.get("name"))
        sid = _unique(base_id, used_strategy_ids)

        conditions, dropped_conditions = _normalize_conditions(raw.get("conditions"), bundle)
        hard_count = sum(1 for c in conditions if c["hard"])
        if len(conditions) >= 2:
            condition_status = "executable"
        elif conditions:
            condition_status = "partial"
        else:
            condition_status = "text"

        hardware = []
        for profile_id in raw.get("hardware_profile_ids") or []:
            if profile_id in bundle.hardware_profiles():
                hardware.append(profile_id)
        hardware = sorted(set(hardware))

        params = _merge_params(raw.get("params"), used_param_ids)
        parameter_values = {p["id"]: p.get("value_in_source") for p in params}

        # 模板
        template = raw.get("template") or {}
        tpl_id = "tpl:%s" % sid.split(":", 1)[1]
        tpl_id = _unique(tpl_id, used_template_ids)
        param_ids = {p["id"] for p in params}
        slots = []
        for slot in template.get("slots") or []:
            slot = dict(slot)
            src = slot.get("source") or ""
            if src.startswith("par:") and src not in param_ids:
                slot["source"] = "derived:isa_defaults"
            slots.append(slot)
        templates.append({
            "id": tpl_id,
            "strategy_id": sid,
            "name": "%s（实施模板）" % raw.get("name", sid),
            "language": template.get("language", "c"),
            "includes": template.get("includes") or [],
            "skeleton": template.get("skeleton") or "/* TODO */",
            "slots": slots,
            "preconditions": template.get("preconditions") or [],
            "check_hints": template.get("check_hints") or [],
            "isa_variant": template.get("isa_variant", "portable"),
            "provenance": raw.get("provenance"),
        })

        pattern_ids = _pick_pattern(op, stage, family, patterns_by_op) if op else []
        status = raw.get("status") if raw.get("status") in bundle.statuses else "raw"
        confidence = raw.get("confidence", 0.7)
        arch_scope = raw.get("arch_scope") or "arch_agnostic"
        if arch_scope not in bundle.vocab_list("arch_scope"):
            arch_scope = "arch_agnostic"
        arch_evidence = []
        for ev in raw.get("arch_evidence") or []:
            if not isinstance(ev, dict):
                continue
            arch_evidence.append({
                "arch": ev.get("arch"),
                "arch_family": ev.get("arch_family"),
                "arch_label": ev.get("arch_label"),
                "isa": ev.get("isa"),
                "artifact_id": ev.get("artifact_id"),
                "file": ev.get("file"),
                "repo_path": ev.get("repo_path"),
                "role": ev.get("role"),
                "kind": ev.get("kind"),
                "line_start": ev.get("line_start"),
                "line_end": ev.get("line_end"),
                "method": ev.get("method"),
            })
        arch_parameters = raw.get("arch_parameters") or {}
        strategy = {
            "id": sid,
            "op": op,
            "name": raw.get("name") or sid,
            "family": family,
            "level": level,
            "action": action,
            "stage": stage,
            "description": raw.get("description") or "",
            "rationale": raw.get("rationale") or "",
            "trade_offs": raw.get("trade_offs"),
            "conditions": conditions,
            "dropped_conditions": dropped_conditions,
            "condition_status": condition_status,
            "hard_condition_count": hard_count,
            "hardware_profile_ids": hardware,
            "parameter_ids": [p["id"] for p in params],
            "parameter_values": parameter_values,
            "template_id": tpl_id,
            "pattern_ids": pattern_ids,
            "isa_requirements": _isa_requirements(raw.get("isa_requirements"), bundle),
            "status": status,
            "support_count": int(raw.get("support_count") or 1),
            "confidence": confidence,
            "source": source,
            "provenance": raw.get("provenance"),
            "origins": raw.get("origins") or [],
            "abstraction_level": 3 if source == "abstract" else (2 if level == "algorithm" else 1),
            "arch_scope": arch_scope,
            "arch": raw.get("arch"),
            "arch_family": raw.get("arch_family") or ("agnostic" if arch_scope == "arch_agnostic" else None),
            "arch_evidence": arch_evidence,
            "arch_parameters": arch_parameters,
            "arches": raw.get("arches") or sorted(arch_parameters.keys()),
            "extraction_method": raw.get("extraction_method") or ("diff" if source == "diff" else "static"),
        }
        strategies.append(strategy)
        by_raw_key[raw.get("raw_key")] = strategy

    # ---- 参数统计与全局节点 ----
    global_params = {}
    for raw in extract_out["strategies"]:
        for p in raw.get("params") or []:
            pid = "par:%s" % slug_of(p.get("name") or "param")
            node = global_params.get(pid)
            if node is None:
                global_params[pid] = {
                    "id": pid,
                    "name": pid.split(":", 1)[1],
                    "label": p.get("label") or p.get("name") or pid,
                    "kind": p.get("kind") or "blocking",
                    "unit": p.get("unit") or "elements",
                    "value_in_source": p.get("value_in_source"),
                    "range": list(p.get("range") or []),
                    "derived_from": p.get("derived_from"),
                    "effect_direction": p.get("effect_direction"),
                    "used_by": 0,
                }
                node = global_params[pid]
            else:
                if node.get("value_in_source") is None and p.get("value_in_source") is not None:
                    node["value_in_source"] = p.get("value_in_source")
                for r in p.get("range") or []:
                    if r not in node["range"]:
                        node["range"].append(r)
                if not node.get("derived_from") and p.get("derived_from"):
                    node["derived_from"] = p.get("derived_from")
                if not node.get("effect_direction") and p.get("effect_direction"):
                    node["effect_direction"] = p.get("effect_direction")
            node["used_by"] += 1
    for node in global_params.values():
        node["range"] = sorted(set(node["range"]))

    # ---- 抽象策略补全（确保引用的节点存在） ----
    existing_ids = {s["id"] for s in strategies}
    for abstract_id, (name, family, action, level) in ABSTRACT_NODES.items():
        if abstract_id in existing_ids:
            continue
        strategies.append({
            "id": abstract_id, "op": None, "name": name, "family": family, "level": level,
            "action": action, "stage": "core", "description": name, "rationale": None, "trade_offs": None,
            "conditions": [
                {"pred": "dtype", "op": "in", "value": ["s", "d", "c", "z"], "hard": True},
                {"pred": "loop_depth", "op": "ge", "value": 2, "hard": True},
                {"pred": "size_regime", "op": "in", "value": ["medium", "large"], "hard": True},
            ],
            "dropped_conditions": [], "condition_status": "executable", "hard_condition_count": 3,
            "hardware_profile_ids": sorted(bundle.hardware_profiles().keys()),
            "parameter_ids": [], "template_id": None, "pattern_ids": [],
            "isa_requirements": [], "status": "merged", "support_count": 1, "confidence": 0.6,
            "source": "abstract", "provenance": None, "origins": [], "abstraction_level": 3,
            "arch_scope": "arch_agnostic", "arch": None, "arch_family": "agnostic",
            "arch_evidence": [], "arch_parameters": {}, "arches": [],
            "extraction_method": "static",
        })
        sid = abstract_id
        tpl_id = "tpl:abstract:%s" % abstract_id.split(":")[-1]
        from .extract import canonical_template

        tpl = canonical_template(action, "portable", {})
        template = {
            "id": tpl_id, "strategy_id": sid, "name": "%s（抽象模板）" % name,
            "language": "c", "includes": tpl["includes"], "skeleton": tpl["skeleton"],
            "slots": tpl["slots"], "preconditions": tpl["preconditions"],
            "check_hints": tpl["check_hints"], "isa_variant": "portable",
            "provenance": None,
        }
        # 用任一同 action 的具体策略的 provenance 兜底
        for concrete in strategies:
            if concrete["source"] != "abstract" and concrete["action"] == action and concrete["provenance"]:
                template["provenance"] = concrete["provenance"]
                strategies[-1]["provenance"] = concrete["provenance"]
                break
        strategies[-1]["template_id"] = tpl_id
        templates.append(template)

    # ---- 抽象策略 support_count 与 GENERALIZES ----
    abstract_support = {}
    generalizes = []
    for strategy in strategies:
        if strategy["source"] == "abstract":
            continue
        abstract_id = ACTION_ABSTRACT.get(strategy["action"])
        if strategy["action"] == "specialize":
            abstract_id = "stg:abstract:stride_dispatch" if strategy["family"] == "指针/索引优化" else "stg:abstract:vector_fma"
        if abstract_id:
            abstract_support[abstract_id] = abstract_support.get(abstract_id, 0) + 1
            generalizes.append({
                "head": strategy["id"], "tail": abstract_id,
                "properties": {"abstraction_gain": strategy["abstraction_level"] - (1 if strategy["level"] != "algorithm" else 2),
                               "conditions_removed": ["dtype", "isa"]},
            })
    for strategy in strategies:
        if strategy["id"] in abstract_support:
            strategy["support_count"] = max(strategy["support_count"], abstract_support[strategy["id"]])

    # ---- 跨架构共有策略：arch_specific -> parameterized 的 GENERALIZES ----
    cross_by_key = {}
    for strategy in strategies:
        if strategy["source"] == "cross_arch":
            cross_by_key.setdefault((strategy["op"], strategy["family"], strategy["action"]), strategy["id"])
    cross_support = {}
    for strategy in strategies:
        if strategy.get("arch_scope") != "arch_specific" or strategy["source"] not in ("multiarch", "diff"):
            continue
        merged_id = cross_by_key.get((strategy["op"], strategy["family"], strategy["action"]))
        if not merged_id:
            continue
        cross_support[merged_id] = cross_support.get(merged_id, 0) + 1
        generalizes.append({
            "head": strategy["id"], "tail": merged_id,
            "properties": {"abstraction_gain": 1, "conditions_removed": ["arch"]},
        })
    for strategy in strategies:
        if strategy["id"] in cross_support:
            strategy["support_count"] = max(strategy["support_count"], cross_support[strategy["id"]])

    # ---- 组合 / 冲突（确定性规则） ----
    composes = []
    conflicts = []
    by_op_action = {}
    for strategy in strategies:
        if strategy["op"]:
            by_op_action.setdefault(strategy["op"], {}).setdefault(strategy["source"], []).append(strategy)
    for op, sources in by_op_action.items():
        diff_strats = sources.get("diff", [])
        static_strats = sources.get("static", [])
        analysis_strats = sources.get("analysis", [])
        core_vec = next((s for s in diff_strats if s["action"] == "vectorize"), None)
        core_tile = next((s for s in static_strats if s["action"] == "tile"), None)
        scalar_tile = next((s for s in analysis_strats if s["action"] == "tile"), None)
        pack_strat = next((s for s in analysis_strats if s["action"] == "pack"), None)
        unroll_strat = next((s for s in diff_strats if s["action"] == "unroll"), None)
        tile_strat = next((s for s in diff_strats if s["action"] == "tile"), None)
        tail_strat = next((s for s in diff_strats if s["action"] == "tail_handle"), None)
        if core_vec and tail_strat:
            composes.append({
                "head": core_vec["id"], "tail": tail_strat["id"],
                "properties": {"order": 1,
                               "condition": "处理主循环剩余元素时保持同一实现的索引、步长、累加状态；动态 vl 已完整覆盖时合并到同一循环",
                               "evidence_id": None},
            })
        if core_vec and core_tile:
            conflicts.append({
                "head": core_vec["id"], "tail": core_tile["id"],
                "properties": {"resource": "register", "severity": "high",
                               "reason": "向量内核与标量累加器微内核是同一热循环的互斥实现，同时生成会重复计算",
                               "condition": "同一 kernel 的热循环", "evidence_id": None},
            })
        if core_vec and scalar_tile:
            conflicts.append({
                "head": core_vec["id"], "tail": scalar_tile["id"],
                "properties": {"resource": "register", "severity": "medium",
                               "reason": "RVV 向量化实现与标量分块实现互斥（同一计算流程的两种落地方式）",
                               "condition": "同一 kernel 的热循环", "evidence_id": None},
            })
        if pack_strat and core_vec:
            composes.append({
                "head": core_vec["id"], "tail": pack_strat["id"],
                "properties": {"order": -1, "synergy": 0.8,
                               "condition": "pack 后的 A/B 布局必须与向量内核读取一致", "evidence_id": None},
            })
        if tile_strat and unroll_strat:
            composes.append({
                "head": tile_strat["id"], "tail": unroll_strat["id"],
                "properties": {"order": 0, "synergy": 0.7,
                               "condition": "展开后的寄存器压力不能超过向量寄存器预算", "evidence_id": None},
            })

    if any(edge["head"].endswith(":vectorize_d") and edge["tail"].endswith(":tail_handle_d")
           for edge in composes):
        composes.append({
            "head": "stg:abstract:vector_fma", "tail": "stg:abstract:tail_handling",
            "properties": {"order": 1,
                           "condition": "选择目标向量路径后，确认余数用同一路径覆盖或由余数路径处理一次",
                           "evidence_id": None},
        })

    return {
        "strategies": sorted(strategies, key=lambda s: s["id"]),
        "templates": sorted(templates, key=lambda t: t["id"]),
        "parameters": sorted(global_params.values(), key=lambda p: p["id"]),
        "isa_nodes": ISA_CATALOG,
        "generalizes": generalizes,
        "composes": composes,
        "conflicts": conflicts,
        "patterns": patterns,
        "operators": extract_out["operators"],
    }


_BUNDLE_CACHE: dict = {}


def _get_bundle(cfg):
    key = str(cfg.schema_dir)
    if key not in _BUNDLE_CACHE:
        from .schema import load_schema

        _BUNDLE_CACHE[key] = load_schema(cfg.schema_dir)
    return _BUNDLE_CACHE[key]
