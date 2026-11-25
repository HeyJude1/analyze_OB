"""检索：条件过滤 + ISA 过滤 + 冲突消解 + 可解释打分 + 7 步推理 trace（SPEC §2.3/§5）。

7 个 step 固定（顺序不可变，每步都有 name/description/candidates/selected，被淘汰的写 dropped.reason）：
  1 pattern_match      算子 -> 计算流程
  2 strategy_recall    沿 PATTERN_OPTIMIZED_BY + GENERALIZES 扩展召回
  3 condition_filter   硬条件过滤（谓词求值）
  4 hardware_filter    ISA/硬件 profile 过滤
  5 conflict_resolve   同族去重 + 冲突边 + 寄存器预算
  6 score_and_select   六项加权打分 -> top_k
  7 pack               生成 StrategyPack + prompt_preview

trace 落盘 data/traces/<trace_id>.json（文件名把 ':' 换成 '_'，Windows 兼容）。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import corpus, graph as graph_mod, schema as schema_mod
from .planning import instantiate_parameters, make_plan

__all__ = ["retrieve", "write_trace", "evaluate_conditions", "score_breakdown", "canonical_op", "default_features"]

TRACE_STEPS = [
    ("pattern_match", "算子计算流程匹配"),
    ("strategy_recall", "沿 PATTERN_OPTIMIZED_BY 召回策略"),
    ("condition_filter", "硬条件过滤（谓词求值）"),
    ("hardware_filter", "ISA/微架构能力过滤"),
    ("conflict_resolve", "同族去重 + 冲突消解 + 寄存器预算"),
    ("score_and_select", "六项加权打分 + top_k"),
    ("pack", "生成 StrategyPack（注入 prompt）"),
]

_SQRT = math.sqrt


# ---------------------------------------------------------------- 查询解析


def canonical_op(graph, op_name: str) -> tuple:
    """`dgemm` -> ('gemm', 'd')；同时返回由 BLAS 前缀推出的 dtype。"""
    name = (op_name or "").strip().lower()
    dtype = None
    if len(name) > 1 and name[0] in "sdcz" and ("op:%s" % name[1:]) in graph.nodes:
        dtype = name[0]
        name = name[1:]
    if ("op:%s" % name) in graph.nodes:
        return name, dtype
    # 兜底：按名字包含匹配（如 dgemm 传成 gemm_op5）
    for node in graph.nodes_of_type("Operator"):
        if node["id"].split(":", 1)[1] in name:
            return node["id"].split(":", 1)[1], dtype
    return None, dtype


def profile_of(graph, profile_name: str):
    name = (profile_name or "").strip()
    if name.startswith("hw:"):
        name = name[3:]
    node_id = "hw:%s" % name
    if node_id in graph.nodes:
        return node_id, graph.nodes[node_id]
    return None, None


def default_features(graph, op: str, dtype: str | None) -> dict:
    """默认代码特征：查询代码缺失时用算子注册表 + 配置的最常见形态。"""
    node = graph.nodes.get("op:%s" % op) or {}
    props = node.get("properties") or {}
    dtypes = props.get("dtype_variants") or ["d"]
    if dtype is None:
        dtype = "d" if "d" in dtypes else dtypes[0]
    op_class = props.get("op_class", "vector")
    depth = 3 if op_class in ("matrix", "triangular") else 1
    return {
        "dtype": dtype,
        "precision": "single" if dtype in ("s", "c") else "double",
        "stride_x": 1,
        "stride_y": 1,
        "loop_depth": depth,
        "size_regime": "large",
        "aligned": True,
        "tail_divisible": False,
        "structural": props.get("structural", "none"),
        "reduction": bool(props.get("reduction", False)),
    }


# ---------------------------------------------------------------- 条件求值


def _cmp(op: str, actual, value) -> bool:
    if op == "eq":
        return actual == value
    if op == "ne":
        return actual != value
    if op == "ge":
        return actual >= value
    if op == "le":
        return actual <= value
    if op == "gt":
        return actual > value
    if op == "lt":
        return actual < value
    if op == "in":
        return actual in value
    if op == "contains":
        return value in actual
    return False


def evaluate_conditions(conditions: list, features: dict) -> dict:
    """逐条求值；返回 {hard_fail: [reason], fit: 0..1, satisfied: [...], violated: [...]}。"""
    hard_fail = []
    satisfied = []
    violated = []
    weight_total = 0.0
    weight_ok = 0.0
    for cond in conditions or []:
        pred = cond.get("pred")
        op = cond.get("op", "eq")
        value = cond.get("value")
        hard = bool(cond.get("hard", False))
        weight = float(cond.get("weight", 1.0) or 1.0)
        actual = features.get(pred)
        if actual is None:
            if hard:
                hard_fail.append("%s 特征缺失（未知），无法满足硬条件 %s %s %r" % (pred, pred, op, value))
            continue
        weight_total += weight
        if _cmp(op, actual, value):
            weight_ok += weight
            satisfied.append(cond)
        else:
            violated.append(cond)
            if hard:
                hard_fail.append("%s=%s 不满足 %s %s %r" % (pred, actual, pred, op, value))
    fit = (weight_ok / weight_total) if weight_total > 0 else 0.5
    if not conditions:
        fit = 0.5
    return {"hard_fail": hard_fail, "fit": fit, "satisfied": satisfied, "violated": violated}


# ---------------------------------------------------------------- 打分


def _edge_map(graph, node_id: str, rel: str) -> list:
    return graph.out_edges(node_id, rel)


def _params_of(graph, node_id: str) -> dict:
    """策略实例的参数值（只取该实例的值，不 fallback 到全局参数节点的值）。"""
    out = {}
    for edge in _edge_map(graph, node_id, "STRATEGY_HAS_PARAMETER"):
        node = graph.nodes.get(edge["tail"]) or {}
        props = node.get("properties") or {}
        name = props.get("name") or edge["tail"].split(":", 1)[1]
        out[name] = (edge.get("properties") or {}).get("value_in_instance")
    return out


def _eligible_speedup(profile, evidence, relation, workload, baseline):
    """Score only matched, measured per-strategy comparisons."""
    value = evidence.get("value")
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0
        and evidence.get("metric") == "speedup"
        and profile.get("measure_env") == evidence.get("measure_env") == "native"
        and evidence.get("platform_id") == "hw:%s" % profile.get("profile_name")
        and evidence.get("rerun_needed") is False
        and relation.get("scope") == "strategy"
        and relation.get("role") in ("supports", "contradicts")
        and bool(profile.get("compiler")) and profile.get("compiler") == evidence.get("compiler")
        and bool(evidence.get("protocol"))
        and bool(workload) and evidence.get("workload") == workload
        and bool(baseline) and evidence.get("baseline") == baseline
    )


def score_breakdown(graph, strategy_id: str, cfg, profile_id: str, features: dict, eval_result: dict,
                    workload=None, baseline=None) -> dict:
    """§5.4 六项加权打分，每项都可在结果里展开。"""
    weights = (cfg.retrieval_cfg.get("weights") or {})
    node = graph.nodes.get(strategy_id) or {}
    props = node.get("properties") or {}
    # 注意：PATTERN_OPTIMIZED_BY 方向是 pattern -> strategy，因此从入边取
    in_edges = graph.in_edges(strategy_id, "PATTERN_OPTIMIZED_BY")
    generalizes_in = graph.in_edges(strategy_id, "STRATEGY_GENERALIZES")
    if in_edges:
        match_p = 1.0
    elif generalizes_in:
        match_p = 0.6
    else:
        match_p = 0.3
    fit_c = float(eval_result.get("fit", 0.5))
    # hw 分：profile 绑定 + arch_scope 匹配度（架构特定实现优先于通用/参数化实现）
    arch_scope = props.get("arch_scope") or "arch_agnostic"
    bound = profile_id in (props.get("hardware_profile_ids") or [])
    if arch_scope == "arch_specific" and bound:
        hw = 1.0
    elif arch_scope == "parameterized":
        hw = 0.85 if bound else 0.4
    elif arch_scope == "arch_agnostic":
        hw = 0.6 if bound else 0.3
    else:
        hw = 0.4
    profile_node = graph.nodes.get("hw:%s" % profile_id) or {}
    uarch_id = (profile_node.get("properties") or {}).get("uarch_id")
    for edge in _edge_map(graph, strategy_id, "STRATEGY_TARGETS_UARCH"):
        if edge["tail"] == uarch_id and (edge.get("properties") or {}).get("fit") in ("ideal", "good"):
            hw = min(1.0, hw + 0.1)
            break
    # Historical QEMU timings and operator-level results never enter either component.
    measured = []
    for edge in _edge_map(graph, strategy_id, "STRATEGY_EVIDENCED_BY"):
        ev_node = graph.nodes.get(edge["tail"]) or {}
        ev_props = ev_node.get("properties") or {}
        if _eligible_speedup(profile_node.get("properties", {}), ev_props,
                             edge.get("properties") or {}, workload, baseline):
            measured.append(max(-1.0, min(1.0, math.log2(ev_props["value"]))))
    ev_score = sum(measured) / len(measured) if measured else 0.0
    support_count = int(props.get("support_count") or 1)
    sup = min(1.0, support_count / 5.0)
    tr_pen = 0.0  # 保留公开 trace 字段。无匹配证据时计中性分。
    total = (
        float(weights.get("match_p", 0.35)) * match_p
        + float(weights.get("fit_c", 0.20)) * fit_c
        + float(weights.get("hw", 0.15)) * hw
        + float(weights.get("ev", 0.20)) * ev_score
        + float(weights.get("sup", 0.10)) * sup
        - float(weights.get("tr_penalty", 0.15)) * tr_pen
    )
    return {
        "match_p": round(match_p, 4),
        "fit_c": round(fit_c, 4),
        "hw": round(hw, 4),
        "ev": round(ev_score, 4),
        "sup": round(sup, 4),
        "tr_penalty": round(tr_pen, 4),
        "total": round(max(0.0, total), 4),
    }


# ---------------------------------------------------------------- trace


def _step(number, name, description, input_obj=None, candidates=None, selected=None, dropped=None,
          reason=None, **extra) -> dict:
    step = {
        "step": number,
        "name": name,
        "description": description,
        "input": input_obj or {},
        "candidates": list(candidates or []),
        "selected": list(selected or []),
        "dropped": list(dropped or []),
    }
    if reason:
        step["reason"] = reason
    step.update(extra)
    return step


def write_trace(trace: dict, traces_dir) -> Path:
    traces_dir = Path(traces_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)
    filename = trace["trace_id"].replace(":", "_").replace("/", "_") + ".json"
    path = traces_dir / filename
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8")
    return path


def _strict_features(cfg, provided):
    if not isinstance(provided, Mapping):
        raise ValueError("严格检索需传入任务的 --code-features JSON 对象")
    specs = schema_mod.load_schema(cfg.schema_dir).predicates
    for name, value in provided.items():
        if name not in specs:
            raise ValueError("未知条件特征：%s" % name)
        spec = specs[name]
        # JSON null 表示未知，硬条件会拒绝该候选。
        if value is None:
            continue
        if ((spec["type"] == "bool" and type(value) is not bool)
                or (spec["type"] == "int" and type(value) is not int)
                or (spec["type"] == "enum" and value not in spec["values"])):
            raise ValueError("条件特征 %s 取值不符合 schema：%r" % (name, value))
    return dict(provided)


def _exact_operator(graph, name):
    name = name.strip().lower()
    if "op:" + name in graph.nodes:
        return name
    if name and name[0] in "sdcz" and "op:" + name[1:] in graph.nodes:
        return name[1:]
    return None


def retrieve(cfg, op: str, profile: str | None = None, top_k: int | None = None,
             code_features: dict | None = None, data_dir=None, traces_dir=None,
             write: bool = True, *, strict: bool = False, source_op: str | None = None,
             independent: bool = False, workload: dict | None = None,
             baseline: dict | None = None) -> dict:
    """一次可回放的检索：返回 trace dict（并落盘 data/traces/）。"""
    data_dir = Path(data_dir) if data_dir else cfg.data_dir
    graph = graph_mod.load_graph(data_dir)
    if not graph.nodes:
        raise FileNotFoundError("图数据为空：%s（先运行 build --offline）" % data_dir)
    top_k = int(top_k or cfg.retrieval_cfg.get("top_k", 6) or 6)
    if top_k < 1:
        raise ValueError("top_k 必须大于零")
    if (source_op or independent) and not strict:
        raise ValueError("类比查询和独立策略卡对照均要求 --strict")
    if strict:
        features = _strict_features(cfg, code_features)
        if not re.fullmatch(r"[A-Za-z0-9_]+", op):
            raise ValueError("严格检索的任务名必须是字母、数字或下划线")
        canonical = _exact_operator(graph, op)
        op_name = canonical or (op.lower() if source_op else None)
        source_name = _exact_operator(graph, source_op) if source_op else canonical
        if source_op and not source_name:
            raise ValueError("类比来源算子未在此图谱登记：%s" % source_op)
    else:
        op_name, dtype = canonical_op(graph, op)
        source_name = op_name
        features = default_features(graph, op_name, dtype)
        features.update(code_features or {})
    analogy = bool(source_op and source_name != op_name)
    if op_name is None:
        raise ValueError("未知算子 %r；可用算子：%s" % (
            op, ", ".join(sorted(n["id"].split(":", 1)[1] for n in graph.nodes_of_type("Operator")))))
    profile = profile or "x86_avx2"
    profile_node_id, profile_node = profile_of(graph, profile)
    if profile_node is None:
        raise ValueError("未知硬件 profile %r；可用：%s" % (
            profile, ", ".join(sorted(n["id"].split(":", 1)[1] for n in graph.nodes_of_type("HardwareProfile")))))
    profile_name = profile_node_id.split(":", 1)[1]
    steps = []
    operator_id = "op:%s" % source_name

    # ---- step 1: pattern_match ----
    pattern_ids = sorted(graph.neighbors(operator_id, "OPERATOR_HAS_PATTERN"))
    steps.append(_step(
        1, "pattern_match", "算子计算流程匹配",
        {"operator": operator_id, "operator_name": op_name, "analogy_source": source_name if analogy else None},
        candidates=pattern_ids, selected=pattern_ids,
        reason="计算流程来源：%s 注册了 %d 个计算流程" % (source_name, len(pattern_ids)),
    ))

    # ---- step 2: strategy_recall ----
    recalled = {}
    for pattern_id in pattern_ids:
        for edge in graph.out_edges(pattern_id, "PATTERN_OPTIMIZED_BY"):
            recalled.setdefault(edge["tail"], []).append(pattern_id)
    expanded = {}
    for strategy_id in list(recalled):
        for edge in graph.out_edges(strategy_id, "STRATEGY_GENERALIZES"):
            expanded.setdefault(edge["tail"], []).append(strategy_id)
    candidates = sorted(set(recalled) | set(expanded))
    if analogy:
        candidates = sorted(sid for sid in expanded if
                            graph.nodes[sid]["properties"].get("source") == "abstract")
    steps.append(_step(
        2, "strategy_recall", "沿 PATTERN_OPTIMIZED_BY 召回策略 + GENERALIZES 抽象扩展",
        {"patterns": pattern_ids, "channels": ["pattern", "generalizes"]},
        candidates=candidates, selected=candidates,
        reason="沿 %d 个计算流程召回 %d 条策略，GENERALIZES 扩展 %d 条抽象策略" % (
            len(pattern_ids), len(recalled), len(expanded)),
    ))

    # ---- step 3: condition_filter ----
    survivors = []
    dropped = []
    eval_results = {}
    for strategy_id in candidates:
        node = graph.nodes.get(strategy_id) or {}
        props = node.get("properties") or {}
        conditions = props.get("conditions") or []
        if props.get("condition_status") == "text":
            if strict:
                dropped.append({"id": strategy_id, "reason": "适用条件尚未结构化并验证"})
                continue
            eval_results[strategy_id] = {"hard_fail": [], "fit": 0.5, "satisfied": [], "violated": []}
            survivors.append(strategy_id)
            continue
        result = evaluate_conditions(conditions, features)
        eval_results[strategy_id] = result
        if result["hard_fail"]:
            dropped.append({"id": strategy_id, "reason": "；".join(result["hard_fail"])})
        else:
            survivors.append(strategy_id)
    steps.append(_step(
        3, "condition_filter", "硬条件过滤：策略 conditions 对查询 code_features 求值",
        {"code_features": features},
        candidates=candidates, selected=survivors, dropped=dropped,
        reason="谓词求值：%d 条通过、%d 条硬条件淘汰" % (len(survivors), len(dropped)),
    ))

    # ---- step 4: hardware_filter ----
    profile_isa = list((profile_node.get("properties") or {}).get("isa") or [])
    hw_survivors = []
    hw_dropped = []
    for strategy_id in survivors:
        node = graph.nodes.get(strategy_id) or {}
        props = node.get("properties") or {}
        bound = props.get("hardware_profile_ids") or []
        if profile_name not in bound:
            hw_dropped.append({"id": strategy_id,
                               "reason": "hardware_profile_ids=%s 不含 %s" % (bound, profile_name)})
            continue
        fail_reason = None
        for edge in graph.out_edges(strategy_id, "STRATEGY_REQUIRES_ISA"):
            if not (edge.get("properties") or {}).get("hard"):
                continue
            isa_node = graph.nodes.get(edge["tail"]) or {}
            isa_props = isa_node.get("properties") or {}
            if isa_props.get("isa") not in profile_isa:
                fail_reason = "requires %s（%s），profile %s 仅支持 %s" % (
                    isa_props.get("isa"), isa_props.get("feature"), profile_name, profile_isa)
                break
        if fail_reason:
            hw_dropped.append({"id": strategy_id, "reason": fail_reason})
        else:
            hw_survivors.append(strategy_id)
    steps.append(_step(
        4, "hardware_filter", "ISA/硬件 profile 能力过滤（profile 绑定 + hard ISA 要求）",
        {"profile": "hw:%s" % profile_name, "profile_isa": profile_isa,
         "vector_regs": (profile_node.get("properties") or {}).get("vector_regs")},
        candidates=survivors, selected=hw_survivors, dropped=hw_dropped,
        reason="%d 条满足 %s 的硬件约束，%d 条被淘汰" % (len(hw_survivors), profile_name, len(hw_dropped)),
    ))

    # ---- step 5: conflict_resolve ----
    # 具体策略优先于抽象策略（否则抽象策略会以高 support_count 挤掉具体实现）；
    # 同级按 support_count 降序，再按 action/id 保证确定性。
    def _rank_key(sid: str):
        node = graph.nodes.get(sid) or {}
        props = node.get("properties") or {}
        scope = props.get("arch_scope") or "arch_agnostic"
        bound = profile_name in (props.get("hardware_profile_ids") or [])
        if scope == "arch_specific" and bound:
            tier = 0
        elif scope == "parameterized":
            tier = 1
        elif scope == "arch_agnostic":
            tier = 2
        else:
            tier = 3
        return (tier, -int(props.get("support_count") or 1), props.get("action") or "", sid)

    scores_before_selection = {
        sid: score_breakdown(graph, sid, cfg, profile_name, features, eval_results.get(sid, {}),
                             workload=workload, baseline=baseline)
        for sid in hw_survivors
    }
    ordered = sorted(hw_survivors, key=(
        (lambda sid: (-scores_before_selection[sid]["total"], sid)) if strict else _rank_key
    ))
    chosen = []
    conflict_dropped = []
    used_family = {}
    reg_budget = 0.0
    uarch_id = (profile_node.get("properties") or {}).get("uarch_id")
    uarch_props = (graph.nodes.get(uarch_id) or {}).get("properties") or {}
    vector_regs = float(uarch_props.get("num_vector_regs") or 16)
    budget_cap = vector_regs * float(cfg.retrieval_cfg.get("register_budget_ratio", 0.8))
    for strategy_id in ordered:
        node = graph.nodes.get(strategy_id) or {}
        props = node.get("properties") or {}
        family = props.get("family")
        action = props.get("action")
        family_key = (family, action)
        if not independent and family_key in used_family:
            conflict_dropped.append({
                "id": strategy_id,
                "reason": "与已选 %s 同族同动作（%s/%s），避免重复注入（架构特性优先保留 %s）" % (
                    used_family[family_key], family, action,
                    ((graph.nodes.get(used_family[family_key]) or {}).get("properties") or {}).get("arch_scope")),
            })
            continue
        conflict_edge = None
        for selected_id in (() if independent else chosen):
            edges = graph.edges_between(selected_id, strategy_id, "STRATEGY_CONFLICTS_WITH")
            if edges:
                conflict_edge = edges[0]
                break
        if conflict_edge is not None:
            edge_props = conflict_edge.get("properties") or {}
            conflict_dropped.append({
                "id": strategy_id,
                "reason": "与 %s 冲突（resource=%s, severity=%s）：%s" % (
                    conflict_edge["head"] if conflict_edge["head"] != strategy_id else conflict_edge["tail"],
                    edge_props.get("resource"), edge_props.get("severity"), edge_props.get("reason")),
            })
            continue
        estimate = _register_estimate(_params_of(graph, strategy_id)) if not strict else 0
        if not strict and reg_budget + estimate > budget_cap:
            conflict_dropped.append({
                "id": strategy_id,
                "reason": "寄存器预算超限：已占用 %.0f + 本条 %.0f > %.0f（profile vector_regs 的 %.0f%%）" % (
                    reg_budget, estimate, budget_cap, float(cfg.retrieval_cfg.get("register_budget_ratio", 0.8)) * 100),
            })
            continue
        reg_budget += estimate
        chosen.append(strategy_id)
        used_family[family_key] = strategy_id
    steps.append(_step(
        5, "conflict_resolve", (
            "逐张筛选，无跨策略约束" if independent else
            "同族去重 + 冲突边消解" if strict else
            "同族去重 + 冲突边消解 + 示范模式寄存器估算"
        ),
        {"register_budget_cap": budget_cap, "vector_regs": vector_regs,
         "uarch": uarch_id},
        candidates=hw_survivors, selected=chosen, dropped=conflict_dropped,
        reason="%d 条通过冲突消解（占用寄存器预算 %.0f / %.0f）" % (len(chosen), reg_budget, budget_cap),
        register_budget_used=round(reg_budget, 2),
    ))

    # ---- step 6: score_and_select ----
    scores = {}
    breakdowns = {}
    for strategy_id in chosen:
        breakdown = scores_before_selection[strategy_id]
        breakdowns[strategy_id] = breakdown
        scores[strategy_id] = breakdown["total"]
    ranked = sorted(chosen, key=lambda sid: (-scores[sid], sid))
    selected = ranked[:top_k]
    score_dropped = [{"id": sid, "reason": "score %.3f 低于 top_k=%d 截断" % (scores[sid], top_k)}
                     for sid in ranked[top_k:]]
    steps.append(_step(
        6, "score_and_select", "六项加权打分（match_p/fit_c/hw/ev/sup - transfer penalty）并取 top_k",
        {"top_k": top_k, "weights": cfg.retrieval_cfg.get("weights")},
        candidates=chosen, selected=selected, dropped=score_dropped,
        scores={sid: round(scores[sid], 4) for sid in chosen},
        score_breakdown=breakdowns,
        reason="按总分降序取 top %d" % len(selected),
    ))

    # ---- step 7: pack ----
    pack_strategies = []
    for strategy_id in selected:
        node = graph.nodes.get(strategy_id) or {}
        props = node.get("properties") or {}
        template = None
        for edge in graph.out_edges(strategy_id, "STRATEGY_HAS_TEMPLATE"):
            tpl_node = graph.nodes.get(edge["tail"]) or {}
            tpl_props = tpl_node.get("properties") or {}
            template = {
                "id": edge["tail"],
                "language": tpl_props.get("language"),
                "skeleton": None if analogy else tpl_props.get("skeleton"),
                "kind": "abstract_analogy" if analogy else "illustrative_skeleton",
                "slots": tpl_props.get("slots"),
                "check_hints": tpl_props.get("check_hints"),
                "preconditions": tpl_props.get("preconditions"),
            }
        parameters = []
        for edge in graph.out_edges(strategy_id, "STRATEGY_HAS_PARAMETER"):
            p_node = graph.nodes.get(edge["tail"]) or {}
            values = dict(p_node.get("properties") or {})
            values["value_in_instance"] = (edge.get("properties") or {}).get("value_in_instance")
            parameters.append({"id": edge["tail"], **values})
        evidence = []
        for edge in graph.out_edges(strategy_id, "STRATEGY_EVIDENCED_BY"):
            ev_node = graph.nodes.get(edge["tail"]) or {}
            ev_props = ev_node.get("properties") or {}
            relation = edge.get("properties") or {}
            eligible = _eligible_speedup(profile_node.get("properties", {}), ev_props, relation,
                                        workload, baseline)
            if not eligible:
                continue
            evidence.append({
                "id": edge["tail"],
                "metric": ev_props.get("metric"),
                "value": ev_props.get("value"),
                "platform_id": ev_props.get("platform_id"),
                "role": (edge.get("properties") or {}).get("role"),
                "scope": (edge.get("properties") or {}).get("scope"),
                "caveat": ev_props.get("caveat"),
                "eligible_for_ranking": eligible,
            })
        pack_strategies.append({
            "id": strategy_id,
            "name": node.get("name"),
            "family": props.get("family"),
            "action": props.get("action"),
            "level": props.get("level"),
            "stage": props.get("stage"),
            "arch_scope": props.get("arch_scope"),
            "arch": props.get("arch"),
            "arches": props.get("arches"),
            "arch_parameters": props.get("arch_parameters"),
            "score": round(scores[strategy_id], 4),
            "score_breakdown": breakdowns[strategy_id],
            "why_selected": {
                "patterns": sorted(p for p in pattern_ids if any(
                    e["tail"] == strategy_id for e in graph.out_edges(p, "PATTERN_OPTIMIZED_BY"))),
                "abstracted_from": sorted(expanded.get(strategy_id, [])),
                "conditions_satisfied": [c["pred"] for c in eval_results.get(strategy_id, {}).get("satisfied", [])],
                "hardware": profile_name,
                "source": (node.get("provenance") or {}).get("source_file"),
            },
            "conditions": props.get("conditions"),
            "template": template,
            "parameters": parameters,
            "evidence": evidence,
            "provenance": node.get("provenance"),
        })
    if strict:
        for card in pack_strategies:
            card["target_parameters"] = instantiate_parameters(
                card, features, profile_isa, uarch_props)
    prompt_preview = _prompt_preview(op_name, features, profile_name, pack_strategies)
    pack = {
        "schema_version": "pack-v1",
        "query": {
            "operator": op_name,
            "dtype": features.get("dtype"),
            "hardware": "hw:%s" % profile_name,
            "patterns": pattern_ids,
            "code_features": features,
            "feature_mode": "explicit" if strict else "demo_defaults",
            "source_operator": source_name if analogy else None,
            "workload": workload,
            "baseline": baseline,
            "top_k": top_k,
        },
        "strategies": pack_strategies,
    }
    if strict:
        pack["plan"] = None if independent else make_plan(graph, pack_strategies)
    steps.append(_step(
        7, "pack", "生成 StrategyPack（策略 + 模板 + 参数 + 证据 + 溯源），供生成端注入 prompt",
        {"selected": selected},
        candidates=selected, selected=selected, dropped=[],
        reason="打包 %d 条策略 / %d 个模板" % (
            len(pack_strategies), sum(1 for s in pack_strategies if s.get("template"))),
        result={"strategies": selected, "templates": [s["template"]["id"] for s in pack_strategies if s.get("template")]},
    ))

    now = datetime.now().astimezone()
    trace_id = "trace:%s:%s:%s:%s" % (now.strftime("%Y%m%dT%H%M%S"), op_name, profile_name, uuid4().hex[:8])
    trace = {
        "trace_id": trace_id,
        "query": {
            "operator": op_name,
            "raw_operator": op,
            "hardware_profile_id": profile_name,
            "hardware_profile_node": profile_node_id,
            "code_features": features,
            "feature_mode": "explicit" if strict else "demo_defaults",
            "source_operator": source_name if analogy else None,
            "top_k": top_k,
        },
        "steps": steps,
        "result": {
            "strategies": selected,
            "pack": pack,
            "prompt_preview": prompt_preview,
        },
        "created_at": now.isoformat(timespec="seconds"),
    }
    if write:
        traces_dir = Path(traces_dir) if traces_dir else (cfg.traces_dir if data_dir == cfg.data_dir else data_dir / "traces")
        path = write_trace(trace, traces_dir)
        trace["trace_path"] = str(path)
    return trace


def _register_estimate(params: dict) -> float:
    """由参数估算向量寄存器占用：reg_block_n × LMUL（无参数则 0）。"""
    reg_block_n = params.get("reg_block_n")
    lmul = params.get("lmul")
    if isinstance(reg_block_n, (int, float)) and isinstance(lmul, (int, float)):
        return float(reg_block_n) * float(lmul)
    if isinstance(reg_block_n, (int, float)):
        return float(reg_block_n)
    return 0.0


def _prompt_preview(op: str, features: dict, profile: str, pack_strategies: list) -> str:
    lines = [
        "目标算子：%s（dtype=%s，loop_depth=%s，stride_x=%s，size_regime=%s）。" % (
            op, features.get("dtype"), features.get("loop_depth"), features.get("stride_x"), features.get("size_regime")),
        "目标硬件 profile：%s。请只使用下列知识块中的策略与模板槽位，不要引入未给出的性能数字。" % profile,
    ]
    for idx, strategy in enumerate(pack_strategies, 1):
        lines.append("%d) %s（%s/%s，score=%.3f）：条件 %s；参数 %s。" % (
            idx, strategy["name"], strategy["family"], strategy["action"], strategy["score"],
            "、".join("%s %s %r" % (c["pred"], c["op"], c["value"]) for c in (strategy.get("conditions") or [])),
            "、".join("%s=%s" % (
                p.get("name"),
                p.get("value_in_instance") if p.get("value_in_instance") is not None else p.get("value_in_source"))
                for p in strategy.get("parameters") or []),
        ))
    text = "\n".join(lines)
    return text[:500]
