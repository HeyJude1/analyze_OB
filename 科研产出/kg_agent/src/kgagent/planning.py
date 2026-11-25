"""Deterministic proposal construction. Parameters and edges describe candidates."""

from __future__ import annotations


def instantiate_parameters(card: dict, features: dict, isa: list, uarch: dict) -> dict:
    """Build a small target-specific search space from declared hardware limits."""
    action = card.get("action")
    if action not in {"vectorize", "tail_handle", "unroll", "tile"}:
        return {}
    if card.get("arch_scope") == "arch_agnostic" and action in {"unroll", "tile"}:
        return {}
    sew = {"s": 32, "d": 64, "c": 32, "z": 64}.get(features.get("dtype"))
    width = uarch.get("vlen_bits")
    registers = uarch.get("num_vector_regs")
    if not sew or not isinstance(width, int) or width < sew or not registers:
        return {}
    rvv = "RVV 1.0" in isa
    if not rvv and not ({"AVX2", "AVX512F", "NEON", "SVE"} & set(isa)):
        return {}
    reserved = 4
    available = max(0, registers - reserved)
    result = {
        "status": "candidate_only",
        "source": "declared_target_profile",
        "element_bits": sew,
        "elements_per_register": width // sew,
        "vector_register_budget": available,
        "reserved_vector_registers": reserved,
    }
    if rvv:
        lmul_max = uarch.get("lmul_max") or 1
        lmul_candidates = [n for n in (1, 2, 4, 8) if n <= lmul_max and available >= 3 * n]
        result["rvv"] = {
            "lmul_candidates": lmul_candidates,
            "vlmax_by_lmul": {str(n): width // sew * n for n in lmul_candidates},
            "max_accumulators_by_lmul": {str(n): available // n - 2 for n in lmul_candidates},
            "tail": "use_vsetvl_return_value",
        }
        return result
    result["simd"] = {
        "fixed_lane_count": width // sew,
        "max_accumulators": max(0, available - 2),
        "tail": "remainder_loop_or_supported_mask",
    }
    return result


def make_plan(graph, cards: list) -> dict:
    """Topologically order observed composition relations among selected cards."""
    position = {card["id"]: index for index, card in enumerate(cards)}
    outgoing = {sid: set() for sid in position}
    incoming = dict.fromkeys(position, 0)
    relations = []
    for sid in position:
        for edge in graph.out_edges(sid, "STRATEGY_COMPOSES_WITH"):
            tail = edge["tail"]
            if tail not in position:
                continue
            props = edge.get("properties") or {}
            order = props.get("order", 0)
            before, after = (tail, sid) if order < 0 else (sid, tail)
            relations.append({
                "edge_id": edge["id"],
                "strategies": [before, after],
                "kind": "compatibility" if order == 0 else "precedence",
                "condition_to_check": props.get("condition"),
                "evidence_id": props.get("evidence_id"),
            })
            if order != 0 and after not in outgoing[before]:
                outgoing[before].add(after)
                incoming[after] += 1
    stage_order = {"prep": 0, "transform": 1, "core": 2, "post": 3}
    order_by = {
        card["id"]: (stage_order.get(card.get("stage"), 4), position[card["id"]]) for card in cards
    }
    remaining = set(position)
    ordered = []
    while remaining:
        ready = [sid for sid in remaining if incoming[sid] == 0]
        if not ready:
            return {"status": "precedence_cycle", "ordered_strategy_ids": [],
                    "relations": relations, "blocked_strategy_ids": sorted(remaining)}
        sid = min(ready, key=order_by.__getitem__)
        ordered.append(sid)
        remaining.remove(sid)
        for after in outgoing[sid]:
            incoming[after] -= 1
    return {
        "status": "candidate_only" if ordered else "no_eligible_strategy",
        "ordered_strategy_ids": ordered,
        "relations": relations,
    }
