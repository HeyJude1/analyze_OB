"""
泛化引擎 — KG v2 核心算法
给定代码特征, 通过直接匹配 + 抽象遍历找到可应用的优化原则

流程: DIRECT_MATCH → ABSTRACTION_EXPAND → CONSTRAINT_FILTER → RANK
"""

import json
from typing import Dict, List, Set, Tuple, Optional
from .vector_store import VectorStore
from .graph import KnowledgeGraph
from .schemas import RelType


class GeneralizationEngine:
    def __init__(self, store: VectorStore):
        self.store = store
        self.kg = KnowledgeGraph()
        self.kg.load_from_milvus()

    def search(self, code_characteristics: List[Dict[str, str]],
               hardware_context: Dict[str, List] = None,
               max_abstraction_depth: int = 3,
               alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> List[Dict]:
        """
        泛化搜索主入口
        code_characteristics: [{type, value_descriptor, metric_range, ...}, ...]
        hardware_context: {"available": ["simd_width_256", "cache_l2_512KB"]}
        """

        if not self.kg.is_loaded() or not code_characteristics:
            return []

        hw = hardware_context or {}
        candidates = {}

        # ====== A. DIRECT_MATCH ======
        candidates = self._direct_match(code_characteristics)

        # ====== B. ABSTRACTION_EXPAND ======
        candidates = self._abstraction_expand(candidates, max_abstraction_depth)

        # ====== C. CONSTRAINT_FILTER ======
        candidates = self._constraint_filter(candidates, code_characteristics, hw)

        # ====== D. RANK ======
        ranked = self._rank(candidates, code_characteristics, alpha, beta, gamma)

        # ====== E. ENRICH ======
        return self._enrich_results(ranked)

    def _direct_match(self, chars: List[Dict]) -> Dict[str, Dict]:
        """向量搜索 + 关系遍历找到直接匹配的原则"""
        candidates = {}

        # 1. 对每个特征做向量搜索
        for c in chars:
            vec = self.store.embed(json.dumps(c, ensure_ascii=False))
            hits = self.store.search("code_characteristic", [vec], limit=10, threshold=0.7)
            for h in hits[0]:
                char_uid = h.id
                # 找到具有此特征的 SourcePattern
                patterns = self.kg.get_neighbors_of_type(char_uid, RelType.HAS_CHARACTERISTIC, "in")
                for p in patterns:
                    p_uid = p.get("source_uid") or p.get("target_uid")
                    # 找到 pattern 实例化的原则
                    principles = self.kg.get_neighbors_of_type(p_uid, RelType.INSTANCE_OF, "out")
                    for pr in principles:
                        pr_uid = pr.get("target_uid") or pr.get("source_uid")
                        key = pr_uid
                        if key not in candidates:
                            candidates[key] = {
                                "principle_uid": key,
                                "match_score": 0, "paths": [], "matched_chars": [],
                                "graph_distance": 0, "abstraction_chain": []
                            }
                        candidates[key]["match_score"] = max(candidates[key]["match_score"], h.distance)
                        candidates[key]["paths"].append("direct:HAS_CHARACTERISTIC→INSTANCE_OF")
                        candidates[key]["matched_chars"].append(c.get("characteristic_type", ""))
                        candidates[key]["graph_distance"] = 0

                # 2. 直接 APPLIES_WHEN 匹配
                principles = self.kg.get_neighbors_of_type(char_uid, RelType.APPLIES_WHEN, "in")
                for pr in principles:
                    pr_uid = pr.get("source_uid") or pr.get("target_uid")
                    key = pr_uid
                    if key not in candidates:
                        candidates[key] = {
                            "principle_uid": key,
                            "match_score": 0, "paths": [], "matched_chars": [],
                            "graph_distance": 0, "abstraction_chain": []
                        }
                    candidates[key]["match_score"] = max(candidates[key]["match_score"], h.distance)
                    candidates[key]["paths"].append("direct:APPLIES_WHEN")
                    candidates[key]["matched_chars"].append(c.get("characteristic_type", ""))
                    candidates[key]["graph_distance"] = 0

        return candidates

    def _abstraction_expand(self, candidates: Dict[str, Dict],
                            max_depth: int) -> Dict[str, Dict]:
        """沿 GENERALIZES / COMPOSES_WITH / ANALOGOUS_TO 扩展候选"""
        expanded = dict(candidates)

        for uid in list(candidates.keys()):
            base_score = candidates[uid]["match_score"]

            # 上行: GENERALIZES (衰减 0.7^depth)
            ancestors = self.kg.traverse_typed(uid, [RelType.GENERALIZES], "out", max_depth)
            for anc_uid, depth, path in ancestors:
                decay = 0.7 ** depth
                score = base_score * decay
                key = anc_uid
                if key not in expanded or expanded[key].get("match_score", 0) < score:
                    expanded[key] = {
                        "principle_uid": anc_uid,
                        "match_score": score,
                        "paths": candidates[uid]["paths"] + [f"generalizes:d{depth}"],
                        "matched_chars": candidates[uid]["matched_chars"],
                        "graph_distance": depth,
                        "abstraction_chain": [uid] + path,
                        "from_generalization": True
                    }

            # 下行: 特殊化 (反向 GENERALIZES, 衰减 0.85^depth)
            specials = self.kg.traverse_typed(uid, [RelType.GENERALIZES], "in", max_depth)
            for spec_uid, depth, path in specials:
                decay = 0.85 ** depth
                score = base_score * decay
                key = spec_uid
                if key not in expanded or expanded[key].get("match_score", 0) < score:
                    expanded[key] = {
                        "principle_uid": spec_uid,
                        "match_score": score,
                        "paths": candidates[uid]["paths"] + [f"specialization:d{depth}"],
                        "matched_chars": candidates[uid]["matched_chars"],
                        "graph_distance": depth,
                        "abstraction_chain": [uid] + path,
                        "from_specialization": True
                    }

            # 横向: COMPOSES_WITH (衰减 0.85)
            comps = self.kg.get_neighbors_of_type(uid, RelType.COMPOSES_WITH, "both")
            for comp in comps:
                comp_uid = comp.get("source_uid") or comp.get("target_uid")
                key = comp_uid
                score = base_score * 0.85
                if key not in expanded or expanded[key].get("match_score", 0) < score:
                    expanded[key] = {
                        "principle_uid": comp_uid,
                        "match_score": score,
                        "paths": candidates[uid]["paths"] + ["composes_with"],
                        "matched_chars": candidates[uid]["matched_chars"],
                        "graph_distance": 1,
                        "abstraction_chain": [uid, comp_uid],
                        "from_composition": True
                    }

            # 横向: ANALOGOUS_TO (衰减 0.6, 低置信度)
            analogs = self.kg.get_neighbors_of_type(uid, RelType.ANALOGOUS_TO, "both")
            for analog in analogs:
                analog_uid = analog.get("source_uid") or analog.get("target_uid")
                key = analog_uid
                score = base_score * 0.6
                if key not in expanded or expanded[key].get("match_score", 0) < score:
                    expanded[key] = {
                        "principle_uid": analog_uid,
                        "match_score": score,
                        "paths": candidates[uid]["paths"] + ["analogous_to"],
                        "matched_chars": candidates[uid]["matched_chars"],
                        "graph_distance": 1,
                        "abstraction_chain": [uid, analog_uid],
                        "from_analogy": True
                    }

        return expanded

    def _constraint_filter(self, candidates: Dict[str, Dict],
                           chars: List[Dict], hw: Dict) -> Dict[str, Dict]:
        """应用 APPLIES_WHEN 约束和 REQUIRES 硬件检查"""
        filtered = {}
        char_types = {c.get("characteristic_type", "") for c in chars}
        available_hw = set(hw.get("available", []))

        for uid, info in candidates.items():
            principle = self.store.query_by_uid("optimization_principle", uid,
                                                 ["constraints", "principle", "scope"])
            if not principle:
                continue

            constraints_str = principle.get("constraints", "{}")
            try:
                constraints = json.loads(constraints_str) if isinstance(constraints_str, str) else constraints_str
            except json.JSONDecodeError:
                constraints = {}

            status = "pass"
            reasons = []

            # 检查 REQUIRES 硬件
            requires = self.kg.get_neighbors_of_type(uid, RelType.REQUIRES, "out")
            for req in requires:
                hw_name = req.get("target_name", "") or req.get("source_name", "")
                if hw_name and hw_name not in available_hw:
                    status = "degraded"
                    reasons.append(f"missing_hw:{hw_name}")

            # 检查 APPLIES_WHEN 约束
            applies = self.kg.get_neighbors_of_type(uid, RelType.APPLIES_WHEN, "out")
            for apply_edge in applies:
                desc = apply_edge.get("description", "{}")
                try:
                    cond = json.loads(desc) if isinstance(desc, str) else desc
                except json.JSONDecodeError:
                    cond = {}
                if cond.get("condition_type") == "requires":
                    required_char = apply_edge.get("target_name", "")
                    if required_char not in char_types:
                        status = "fail"
                        reasons.append(f"missing_char:{required_char}")
                        break
                elif cond.get("condition_type") == "conflicts":
                    conflict_char = apply_edge.get("target_name", "")
                    if conflict_char in char_types:
                        status = "fail"
                        reasons.append(f"conflict_char:{conflict_char}")
                        break

            if status == "fail":
                continue

            info["constraint_status"] = status
            info["constraint_reasons"] = reasons
            filtered[uid] = info

        return filtered

    def _rank(self, candidates: Dict[str, Dict], chars: List[Dict],
              alpha: float, beta: float, gamma: float) -> List[Dict]:
        """计算最终得分并排序"""
        ranked = []
        for uid, info in candidates.items():
            principle = self.store.query_by_uid("optimization_principle", uid,
                                                 ["evidence_strength", "source_count", "principle", "name", "scope", "level"])
            if not principle:
                continue

            graph_dist = info.get("graph_distance", 0)
            graph_score = 1.0 / (1.0 + graph_dist)
            evidence = float(principle.get("evidence_strength", 0.5))
            char_score = info.get("match_score", 0)

            final_score = alpha * char_score + beta * graph_score + gamma * evidence

            if info.get("constraint_status") == "degraded":
                final_score *= 0.5

            ranked.append({
                "principle_uid": uid,
                "principle_name": principle.get("name", ""),
                "principle_text": principle.get("principle", ""),
                "scope": principle.get("scope", ""),
                "level": principle.get("level", ""),
                "final_score": round(final_score, 4),
                "score_breakdown": {
                    "characteristic_similarity": round(char_score, 3),
                    "graph_proximity": round(graph_score, 3),
                    "evidence_strength": round(evidence, 3)
                },
                "paths": info.get("paths", []),
                "graph_distance": graph_dist,
                "constraint_status": info.get("constraint_status", "pass"),
                "constraint_reasons": info.get("constraint_reasons", []),
                "is_generalized": info.get("from_generalization", False) or
                                   info.get("from_specialization", False) or
                                   info.get("from_composition", False) or
                                   info.get("from_analogy", False),
                "generalization_type": "generalization" if info.get("from_generalization") else
                                       "specialization" if info.get("from_specialization") else
                                       "composition" if info.get("from_composition") else
                                       "analogy" if info.get("from_analogy") else
                                       "direct"
            })

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        return ranked

    def _enrich_results(self, ranked: List[Dict]) -> List[Dict]:
        """富化结果: 添加图上下文"""
        for r in ranked:
            uid = r["principle_uid"]
            r["graph_context"] = self.kg.get_strategy_context(uid)
        return ranked
