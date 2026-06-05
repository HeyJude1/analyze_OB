#!/usr/bin/env python3
"""
轻量级知识图谱引擎
在 Milvus relation collection 之上构建邻接图结构，支持：
- 多跳遍历 (BFS/DFS)
- 路径发现
- 子图提取
- PageRank 式共现排序
- 无需 Neo4j 或其他图数据库
"""

import json
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple, Optional, Any
from pymilvus import Collection


class KnowledgeGraph:
    """内存图结构，从 Milvus relations 构建"""

    def __init__(self):
        self._adj_out: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # uid -> [(rel_type, target_uid, description)]
        self._adj_in: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)   # uid <- [(rel_type, source_uid, description)]
        self._entity_index: Dict[str, Dict[str, Any]] = {}  # uid -> entity info
        self._loaded = False

    def load_from_milvus(self, collection_name: str = "relation",
                         entity_collections: List[str] = None):
        """从 Milvus 加载所有关系和实体"""
        self._adj_out.clear()
        self._adj_in.clear()
        self._entity_index.clear()

        # 加载关系
        try:
            rel_col = Collection(collection_name)
            rel_col.load()
            limit = 50000
            offset = 0
            while True:
                results = rel_col.query(
                    expr="relation_id != ''",
                    output_fields=["relation_type", "head_entity_uid", "tail_entity_uid",
                                   "head_name", "tail_name", "description"],
                    limit=limit, offset=offset
                )
                if not results:
                    break
                for r in results:
                    head = r["head_entity_uid"]
                    tail = r["tail_entity_uid"]
                    rtype = r["relation_type"]
                    desc = r.get("description", "")
                    self._adj_out[head].append((rtype, tail, desc))
                    self._adj_in[tail].append((rtype, head, desc))
                offset += limit
        except Exception as e:
            print(f"⚠️ 加载关系失败: {e}")

        # 加载实体元数据
        if entity_collections is None:
            entity_collections = ["optimization_strategy", "computational_pattern",
                                  "hardware_feature", "tunable_parameter", "code_example"]
        for ec in entity_collections:
            try:
                col = Collection(ec)
                col.load()
                limit = 50000
                offset = 0
                while True:
                    results = col.query(
                        expr="uid != ''",
                        output_fields=["uid", "name"],
                        limit=limit, offset=offset
                    )
                    if not results:
                        break
                    for r in results:
                        self._entity_index[r["uid"]] = {
                            "name": r.get("name", ""),
                            "type": ec
                        }
                    offset += limit
            except Exception as e:
                print(f"⚠️ 加载实体 {ec} 失败: {e}")

        self._loaded = True
        n_entities = len(self._entity_index)
        n_edges = sum(len(v) for v in self._adj_out.values())
        print(f"✅ 知识图谱已加载: {n_entities} 实体, {n_edges} 边")

    def get_neighbors(self, uid: str, direction: str = "both") -> List[Dict[str, Any]]:
        """获取实体的邻居"""
        neighbors = []
        if direction in ("out", "both"):
            for rtype, target, desc in self._adj_out.get(uid, []):
                neighbors.append({
                    "direction": "out", "relation_type": rtype,
                    "target_uid": target, "description": desc,
                    "target_name": self._entity_index.get(target, {}).get("name", ""),
                    "target_type": self._entity_index.get(target, {}).get("type", "")
                })
        if direction in ("in", "both"):
            for rtype, source, desc in self._adj_in.get(uid, []):
                neighbors.append({
                    "direction": "in", "relation_type": rtype,
                    "source_uid": source, "description": desc,
                    "source_name": self._entity_index.get(source, {}).get("name", ""),
                    "source_type": self._entity_index.get(source, {}).get("type", "")
                })
        return neighbors

    def traverse(self, start_uid: str, max_depth: int = 2) -> Dict[str, Any]:
        """BFS 多跳遍历，返回子图"""
        visited = set()
        edges = []
        nodes = set()
        queue = deque([(start_uid, 0)])
        visited.add(start_uid)
        nodes.add(start_uid)

        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for rtype, neighbor, desc in self._adj_out.get(current, []):
                edges.append({
                    "source": current, "target": neighbor,
                    "relation_type": rtype, "description": desc, "depth": depth + 1
                })
                nodes.add(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
            for rtype, neighbor, desc in self._adj_in.get(current, []):
                edges.append({
                    "source": neighbor, "target": current,
                    "relation_type": rtype, "description": desc, "depth": depth + 1
                })
                nodes.add(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        node_details = []
        for n in nodes:
            info = self._entity_index.get(n, {})
            node_details.append({
                "uid": n, "name": info.get("name", ""), "type": info.get("type", ""),
                "out_degree": len(self._adj_out.get(n, [])),
                "in_degree": len(self._adj_in.get(n, []))
            })
        return {"start_uid": start_uid, "nodes": node_details, "edges": edges,
                "total_nodes": len(nodes), "total_edges": len(edges)}

    def find_paths(self, source: str, target: str, max_depth: int = 3) -> List[List[Dict]]:
        """查找两个实体间的所有路径"""
        all_paths = []

        def dfs(current, path, visited, depth):
            if depth > max_depth:
                return
            if current == target and len(path) > 0:
                all_paths.append(list(path))
                return
            for rtype, neighbor, desc in self._adj_out.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append({"from": current, "to": neighbor, "relation_type": rtype, "description": desc})
                    dfs(neighbor, path, visited, depth + 1)
                    path.pop()
                    visited.discard(neighbor)

        dfs(source, [], {source}, 0)
        return all_paths

    def get_strategy_context(self, strategy_uid: str) -> Dict[str, Any]:
        """获取优化策略的完整上下文（邻居子图）"""
        subgraph = self.traverse(strategy_uid, max_depth=2)

        patterns = []
        params = []
        hw_features = []
        code_examples = []

        for node in subgraph["nodes"]:
            if node["uid"] == strategy_uid:
                continue
            ntype = node["type"]
            if ntype == "computational_pattern":
                patterns.append(node)
            elif ntype == "tunable_parameter":
                params.append(node)
            elif ntype == "hardware_feature":
                hw_features.append(node)
            elif ntype == "code_example":
                code_examples.append(node)

        return {
            "strategy_uid": strategy_uid,
            "strategy_name": self._entity_index.get(strategy_uid, {}).get("name", ""),
            "related_patterns": patterns,
            "tunable_parameters": params,
            "hardware_features": hw_features,
            "code_examples": code_examples,
            "total_connections": subgraph["total_edges"]
        }

    def co_occurrence_rank(self, strategy_uids: List[str],
                           query_pattern_types: Set[str]) -> List[Tuple[str, float]]:
        """基于图结构对策略排序
        评分 = 直连模式匹配 + 邻居多样性 + 中心度
        """
        scores = {}
        for uid in strategy_uids:
            score = 0.0
            neighbors = self.get_neighbors(uid, direction="both")

            # 1. 模式匹配分：直连的 computational_pattern 有多少匹配查询
            pattern_match = 0
            for n in neighbors:
                if n.get("target_type") == "computational_pattern":
                    pname = n.get("target_name", "")
                    for qp in query_pattern_types:
                        if qp.lower() in pname.lower():
                            pattern_match += 1
                            break
            score += pattern_match * 2.0

            # 2. 邻居多样性：连接的实体类型种类越多越好
            neighbor_types = set()
            for n in neighbors:
                nt = n.get("target_type") or n.get("source_type") or ""
                if nt:
                    neighbor_types.add(nt)
            score += len(neighbor_types) * 0.5

            # 3. 中心度：出入度之和
            degree = len(self._adj_out.get(uid, [])) + len(self._adj_in.get(uid, []))
            score += min(degree / 10.0, 1.0)

            scores[uid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def is_loaded(self) -> bool:
        return self._loaded
