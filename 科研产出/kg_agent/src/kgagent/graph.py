"""轻量图结构（纯 dict 邻接表 + BFS/最短路，不依赖 networkx）。

数据源是 data/nodes.jsonl + data/edges.jsonl（文件即真源）。规模 <= 数万边，纯 Python 足够。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from . import corpus

__all__ = ["Graph", "load_graph", "shortest_path", "subgraph_ids"]


class Graph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = {}
        self.edges = list(edges or [])
        self.out_adj = {}
        self.in_adj = {}
        for node in nodes or []:
            self.add_node(node)
        for edge in self.edges:
            self._index_edge(edge)

    # ---- 构建 ----
    def add_node(self, node: dict) -> None:
        self.nodes[node["id"]] = node
        self.out_adj.setdefault(node["id"], [])
        self.in_adj.setdefault(node["id"], [])

    def _index_edge(self, edge: dict) -> None:
        self.out_adj.setdefault(edge["head"], []).append(edge)
        self.in_adj.setdefault(edge["tail"], []).append(edge)

    # ---- 查询 ----
    def node(self, node_id: str) -> dict:
        return self.nodes.get(node_id)

    def node_type(self, node_id: str):
        node = self.nodes.get(node_id)
        return node["type"] if node else None

    def nodes_of_type(self, node_type: str) -> list:
        return [n for n in self.nodes.values() if n["type"] == node_type]

    def edges_of_type(self, rel: str) -> list:
        return [e for e in self.edges if e["type"] == rel]

    def out_edges(self, node_id: str, rel: str = None) -> list:
        edges = self.out_adj.get(node_id, [])
        return [e for e in edges if rel is None or e["type"] == rel]

    def in_edges(self, node_id: str, rel: str = None) -> list:
        edges = self.in_adj.get(node_id, [])
        return [e for e in edges if rel is None or e["type"] == rel]

    def neighbors(self, node_id: str, rel: str = None, direction: str = "out") -> list:
        if direction == "out":
            return [e["tail"] for e in self.out_edges(node_id, rel)]
        return [e["head"] for e in self.in_edges(node_id, rel)]

    def has_edge(self, head: str, rel: str, tail: str) -> bool:
        return any(e["tail"] == tail for e in self.out_edges(head, rel))

    def degree(self, node_id: str) -> int:
        return len(self.out_adj.get(node_id, [])) + len(self.in_adj.get(node_id, []))

    def edges_between(self, a: str, b: str, rel: str = None) -> list:
        out = []
        for edge in self.out_adj.get(a, []):
            if edge["tail"] == b and (rel is None or edge["type"] == rel):
                out.append(edge)
        for edge in self.out_adj.get(b, []):
            if edge["tail"] == a and (rel is None or edge["type"] == rel):
                out.append(edge)
        return out

    # ---- 路径 ----
    def bfs(self, start: str, goal: str, max_depth: int = 8) -> list:
        """BFS 最短路径（无向），返回节点 id 列表；不可达返回 []。"""
        if start not in self.nodes or goal not in self.nodes:
            return []
        if start == goal:
            return [start]
        prev = {start: None}
        queue = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self.out_adj.get(current, []) + self.in_adj.get(current, []):
                nxt = edge["tail"] if edge["head"] == current else edge["head"]
                if nxt in prev:
                    continue
                prev[nxt] = current
                if nxt == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append((nxt, depth + 1))
        return []

    def path_edges(self, path: list) -> list:
        """把节点序列翻成带上关系的路径：[(head, rel, edge, tail)]。"""
        out = []
        for head, tail in zip(path, path[1:]):
            for edge in self.out_adj.get(head, []):
                if edge["tail"] == tail:
                    out.append((head, edge["type"], edge, tail))
                    break
            else:
                for edge in self.out_adj.get(tail, []):
                    if edge["tail"] == head:
                        out.append((tail, edge["type"], edge, head))
                        break
        return out

    def subgraph(self, node_ids) -> "Graph":
        keep = set(node_ids)
        nodes = [n for n in self.nodes.values() if n["id"] in keep]
        edges = [e for e in self.edges if e["head"] in keep and e["tail"] in keep]
        return Graph(nodes, edges)

    def stats(self) -> dict:
        by_type = {}
        for node in self.nodes.values():
            by_type[node["type"]] = by_type.get(node["type"], 0) + 1
        edge_by_type = {}
        for edge in self.edges:
            edge_by_type[edge["type"]] = edge_by_type.get(edge["type"], 0) + 1
        return {"nodes": len(self.nodes), "edges": len(self.edges),
                "nodes_by_type": by_type, "edges_by_type": edge_by_type}


def load_graph(data_dir) -> Graph:
    data_dir = Path(data_dir)
    nodes = corpus.read_jsonl(data_dir / "nodes.jsonl")
    edges = corpus.read_jsonl(data_dir / "edges.jsonl")
    return Graph(nodes, edges)


def shortest_path(graph: Graph, start: str, goal: str, max_depth: int = 8) -> dict:
    path = graph.bfs(start, goal, max_depth=max_depth)
    return {"from": start, "to": goal, "path": path, "edges": [
        {"head": h, "rel": r, "tail": t} for h, r, _, t in graph.path_edges(path)
    ]}


def subgraph_ids(graph: Graph, node_id: str, depth: int = 1) -> list:
    """以 node_id 为中心、双向 depth 跳内的节点 id（可视化/调试用）。"""
    if node_id not in graph.nodes:
        return []
    seen = {node_id}
    frontier = {node_id}
    for _ in range(depth):
        nxt = set()
        for current in frontier:
            for edge in graph.out_adj.get(current, []) + graph.in_adj.get(current, []):
                other = edge["tail"] if edge["head"] == current else edge["head"]
                if other not in seen:
                    nxt.add(other)
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return sorted(seen)
