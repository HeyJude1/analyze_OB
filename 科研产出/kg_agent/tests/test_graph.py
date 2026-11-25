"""图结构测试：纯 dict 邻接表、BFS 最短路、子图、度数。"""

import unittest

from kgagent.graph import Graph, shortest_path, subgraph_ids


def _graph():
    nodes = [
        {"id": "op:x", "type": "Operator", "name": "x", "properties": {}, "provenance": {}},
        {"id": "pat:x", "type": "ComputationPattern", "name": "p", "properties": {}, "provenance": {}},
        {"id": "stg:x", "type": "OptimizationStrategy", "name": "s", "properties": {}, "provenance": {}},
        {"id": "tpl:x", "type": "ImplementationTemplate", "name": "t", "properties": {}, "provenance": {}},
    ]
    edges = [
        {"id": "e1", "type": "OPERATOR_HAS_PATTERN", "head": "op:x", "tail": "pat:x", "properties": {}},
        {"id": "e2", "type": "PATTERN_OPTIMIZED_BY", "head": "pat:x", "tail": "stg:x", "properties": {}},
        {"id": "e3", "type": "STRATEGY_HAS_TEMPLATE", "head": "stg:x", "tail": "tpl:x", "properties": {}},
    ]
    return Graph(nodes, edges)


class GraphTest(unittest.TestCase):
    def test_edges_and_neighbors(self):
        g = _graph()
        self.assertEqual(g.neighbors("op:x", "OPERATOR_HAS_PATTERN"), ["pat:x"])
        self.assertTrue(g.has_edge("pat:x", "PATTERN_OPTIMIZED_BY", "stg:x"))
        self.assertEqual(len(g.out_edges("stg:x")), 1)

    def test_bfs_shortest_path(self):
        g = _graph()
        path = g.bfs("op:x", "tpl:x")
        self.assertEqual(path, ["op:x", "pat:x", "stg:x", "tpl:x"])
        result = shortest_path(g, "op:x", "tpl:x")
        self.assertEqual([e["rel"] for e in result["edges"]],
                         ["OPERATOR_HAS_PATTERN", "PATTERN_OPTIMIZED_BY", "STRATEGY_HAS_TEMPLATE"])

    def test_unreachable_and_self(self):
        g = _graph()
        self.assertEqual(g.bfs("tpl:x", "tpl:x"), ["tpl:x"])
        g2 = Graph([{"id": "a", "type": "Operator", "name": "a", "properties": {}, "provenance": {}},
                    {"id": "b", "type": "Operator", "name": "b", "properties": {}, "provenance": {}}], [])
        self.assertEqual(g2.bfs("a", "b"), [])
        self.assertEqual(g2.bfs("a", "missing"), [])

    def test_degree_and_subgraph(self):
        g = _graph()
        self.assertEqual(g.degree("stg:x"), 2)
        ids = subgraph_ids(g, "pat:x", depth=1)
        self.assertEqual(set(ids), {"op:x", "pat:x", "stg:x"})
        sub = g.subgraph(["op:x", "pat:x"])
        self.assertEqual(len(sub.nodes), 2)
        self.assertEqual(len(sub.edges), 1)

    def test_bidirectional_traversal(self):
        g = _graph()
        # 从模板反向 BFS 也能回到算子（检索回路兼容无向遍历）
        self.assertEqual(g.bfs("tpl:x", "op:x"), ["tpl:x", "stg:x", "pat:x", "op:x"])


if __name__ == "__main__":
    unittest.main()
