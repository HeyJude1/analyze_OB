"""生成可视化数据：viz/graph_data.js + viz/traces.js（SPEC §5 的唯一 pipeline 职责）。

约定（与 SPEC §7 写权限的例外）：pipeline 只写这两个 js 文件；viz/graph.html 与 assets/ 归 viz engineer。
输出格式：
  window.KG_DATA   = {nodes:[...], edges:[...], stats:{...}};
  window.KG_TRACES = {trace_id: {...}, ...};
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from . import corpus, graph as graph_mod

__all__ = ["export", "write_js"]

_TYPE_GROUP = {
    "Operator": 1,
    "ComputationPattern": 2,
    "CodeCharacteristic": 3,
    "OptimizationStrategy": 4,
    "ImplementationTemplate": 5,
    "TunableParameter": 6,
    "ISACapability": 7,
    "Microarchitecture": 8,
    "HardwareProfile": 9,
    "PerformanceEvidence": 10,
    "SourceArtifact": 11,
    "FailureCase": 12,
}


def _escape(text) -> str:
    if text is None:
        return ""
    return str(text).replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", " ").replace("\r", " ")


def _node_for_viz(node: dict, degree: int) -> dict:
    props = node.get("properties") or {}
    prov = node.get("provenance") or {}
    name = node.get("name") or node["id"]
    label = name if len(name) <= 42 else name[:40] + "…"
    title_lines = [
        "<b>%s</b>" % _escape(node.get("name")),
        "id: %s" % _escape(node["id"]),
        "type: %s" % _escape(node.get("type")),
        "source: %s" % _escape(prov.get("source_file")),
        "lines: %s-%s" % (prov.get("line_start"), prov.get("line_end")),
        "confidence: %s" % node.get("confidence"),
    ]
    if props.get("hardware_profile_ids"):
        title_lines.append("hardware: %s" % _escape(", ".join(props["hardware_profile_ids"])))
    if props.get("conditions"):
        title_lines.append("conditions: %s" % _escape(
            "、".join("%s %s %s" % (c.get("pred"), c.get("op"), c.get("value")) for c in props["conditions"])))
    if props.get("template_id"):
        title_lines.append("template: %s" % _escape(props["template_id"]))
    return {
        "id": node["id"],
        "name": node.get("name") or label,
        "label": label,
        "type": node.get("type"),
        "group": _TYPE_GROUP.get(node.get("type"), 0),
        "value": max(1, int(degree)),
        "title": "\n".join(title_lines),
        "properties": props,
        "provenance": prov,
        "confidence": node.get("confidence"),
    }


def _edge_for_viz(edge: dict) -> dict:
    props = edge.get("properties") or {}
    title_lines = ["<b>%s</b>" % _escape(edge["type"]), "%s -> %s" % (_escape(edge["head"]), _escape(edge["tail"]))]
    if props:
        title_lines.append(_escape(json.dumps(props, ensure_ascii=False)))
    return {
        "id": edge["id"],
        "from": edge["head"],
        "to": edge["tail"],
        # SPEC §2.2 也要求 head/tail；viz 的 normalize() 以 head/tail 为准，缺了会显示 0 条边
        "head": edge["head"],
        "tail": edge["tail"],
        "type": edge["type"],
        "label": edge["type"],
        "title": "\n".join(title_lines),
        "properties": props,
        "confidence": edge.get("confidence"),
    }


def write_js(path: Path, variable: str, payload) -> int:
    """写 `window.<variable> = <json>;`（UTF-8，无 BOM）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "window.%s = %s;\n" % (variable, json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    path.write_text(text, encoding="utf-8", newline="\n")
    return len(text.encode("utf-8"))


def load_traces(traces_dir: Path) -> dict:
    traces = {}
    traces_dir = Path(traces_dir)
    if not traces_dir.exists():
        return traces
    for path in sorted(traces_dir.glob("*.json")):
        try:
            trace = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        trace_id = trace.get("trace_id") or path.stem
        traces[trace_id] = trace
    return traces


def export(cfg, data_dir=None, viz_dir=None) -> dict:
    """读 data/nodes.jsonl + edges.jsonl + traces/*.json，写 viz 两个 js 文件。"""
    data_dir = Path(data_dir) if data_dir else cfg.data_dir
    viz_dir = Path(viz_dir) if viz_dir else cfg.viz_dir
    loaded = graph_mod.load_graph(data_dir)
    if not loaded.nodes:
        raise FileNotFoundError("图数据为空：%s（先运行 build --offline）" % data_dir)
    stats = loaded.stats()
    nodes = [_node_for_viz(node, loaded.degree(node["id"])) for node in sorted(
        loaded.nodes.values(), key=lambda n: (n["type"], n["id"]))]
    edges = [_edge_for_viz(edge) for edge in sorted(
        loaded.edges, key=lambda e: (e["type"], e["head"], e["tail"]))]
    traces = load_traces(data_dir / "traces")
    stats.update({
        "traces": len(traces),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "schema_version": cfg.schema_version,
    })
    graph_data = {"nodes": nodes, "edges": edges, "stats": stats}
    graph_path = viz_dir / "graph_data.js"
    traces_path = viz_dir / "traces.js"
    size_graph = write_js(graph_path, "KG_DATA", graph_data)
    size_traces = write_js(traces_path, "KG_TRACES", traces)
    return {
        "graph_data": str(graph_path),
        "traces": str(traces_path),
        "nodes": len(nodes),
        "edges": len(edges),
        "trace_count": len(traces),
        "bytes": {"graph_data": size_graph, "traces": size_traces},
    }
