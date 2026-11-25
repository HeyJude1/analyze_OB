#!/usr/bin/env python3
"""export_web.py —— 把知识图谱导出成单文件 HTML（离线可开，无需服务器）。

输入二选一：
  --kg kg.json                       # 规范格式：{entities:[], relations:[]}
  --nodes nodes.jsonl --edges edges.jsonl   # kg_agent / 常见流水线格式

输出：一个 HTML，内嵌 vis-network 与全部数据；支持类型筛选、搜索、详情面板、邻居跳转、
      溯源与质量徽标。适合答辩演示与"建图结果自检"。

用法：
  python export_web.py --nodes data/nodes.jsonl --edges data/edges.jsonl --out out/graph.html
  python export_web.py --kg out/kg.json --out out/graph.html --title "OpenBLAS 优化策略图谱"
  python export_web.py --kg out/kg.json --assets path/to/vis-network.min.js --out g.html
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ASSET_CANDIDATES = [
    HERE / "assets" / "vis-network.min.js",
    HERE.parents[2] / "kg_agent" / "viz" / "assets" / "vis-network.min.js",
]

PALETTE = [
    "#4C9AFF", "#36B37E", "#FFAB00", "#FF7452", "#8777D9",
    "#00B8D9", "#79F2C0", "#FF8F73", "#B3D4FF", "#C0B6F2",
    "#57D9A3", "#F5CD47",
]


# ---------------------------------------------------------------- 读取与归一


def _read_jsonl(path: Path) -> list:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_graph(args) -> tuple[list, list]:
    if args.kg:
        data = json.loads(Path(args.kg).read_text(encoding="utf-8"))
        return data.get("entities") or data.get("nodes") or [], data.get("relations") or data.get("edges") or []
    if args.nodes and args.edges:
        return _read_jsonl(Path(args.nodes)), _read_jsonl(Path(args.edges))
    raise SystemExit("需要 --kg，或同时给 --nodes 与 --edges")


def _trim(value, limit: int):
    if limit and isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…（截断，原长 %d 字符）" % len(value)
    if isinstance(value, list):
        return [_trim(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, limit) for k, v in value.items()}
    return value


def norm_entity(raw: dict, used: set, max_prop_chars: int = 400) -> dict:
    eid = str(raw.get("id") or raw.get("uid") or raw.get("name") or "")
    if not eid or eid in used:
        eid = "%s#%s" % (eid or "entity", hashlib.md5(json.dumps(raw, sort_keys=True, default=str).encode()).hexdigest()[:8])
    used.add(eid)
    return {
        "id": eid,
        "type": raw.get("type") or raw.get("entity_type") or "Unknown",
        "name": raw.get("name") or raw.get("label") or eid,
        "properties": _trim(raw.get("properties") or raw.get("attributes") or {}, max_prop_chars),
        "provenance": raw.get("provenance") or {},
        "confidence": raw.get("confidence"),
        "status": raw.get("status") or "raw",
    }


def norm_relation(raw: dict, index: int) -> dict:
    head = raw.get("head") or raw.get("source") or raw.get("from")
    tail = raw.get("tail") or raw.get("target") or raw.get("to")
    if isinstance(head, dict):
        head = head.get("id") or head.get("uid") or head.get("name")
    if isinstance(tail, dict):
        tail = tail.get("id") or tail.get("uid") or tail.get("name")
    return {
        "id": raw.get("id") or raw.get("relation_id") or "rel:%d" % index,
        "type": raw.get("type") or raw.get("relation_type") or "RELATED",
        "head": str(head),
        "tail": str(tail),
        "properties": raw.get("properties") or {},
        "provenance": raw.get("provenance") or {},
    }


# ---------------------------------------------------------------- 统计


def quality_stats(entities: list, relations: list) -> dict:
    by_type = Counter(e["type"] for e in entities)
    status = Counter(e["status"] for e in entities)

    def has_prov(e):
        p = e.get("provenance") or {}
        return bool(p.get("source_file")) and (p.get("line_start") is not None or p.get("line"))

    prov_ok = sum(1 for e in entities if has_prov(e))
    ids = {e["id"] for e in entities}
    dangling = [r for r in relations if r["head"] not in ids or r["tail"] not in ids]
    degree = Counter()
    for r in relations:
        degree[r["head"]] += 1
        degree[r["tail"]] += 1
    # 叶子型节点（只靠属性绑定、不指望有边）不计入"异常孤立"
    leaf_types = {"SourceArtifact", "HardwareProfile"}
    orphan_all = [e for e in entities if degree[e["id"]] == 0]
    orphan_bad = [e for e in orphan_all if e["type"] not in leaf_types]
    name_key = Counter((e["type"], e["name"].strip().lower()) for e in entities)
    dup_names = sum(1 for k, n in name_key.items() if n > 1)
    return {
        "entities": len(entities),
        "relations": len(relations),
        "by_type": dict(by_type.most_common()),
        "by_status": dict(status),
        "provenance_coverage": round(prov_ok / max(len(entities), 1), 3),
        "dangling_relations": len(dangling),
        "orphan_entities": len(orphan_bad),
        "orphan_all": len(orphan_all),
        "orphan_by_type": dict(Counter(e["type"] for e in orphan_all).most_common()),
        "duplicate_name_groups": dup_names,
        "relation_types": dict(Counter(r["type"] for r in relations).most_common()),
    }


# ---------------------------------------------------------------- HTML


def _asset_text(path: Path | None) -> str:
    for candidate in ([path] if path else []) + DEFAULT_ASSET_CANDIDATES:
        if candidate and Path(candidate).exists():
            return Path(candidate).read_text(encoding="utf-8", errors="ignore")
    raise SystemExit(
        "找不到 vis-network.min.js。用 --assets 指定，或放到 scripts/assets/ 下。\n"
        "（可从 科研产出/kg_agent/viz/assets/vis-network.min.js 复制）"
    )


def build_html(entities: list, relations: list, stats: dict, title: str, asset_js: str) -> str:
    payload = json.dumps({"entities": entities, "relations": relations, "stats": stats},
                         ensure_ascii=False).replace("</", "<\\/")
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root{--bg:#0f1420;--panel:#161d2c;--line:#243044;--text:#e6edf7;--muted:#8aa0bd;--accent:#4c9aff;}
  *{box-sizing:border-box}
  body{margin:0;height:100vh;display:flex;flex-direction:column;background:var(--bg);color:var(--text);
       font:14px/1.5 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
  header{display:flex;align-items:center;gap:16px;padding:10px 16px;background:var(--panel);border-bottom:1px solid var(--line);flex-wrap:wrap}
  header h1{font-size:16px;margin:0;font-weight:600}
  .badge{font-size:12px;padding:2px 8px;border-radius:10px;background:#1f2b3d;color:var(--muted);border:1px solid var(--line)}
  .badge.ok{color:#57d9a3;border-color:#1f5b45}
  .badge.warn{color:#f5cd47;border-color:#5b5320}
  .badge.bad{color:#ff8f73;border-color:#5b2a20}
  main{flex:1;display:flex;min-height:0}
  aside{width:240px;padding:12px;overflow:auto;background:var(--panel);border-right:1px solid var(--line)}
  aside h3{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin:14px 0 6px}
  .filter{display:flex;align-items:center;gap:6px;padding:3px 0;cursor:pointer;font-size:13px}
  .dot{width:9px;height:9px;border-radius:50%;flex:none}
  .count{margin-left:auto;color:var(--muted);font-size:12px}
  #search{width:100%;padding:7px 9px;border-radius:7px;border:1px solid var(--line);background:#0d1320;color:var(--text)}
  #net{flex:1;min-width:0}
  #detail{width:340px;padding:14px;overflow:auto;background:var(--panel);border-left:1px solid var(--line)}
  #detail h2{font-size:15px;margin:0 0 4px;word-break:break-all}
  #detail .sub{color:var(--muted);font-size:12px;margin-bottom:10px}
  table{width:100%;border-collapse:collapse;font-size:12px}
  td{padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top;word-break:break-all}
  td.k{color:var(--muted);width:34%}
  .chip{display:inline-block;padding:1px 7px;border-radius:9px;background:#1c2740;margin:2px 4px 2px 0;font-size:12px;cursor:pointer}
  .chip:hover{background:#26375a}
  .empty{color:var(--muted);font-size:13px}
  footer{padding:6px 16px;background:var(--panel);border-top:1px solid var(--line);color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <span class="badge" id="b-entities"></span>
  <span class="badge" id="b-relations"></span>
  <span class="badge" id="b-prov"></span>
  <span class="badge" id="b-orphan"></span>
  <span class="badge" id="b-dup"></span>
</header>
<main>
  <aside>
    <input id="search" placeholder="搜名称 / id / 属性值…">
    <h3>实体类型</h3>
    <div id="type-filters"></div>
    <h3>关系类型</h3>
    <div id="rel-filters"></div>
    <h3>状态</h3>
    <label class="filter"><input type="checkbox" id="only-verified"> 只看 verified</label>
  </aside>
  <div id="net"></div>
  <section id="detail"><div class="empty">点一个节点看详情与邻居。</div></section>
</main>
<footer>离线单文件 · vis-network · 数据内嵌 · 由 kg-build skill 生成</footer>
<script>__VIS__</script>
<script>
const KG = __DATA__;
const PALETTE = __PALETTE__;
const stats = KG.stats;
const typeColor = {};
Object.keys(stats.by_type).forEach((t, i) => { typeColor[t] = PALETTE[i % PALETTE.length]; });

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;"}[c]));
}

const nodes = KG.entities.map(e => ({
  id: e.id, label: e.name.length > 26 ? e.name.slice(0, 24) + "…" : e.name,
  title: e.name + "\\n" + e.type,
  group: e.type, color: typeColor[e.type] || "#8aa0bd", raw: e
}));
const edges = KG.relations.map(r => ({
  id: r.id, from: r.head, to: r.tail, label: r.type, arrows: "to",
  font: {size: 9, color: "#8aa0bd", strokeWidth: 0}, color: {color: "#2c3a52", highlight: "#4c9aff"},
  width: 0.7, title: r.type, raw: r
}));

const nodeDS = new vis.DataSet(nodes);
const edgeDS = new vis.DataSet(edges);
const container = document.getElementById("net");
const network = new vis.Network(container, {nodes: nodeDS, edges: edgeDS}, {
  physics: {barnesHut: {gravitationalConstant: -9000, springLength: 120, springConstant: 0.03, damping: 0.6}, stabilization: {iterations: 220}},
  interaction: {hover: true, tooltipDelay: 150},
  nodes: {shape: "dot", size: 9, borderWidth: 1, font: {size: 11, color: "#c9d6e8"}},
  edges: {smooth: {type: "continuous"}},
  layout: {improvedLayout: true}
});

// ---- 徽标
function setBadge(id, text, cls) { const el = document.getElementById(id); el.textContent = text; el.className = "badge " + (cls || ""); }
setBadge("b-entities", "实体 " + stats.entities);
setBadge("b-relations", "关系 " + stats.relations);
setBadge("b-prov", "溯源 " + Math.round(stats.provenance_coverage * 100) + "%",
         stats.provenance_coverage >= 0.9 ? "ok" : stats.provenance_coverage >= 0.5 ? "warn" : "bad");
setBadge("b-orphan", "孤儿 " + stats.orphan_entities, stats.orphan_entities ? "warn" : "ok");
setBadge("b-dup", "重名组 " + stats.duplicate_name_groups, stats.duplicate_name_groups ? "warn" : "ok");

// ---- 过滤器
const activeTypes = new Set(Object.keys(stats.by_type));
const activeRels = new Set(Object.keys(stats.relation_types));
const tf = document.getElementById("type-filters");
Object.entries(stats.by_type).forEach(([t, n]) => {
  const row = document.createElement("label"); row.className = "filter";
  row.innerHTML = '<input type="checkbox" checked><span class="dot" style="background:' + (typeColor[t] || "#888") +
                  '"></span><span>' + esc(t) + '</span><span class="count">' + n + "</span>";
  row.querySelector("input").onchange = e => { e.target.checked ? activeTypes.add(t) : activeTypes.delete(t); applyFilter(); };
  tf.appendChild(row);
});
const rf = document.getElementById("rel-filters");
Object.entries(stats.relation_types).forEach(([t, n]) => {
  const row = document.createElement("label"); row.className = "filter";
  row.innerHTML = '<input type="checkbox" checked><span>' + esc(t) + '</span><span class="count">' + n + "</span>";
  row.querySelector("input").onchange = e => { e.target.checked ? activeRels.add(t) : activeRels.delete(t); applyFilter(); };
  rf.appendChild(row);
});

function edgeVisible(e) { return activeRels.has(e.raw.type); }
function nodeVisible(n) {
  if (!activeTypes.has(n.raw.type)) return false;
  if (document.getElementById("only-verified").checked && n.raw.status !== "verified") return false;
  return true;
}
function applyFilter() {
  nodeDS.update(nodes.map(n => ({id: n.id, hidden: !nodeVisible(n)})));
  edgeDS.update(edges.map(e => ({id: e.id, hidden: !edgeVisible(e)})));
}
document.getElementById("only-verified").onchange = applyFilter;

// ---- 搜索
document.getElementById("search").addEventListener("input", ev => {
  const q = ev.target.value.trim().toLowerCase();
  if (!q) { nodeDS.update(nodes.map(n => ({id: n.id, opacity: 1, size: 9}))); return; }
  nodeDS.update(nodes.map(n => {
    const hay = (n.raw.name + " " + n.raw.id + " " + JSON.stringify(n.raw.properties)).toLowerCase();
    const hit = hay.includes(q);
    return {id: n.id, opacity: hit ? 1 : 0.15, size: hit ? 16 : 9};
  }));
});

// ---- 详情面板
const detail = document.getElementById("detail");
function kvTable(obj) {
  const rows = Object.entries(obj || {}).map(([k, v]) =>
    "<tr><td class='k'>" + esc(k) + "</td><td>" + esc(typeof v === "object" ? JSON.stringify(v) : v) + "</td></tr>").join("");
  return rows ? "<table>" + rows + "</table>" : '<div class="empty">无</div>';
}
function showEntity(id) {
  const e = nodes.find(n => n.id === id); if (!e) return;
  const r = e.raw;
  const p = r.provenance || {};
  const prov = (p.source_file || p.line_start != null)
    ? "<table><tr><td class='k'>source</td><td>" + esc(p.source_file || "-") + "</td></tr>" +
      "<tr><td class='k'>lines</td><td>" + esc((p.line_start ?? "-") + " – " + (p.line_end ?? "-")) + "</td></tr>" +
      (p.sha256 ? "<tr><td class='k'>sha256</td><td>" + esc(String(p.sha256).slice(0, 16)) + "…</td></tr>" : "") + "</table>"
    : '<div class="empty">无行号溯源（质量告警）</div>';
  const nb = edges.filter(x => x.from === id || x.to === id).slice(0, 60).map(x => {
    const other = x.from === id ? x.to : x.from;
    const dir = x.from === id ? "→" : "←";
    return '<span class="chip" data-id="' + esc(other) + '">' + dir + " " + esc(x.raw.type) + " · " + esc(other) + "</span>";
  }).join("") || '<div class="empty">无邻居</div>';
  detail.innerHTML = "<h2>" + esc(r.name) + "</h2>" +
    '<div class="sub">' + esc(r.type) + " · " + esc(r.id) + " · status " + esc(r.status) + "</div>" +
    "<h3>属性</h3>" + kvTable(r.properties) + "<h3>溯源</h3>" + prov +
    '<h3>邻居（' + edges.filter(x => x.from === id || x.to === id).length + "）</h3>" + nb;
  detail.querySelectorAll(".chip").forEach(c => c.onclick = () => { showEntity(c.dataset.id); network.focus(c.dataset.id, {scale: 1.2, animation: true}); });
}
network.on("click", params => {
  if (params.nodes.length) showEntity(params.nodes[0]);
  else detail.innerHTML = '<div class="empty">点一个节点看详情与邻居。</div>';
});
</script>
</body>
</html>
""".replace("__TITLE__", title).replace("__VIS__", asset_js).replace("__DATA__", payload).replace(
        "__PALETTE__", json.dumps(PALETTE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=None)
    ap.add_argument("--nodes", default=None)
    ap.add_argument("--edges", default=None)
    ap.add_argument("--out", default="graph.html")
    ap.add_argument("--assets", default=None, help="vis-network.min.js 路径")
    ap.add_argument("--title", default="知识图谱")
    ap.add_argument("--max-prop-chars", type=int, default=400, help="属性字符串超过该长度截断（0=不截断）")
    args = ap.parse_args(argv)

    raw_entities, raw_relations = load_graph(args)
    used = set()
    entities = [norm_entity(e, used, args.max_prop_chars) for e in raw_entities]
    relations = [norm_relation(r, i) for i, r in enumerate(raw_relations)]
    stats = quality_stats(entities, relations)

    html = build_html(entities, relations, stats, args.title, _asset_text(Path(args.assets) if args.assets else None))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(json.dumps({"out": str(out), "bytes": out.stat().st_size, **stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
