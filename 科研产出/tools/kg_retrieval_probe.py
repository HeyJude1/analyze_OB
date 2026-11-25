#!/usr/bin/env python3
"""kg_retrieval_probe.py —— 检索层最小诊断：图检索 vs 扁平文本检索 vs 随机。

目的（回答"知识图谱到底怎么检索、比向量检索多给了什么"）：
  对同一批查询，比较三种"检索器"给出的 top-k 策略卡，
  看它们"选出来的知识到底能不能用"——而不是看文本像不像。

三种检索器：
  graph   现有七步检索（retrieve.py），输出的 pack
  flat    BM25（字符 2-gram）在 262 张策略卡全文上打的 top-k
  random  固定随机种子抽 k 张（下界参照）

指标（全部可复算）：
  applicable@k    top-k 中"硬条件全部满足且条件可判定"的比例
  decidable@k     top-k 中条件可判定（非纯文本条件）的比例
  op_match@k      top-k 中属于该算子的比例（含 GENERALIZES 抽象策略）
  arch_bound@k    top-k 中 hardware_profile_ids 绑定目标硬件的比例
  ctx_chars       注入上下文的字符数（成本代理）

注意：flat 用 BM25 只是"轻量、无依赖、可复现"的替身；
     正式论文里必须换成真 embedding（如 bge-m3）+ 同一 top-k，见文档 23 第 6 节。

用法：
    python kg_retrieval_probe.py                 # 默认 4 组查询
    python kg_retrieval_probe.py --k 5 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KG_ROOT = HERE.parent / "kg_agent"
sys.path.insert(0, str(KG_ROOT / "src"))

from kgagent import graph as graph_mod  # noqa: E402
from kgagent import retrieve as retrieve_mod  # noqa: E402
from kgagent.config import load_config  # noqa: E402

DEFAULT_QUERIES = [
    ("daxpy", "x86_avx2", "daxpy：在 x86 AVX2 上把通用 C 实现优化成高性能 kernel"),
    ("daxpy", "k1_rvv256", "daxpy：在 RISC-V K1（RVV，VLEN=256）上写向量化 kernel"),
    ("dgemm", "x86_avx2", "dgemm：x86 AVX2 上做矩阵分块和打包"),
    ("dot", "k1_rvv256", "ddot：RISC-V 向量化 + 尾部处理 + 归约"),
]

# ---------------------------------------------------------------- 文本特征


def _norm(text: str) -> str:
    text = (text or "").lower()
    return re.sub(r"\s+", " ", text)


def _grams(text: str, n: int = 2) -> list:
    """字符 n-gram：中文无空格，字符 n-gram 比词切分稳且零依赖。"""
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", _norm(text))
    out = []
    for chunk in text.split():
        if len(chunk) <= n:
            out.append(chunk)
        else:
            out.extend(chunk[i:i + n] for i in range(len(chunk) - n + 1))
    return out


def card_text(node: dict) -> str:
    """把一条策略序列化成"扁平文本"，近似向量库里存的那段 chunk。"""
    props = node.get("properties") or {}
    parts = [
        node.get("name") or "",
        props.get("family") or "",
        props.get("action") or "",
        props.get("level") or "",
        props.get("stage") or "",
        props.get("arch") or "",
        props.get("description") or "",
    ]
    for cond in props.get("conditions") or []:
        parts.append("%s %s %s" % (cond.get("pred"), cond.get("op"), cond.get("value")))
    parts.append(props.get("arch_scope") or "")
    return " ".join(str(p) for p in parts if p)


class BM25:
    """极简 BM25（k1=1.5, b=0.75），字符 2-gram 词表。"""

    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.docs = [_grams(d) for d in docs]
        self.k1, self.b = k1, b
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.df = {}
        for doc in self.docs:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in self.df.items()}
        self.tf = []
        for doc in self.docs:
            counts = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def score(self, query: str, idx: int) -> float:
        counts, dl = self.tf[idx], self.doc_len[idx]
        total = 0.0
        for term in _grams(query):
            f = counts.get(term)
            if not f:
                continue
            idf = self.idf.get(term, 0.0)
            total += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9)))
        return total


# ---------------------------------------------------------------- 指标


def judge(graph, cfg, strategy_ids: list, features: dict, profile: str, op: str) -> dict:
    """对一组被选中的策略，逐条判定"能不能用"。

    applicable = 属于该算子（或抽象策略） 且 硬条件全部满足 且 硬件绑定匹配。
    三个条件缺一不可：漏掉"属于该算子"，文本检索靠共享词汇捞回别的算子的卡也会被算成可用。
    """
    applicable = decidable = op_match = arch_bound = 0
    for sid in strategy_ids:
        node = graph.nodes.get(sid) or {}
        props = node.get("properties") or {}
        bound = profile in (props.get("hardware_profile_ids") or [])
        if bound:
            arch_bound += 1
        name = (node.get("name") or "") + sid
        patterns = [e["head"] for e in graph.in_edges(sid, "PATTERN_OPTIMIZED_BY")]
        owner_ops = set()
        for pattern_id in patterns:
            for e in graph.in_edges(pattern_id, "OPERATOR_HAS_PATTERN"):
                owner_ops.add(e["head"].split(":", 1)[1])
        owned = op in name or op in owner_ops
        if owned:
            op_match += 1
        if props.get("condition_status") == "text":
            continue
        decidable += 1
        result = retrieve_mod.evaluate_conditions(props.get("conditions") or [], features)
        if not result["hard_fail"] and bound and owned:
            applicable += 1
    k = max(len(strategy_ids), 1)
    return {
        "k": k,
        "applicable@k": round(applicable / k, 3),
        "decidable@k": round(decidable / k, 3),
        "op_match@k": round(op_match / k, 3),
        "arch_bound@k": round(arch_bound / k, 3),
    }


def run(cfg, queries, k: int, seed: int = 13) -> list:
    graph = graph_mod.load_graph(cfg.data_dir)
    strategies = [n for n in graph.nodes.values() if n["type"] == "OptimizationStrategy"]
    strategies.sort(key=lambda n: n["id"])
    corpus = [card_text(n) for n in strategies]
    bm25 = BM25(corpus)
    rng = random.Random(seed)
    rows = []

    for op, profile, query_text in queries:
        # ---- 1) 图检索
        trace = retrieve_mod.retrieve(cfg, op, profile=profile, top_k=k, write=False)
        graph_ids = trace["result"]["strategies"]
        canonical = trace["query"]["operator"]
        features = trace["query"]["code_features"]
        # 与 mpsa 实际注入的一致：STRATEGIES_JSON = json.dumps(pack["strategies"])
        pack_chars = len(json.dumps(trace["result"]["pack"]["strategies"], ensure_ascii=False))

        # ---- 2) 扁平文本检索（BM25 替身）
        ranked = sorted(range(len(strategies)), key=lambda i: -bm25.score(query_text, i))[:k]
        flat_ids = [strategies[i]["id"] for i in ranked]
        flat_chars = sum(len(corpus[i]) for i in ranked) + len(query_text)

        # ---- 3) 随机
        random_ids = [n["id"] for n in rng.sample(strategies, k)]
        random_chars = sum(len(card_text(graph.nodes[sid])) for sid in random_ids)

        rows.append({
            "operator": op,
            "profile": profile,
            "query": query_text,
            "graph": {**judge(graph, cfg, graph_ids, features, profile, canonical),
                      "ids": graph_ids, "ctx_chars": pack_chars},
            "flat_bm25": {**judge(graph, cfg, flat_ids, features, profile, canonical),
                          "ids": flat_ids, "ctx_chars": flat_chars},
            "random": {**judge(graph, cfg, random_ids, features, profile, canonical),
                       "ids": random_ids, "ctx_chars": random_chars},
        })
    return rows


def to_markdown(rows: list, k: int) -> str:
    head = ("| 查询 | 检索器 | 可适用@%d | 条件可判定@%d | 算子匹配@%d | 硬件绑定@%d | 上下文字符 |\n"
            "|---|---|---|---|---|---|---|" % (k, k, k, k))
    lines = [head]
    order = [("graph", "图检索（七步）"), ("flat_bm25", "扁平文本 BM25"), ("random", "随机")]
    for row in rows:
        q = "%s / %s" % (row["operator"], row["profile"])
        for key, label in order:
            m = row[key]
            lines.append("| %s | %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %d |" % (
                q, label, m["applicable@k"] * 100, m["decidable@k"] * 100,
                m["op_match@k"] * 100, m["arch_bound@k"] * 100, m["ctx_chars"]))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", default=None, help="把明细写到这个路径")
    ap.add_argument("--markdown", default=None, help="把表格写到这个路径")
    args = ap.parse_args(argv)

    cfg = load_config()
    rows = run(cfg, DEFAULT_QUERIES, args.k)
    table = to_markdown(rows, args.k)
    print(table)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.markdown:
        Path(args.markdown).write_text(table + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
