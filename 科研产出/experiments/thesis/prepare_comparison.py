#!/usr/bin/env python3
"""Prepare auditable, paired prompt inputs without calling a model or timing any code."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
sys.path.insert(0, str(RESEARCH / "kg_agent" / "src"))
from kgagent.config import load_config
from kgagent.graph import load_graph
from kgagent.retrieve import retrieve
from kgagent.schema import load_schema


METHODS = ("no_knowledge", "text_rag", "strategy_cards", "constraint_plan")
PROFILES = ("x86_avx2", "k1_rvv256")
SOURCE_ANALOGIES = {
    "dscal": "axpy", "dasum": "dot", "dsymv": "gemv", "dsymm": "gemm", "conv1d": "dot",
}
CONTRACT_FIELDS = (
    "symbol", "pattern", "dimensions", "parameters", "signature", "interface",
    "semantics", "memory_contract", "numeric_contract", "layout", "dtype",
    "preconditions", "invariants", "restrictions", "aliasing", "numerical",
    "features", "tolerance",
)
CARD_FIELDS = (
    "id", "name", "family", "action", "level", "stage", "arch_scope", "arch",
    "conditions", "template", "parameters", "provenance",
)
SYSTEM = (
    "Implement an optimized C11 operator for the supplied task. Preserve the ABI, "
    "domain, floating-point error contract and support for all allowed dimensions. "
    "Treat knowledge as suggestions bounded by the task contract. Return a single "
    "compilable C translation unit, including opbench.h as needed."
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_text(obj, *, pretty=False):
    kwargs = {"indent": 2} if pretty else {"separators": (",", ":")}
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, **kwargs)


def write_json(path, obj):
    path.write_text(json_text(obj, pretty=True) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokens(value):
    text = value.lower()
    terms = re.findall(r"[a-z0-9_]+", text)
    for group in re.findall(r"[\u3400-\u9fff]+", text):
        terms.extend(group[index:index + 2] for index in range(max(1, len(group) - 1)))
    return terms


def rank_text(query, documents):
    """BM25 on the same admissible cards as the structured arm."""
    bags = [Counter(tokens(text)) for _, text in documents]
    lengths = [sum(bag.values()) for bag in bags]
    average = max(1.0, sum(lengths) / max(1, len(bags)))
    query_terms = sorted(set(tokens(query)))
    frequencies = {term: sum(term in bag for bag in bags) for term in query_terms}
    scores = []
    for (card, _), bag, size in zip(documents, bags, lengths):
        norm = 1.2 * (0.25 + 0.75 * size / average)
        score = 0.0
        for term in query_terms:
            if term in bag:
                df = frequencies[term]
                idf = math.log1p((len(bags) - df + 0.5) / (df + 0.5))
                score += idf * 2.2 * bag[term] / (bag[term] + norm)
        scores.append((score, card["id"], card))
    return [card for _, _, card in sorted(scores, key=lambda row: (-row[0], row[1]))]


def canonical_card(raw):
    return {field: raw[field] for field in CARD_FIELDS if field in raw}


def render_text(card):
    return "\n".join(f"{key}: {json_text(value)}" for key, value in card.items())


def pack(trace):
    return trace["result"]["pack"]


def prompt_versions(cfg, graph, op, profile, features, public_task, source, top_k, budget):
    options = dict(cfg=cfg, op=op, profile=profile, data_dir=graph,
                   code_features=features, write=False, strict=True,
                   source_op=SOURCE_ANALOGIES.get(op))
    independent = retrieve(**options, independent=True, top_k=1000)
    cards = [canonical_card(raw) for raw in pack(independent)["strategies"]]
    # Use the task and platform vocabulary without using expert code.
    query = " ".join((op, profile, json_text(public_task), source))
    ranked = rank_text(query, [(c, render_text(c)) for c in cards])
    for limit in range(min(top_k, len(cards)), 0, -1):
        trial = retrieve(**options, top_k=limit)
        full_pack = pack(trial)
        chosen = [canonical_card(raw) for raw in full_pack["strategies"]]
        planned = [
            {**card, "target_parameters": raw.get("target_parameters")}
            for card, raw in zip(chosen, full_pack["strategies"])
        ]
        # Match the number of whole cards, even when constraints cut down the plan.
        count = len(chosen)
        candidate = {
            "no_knowledge": ("", []),
            "text_rag": ("\n\n".join(map(render_text, ranked[:count])), ranked[:count]),
            "strategy_cards": (json_text(cards[:count]), cards[:count]),
            "constraint_plan": (
                json_text({"cards": planned, "plan": full_pack.get("plan")}), chosen),
        }
        if max(len(content.encode("utf-8")) for content, _ in candidate.values()) <= budget:
            return candidate, independent, trial

    if cards:
        raise ValueError(f"{op}/{profile}: no complete matched context fits the budget")
    empty = retrieve(**options, top_k=1)
    return {method: ("", []) for method in METHODS}, independent, empty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=RESEARCH / "算子输入")
    parser.add_argument("--graph", type=Path, default=RESEARCH / "kg_agent" / "data_thesis")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", action="append")
    parser.add_argument("--profile", action="append", choices=PROFILES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--knowledge-byte-cap", type=int, default=18000)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if min(args.top_k, args.knowledge_byte_cap, args.repetitions) <= 0:
        parser.error("top-k, byte cap and repetitions must be positive")

    corpus_manifest = args.corpus / "manifest.json"
    graph_manifest = args.graph / "manifest.json"
    tasks = read_json(corpus_manifest)["tasks"]
    cfg = load_config()
    selected_tasks = args.task or list(tasks)
    unknown = set(selected_tasks) - tasks.keys()
    if unknown:
        parser.error(f"unknown tasks: {sorted(unknown)}")
    if len(set(selected_tasks)) != len(selected_tasks):
        parser.error("task selection contains duplicates")
    profiles = args.profile or PROFILES
    if len(set(profiles)) != len(profiles):
        parser.error("profile selection contains duplicates")
    graph_metadata = read_json(graph_manifest)
    if graph_metadata.get("split") != "thesis":
        split = graph_metadata.get("dataset_split") or graph_metadata.get("split")
        if not (isinstance(split, dict) and split.get("name") == "thesis"):
            parser.error("use the isolated thesis graph")
    for name in ("nodes.jsonl", "edges.jsonl"):
        expected = graph_metadata["data"][name]["sha256"]
        if sha256(args.graph / name) != expected:
            parser.error(f"graph snapshot hash mismatch: {name}")

    graph = load_graph(args.graph)
    predicates = load_schema(cfg.schema_dir).predicates
    hardware = {}
    for profile in profiles:
        hardware_node = graph.nodes[f"hw:{profile}"]
        details = hardware_node["properties"]
        hardware[profile] = {
            "profile": details,
            "microarchitecture": graph.nodes[details["uarch_id"]]["properties"],
            "provisional": True,
        }
    shared_header = (args.corpus / "include" / "opbench.h").read_text(encoding="utf-8")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for op in selected_tasks:
        task = tasks[op]
        contract_path = args.corpus / task["contract"]
        source_path = args.corpus / task["generic"]
        contract = read_json(contract_path)
        source = source_path.read_text(encoding="utf-8")
        features = {field: value for field, value in contract["features"].items() if field in predicates}
        public_task = {field: contract[field] for field in CONTRACT_FIELDS if field in contract}
        for profile in profiles:
            choices, independent, full = prompt_versions(
                cfg, args.graph, op, profile, features, public_task, source,
                args.top_k, args.knowledge_byte_cap)
            case = args.out / op / profile
            case.mkdir(parents=True)
            write_json(case / "trace_cards.json", independent)
            write_json(case / "trace_plan.json", full)
            for method, (knowledge, cards) in choices.items():
                payload = {
                    "operator": op, "target": hardware[profile], "task": public_task,
                    "interface_header": shared_header, "input_c": source,
                    "knowledge": knowledge,
                }
                prompt_path = case / f"{method}.json"
                write_json(prompt_path, {"system": SYSTEM, "user": json_text(payload)})
                for repetition in range(1, args.repetitions + 1):
                    rows.append({
                        "operator": op, "split": task["split"], "profile": profile,
                        "method": method, "repetition": repetition,
                        "prompt": prompt_path.relative_to(args.out).as_posix(),
                        "prompt_sha256": sha256(prompt_path),
                        "source_sha256": sha256(source_path),
                        "contract_sha256": sha256(contract_path),
                        "knowledge_bytes": len(knowledge.encode("utf-8")),
                        "card_ids": [card["id"] for card in cards],
                        "analogy_source": SOURCE_ANALOGIES.get(op),
                        "status": "prepared",
                    })
    write_json(args.out / "manifest.json", {
        "graph_sha256": sha256(graph_manifest), "corpus_sha256": sha256(corpus_manifest),
        "knowledge_byte_cap": args.knowledge_byte_cap, "cases": rows,
    })
    print(f"Prepared {len(rows)} generation cells in {args.out}.")


if __name__ == "__main__":
    main()
