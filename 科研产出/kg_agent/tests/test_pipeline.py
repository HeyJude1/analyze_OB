"""流水线测试：离线 build 产物、validate 通过、验收条数、幂等、export-viz、LLM offline 降级。"""

import hashlib
import json
import unittest
from pathlib import Path

from tests.helpers import KG_ROOT, built, config, fresh_data_dir

from kgagent import corpus, validate, viz_export
from kgagent.llm import LLMClient, OfflineLLMUnavailable


class PipelineBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.data_dir, cls.report = built()

    def test_data_files_exist_and_non_empty(self):
        for name in ("corpus_manifest.jsonl", "characteristics.jsonl", "diffs.jsonl",
                     "strategies_raw.jsonl", "nodes.jsonl", "edges.jsonl", "manifest.json"):
            path = Path(self.data_dir) / name
            self.assertTrue(path.exists(), "缺少 %s" % name)
            self.assertGreater(path.stat().st_size, 0, "%s 为空" % name)
        self.assertTrue((Path(self.data_dir) / "traces").is_dir(), "build 应创建 data/traces/")
        nodes = corpus.read_jsonl(Path(self.data_dir) / "nodes.jsonl")
        edges = corpus.read_jsonl(Path(self.data_dir) / "edges.jsonl")
        self.assertGreater(len(nodes), 100)
        self.assertGreater(len(edges), 500)

    def test_validate_catches_dangling_edge_and_bad_condition(self):
        cfg = config()
        node = {
            "id": "stg:demo:vectorize", "type": "OptimizationStrategy", "name": "demo",
            "properties": {
                "family": "SIMD向量指令", "level": "code", "action": "vectorize",
                "description": "d", "conditions": [{"pred": "not_a_pred", "op": "eq", "value": 1, "hard": True}],
                "condition_status": "executable", "template_id": "tpl:missing",
                "hardware_profile_ids": ["hw:missing"], "status": "raw",
            },
            "provenance": {"source_file": "a/b.c", "line_start": 1, "line_end": 2,
                           "extractor_version": "kgagent-v1.5"},
        }
        edge = {
            "id": "e:stg:demo:vectorize::PATTERN_OPTIMIZED_BY::pat:missing",
            "type": "PATTERN_OPTIMIZED_BY", "head": "stg:demo:vectorize", "tail": "pat:missing",
            "properties": {}, "provenance": {"source_file": "a/b.c"},
        }
        report = validate.check(cfg, [node], [edge], require_acceptance=False)
        self.assertFalse(report["ok"])
        joined = "\n".join(report["errors"])
        self.assertIn("tail 不存在", joined)
        self.assertIn("not_a_pred", joined)
        self.assertIn("hardware profile", joined)

    def test_manifest_has_version_quadruple_and_stats(self):
        manifest = json.loads((Path(self.data_dir) / "manifest.json").read_text(encoding="utf-8"))
        for key in ("schema_version", "extractor_version", "prompt_version", "model_version"):
            self.assertTrue(manifest.get(key), "manifest 缺 %s" % key)
        self.assertEqual(manifest["stats"]["nodes"], len(corpus.read_jsonl(Path(self.data_dir) / "nodes.jsonl")))
        self.assertGreater(len(manifest["inputs"]), 20, "manifest 应记录所有输入文件 sha256")
        for item in manifest["inputs"]:
            self.assertEqual(len(item["sha256"]), 64)

    def test_validate_passes(self):
        report = validate.validate_data_dir(self.cfg, self.data_dir)
        self.assertTrue(report["ok"], "validate 失败：%s" % report["errors"][:5])
        self.assertEqual(report["errors"], [])

    def test_acceptance_at_least_20_strategies(self):
        report = validate.validate_data_dir(self.cfg, self.data_dir)
        self.assertGreaterEqual(report["stats"]["acceptance_count"], 20)
        self.assertGreaterEqual(len(report["acceptance"]), 20)
        # 至少 20 条是具体策略（非抽象层）
        concrete = [sid for sid in report["acceptance"] if not sid.startswith("stg:abstract:")]
        self.assertGreaterEqual(len(concrete), 20)

    def test_build_is_idempotent(self):
        cfg = config()
        second_dir = fresh_data_dir("data2")
        validate.build(cfg, mode="offline", data_dir=second_dir)
        for name in ("nodes.jsonl", "edges.jsonl"):
            first = hashlib.sha256((Path(self.data_dir) / name).read_bytes()).hexdigest()
            second = hashlib.sha256((Path(second_dir) / name).read_bytes()).hexdigest()
            self.assertEqual(first, second, "%s 两次 build 不一致（破坏幂等）" % name)

    def test_positive_and_negative_evidence(self):
        nodes = corpus.read_jsonl(Path(self.data_dir) / "nodes.jsonl")
        evidences = [n for n in nodes if n["type"] == "PerformanceEvidence"]
        self.assertGreaterEqual(len(evidences), 80)
        for ev in evidences:
            self.assertTrue(ev["properties"]["rerun_needed"])
            self.assertEqual(ev["properties"]["platform_id"], "hw:qemu-riscv64-vlen256")
        negatives = [e for e in evidences if e["properties"].get("negative_example")]
        self.assertTrue(negatives, "应保留 idamax/dnrm2 等负例（G6）")

    def test_export_viz_creates_two_js_files(self):
        from kgagent import retrieve as retrieve_mod

        # 先产生一条 trace，验证 traces.js 的"检索回放"数据源
        retrieve_mod.retrieve(self.cfg, "dgemm", "x86_avx2", data_dir=self.data_dir, write=True)
        viz_dir = fresh_data_dir("viz")
        result = viz_export.export(self.cfg, data_dir=self.data_dir, viz_dir=viz_dir)
        graph_js = Path(result["graph_data"]).read_text(encoding="utf-8")
        traces_js = Path(result["traces"]).read_text(encoding="utf-8")
        self.assertTrue(graph_js.startswith("window.KG_DATA = "))
        self.assertTrue(traces_js.startswith("window.KG_TRACES = "))
        self.assertIn('"nodes"', graph_js)
        self.assertIn('"steps"', traces_js)
        self.assertGreater(result["nodes"], 100)
        self.assertGreaterEqual(result["trace_count"], 1)


class LlmOfflineTest(unittest.TestCase):
    def test_offline_backend_never_calls_network(self):
        cfg = config()
        client = LLMClient(cfg, backend="offline")
        with self.assertRaises(OfflineLLMUnavailable):
            client.generate_json("测试 prompt：请输出 JSON", purpose="unit_test")


class DiffExtractionTest(unittest.TestCase):
    def test_diff_actions_for_dgemm(self):
        cfg = config()
        artifacts = corpus.run(cfg)
        from kgagent import diff, static

        chars, impls = static.run(cfg, artifacts)
        diffs = {d["op"]: d for d in diff.run(cfg, impls)}
        self.assertIn("gemm", diffs)
        actions = diffs["gemm"]["actions"]
        self.assertIn("vectorize", actions)
        self.assertIn("blocking", actions)
        self.assertIn("unroll", actions)
        # 差分锚点必须带行号（可核查）
        self.assertGreater(diffs["gemm"]["anchors"]["after"]["line_end"],
                           diffs["gemm"]["anchors"]["after"]["line_start"])
        self.assertTrue(chars)


if __name__ == "__main__":
    unittest.main()
