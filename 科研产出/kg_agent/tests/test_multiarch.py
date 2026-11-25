"""多架构扩展测试：arch 自动发现/复制、汇编静态特征、arch_scope、跨架构合并、按 arch 检索。"""

import json
import unittest
from pathlib import Path

from tests.helpers import built, config

from kgagent import corpus, graph as graph_mod, retrieve as retrieve_mod, static, validate

ARCHES = {"x86_avx2", "x86_avx512", "arm_neon", "arm_sve", "rvv"}


class MultiArchPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.data_dir, cls.report = built()
        cls.nodes = corpus.read_jsonl(Path(cls.data_dir) / "nodes.jsonl")
        cls.edges = corpus.read_jsonl(Path(cls.data_dir) / "edges.jsonl")
        cls.by_id = {n["id"]: n for n in cls.nodes}
        cls.arch = cls.report.get("arch") or {}

    def test_discovery_covers_multiple_architectures(self):
        path = Path(self.data_dir) / "multiarch_manifest.jsonl"
        self.assertTrue(path.exists())
        artifacts = corpus.read_jsonl(path)
        self.assertGreaterEqual(len(artifacts), 30)
        arches = {a["arch"] for a in artifacts}
        self.assertEqual(arches, ARCHES)
        self.assertGreaterEqual(self.arch["ops_with_multi_arch"], 8,
                                "至少 8 个算子应有 >=2 架构实现")
        self.assertGreaterEqual(len(self.arch["ops_with_multi_arch_list"]), 8)

    def test_copied_files_are_self_contained(self):
        artifacts = corpus.read_jsonl(Path(self.data_dir) / "multiarch_manifest.jsonl")
        cfg = config()
        for art in artifacts:
            self.assertTrue(art["path"].startswith("data/corpus_multiarch/"),
                            "多架构源文件必须复制进 kg_agent：%s" % art["path"])
            abs_path = cfg.root / art["path"]
            self.assertTrue(abs_path.exists(), "复制件不存在：%s" % abs_path)
            self.assertEqual(art["sha256"], corpus.sha256_file(abs_path),
                             "sha256 与复制件不一致：%s" % art["path"])
            self.assertGreater(art["line_count"], 0)

    def test_asm_static_features_extracted(self):
        """汇编-only 实现必须能抽出 mnemonics/向量特征（method=static）。"""
        cfg = config()
        asm_files = sorted((cfg.root / "data" / "corpus_multiarch" / "x86_avx2" / "gemm").glob("*.S"))
        self.assertTrue(asm_files, "缺少 x86 gemm 汇编复制件")
        feature = static.parse_arch_impl(asm_files[0], "gemm", "x86_avx2", "x86", vector_bits=256)
        self.assertEqual(feature["kind"], "asm")
        self.assertGreater(feature["instruction_count"], 100)
        self.assertTrue(feature["mnemonic_counts"])
        self.assertGreater(feature["vector_instr_count"], 0)
        self.assertTrue(feature["detected_isa"].startswith(("AVX", "SSE")))
        # 汇编实现的差分记录 method 必须是 static（没有 generic vs optimized 的 C 对）
        diffs = corpus.read_jsonl(Path(self.data_dir) / "diffs.jsonl")
        ma_diffs = [d for d in diffs if d.get("diff_id", "").startswith("ma_diff:")]
        self.assertTrue(ma_diffs)
        gemm_x86 = [d for d in ma_diffs if d["op"] == "gemm" and d["arch"] == "x86_avx2"][0]
        self.assertEqual(gemm_x86["method"], "static")
        self.assertIn("vectorize", gemm_x86["actions"])

    def test_every_strategy_has_arch_scope(self):
        strategies = [n for n in self.nodes if n["type"] == "OptimizationStrategy"]
        scopes = {}
        for strategy in strategies:
            scope = strategy["properties"].get("arch_scope")
            self.assertIn(scope, ("arch_agnostic", "parameterized", "arch_specific"))
            scopes[scope] = scopes.get(scope, 0) + 1
            if scope == "arch_specific":
                self.assertTrue(strategy["properties"].get("arch"))
                self.assertTrue(strategy["properties"].get("arch_evidence"))
        self.assertGreaterEqual(len(strategies), 20)
        self.assertEqual(set(scopes), {"arch_agnostic", "parameterized", "arch_specific"})

    def test_cross_arch_shared_strategies(self):
        shared = self.arch["cross_arch_shared"]
        self.assertGreaterEqual(len(shared), 1)
        for item in shared:
            self.assertGreaterEqual(len(item["arches"]), 2)
            node = self.by_id[item["id"]]
            props = node["properties"]
            self.assertEqual(props["arch_scope"], "parameterized")
            for arch in item["arches"]:
                self.assertIn(arch, props["arch_parameters"])
            self.assertGreaterEqual(len(props["arch_evidence"]), 2)
        # arch_specific -> parameterized 的 GENERALIZES 边必须存在
        generalized = [e for e in self.edges if e["type"] == "STRATEGY_GENERALIZES"]
        self.assertTrue(any(e["tail"] in {item["id"] for item in shared} for e in generalized))

    def test_arch_matrix_and_validate(self):
        matrix_path = Path(self.data_dir) / "arch_coverage.json"
        self.assertTrue(matrix_path.exists())
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(matrix["ops_with_multi_arch"], 8)
        self.assertEqual(set(matrix["arch_matrix"].keys()), ARCHES)
        report = validate.validate_data_dir(config(), self.data_dir)
        self.assertTrue(report["ok"], "validate 失败：%s" % report["errors"][:5])
        self.assertGreaterEqual(report["stats"]["arch"]["cross_arch_shared_count"], 1)

    def test_retrieve_filters_by_architecture(self):
        cfg = config()
        x86 = retrieve_mod.retrieve(cfg, "dgemm", "x86_avx2", data_dir=self.data_dir, write=False)
        rvv = retrieve_mod.retrieve(cfg, "dgemm", "k1_rvv256", data_dir=self.data_dir, write=False)
        arm = retrieve_mod.retrieve(cfg, "dgemm", "arm_neon", data_dir=self.data_dir, write=False)
        x86_ids = x86["result"]["strategies"]
        rvv_ids = rvv["result"]["strategies"]
        arm_ids = arm["result"]["strategies"]
        self.assertTrue(any(sid.endswith("_x86_avx2") for sid in x86_ids))
        self.assertTrue(any(sid.endswith("_d") for sid in rvv_ids))
        self.assertTrue(any(sid.endswith("_arm_neon") for sid in arm_ids))
        # 架构过滤：x86 结果不应选到 arm/rvv 的 arch_specific 策略
        graph = graph_mod.load_graph(self.data_dir)
        for sid in x86_ids:
            props = graph.node(sid)["properties"]
            if props.get("arch_scope") == "arch_specific":
                self.assertEqual(props.get("arch"), "x86_avx2")

    def test_scale_not_regressed(self):
        self.assertGreater(len(self.nodes), 527)
        self.assertGreater(len(self.edges), 2218)


if __name__ == "__main__":
    unittest.main()
