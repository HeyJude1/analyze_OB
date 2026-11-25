"""Regression checks for the isolated snapshot and the strict query path."""

import unittest

from kgagent import retrieve, validate
from tests.helpers import config, fresh_data_dir


class ThesisSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = config()
        cls.data_dir = fresh_data_dir("data_thesis")
        cls.report = validate.build(cls.cfg, data_dir=cls.data_dir, split="thesis")

    def query(self, op, profile="k1_rvv256", **kwargs):
        return retrieve.retrieve(self.cfg, op, profile, strict=True, write=False,
                                 data_dir=self.data_dir, **kwargs)["result"]["pack"]

    def test_split_rebuilds_only_training_families(self):
        nodes, _, manifest = validate.load_data(self.data_dir)
        self.assertEqual(manifest["split"], "thesis")
        self.assertEqual({n["id"] for n in nodes if n["type"] == "Operator"},
                         {"op:axpy", "op:dot", "op:gemv", "op:gemm"})
        self.assertEqual({n["properties"]["op"] for n in nodes if n["type"] == "SourceArtifact"},
                         {"axpy", "dot", "gemv", "gemm"})
        self.assertFalse(any(n["type"] == "PerformanceEvidence" for n in nodes))
        self.assertTrue(validate.validate_data_dir(self.cfg, self.data_dir)["ok"])
        audit = manifest["source_stats"]["split_audit"]
        self.assertTrue(audit["overlaps_with_shared_source_cases"])
        self.assertEqual({row["op"] for row in audit["overlaps_with_shared_source_cases"]}, {"symm"})

    def test_explicit_analogy_and_target_parameters(self):
        features = {"dtype": "d", "loop_depth": 2, "stride_x": 1, "reduction": True}
        with self.assertRaises(ValueError):
            self.query("conv1d", code_features=features)
        rvv = self.query("conv1d", code_features=features, source_op="dot")
        simd = self.query("conv1d", profile="x86_avx2", code_features=features, source_op="dot")
        self.assertEqual(rvv["query"]["code_features"], features)
        self.assertEqual(rvv["query"]["source_operator"], "dot")
        self.assertTrue(rvv["strategies"])
        self.assertTrue(all(card["id"].startswith("stg:abstract:") and not card["template"]["skeleton"]
                            for card in rvv["strategies"]))
        by_action = lambda pack, action: next(c for c in pack["strategies"] if c["action"] == action)
        self.assertIn("rvv", by_action(rvv, "vectorize")["target_parameters"])
        self.assertEqual(by_action(simd, "vectorize")["target_parameters"]["simd"]["fixed_lane_count"], 4)
        self.assertTrue(rvv["plan"]["relations"])
        ids = rvv["plan"]["ordered_strategy_ids"]
        self.assertLess(ids.index(by_action(rvv, "vectorize")["id"]), ids.index(by_action(rvv, "tail_handle")["id"]))

    def test_independent_cards_keep_conditions_and_target_parameters(self):
        features = {"dtype": "d", "loop_depth": 1, "size_regime": "medium", "stride_x": 1,
                    "stride_y": 1, "reduction": True}
        full = self.query("ddot", code_features=features)
        independent = self.query("ddot", code_features=features, independent=True, top_k=50)
        self.assertIsNone(independent["plan"])
        candidates = {c["id"]: c for c in independent["strategies"]}
        self.assertTrue(full["plan"]["relations"])
        for card in full["strategies"]:
            self.assertEqual(card["conditions"], candidates[card["id"]]["conditions"])
            self.assertEqual(card["target_parameters"], candidates[card["id"]]["target_parameters"])
        unknown = self.query("ddot", code_features={})
        self.assertEqual(unknown["strategies"], [])
        self.assertEqual(unknown["plan"]["status"], "no_eligible_strategy")

    def test_speedup_requires_verified_matching_native_workload(self):
        profile = {"measure_env": "native", "profile_name": "x86_avx2", "compiler": "gcc 13"}
        workload, baseline = {"n": 257}, {"flags": "-O3"}
        record = {"measure_env": "native", "platform_id": "hw:x86_avx2", "compiler": "gcc 13",
                  "metric": "speedup", "value": 1.20, "rerun_needed": False, "protocol": "test fixture",
                  "workload": workload, "baseline": baseline}
        relation = {"role": "supports", "scope": "strategy"}
        eligible = retrieve._eligible_speedup
        self.assertTrue(eligible(profile, record, relation, workload, baseline))
        self.assertFalse(eligible(profile, record, relation, {"n": 258}, baseline))
        self.assertFalse(eligible(profile, record, {**relation, "scope": "operator"}, workload, baseline))
        self.assertFalse(eligible(profile, {**record, "measure_env": "qemu"}, relation, workload, baseline))
