"""schema 层测试：YAML 子集解析、闭集、JSON Schema 子集校验、条件谓词定义。"""

import json
import unittest
from pathlib import Path

from kgagent import yamlmini
from kgagent.config import load_config
from kgagent.schema import load_schema, validate_json_schema

KG_ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(KG_ROOT / "config.yaml")
BUNDLE = load_schema(CFG.schema_dir)


class YamlMiniTest(unittest.TestCase):
    def test_nested_map_and_inline_structures(self):
        text = (
            "root:\n"
            "  a: 1\n"
            "  b: [x, y, 2]\n"
            "  c: {k: v, n: 3}\n"
            "  list:\n"
            "    - one\n"
            "    - name: two\n"
            "      value: 2\n"
        )
        data = yamlmini.loads(text)
        self.assertEqual(data["root"]["a"], 1)
        self.assertEqual(data["root"]["b"], ["x", "y", 2])
        self.assertEqual(data["root"]["c"], {"k": "v", "n": 3})
        self.assertEqual(data["root"]["list"][0], "one")
        self.assertEqual(data["root"]["list"][1], {"name": "two", "value": 2})

    def test_comment_and_quotes(self):
        data = yamlmini.loads('a: "x # not comment"  # real comment\nb: true\nc: null\n')
        self.assertEqual(data["a"], "x # not comment")
        self.assertTrue(data["b"])
        self.assertIsNone(data["c"])

    def test_config_loads_real_yaml_files(self):
        self.assertEqual(CFG.schema_version, "kg-v1.5")
        self.assertEqual(len(CFG.operators), 12)
        self.assertEqual(len(BUNDLE.node_types), 12)
        self.assertEqual(len(BUNDLE.relation_types), 14)


class ClosedSetTest(unittest.TestCase):
    def test_entity_and_relation_closed_sets(self):
        expected_entities = {
            "Operator", "ComputationPattern", "CodeCharacteristic", "OptimizationStrategy",
            "ImplementationTemplate", "TunableParameter", "ISACapability", "Microarchitecture",
            "HardwareProfile", "PerformanceEvidence", "SourceArtifact", "FailureCase",
        }
        expected_relations = {
            "OPERATOR_HAS_PATTERN", "PATTERN_HAS_CHARACTERISTIC", "PATTERN_OPTIMIZED_BY",
            "STRATEGY_HAS_PARAMETER", "STRATEGY_HAS_TEMPLATE", "STRATEGY_APPLIES_WHEN",
            "STRATEGY_REQUIRES_ISA", "STRATEGY_TARGETS_UARCH", "STRATEGY_EVIDENCED_BY",
            "EVIDENCE_ON", "EVIDENCE_SOURCE", "STRATEGY_COMPOSES_WITH", "STRATEGY_CONFLICTS_WITH",
            "STRATEGY_GENERALIZES",
        }
        self.assertEqual(set(BUNDLE.node_types), expected_entities)
        self.assertEqual(set(BUNDLE.relation_types), expected_relations)

    def test_condition_vocab_has_required_predicates(self):
        required = {"dtype", "precision", "stride_x", "stride_y", "loop_depth",
                    "size_regime", "aligned", "tail_divisible", "structural", "reduction"}
        self.assertEqual(set(BUNDLE.predicates), required)
        self.assertEqual(BUNDLE.predicates["dtype"]["values"], ["s", "d", "c", "z"])
        self.assertEqual(BUNDLE.char_type_map["dtype"], "dtype")

    def test_hardware_profiles_have_numeric_fields(self):
        profiles = BUNDLE.hardware_profiles()
        for name in ("x86_avx2", "x86_avx512", "k1_rvv256", "qemu-riscv64-vlen256"):
            self.assertIn(name, profiles)
            self.assertIn("vector_bits", profiles[name])
            self.assertIn("isa", profiles[name])
            self.assertTrue(profiles[name]["isa"])
        self.assertGreaterEqual(len(profiles["k1_rvv256"]), 12)


class NodeSchemaTest(unittest.TestCase):
    def _valid_node(self):
        return {
            "id": "stg:demo:vectorize",
            "type": "OptimizationStrategy",
            "name": "demo",
            "properties": {"family": "SIMD向量指令"},
            "provenance": {"source_file": "a/b.c", "extractor_version": "kgagent-v1.5"},
        }

    def test_valid_node_passes(self):
        self.assertEqual(BUNDLE.check_node(self._valid_node()), [])

    def test_missing_provenance_fails(self):
        node = self._valid_node()
        del node["provenance"]
        errors = BUNDLE.check_node(node)
        self.assertTrue(any("provenance" in e for e in errors))

    def test_invalid_type_and_id_fail(self):
        node = self._valid_node()
        node["type"] = "NotAnEntity"
        node["id"] = "BAD ID!"
        errors = BUNDLE.check_node(node)
        self.assertGreaterEqual(len(errors), 2)

    def test_edge_schema_and_relation_enum(self):
        edge = {
            "id": "e:op:gemm::OPERATOR_HAS_PATTERN::pat:gemm:core",
            "type": "OPERATOR_HAS_PATTERN",
            "head": "op:gemm",
            "tail": "pat:gemm:core",
            "properties": {},
            "provenance": {"source_file": "a/b.c"},
        }
        self.assertEqual(BUNDLE.check_edge(edge), [])
        edge["type"] = "NOT_A_REL"
        self.assertTrue(BUNDLE.check_edge(edge))

    def test_json_schema_subset_type_check(self):
        schema = {"type": "object", "required": ["x"], "properties": {"x": {"type": "integer", "minimum": 2}}}
        self.assertEqual(validate_json_schema({"x": 3}, schema), [])
        self.assertTrue(validate_json_schema({"x": 1}, schema))
        self.assertTrue(validate_json_schema({"x": "3"}, schema))
        self.assertTrue(validate_json_schema({}, schema))


if __name__ == "__main__":
    unittest.main()
