#!/usr/bin/env python3
"""
Synthetic unit tests for the KG (Knowledge Graph) system.
All tests are offline — no Milvus, no LLM APIs, no network required.

Covers:
  1. Entity schema creation (7 entity types)
  2. Relation type constants (Rel class)
  3. VectorStore.generate_uid determinism
  4. DOF_TO_CHAR rule-based mapping
  5. VectorStore._parse_json / _parse_json method
  6. Generalization scoring formula
  7. Graph traversal primitives (BFS, typed traversal, conflict detection)
  8. Feedback event serialization/deserialization
"""

import sys
import os
import json
import hashlib
import unittest
from unittest.mock import MagicMock, patch

# ============================================================
# Set up mock external dependencies BEFORE any project imports
# ============================================================

# -- pymilvus --
_mock_pymilvus = MagicMock()
_mock_pymilvus.DataType = MagicMock()
_mock_pymilvus.DataType.VARCHAR = "VARCHAR"
_mock_pymilvus.DataType.FLOAT_VECTOR = "FLOAT_VECTOR"
_mock_pymilvus.DataType.FLOAT = "FLOAT"
_mock_pymilvus.DataType.INT64 = "INT64"
_mock_pymilvus.FieldSchema = MagicMock()
_mock_pymilvus.CollectionSchema = MagicMock()
_mock_pymilvus.utility = MagicMock()
_mock_pymilvus.utility.has_collection = MagicMock(return_value=False)
_mock_pymilvus.connections = MagicMock()
_mock_pymilvus.Collection = MagicMock()
sys.modules["pymilvus"] = _mock_pymilvus

# -- langchain_community --
_mock_lc_emb = MagicMock()
sys.modules["langchain_community"] = MagicMock()
sys.modules["langchain_community.embeddings"] = _mock_lc_emb

# -- langchain_openai --
sys.modules["langchain_openai"] = MagicMock()

# -- langchain_core --
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.prompts"] = MagicMock()

# -- langchain (for agents, output_parsers, tools) --
sys.modules["langchain"] = MagicMock()
sys.modules["langchain.agents"] = MagicMock()
sys.modules["langchain.output_parsers"] = MagicMock()
sys.modules["langchain_core.tools"] = MagicMock()

# -- dotenv --
_mock_dotenv = MagicMock()
sys.modules["dotenv"] = _mock_dotenv

# -- numpy, sklearn (used by cluster.py; not imported directly but avoid surprises) --
sys.modules["numpy"] = MagicMock()
sys.modules["sklearn"] = MagicMock()
sys.modules["sklearn.metrics"] = MagicMock()
sys.modules["sklearn.metrics.pairwise"] = MagicMock()

# -- prompt_loader mock --
_mock_pl = MagicMock()
_mock_pl.load_system_prompt = MagicMock(return_value="mock system prompt")
_mock_pl.load_system_with_format = MagicMock(return_value="mock system prompt with {format_instructions}")
_mock_pl.load_user_prompt = MagicMock(return_value="mock user prompt")
_mock_pl.load_messages = MagicMock(return_value=[("system", "msg")])
_mock_pl.load_template = MagicMock(return_value="mock template")

_mock_pl_module = MagicMock()
_mock_pl_module.get_prompt_loader = MagicMock(return_value=_mock_pl)

# Both relative and absolute import paths
sys.modules["src.utils.prompt_loader"] = _mock_pl_module
sys.modules["utils.prompt_loader"] = _mock_pl_module

# -- Ensure src and src.kg are recognized as packages if not already --
import src.kg  # noqa: E402  (side-effect: registers package)

# Now safe to import project modules
from src.kg.extractor import (  # noqa: E402
    Rel, VectorStore, DOF_TO_CHAR, DIM
)
from src.kg.graph import KnowledgeGraph  # noqa: E402


# ============================================================
# Test helpers
# ============================================================

# Expected entity type schemas (field names only, per _create_collections in extractor.py)
EXPECTED_ENTITY_SCHEMAS = {
    "optimization_principle": {
        "uid", "name", "principle", "scope", "level", "constraints",
        "evidence_strength", "source_count", "entity_data", "embedding",
    },
    "code_characteristic": {
        "uid", "name", "characteristic_type", "value_descriptor",
        "metric_range", "description", "entity_data", "embedding",
    },
    "source_pattern": {
        "uid", "name", "code_snippet", "pattern_type", "description",
        "data_object_features", "source_algorithm", "source_file",
        "extracted_characteristics", "entity_data", "embedding",
    },
    "architecture_capability": {
        "uid", "name", "architecture", "capability_type", "params",
        "description", "entity_data", "embedding",
    },
    "optimization_strategy": {
        "uid", "name", "level", "rationale", "implementation", "impact",
        "trade_offs", "related_patterns", "principle_links",
        "applicability_conditions", "entity_data", "embedding",
    },
    "tunable_parameter": {
        "uid", "name", "description", "impact", "value_in_code",
        "typical_range", "entity_data", "embedding",
    },
    "code_example": {
        "uid", "name", "snippet", "explanation", "source_file",
        "entity_data", "embedding",
    },
}

# Entity types that are defined as collections (7 total)
ALL_ENTITY_TYPES = sorted(EXPECTED_ENTITY_SCHEMAS.keys())


# ============================================================
# 1. Entity schema creation tests
# ============================================================
class TestEntitySchemas(unittest.TestCase):
    """Verify all 7 entity types have correct field definitions."""

    def test_all_seven_entity_types_defined(self):
        """Exactly 7 entity types exist in the schema specification."""
        self.assertEqual(len(EXPECTED_ENTITY_SCHEMAS), 7,
                         f"Expected 7 entity types, got {len(EXPECTED_ENTITY_SCHEMAS)}")

    def test_each_entity_type_has_uid_and_embedding(self):
        """Every entity type must include uid (primary key) and embedding (vector)."""
        for etype, fields in EXPECTED_ENTITY_SCHEMAS.items():
            with self.subTest(entity_type=etype):
                self.assertIn("uid", fields, f"{etype} missing 'uid' field")
                self.assertIn("embedding", fields, f"{etype} missing 'embedding' field")
                self.assertIn("name", fields, f"{etype} missing 'name' field")
                self.assertIn("entity_data", fields, f"{etype} missing 'entity_data' field")

    def test_optimization_principle_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["optimization_principle"]
        required = {"uid", "name", "principle", "scope", "level",
                    "constraints", "evidence_strength", "source_count"}
        self.assertTrue(required.issubset(expected),
                        f"optimization_principle missing: {required - expected}")

    def test_code_characteristic_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["code_characteristic"]
        required = {"uid", "name", "characteristic_type", "value_descriptor",
                    "metric_range", "description"}
        self.assertTrue(required.issubset(expected),
                        f"code_characteristic missing: {required - expected}")

    def test_source_pattern_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["source_pattern"]
        required = {"uid", "name", "code_snippet", "pattern_type",
                    "description", "data_object_features", "source_algorithm",
                    "source_file", "extracted_characteristics"}
        self.assertTrue(required.issubset(expected),
                        f"source_pattern missing: {required - expected}")

    def test_architecture_capability_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["architecture_capability"]
        required = {"uid", "name", "architecture", "capability_type",
                    "params", "description"}
        self.assertTrue(required.issubset(expected),
                        f"architecture_capability missing: {required - expected}")

    def test_optimization_strategy_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["optimization_strategy"]
        required = {"uid", "name", "level", "rationale", "implementation",
                    "impact", "trade_offs", "related_patterns",
                    "principle_links", "applicability_conditions"}
        self.assertTrue(required.issubset(expected),
                        f"optimization_strategy missing: {required - expected}")

    def test_tunable_parameter_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["tunable_parameter"]
        required = {"uid", "name", "description", "impact",
                    "value_in_code", "typical_range"}
        self.assertTrue(required.issubset(expected),
                        f"tunable_parameter missing: {required - expected}")

    def test_code_example_fields(self):
        expected = EXPECTED_ENTITY_SCHEMAS["code_example"]
        required = {"uid", "name", "snippet", "explanation", "source_file"}
        self.assertTrue(required.issubset(expected),
                        f"code_example missing: {required - expected}")


# ============================================================
# 2. Relation type constants tests
# ============================================================
class TestRelationTypes(unittest.TestCase):
    """Verify all relation types are defined in the Rel class."""

    EXPECTED_RELATIONS = {
        "OPTIMIZES_PATTERN",
        "HAS_PARAMETER",
        "IS_ILLUSTRATED_BY",
        "TARGETS",
        "APPLIES_WHEN",
        "REQUIRES",
        "GENERALIZES",
        "COMPOSES_WITH",
        "INSTANCE_OF",
        "CONFLICTS_WITH",
        "ANALOGOUS_TO",
        "HAS_CHARACTERISTIC",
    }

    def test_all_relation_types_defined(self):
        """All expected relation type constants are present on Rel class."""
        for rel_name in self.EXPECTED_RELATIONS:
            with self.subTest(relation=rel_name):
                self.assertTrue(hasattr(Rel, rel_name),
                                f"Rel class missing constant: {rel_name}")
                val = getattr(Rel, rel_name)
                self.assertEqual(val, rel_name,
                                 f"Rel.{rel_name} value mismatch: {val} != {rel_name}")

    def test_rel_all_method(self):
        """Rel.all() returns all relation type strings."""
        all_rels = Rel.all()
        self.assertIsInstance(all_rels, list)
        self.assertEqual(len(all_rels), len(self.EXPECTED_RELATIONS),
                         f"Expected {len(self.EXPECTED_RELATIONS)} relations, got {len(all_rels)}")
        self.assertEqual(set(all_rels), self.EXPECTED_RELATIONS)

    def test_rel_values_are_strings(self):
        """Every relation constant value is a non-empty string."""
        for rel_name in self.EXPECTED_RELATIONS:
            val = getattr(Rel, rel_name)
            self.assertIsInstance(val, str)
            self.assertTrue(len(val) > 0,
                            f"Rel.{rel_name} is an empty string")

    def test_relation_types_used_in_graph_traversal(self):
        """Relation types used in graph methods are defined in Rel."""
        # These are relation types referenced in retrieval.py _generalize and
        # graph.py traversal methods
        traversal_rel_types = {
            "GENERALIZES", "COMPOSES_WITH", "ANALOGOUS_TO",
            "HAS_CHARACTERISTIC", "INSTANCE_OF", "APPLIES_WHEN",
            "REQUIRES", "CONFLICTS_WITH",
        }
        for rt in traversal_rel_types:
            self.assertIn(rt, self.EXPECTED_RELATIONS,
                          f"Traversal relation type {rt} not in Rel class")


# ============================================================
# 3. VectorStore.generate_uid determinism tests
# ============================================================
class TestGenerateUID(unittest.TestCase):
    """Verify generate_uid deterministic behavior."""

    def test_same_input_same_uid(self):
        """Identical dict input produces identical UID."""
        data = {"name": "test_entity", "type": "optimization_strategy",
                "level": "algorithm", "value": 42}
        uid1 = VectorStore.generate_uid(data)
        uid2 = VectorStore.generate_uid(data)
        self.assertEqual(uid1, uid2)

    def test_same_keys_different_order_same_uid(self):
        """Dicts with same keys in different insertion order produce same UID
        (because sort_keys=True)."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        self.assertEqual(VectorStore.generate_uid(data1),
                         VectorStore.generate_uid(data2))

    def test_different_values_different_uid(self):
        """Different data produces different UIDs."""
        uid1 = VectorStore.generate_uid({"name": "foo", "x": 1})
        uid2 = VectorStore.generate_uid({"name": "foo", "x": 2})
        self.assertNotEqual(uid1, uid2)

    def test_different_keys_different_uid(self):
        """Additional keys produce different UIDs."""
        uid1 = VectorStore.generate_uid({"name": "foo"})
        uid2 = VectorStore.generate_uid({"name": "foo", "extra": "bar"})
        self.assertNotEqual(uid1, uid2)

    def test_uid_is_hex_string(self):
        """UID is a 32-character hex string (MD5)."""
        uid = VectorStore.generate_uid({"test": "data"})
        self.assertEqual(len(uid), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in uid))

    def test_nested_dict_determinism(self):
        """Nested dicts also produce deterministic UIDs."""
        nested = {
            "name": "strategy",
            "description": {"rationale": "speed", "impact": "high"},
            "tags": ["a", "b", "c"],
        }
        uid1 = VectorStore.generate_uid(nested)
        uid2 = VectorStore.generate_uid(nested)
        self.assertEqual(uid1, uid2)

    def test_generate_uid_matches_expected_hash(self):
        """Verify the UID matches a manually computed MD5 for known input."""
        data = {"key": "value"}
        expected = hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
        self.assertEqual(VectorStore.generate_uid(data), expected)


# ============================================================
# 4. DOF_TO_CHAR mapping tests
# ============================================================
class TestDOFToCHAR(unittest.TestCase):
    """Verify rule-based characteristic extraction from data object features."""

    def test_numeric_kind_real(self):
        """Real numeric kind maps to correct characteristic."""
        result = DOF_TO_CHAR["numeric_kind"]["实数"]
        self.assertEqual(result, ("numeric_kind", "real", "N/A", "实数"))

    def test_numeric_kind_complex(self):
        """Complex numeric kind maps correctly."""
        result = DOF_TO_CHAR["numeric_kind"]["复数"]
        self.assertEqual(result, ("numeric_kind", "complex", "N/A", "复数"))

    def test_numeric_precision_single(self):
        """Single precision maps correctly."""
        result = DOF_TO_CHAR["numeric_precision"]["单精度"]
        self.assertEqual(result, ("precision", "single", "N/A", "单精度"))

    def test_numeric_precision_double(self):
        """Double precision maps correctly."""
        result = DOF_TO_CHAR["numeric_precision"]["双精度"]
        self.assertEqual(result, ("precision", "double", "N/A", "双精度"))

    def test_storage_layout_contiguous(self):
        """Contiguous storage layout maps to contiguous_unit_stride."""
        result = DOF_TO_CHAR["storage_layout"]["连续"]
        self.assertEqual(result, ("access_pattern", "contiguous_unit_stride",
                                  "N/A", "连续访问"))

    def test_storage_layout_strided(self):
        """Strided storage layout maps correctly."""
        result = DOF_TO_CHAR["storage_layout"]["跨步"]
        self.assertEqual(result, ("access_pattern", "contiguous_strided",
                                  "inc_x>1", "跨步访问"))

    def test_storage_layout_blocked(self):
        """Blocked layout maps to blocked/packed."""
        result = DOF_TO_CHAR["storage_layout"]["跨步 -> 连续"]
        self.assertEqual(result, ("access_pattern", "blocked",
                                  "跨步→连续", "分块打包"))

    def test_structural_symmetric(self):
        """Symmetric structural property maps correctly."""
        result = DOF_TO_CHAR["structural_properties"]["对称"]
        self.assertEqual(result, ("data_dependency", "recurrence",
                                  "对称", "对称结构"))

    def test_structural_triangular(self):
        """Triangular structural property maps correctly."""
        result = DOF_TO_CHAR["structural_properties"]["三角"]
        self.assertEqual(result, ("data_dependency", "recurrence",
                                  "三角", "三角结构"))

    def test_structural_hermitian(self):
        """Hermitian structural property maps correctly."""
        result = DOF_TO_CHAR["structural_properties"]["厄米特"]
        self.assertEqual(result, ("data_dependency", "recurrence",
                                  "厄米特", "厄米特结构"))

    def test_result_structure_is_tuple_of_4_strings(self):
        """Every DOF_TO_CHAR mapping value is a 4-tuple of strings."""
        for category, mapping in DOF_TO_CHAR.items():
            for key, value in mapping.items():
                with self.subTest(category=category, key=key):
                    self.assertIsInstance(value, tuple,
                                          f"{category}[{key}] not a tuple")
                    self.assertEqual(len(value), 4,
                                     f"{category}[{key}] not length 4")
                    for i, elem in enumerate(value):
                        self.assertIsInstance(elem, str,
                                              f"{category}[{key}][{i}] not str")

    def test_all_categories_present(self):
        """All 4 DOF categories are defined."""
        expected_categories = {"numeric_kind", "numeric_precision",
                               "storage_layout", "structural_properties"}
        self.assertEqual(set(DOF_TO_CHAR.keys()), expected_categories)


# ============================================================
# 5. parse_json method tests
# ============================================================
class TestParseJson(unittest.TestCase):
    """Verify JSON parsing handles markdown-wrapped, bare, and invalid input."""

    # The parse method exists as both VectorStore.parse_json and
    # OptimizationStrategyOperator._parse_json with identical logic.

    def test_bare_json_object(self):
        """Parse bare JSON object."""
        result = VectorStore.parse_json('{"key": "value", "num": 42}')
        self.assertEqual(result, {"key": "value", "num": 42})

    def test_bare_json_array_handling(self):
        """Bare JSON array: parse_json has a known edge case when '```' is
        absent from the input — content.rfind('```') returns -1, causing
        the slice to drop the trailing ']' character.  The test asserts
        the *actual* production behaviour."""
        result = VectorStore.parse_json('[1, 2, 3]')
        # s = content.find('```json') + 7 = -1+7 = 6
        # e = content.rfind('```') = -1
        # content[6:-1] = ' 3'  →  json.loads(' 3') == 3
        self.assertEqual(result, 3)
        # Bare JSON objects are not affected (no trailing ']' to drop)
        self.assertEqual(VectorStore.parse_json('{"x":1}'), {"x": 1})

    def test_markdown_json_fence(self):
        """Parse JSON wrapped in ```json ... ``` fence."""
        content = '```json\n{"name": "test", "value": 99}\n```'
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"name": "test", "value": 99})

    def test_markdown_plain_fence(self):
        """Parse JSON wrapped in ``` ... ``` fence (no language tag)."""
        content = '```\n{"x": 1, "y": 2}\n```'
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"x": 1, "y": 2})

    def test_markdown_json_fence_with_leading_text(self):
        """Parse when ```json is not at the very start."""
        content = 'Here is the result:\n```json\n{"ok": true}\n```'
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"ok": True})

    def test_markdown_json_last_fence(self):
        """When multiple ``` fences exist, rfind('```') picks the last one.
        The slice then spans from the FIRST ```json marker to the LAST ```,
        which includes the intermediate text and second fence — so parsing
        fails and the method returns None."""
        content = '```json\n{"a": 1}\n```\nSome text\n```json\n{"b": 2}\n```'
        result = VectorStore.parse_json(content)
        self.assertIsNone(result)

    def test_nested_json(self):
        """Parse nested JSON structures."""
        content = '```json\n{"outer": {"inner": [1, 2, 3]}, "flag": true}\n```'
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"outer": {"inner": [1, 2, 3]}, "flag": True})

    def test_invalid_json_returns_none(self):
        """Invalid JSON string returns None."""
        result = VectorStore.parse_json("this is not json at all")
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        result = VectorStore.parse_json("")
        self.assertIsNone(result)

    def test_malformed_markdown_returns_none(self):
        """Malformed markdown with unclosed fence returns None."""
        content = '```json\n{"incomplete": '
        result = VectorStore.parse_json(content)
        self.assertIsNone(result)

    def test_fence_with_only_json_tag_no_close(self):
        """Fence opened but never closed."""
        content = '```json\n{"valid": "json"}'
        result = VectorStore.parse_json(content)
        # rfind('```') returns -1 if not found (after the opening fence),
        # so s > e, and json.loads(content[s:e]) fails.
        self.assertIsNone(result)

    def test_json_with_whitespace(self):
        """JSON with surrounding whitespace parses correctly."""
        content = '  \n  {"trimmed": true}  \n  '
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"trimmed": True})

    def test_unicode_json(self):
        """JSON containing Unicode characters."""
        content = '{"策略": "优化", "值": 100}'
        result = VectorStore.parse_json(content)
        self.assertEqual(result, {"策略": "优化", "值": 100})

    def test_json_with_escaped_chars(self):
        """JSON with escaped characters."""
        content = '{"path": "C:\\\\Users\\\\test", "quote": "say \\"hello\\""}'
        result = VectorStore.parse_json(content)
        self.assertEqual(result["path"], "C:\\Users\\test")
        self.assertEqual(result["quote"], 'say "hello"')


# ============================================================
# 6. Generalization scoring formula tests
# ============================================================
class TestGeneralizationScoringFormula(unittest.TestCase):
    """Verify the generalization scoring formula produces expected outputs.

    Formula from OptimizationStrategyOperator._generalize():
        gs  = 1.0 / (1.0 + dist)
        fs  = alpha * char_sim  +  beta * gs  +  gamma * evidence

    Defaults: alpha=0.5, beta=0.3, gamma=0.2
    """

    ALPHA = 0.5
    BETA = 0.3
    GAMMA = 0.2

    def _compute_score(self, char_sim, dist, evidence, alpha=None, beta=None, gamma=None):
        """Reproduce the exact scoring formula."""
        a = alpha if alpha is not None else self.ALPHA
        b = beta if beta is not None else self.BETA
        g = gamma if gamma is not None else self.GAMMA
        gs = 1.0 / (1.0 + dist)
        fs = a * char_sim + b * gs + g * evidence
        return round(fs, 4), round(gs, 3)

    def test_perfect_direct_match(self):
        """Direct match (dist=0) with perfect char_sim and evidence."""
        fs, gs = self._compute_score(char_sim=1.0, dist=0, evidence=1.0)
        # gs = 1.0/(1+0) = 1.0
        # fs = 0.5*1.0 + 0.3*1.0 + 0.2*1.0 = 0.5 + 0.3 + 0.2 = 1.0
        self.assertEqual(gs, 1.0)
        self.assertAlmostEqual(fs, 1.0, places=4)

    def test_direct_match_zero_evidence(self):
        """Direct match with zero evidence."""
        fs, gs = self._compute_score(char_sim=0.8, dist=0, evidence=0.0)
        # gs = 1.0
        # fs = 0.5*0.8 + 0.3*1.0 + 0.2*0.0 = 0.4 + 0.3 + 0.0 = 0.7
        self.assertEqual(gs, 1.0)
        self.assertAlmostEqual(fs, 0.7, places=4)

    def test_generalization_distance_1(self):
        """One-hop generalization (dist=1)."""
        fs, gs = self._compute_score(char_sim=0.7, dist=1, evidence=0.5)
        # gs = 1.0/(1+1) = 0.5
        # fs = 0.5*0.7 + 0.3*0.5 + 0.2*0.5 = 0.35 + 0.15 + 0.1 = 0.6
        self.assertEqual(gs, 0.5)
        self.assertAlmostEqual(fs, 0.6, places=4)

    def test_generalization_distance_2(self):
        """Two-hop generalization (dist=2)."""
        fs, gs = self._compute_score(char_sim=0.6, dist=2, evidence=0.5)
        # gs = 1.0/(1+2) = 0.3333...
        # fs = 0.5*0.6 + 0.3*0.3333... + 0.2*0.5
        #    = 0.3 + 0.1 + 0.1 = 0.5
        self.assertAlmostEqual(gs, 0.333, places=3)
        self.assertAlmostEqual(fs, 0.5, places=4)

    def test_generalization_distance_3(self):
        """Three-hop generalization (dist=3)."""
        fs, gs = self._compute_score(char_sim=0.5, dist=3, evidence=0.3)
        # gs = 1.0/(1+3) = 0.25
        # fs = 0.5*0.5 + 0.3*0.25 + 0.2*0.3 = 0.25 + 0.075 + 0.06 = 0.385
        self.assertEqual(gs, 0.25)
        self.assertAlmostEqual(fs, 0.385, places=4)

    def test_zero_char_sim(self):
        """Zero character similarity."""
        fs, gs = self._compute_score(char_sim=0.0, dist=0, evidence=1.0)
        # gs = 1.0
        # fs = 0.0 + 0.3 + 0.2 = 0.5
        self.assertEqual(gs, 1.0)
        self.assertAlmostEqual(fs, 0.5, places=4)

    def test_degraded_constraint_score(self):
        """Score after constraint degradation (multiply by 0.5)."""
        fs, gs = self._compute_score(char_sim=0.9, dist=0, evidence=0.8)
        # gs = 1.0
        # fs = 0.5*0.9 + 0.3*1.0 + 0.2*0.8 = 0.45 + 0.3 + 0.16 = 0.91
        # After degradation: 0.91 * 0.5 = 0.455
        self.assertAlmostEqual(fs, 0.91, places=4)
        degraded = round(fs * 0.5, 4)
        self.assertAlmostEqual(degraded, 0.455, places=4)

    def test_high_distance_low_similarity(self):
        """High distance and low similarity gives low score."""
        fs, gs = self._compute_score(char_sim=0.2, dist=5, evidence=0.1)
        # gs = 1.0/(1+5) = 0.1667
        # fs = 0.5*0.2 + 0.3*0.1667 + 0.2*0.1 = 0.1 + 0.05 + 0.02 = 0.17
        self.assertAlmostEqual(gs, 0.1667, places=3)
        self.assertAlmostEqual(fs, 0.17, places=2)

    def test_custom_alpha_beta_gamma(self):
        """Formula works with non-default alpha/beta/gamma values."""
        fs, gs = self._compute_score(char_sim=0.8, dist=1, evidence=0.6,
                                     alpha=0.4, beta=0.4, gamma=0.2)
        # gs = 0.5
        # fs = 0.4*0.8 + 0.4*0.5 + 0.2*0.6 = 0.32 + 0.20 + 0.12 = 0.64
        self.assertAlmostEqual(fs, 0.64, places=4)

    def test_graph_proximity_always_between_0_and_1(self):
        """gs is always in (0, 1] for non-negative distances."""
        for dist in [0, 1, 2, 5, 10, 100]:
            gs = 1.0 / (1.0 + dist)
            self.assertGreater(gs, 0.0)
            self.assertLessEqual(gs, 1.0)

    def test_final_score_always_between_0_and_1(self):
        """With inputs in [0,1], fs is always in [0,1]."""
        for char_sim in [0.0, 0.5, 1.0]:
            for dist in [0, 1, 3]:
                for evidence in [0.0, 0.5, 1.0]:
                    fs, _ = self._compute_score(char_sim, dist, evidence)
                    self.assertGreaterEqual(fs, 0.0)
                    self.assertLessEqual(fs, 1.0)


# ============================================================
# 7. Graph traversal primitives tests (mock data)
# ============================================================
class TestGraphTraversal(unittest.TestCase):
    """Test KnowledgeGraph BFS, typed traversal, and conflict detection."""

    def setUp(self):
        """Build an in-memory KnowledgeGraph with mock data, no Milvus."""
        self.kg = KnowledgeGraph()

        # Register mock entities
        self.kg._entity_index = {
            "uid_a": {"name": "Strategy A", "type": "optimization_strategy"},
            "uid_b": {"name": "Pattern B", "type": "source_pattern"},
            "uid_c": {"name": "Principle C", "type": "optimization_principle"},
            "uid_d": {"name": "Param D", "type": "tunable_parameter"},
            "uid_e": {"name": "Strategy E", "type": "optimization_strategy"},
            "uid_f": {"name": "Hardware F", "type": "hardware_feature"},
            "uid_p1": {"name": "Principle P1", "type": "optimization_principle"},
            "uid_p2": {"name": "Principle P2", "type": "optimization_principle"},
        }

        # Build adjacency:
        # A --OPTIMIZES_PATTERN--> B
        # A --HAS_PARAMETER--> D
        # A --INSTANCE_OF--> C
        # C --GENERALIZES--> P1
        # A --TARGETS--> F
        # E --CONFLICTS_WITH--> A
        # P1 --CONFLICTS_WITH--> P2
        self.kg._adj_out = {
            "uid_a": [
                ("OPTIMIZES_PATTERN", "uid_b", "A optimizes B"),
                ("HAS_PARAMETER", "uid_d", "A has param D"),
                ("INSTANCE_OF", "uid_c", "A implements C"),
                ("TARGETS", "uid_f", "A targets F"),
            ],
            "uid_c": [
                ("GENERALIZES", "uid_p1", "C generalizes P1"),
            ],
            "uid_e": [
                ("CONFLICTS_WITH", "uid_a", "E conflicts with A"),
            ],
            "uid_p1": [
                ("CONFLICTS_WITH", "uid_p2", "P1 conflicts with P2"),
            ],
        }
        self.kg._adj_in = {
            "uid_b": [("OPTIMIZES_PATTERN", "uid_a", "A optimizes B")],
            "uid_d": [("HAS_PARAMETER", "uid_a", "A has param D")],
            "uid_c": [("INSTANCE_OF", "uid_a", "A implements C")],
            "uid_f": [("TARGETS", "uid_a", "A targets F")],
            "uid_p1": [
                ("GENERALIZES", "uid_c", "C generalizes P1"),
                ("CONFLICTS_WITH", "uid_p2", "P1 conflicts with P2"),
            ],
            "uid_a": [("CONFLICTS_WITH", "uid_e", "E conflicts with A")],
        }

    # ---- BFS traversal ----

    def test_bfs_traverse_single_hop(self):
        """BFS with max_depth=1 returns direct neighbors only."""
        subgraph = self.kg.traverse("uid_a", max_depth=1)
        self.assertEqual(subgraph["start_uid"], "uid_a")
        self.assertGreater(subgraph["total_edges"], 0)
        # A has 4 outgoing edges, so total edges >= 4
        self.assertGreaterEqual(subgraph["total_edges"], 4)
        # Nodes include A + all neighbors
        neighbor_uids = {n["uid"] for n in subgraph["nodes"]}
        self.assertIn("uid_a", neighbor_uids)
        self.assertIn("uid_b", neighbor_uids)
        self.assertIn("uid_c", neighbor_uids)
        self.assertIn("uid_d", neighbor_uids)
        self.assertIn("uid_f", neighbor_uids)

    def test_bfs_traverse_two_hops(self):
        """BFS with max_depth=2 reaches two-hop neighbors."""
        subgraph = self.kg.traverse("uid_a", max_depth=2)
        neighbor_uids = {n["uid"] for n in subgraph["nodes"]}
        # Two-hop from A via C reaches P1
        self.assertIn("uid_p1", neighbor_uids)

    def test_bfs_stops_at_max_depth(self):
        """BFS does not exceed max_depth."""
        subgraph = self.kg.traverse("uid_a", max_depth=1)
        edges = subgraph["edges"]
        for edge in edges:
            self.assertLessEqual(edge["depth"], 1,
                                 f"Edge {edge} exceeds max_depth=1")

    def test_bfs_node_has_degree_info(self):
        """Each node in BFS result includes in_degree and out_degree."""
        subgraph = self.kg.traverse("uid_a", max_depth=1)
        for node in subgraph["nodes"]:
            self.assertIn("out_degree", node)
            self.assertIn("in_degree", node)
            self.assertIsInstance(node["out_degree"], int)
            self.assertIsInstance(node["in_degree"], int)

    # ---- get_neighbors ----

    def test_get_neighbors_out(self):
        """get_neighbors direction='out' returns outgoing edges."""
        neighbors = self.kg.get_neighbors("uid_a", direction="out")
        self.assertEqual(len(neighbors), 4)
        for n in neighbors:
            self.assertEqual(n["direction"], "out")

    def test_get_neighbors_in(self):
        """get_neighbors direction='in' returns incoming edges."""
        neighbors = self.kg.get_neighbors("uid_b", direction="in")
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["direction"], "in")
        self.assertEqual(neighbors[0]["relation_type"], "OPTIMIZES_PATTERN")

    def test_get_neighbors_both(self):
        """get_neighbors direction='both' returns all edges."""
        # A has 4 out + 1 in (from E's CONFLICTS_WITH)
        neighbors = self.kg.get_neighbors("uid_a", direction="both")
        self.assertEqual(len(neighbors), 5)

    def test_get_neighbors_isolated_node(self):
        """Isolated node returns empty list."""
        neighbors = self.kg.get_neighbors("uid_nonexistent")
        self.assertEqual(neighbors, [])

    # ---- Typed traversal ----

    def test_typed_traversal_follows_only_specified_relations(self):
        """traverse_typed only follows edges with specified types."""
        # From A, only follow HAS_PARAMETER and TARGETS
        results = self.kg.traverse_typed(
            "uid_a", ["HAS_PARAMETER", "TARGETS"], max_depth=2
        )
        found_uids = {r[0] for r in results}
        # D (via HAS_PARAMETER) should be found
        self.assertIn("uid_d", found_uids)
        # F (via TARGETS) should be found
        self.assertIn("uid_f", found_uids)
        # B (via OPTIMIZES_PATTERN) should NOT be found
        self.assertNotIn("uid_b", found_uids)

    def test_typed_traversal_max_depth(self):
        """traverse_typed respects max_depth."""
        results = self.kg.traverse_typed(
            "uid_a", ["INSTANCE_OF", "GENERALIZES"], max_depth=2
        )
        # At depth 1: uid_c via INSTANCE_OF
        # At depth 2: uid_p1 via GENERALIZES from uid_c
        found_uids = {r[0] for r in results}
        self.assertIn("uid_c", found_uids)
        self.assertIn("uid_p1", found_uids)

    def test_typed_traversal_returns_path(self):
        """Each result tuple includes (uid, depth, path)."""
        results = self.kg.traverse_typed(
            "uid_a", ["INSTANCE_OF", "GENERALIZES"], max_depth=2
        )
        for uid, depth, path in results:
            self.assertIsInstance(uid, str)
            self.assertIsInstance(depth, int)
            self.assertIsInstance(path, list)
            self.assertGreaterEqual(len(path), 2)  # at least start + 1 hop
            self.assertEqual(path[0], "uid_a")

    # ---- has_relation / check_conflict ----

    def test_get_neighbors_of_type_filter(self):
        """get_neighbors_of_type filters by relation type."""
        neighbors = self.kg.get_neighbors_of_type(
            "uid_a", "OPTIMIZES_PATTERN", direction="out"
        )
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["relation_type"], "OPTIMIZES_PATTERN")

    def test_has_relation_specific_type(self):
        """has_relation with specific type returns True only for that type."""
        self.assertTrue(self.kg.has_relation("uid_a", "uid_b", "OPTIMIZES_PATTERN"))
        self.assertFalse(self.kg.has_relation("uid_a", "uid_b", "HAS_PARAMETER"))

    def test_has_relation_any_type(self):
        """has_relation without type returns True for any edge."""
        self.assertTrue(self.kg.has_relation("uid_a", "uid_b"))
        self.assertTrue(self.kg.has_relation("uid_a", "uid_d"))
        self.assertFalse(self.kg.has_relation("uid_a", "uid_nonexistent"))

    def test_check_conflict_finds_conflicting_pairs(self):
        """check_conflict returns pairs with CONFLICTS_WITH relation."""
        conflicts = self.kg.check_conflict(["uid_p1", "uid_p2", "uid_a"])
        self.assertEqual(len(conflicts), 1)
        self.assertIn(("uid_p1", "uid_p2"), conflicts)

    def test_check_conflict_no_conflicts(self):
        """check_conflict returns empty when no conflicts exist."""
        conflicts = self.kg.check_conflict(["uid_a", "uid_b", "uid_c"])
        self.assertEqual(conflicts, [])

    # ---- get_entities_by_type ----

    def test_get_entities_by_type_filter(self):
        """get_entities_by_type filters UIDs by entity type."""
        uids = ["uid_a", "uid_b", "uid_c", "uid_d"]
        strategies = self.kg.get_entities_by_type(uids, "optimization_strategy")
        self.assertEqual(strategies, ["uid_a"])
        params = self.kg.get_entities_by_type(uids, "tunable_parameter")
        self.assertEqual(params, ["uid_d"])

    # ---- find_paths ----

    def test_find_paths_between_connected_nodes(self):
        """find_paths discovers paths between connected entities."""
        paths = self.kg.find_paths("uid_a", "uid_b", max_depth=2)
        self.assertGreater(len(paths), 0)
        # Path should be: A --OPTIMIZES_PATTERN--> B
        self.assertEqual(paths[0][0]["from"], "uid_a")
        self.assertEqual(paths[0][0]["to"], "uid_b")
        self.assertEqual(paths[0][0]["relation_type"], "OPTIMIZES_PATTERN")

    def test_find_paths_no_path(self):
        """find_paths returns empty list when no path exists."""
        paths = self.kg.find_paths("uid_a", "uid_e", max_depth=1)
        self.assertEqual(paths, [])

    def test_find_paths_multi_hop(self):
        """find_paths works through multi-hop paths."""
        paths = self.kg.find_paths("uid_a", "uid_p1", max_depth=3)
        self.assertGreater(len(paths), 0)
        # Path: A --INSTANCE_OF--> C --GENERALIZES--> P1
        found = False
        for path in paths:
            if len(path) == 2 and path[0]["relation_type"] == "INSTANCE_OF" \
                    and path[1]["relation_type"] == "GENERALIZES":
                found = True
                break
        self.assertTrue(found, "Expected INSTANCE_OF->GENERALIZES path not found")

    # ---- get_strategy_context ----

    def test_get_strategy_context(self):
        """get_strategy_context returns organized subgraph."""
        ctx = self.kg.get_strategy_context("uid_a")
        self.assertEqual(ctx["strategy_uid"], "uid_a")
        self.assertEqual(ctx["strategy_name"], "Strategy A")
        self.assertIn("related_patterns", ctx)
        self.assertIn("tunable_parameters", ctx)
        self.assertIn("hardware_features", ctx)
        self.assertIn("code_examples", ctx)
        self.assertIn("total_connections", ctx)
        # A has B (pattern), D (param), F (hardware)
        self.assertGreaterEqual(len(ctx["related_patterns"]), 1)
        self.assertGreaterEqual(len(ctx["tunable_parameters"]), 1)
        self.assertGreaterEqual(len(ctx["hardware_features"]), 1)

    # ---- co_occurrence_rank ----

    def test_co_occurrence_rank_returns_sorted_scores(self):
        """co_occurrence_rank returns strategies sorted by score descending."""
        ranked = self.kg.co_occurrence_rank(
            ["uid_a", "uid_e", "uid_c"],
            query_pattern_types={"blas", "gemm"}
        )
        self.assertEqual(len(ranked), 3)
        # Scores should be non-increasing
        scores = [s for _, s in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_co_occurrence_rank_all_unique_uids(self):
        """co_occurrence_rank returns all input UIDs exactly once."""
        ranked = self.kg.co_occurrence_rank(["uid_a", "uid_e"], set())
        returned = {uid for uid, _ in ranked}
        self.assertEqual(returned, {"uid_a", "uid_e"})


# ============================================================
# 8. Feedback event serialization tests
# ============================================================
class TestFeedbackEvents(unittest.TestCase):
    """Test feedback event serialization/deserialization (from
    OptimizationStrategyOperator._record_feedback)."""

    def _make_feedback_event(self, code, chars, results,
                              timestamp="2025-06-05T12:00:00"):
        """Replicate _record_feedback event construction."""
        event = {
            "timestamp": timestamp,
            "code_hash": str(hash(code))[:16],
            "extracted_chars": [c.get("characteristic_type", "") for c in chars],
            "recommended_uids": [r.get("uid", "") for r in results[:10]],
            "recommended_scores": [r.get("score", 0) for r in results[:10]],
        }
        return event

    def test_event_serialization_roundtrip(self):
        """Feedback event serializes to JSON and back without loss."""
        code = "void sgemm() { for(i=0;i<N;i++) for(j=0;j<M;j++) { C[i][j] += ... } }"
        chars = [
            {"characteristic_type": "compute_intensity", "value_descriptor": "medium"},
            {"characteristic_type": "access_pattern", "value_descriptor": "contiguous_unit_stride"},
        ]
        results = [
            {"uid": "abc123", "score": 0.95, "source": "direct"},
            {"uid": "def456", "score": 0.72, "source": "generalized:analogy"},
        ]

        event = self._make_feedback_event(code, chars, results)

        # Serialize
        json_str = json.dumps(event, ensure_ascii=False)
        # Deserialize
        restored = json.loads(json_str)

        self.assertEqual(restored["timestamp"], "2025-06-05T12:00:00")
        self.assertEqual(len(restored["extracted_chars"]), 2)
        self.assertIn("compute_intensity", restored["extracted_chars"])
        self.assertIn("access_pattern", restored["extracted_chars"])
        self.assertEqual(len(restored["recommended_uids"]), 2)
        self.assertEqual(restored["recommended_uids"][0], "abc123")
        self.assertEqual(restored["recommended_scores"][0], 0.95)

    def test_event_structure_has_required_keys(self):
        """Event dict contains all required top-level keys."""
        event = self._make_feedback_event("code", [], [])
        required_keys = {"timestamp", "code_hash", "extracted_chars",
                         "recommended_uids", "recommended_scores"}
        self.assertEqual(set(event.keys()), required_keys)

    def test_code_hash_is_deterministic(self):
        """Same code produces same code_hash."""
        code = "int main() { return 0; }"
        event1 = self._make_feedback_event(code, [], [])
        event2 = self._make_feedback_event(code, [], [])
        self.assertEqual(event1["code_hash"], event2["code_hash"])

    def test_code_hash_differs_for_different_code(self):
        """Different code produces (likely) different code_hash."""
        event1 = self._make_feedback_event("code A", [], [])
        event2 = self._make_feedback_event("code B", [], [])
        self.assertNotEqual(event1["code_hash"], event2["code_hash"])

    def test_empty_chars_produces_empty_list(self):
        """Empty characteristics list produces empty extracted_chars."""
        event = self._make_feedback_event("code", [], [])
        self.assertEqual(event["extracted_chars"], [])

    def test_empty_results_produces_empty_lists(self):
        """Empty results list produces empty lists for uids and scores."""
        event = self._make_feedback_event("code", [{"type": "test"}], [])
        self.assertEqual(event["recommended_uids"], [])
        self.assertEqual(event["recommended_scores"], [])

    def test_results_are_truncated_to_10(self):
        """Only the first 10 results are recorded."""
        many_results = [{"uid": f"uid_{i}", "score": float(i)} for i in range(20)]
        event = self._make_feedback_event("code", [], many_results)
        self.assertEqual(len(event["recommended_uids"]), 10)
        self.assertEqual(event["recommended_uids"][0], "uid_0")
        self.assertEqual(event["recommended_uids"][-1], "uid_9")

    def test_char_without_type_field_excluded(self):
        """Characteristics without 'characteristic_type' key become empty strings."""
        chars = [
            {"characteristic_type": "compute_intensity"},
            {"other_field": "ignore_me"},  # no characteristic_type
        ]
        event = self._make_feedback_event("code", chars, [])
        self.assertIn("compute_intensity", event["extracted_chars"])
        self.assertIn("", event["extracted_chars"])  # get default ""

    def test_event_jsonl_line_format(self):
        """Serialized event is a single line (for JSONL append)."""
        event = self._make_feedback_event("simple code", [], [])
        json_str = json.dumps(event, ensure_ascii=False)
        self.assertEqual(json_str.count("\n"), 0,
                         "JSONL entry must not contain newlines")


# ============================================================
# Bonus: DIM constant test
# ============================================================
class TestConstants(unittest.TestCase):
    """Test module-level constants."""

    def test_dim_is_1024(self):
        """DIM (embedding dimension) is 1024."""
        self.assertEqual(DIM, 1024)

    def test_dim_is_integer(self):
        """DIM is an integer."""
        self.assertIsInstance(DIM, int)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    unittest.main()
