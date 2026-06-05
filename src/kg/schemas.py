"""
KG v2 实体与关系 Schema 定义
7 个实体类型, 13 种关系类型, 均在 Milvus 中以 collection 形式存储
"""

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import Dict, List


DIM = 1024

# ============================================================
# 关系类型常量
# ============================================================
class RelType:
    OPTIMIZES_PATTERN    = "OPTIMIZES_PATTERN"      # SourcePattern → OptimizationStrategy
    HAS_PARAMETER        = "HAS_PARAMETER"          # Strategy → TunableParameter
    IS_ILLUSTRATED_BY    = "IS_ILLUSTRATED_BY"      # Strategy → CodeExample
    TARGETS              = "TARGETS"                # Strategy → ArchitectureCapability
    APPLIES_WHEN         = "APPLIES_WHEN"           # Principle → Characteristic (约束 JSON 在 description)
    REQUIRES             = "REQUIRES"               # Principle → ArchitectureCapability
    GENERALIZES          = "GENERALIZES"            # Principle → Principle (子→父)
    COMPOSES_WITH        = "COMPOSES_WITH"          # Principle ↔ Principle (双向)
    INSTANCE_OF          = "INSTANCE_OF"            # SourcePattern → Principle
    CONFLICTS_WITH       = "CONFLICTS_WITH"         # Principle ↔ Principle (双向)
    ANALOGOUS_TO         = "ANALOGOUS_TO"           # Principle ↔ Principle (双向)
    HAS_CHARACTERISTIC   = "HAS_CHARACTERISTIC"     # SourcePattern → Characteristic
    FEEDBACK_SUPPORTS    = "FEEDBACK_SUPPORTS"      # RetrievalOutcome → Principle

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, str)]


# ============================================================
# 实体 Schema
# ============================================================

_uid = lambda max_len=100: FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=max_len, is_primary=True)
_embed = lambda: FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM)
_edata = lambda: FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535)
_name = lambda max_len=500: FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=max_len)
_desc = lambda max_len=5000: FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=max_len)


def optimization_principle_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="principle", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="scope", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="constraints", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="evidence_strength", dtype=DataType.FLOAT),
        FieldSchema(name="source_count", dtype=DataType.INT64),
        _edata(), _embed(),
    ], "optimization_principle")


def code_characteristic_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="characteristic_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="value_descriptor", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="metric_range", dtype=DataType.VARCHAR, max_length=500),
        _desc(), _edata(), _embed(),
    ], "code_characteristic")


def source_pattern_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="code_snippet", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="pattern_type", dtype=DataType.VARCHAR, max_length=100),
        _desc(5000),
        FieldSchema(name="data_object_features", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="source_algorithm", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="extracted_characteristics", dtype=DataType.VARCHAR, max_length=2000),
        _edata(), _embed(),
    ], "source_pattern")


def architecture_capability_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="architecture", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="capability_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="params", dtype=DataType.VARCHAR, max_length=500),
        _desc(), _edata(), _embed(),
    ], "architecture_capability")


def optimization_strategy_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="rationale", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="implementation", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="impact", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="trade_offs", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="related_patterns", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="principle_links", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="applicability_conditions", dtype=DataType.VARCHAR, max_length=5000),
        _edata(), _embed(),
    ], "optimization_strategy")


def tunable_parameter_schema():
    return CollectionSchema([
        _uid(), _name(),
        _desc(), FieldSchema(name="impact", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="value_in_code", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="typical_range", dtype=DataType.VARCHAR, max_length=500),
        _edata(), _embed(),
    ], "tunable_parameter")


def code_example_schema():
    return CollectionSchema([
        _uid(), _name(),
        FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="explanation", dtype=DataType.VARCHAR, max_length=5000),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
        _edata(), _embed(),
    ], "code_example")


def relation_schema():
    return CollectionSchema([
        FieldSchema(name="relation_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="relation_type", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="head_entity_uid", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="tail_entity_uid", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="head_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="tail_name", dtype=DataType.VARCHAR, max_length=500),
        _desc(), _embed(),
    ], "relation")


# ============================================================
# Collection 创建
# ============================================================

ENTITY_SCHEMAS = {
    "optimization_principle": optimization_principle_schema,
    "code_characteristic": code_characteristic_schema,
    "source_pattern": source_pattern_schema,
    "architecture_capability": architecture_capability_schema,
    "optimization_strategy": optimization_strategy_schema,
    "tunable_parameter": tunable_parameter_schema,
    "code_example": code_example_schema,
    "relation": relation_schema,
}


def create_all_collections(drop_existing: bool = False):
    """创建所有 KG v2 的 Milvus collection"""
    for name, schema_fn in ENTITY_SCHEMAS.items():
        if utility.has_collection(name):
            if drop_existing:
                utility.drop_collection(name)
            else:
                continue
        Collection(name, schema_fn())
        print(f"  ✓ {name}")


def get_collection_names() -> List[str]:
    return list(ENTITY_SCHEMAS.keys())
