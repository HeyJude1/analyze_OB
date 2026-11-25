"""kgagent 包入口：版本与根目录常量。"""

__version__ = "1.5.0"

SCHEMA_VERSION = "kg-v1.5"
EXTRACTOR_VERSION = "kgagent-v1.5"

#: 12 类实体（闭集，与 schema/entities.yaml 一致）
NODE_TYPES = [
    "Operator",
    "ComputationPattern",
    "CodeCharacteristic",
    "OptimizationStrategy",
    "ImplementationTemplate",
    "TunableParameter",
    "ISACapability",
    "Microarchitecture",
    "HardwareProfile",
    "PerformanceEvidence",
    "SourceArtifact",
    "FailureCase",
]

#: 14 类关系（闭集，与 schema/relations.yaml 一致）
RELATION_TYPES = [
    "OPERATOR_HAS_PATTERN",
    "PATTERN_HAS_CHARACTERISTIC",
    "PATTERN_OPTIMIZED_BY",
    "STRATEGY_HAS_PARAMETER",
    "STRATEGY_HAS_TEMPLATE",
    "STRATEGY_APPLIES_WHEN",
    "STRATEGY_REQUIRES_ISA",
    "STRATEGY_TARGETS_UARCH",
    "STRATEGY_EVIDENCED_BY",
    "EVIDENCE_ON",
    "EVIDENCE_SOURCE",
    "STRATEGY_COMPOSES_WITH",
    "STRATEGY_CONFLICTS_WITH",
    "STRATEGY_GENERALIZES",
]

__all__ = ["__version__", "SCHEMA_VERSION", "EXTRACTOR_VERSION", "NODE_TYPES", "RELATION_TYPES"]
