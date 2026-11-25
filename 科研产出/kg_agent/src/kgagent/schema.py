"""Schema 装载 + JSON Schema 子集校验（标准库实现，不依赖 jsonschema）。

schema/ 目录是唯一闭集来源：
  entities.yaml / relations.yaml / condition_vocab.yaml / hardware_profiles.yaml
  node.schema.json / edge.schema.json   （draft-07 子集，本模块实现其校验器）

之所以自己写校验器：任务硬约束不允许引入 jsonschema。支持的关键字是 node/edge schema 实际用到的
（type/required/properties/additionalProperties/items/enum/const/pattern/minLength/maxLength/
minimum/maximum/minItems/maxItems/oneOf/anyOf/allOf）。遇到 `$ref` 直接报错，避免静默漏校验。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import yamlmini

__all__ = [
    "SchemaBundle",
    "load_schema",
    "validate_json_schema",
    "type_of",
    "PREDICATE_INFO",
]

#: 条件谓词扩展信息（从 condition_vocab.yaml 载入后会被覆盖）
PREDICATE_INFO: dict = {}

_TYPE_MAP = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def type_of(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _matches_type(value, expected: str) -> bool:
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    py = _TYPE_MAP.get(expected)
    if py is None:
        raise ValueError("未知 JSON Schema type: %r" % expected)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "null":
        return value is None
    return isinstance(value, py) and not isinstance(value, bool)


def validate_json_schema(instance, schema: dict, path: str = "$") -> list:
    """校验 instance 是否满足 schema 子集；返回错误字符串列表（空列表=通过）。"""
    errors: list = []
    if not isinstance(schema, dict):
        return ["%s: schema 必须是 object" % path]
    if "$ref" in schema:
        errors.append("%s: 不支持 $ref（请把 schema 内联）" % path)
        return errors

    if "type" in schema:
        expected = schema["type"]
        types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(instance, t) for t in types):
            errors.append("%s: 类型应为 %s，实际 %s" % (path, types, type_of(instance)))
            return errors

    if "const" in schema and instance != schema["const"]:
        errors.append("%s: 期望常量 %r" % (path, schema["const"]))
    if "enum" in schema and instance not in schema["enum"]:
        errors.append("%s: 值 %r 不在枚举 %r 中" % (path, instance, schema["enum"]))

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            errors.append("%s: 字符串长度 < %s" % (path, schema["minLength"]))
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            errors.append("%s: 字符串长度 > %s" % (path, schema["maxLength"]))
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append("%s: 不匹配 pattern %r" % (path, schema["pattern"]))

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append("%s: %r < minimum %r" % (path, instance, schema["minimum"]))
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append("%s: %r > maximum %r" % (path, instance, schema["maximum"]))

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append("%s: 元素数 < %s" % (path, schema["minItems"]))
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append("%s: 元素数 > %s" % (path, schema["maxItems"]))
        if "items" in schema:
            for idx, item in enumerate(instance):
                errors.extend(validate_json_schema(item, schema["items"], "%s[%d]" % (path, idx)))

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append("%s: 缺少必需字段 %r" % (path, key))
        props = schema.get("properties", {}) or {}
        for key, value in instance.items():
            if key in props:
                errors.extend(validate_json_schema(value, props[key], "%s.%s" % (path, key)))
            elif schema.get("additionalProperties") is False:
                errors.append("%s: 不允许的字段 %r" % (path, key))

    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword not in schema:
            continue
        sub = schema[keyword]
        if keyword == "allOf":
            for idx, s in enumerate(sub):
                errors.extend(validate_json_schema(instance, s, "%s.%s[%d]" % (path, keyword, idx)))
        else:
            outcomes = [not validate_json_schema(instance, s, path) for s in sub]
            passed = sum(outcomes)
            if keyword == "anyOf" and passed == 0:
                errors.append("%s: anyOf 无一满足" % path)
            if keyword == "oneOf" and passed != 1:
                errors.append("%s: oneOf 需恰好 1 个满足，实际 %d" % (path, passed))
    return errors


class SchemaBundle:
    """schema/ 目录的只读视图。"""

    def __init__(self, schema_dir: Path):
        self.dir = Path(schema_dir)
        self.entities = yamlmini.load(self.dir / "entities.yaml")["entities"]
        self.relations = yamlmini.load(self.dir / "relations.yaml")["relations"]
        self.condition_vocab = yamlmini.load(self.dir / "condition_vocab.yaml")
        self.hardware = yamlmini.load(self.dir / "hardware_profiles.yaml")
        self.vocab = yamlmini.load(self.dir / "vocab.yaml")
        self.node_schema = json.loads((self.dir / "node.schema.json").read_text(encoding="utf-8"))
        self.edge_schema = json.loads((self.dir / "edge.schema.json").read_text(encoding="utf-8"))
        global PREDICATE_INFO
        PREDICATE_INFO = self.condition_vocab.get("predicates", {}) or {}

    # ---- 闭集 ----
    @property
    def node_types(self) -> list:
        return sorted(self.entities.keys())

    @property
    def relation_types(self) -> list:
        return sorted(self.relations.keys())

    @property
    def predicates(self) -> dict:
        return self.condition_vocab.get("predicates", {}) or {}

    @property
    def char_type_map(self) -> dict:
        return self.condition_vocab.get("char_type_map", {}) or {}

    def prefix(self, node_type: str) -> str:
        return self.entities[node_type]["prefix"]

    def relation_spec(self, rel: str) -> dict:
        return self.relations[rel]

    def hardware_profiles(self) -> dict:
        return self.hardware.get("profiles", {}) or {}

    # ---- 受控词表（vocab.yaml） ----
    def vocab_list(self, name: str) -> list:
        return list(self.vocab.get(name, []) or [])

    @property
    def families(self) -> list:
        return self.vocab_list("strategy_family")

    @property
    def actions(self) -> list:
        return self.vocab_list("strategy_action")

    @property
    def levels(self) -> list:
        return self.vocab_list("strategy_level")

    @property
    def stages(self) -> list:
        return self.vocab_list("strategy_stage")

    @property
    def statuses(self) -> list:
        return self.vocab_list("strategy_status")

    @property
    def pattern_categories(self) -> list:
        return self.vocab_list("pattern_category")

    @property
    def char_types(self) -> list:
        return self.vocab_list("char_type")

    # ---- 单条对象校验 ----
    def check_node(self, node: dict) -> list:
        return validate_json_schema(node, self.node_schema)

    def check_edge(self, edge: dict) -> list:
        return validate_json_schema(edge, self.edge_schema)


_BUNDLE_CACHE: dict = {}


def load_schema(schema_dir) -> SchemaBundle:
    """按绝对路径缓存（build 内多处复用）。"""
    key = str(Path(schema_dir).resolve())
    bundle = _BUNDLE_CACHE.get(key)
    if bundle is None:
        bundle = SchemaBundle(Path(schema_dir))
        _BUNDLE_CACHE[key] = bundle
    return bundle
