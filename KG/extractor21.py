#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS知识图谱实体抽取器 - (V13 - Embedding生成逻辑最终修复)
- "analysis_results_dir" 现在是基准目录。
- JSON文件从基准目录下的 "analysis_results" 子目录读取。
- 输出文件 (relations, checkpoints) 直接保存在基准目录下。
- 每个提取的实体都被视为全新实体，UID根据其完整数据生成。
- --fresh 参数用于强制从头开始处理。
- 修正了所有已知的bug。
- 新增：为关系实体自动生成描述。
- 新增：为硬件特征实体填充架构信息。
- 修正：确保在生成embedding时，entity_data中不包含uid。
- 新增：丰富optimization_strategy和computational_pattern的entity_data字段。
"""

import os
import json
import hashlib
from typing import Dict, List, Any
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_community.embeddings import DashScopeEmbeddings
import argparse
from dotenv import load_dotenv

load_dotenv()


class KnowledgeGraphExtractor:
    """知识图谱实体抽取器"""
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = config
        self.milvus_config = self.config.get("milvus", {})
        self.embedding_config = self.config.get("dashscope_embeddings", {})
        self.data_source_config = self.config.get("data_source", {})
        
        self.embedding_model_name = self.embedding_config.get("name", "text-embedding-v3")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is required")
        
        self.embedding_model = DashScopeEmbeddings(
            model=self.embedding_model_name, 
            dashscope_api_key=api_key
        )
        
        self._connect_milvus()
        self._create_collections()
        
        self.checkpoint_file = checkpoint_path
        self.processed_files = self._load_checkpoint()
        
        self.all_relations_for_txt = []
        self.all_relations_for_json = []

        self.code_counter = 1

        print("✅ 知识图谱抽取器初始化完成")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "dashscope_embeddings": {"name": "text-embedding-v3", "dimension": 1024}
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _connect_milvus(self):
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        database = self.milvus_config.get("database", "code_op")
        
        connections.connect(alias="default", host=host, port=port, db_name=database)
        print(f"✅ 已连接到Milvus: {host}:{port}/{database}")
    
    def _create_collections(self):
        dimension = self.embedding_config.get("dimension", 1024)
        collections_schema = {
            "optimization_strategy": [
                FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="level", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="rationale", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="implementation", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="impact", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="trade_offs", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            "computational_pattern": [
                FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="code", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="numeric_kind", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="numeric_precision", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="structural_properties", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="storage_layout", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            "hardware_feature": [
                FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="architecture", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            "tunable_parameter": [
                FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="impact", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="value_in_code", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="typical_range", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            "code_example": [
                FieldSchema(name="uid", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="snippet", dtype=DataType.VARCHAR, max_length=10000),
                FieldSchema(name="explanation", dtype=DataType.VARCHAR, max_length=5000),
                FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="entity_data", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            "relation": [
                FieldSchema(name="relation_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
                FieldSchema(name="relation_type", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="head_entity_uid", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="tail_entity_uid", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="head_name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="tail_name", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ]
        }
        for collection_name, fields in collections_schema.items():
            if not utility.has_collection(collection_name):
                Collection(collection_name, CollectionSchema(fields, f"{collection_name} collection"))

    def _build_index_for_collection(self, collection_name: str):
        try:
            collection = Collection(collection_name)
            collection.flush()
            num_entities = collection.num_entities
            if num_entities == 0: return
            if not collection.has_index():
                if num_entities < 1000:
                    index_params = {"index_type": "FLAT", "metric_type": "L2"}
                else:
                    nlist = max(128, min(1024, int((num_entities ** 0.5) * 2)))
                    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": nlist}}
                collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
        except Exception as e:
            print(f"⚠️ 处理集合 {collection_name} 时出错: {e}")

    def _build_indexes_for_all_collections(self):
        collection_names = ["optimization_strategy", "computational_pattern", "hardware_feature", 
                            "tunable_parameter", "code_example", "relation"]
        for name in collection_names: self._build_index_for_collection(name)

    def _load_checkpoint(self) -> set:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return set(json.load(f).get("processed_files", []))
        return set()

    def _save_checkpoint(self):
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({"processed_files": list(self.processed_files)}, f, ensure_ascii=False, indent=2)

    def _generate_uid_from_dict(self, data: Dict[str, Any]) -> str:
        dhash = hashlib.md5()
        encoded = json.dumps(data, sort_keys=True).encode('utf-8')
        dhash.update(encoded)
        return dhash.hexdigest()

    def _get_embedding(self, text: str) -> List[float]:
        try:
            return self.embedding_model.embed_query(text)
        except Exception as e:
            print(f"⚠️ 向量化失败: {e}")
            return [0.0] * self.embedding_config.get("dimension", 1024)

    def _get_embedding_text(self, entity_data_without_uid: Dict[str, Any]) -> str:
        """从不含UID的entity_data字典生成用于embedding的文本"""
        return json.dumps(entity_data_without_uid, ensure_ascii=False, sort_keys=True)

    # <<< MODIFIED: Corrected the logic to remove uid BEFORE embedding
    def _save_entity(self, collection_name: str, entity_data: Dict[str, Any]) -> str:
        # 1. 提取UID
        uid = entity_data["uid"]
        
        # 2. 创建一个干净的副本用于embedding和存储
        data_for_processing = entity_data.copy()
        data_for_processing.pop("uid", None)
        
        # 3. 基于干净的数据生成embedding
        embedding_text = self._get_embedding_text(data_for_processing)
        embedding = self._get_embedding(embedding_text)
        
        # 4. 准备插入数据
        schema = Collection(collection_name).schema
        field_names = [field.name for field in schema.fields]
        
        insert_data = []
        for name in field_names:
            if name == "uid":
                insert_data.append([uid]) # 使用原始UID
            elif name == "embedding":
                insert_data.append([embedding])
            elif name == "entity_data":
                # 存储不含uid的entity_data
                insert_data.append([json.dumps(data_for_processing, ensure_ascii=False)])
            else:
                # 从原始的entity_data中获取其他字段值
                if name in ["numeric_kind", "numeric_precision", "structural_properties", "storage_layout"]:
                    value = entity_data.get("data_object_features", {}).get(name, "")
                else:
                    value = entity_data.get(name, "")
                insert_data.append([value])

        Collection(collection_name).insert(insert_data)
        print(f"    ✓ 保存新实体 {collection_name}: {entity_data.get('name', '')} -> {uid[:8]}...")
        return uid
    
    def _save_relation(self, head_uid: str, tail_uid: str, relation_type: str,
                       head_name: str, tail_name: str):
        description = ""
        if relation_type == "OPTIMIZES_PATTERN":
            description = f"{head_name}可使用{tail_name}优化"
        elif relation_type == "HAS_PARAMETER":
            description = "该优化策略包含此可调参数"
        elif relation_type == "IS_ILLUSTRATED_BY":
            description = "该代码示例展示了此优化策略"
        elif relation_type == "TARGETS":
            description = "该优化策略针对此硬件特性"
            
        relation_content = {"type": relation_type, "head": head_uid, "tail": tail_uid, "desc": description}
        relation_uid = self._generate_uid_from_dict(relation_content)

        embedding_text = f"{relation_type} from {head_name} to {tail_name}: {description}"
        embedding = self._get_embedding(embedding_text)
        
        Collection("relation").insert([
            [relation_uid], [relation_type], [head_uid], [tail_uid],
            [head_name], [tail_name], [description], [embedding]
        ])
        print(f"    ✓ 保存新关系: {relation_type} ({head_name} -> {tail_name})")

        self.all_relations_for_txt.append((head_name, relation_type, tail_name))
        self.all_relations_for_json.append({
            "relation_type": relation_type,
            "relation_id": relation_uid,
            "head": {"name": head_name, "uid": head_uid},
            "tail": {"name": tail_name, "uid": tail_uid}
        })

    def extract_from_file(self, file_path: str):
        if file_path in self.processed_files:
            print(f"⏭️ 跳过已处理文件: {file_path}")
            return
        
        print(f"📄 处理: {os.path.basename(file_path)}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_algorithm = data.get("algorithm", "unknown")

        for analysis in data.get("individual_analyses", []):
            operator_name = analysis.get("file_path", "").split("/")[-1]
            architecture = analysis.get("architecture", "通用")
            print(f"  🔍 算子: {operator_name} (算法: {source_algorithm}, 架构: {architecture})")
            
            entity_count = 0
            relation_count = 0
            
            pattern_map = {}
            for pattern in analysis.get("computational_patterns", []):
                pattern_entity = {
                    "name": pattern.get("name", ""),
                    "type": pattern.get("pattern_type", "") or pattern.get("type", ""),
                    "description": pattern.get("description", ""),
                    "code": pattern.get("code", ""),
                    "data_object_features": pattern.get("data_object_features", {}),
                    "source_algorithm": source_algorithm,
                    "source_file": operator_name,
                    "architecture": architecture
                }
                uid = self._generate_uid_from_dict(pattern_entity)
                pattern_entity["uid"] = uid
                self._save_entity("computational_pattern", pattern_entity)
                entity_count += 1
                if pattern_entity["type"]:
                    pattern_map[pattern_entity["type"]] = {"uid": uid, "name": pattern_entity["name"]}

            for level in ["algorithm_level_optimizations", "code_level_optimizations", "instruction_level_optimizations"]:
                for opt in analysis.get(level, []):
                    desc = opt.get('description', {})
                    strategy_entity = {
                        "name": opt.get("optimization_name", ""),
                        "level": opt.get("level", ""),
                        "rationale": desc.get("strategy_rationale", ""),
                        "implementation": desc.get("implementation_pattern", ""),
                        "impact": desc.get("performance_impact", ""),
                        "trade_offs": desc.get("trade_offs", ""),
                        "related_patterns": opt.get("related_patterns", []),
                        "source_algorithm": source_algorithm,
                        "source_file": operator_name,
                        "architecture": architecture
                    }
                    strategy_uid = self._generate_uid_from_dict(strategy_entity)
                    strategy_entity["uid"] = strategy_uid
                    self._save_entity("optimization_strategy", strategy_entity)
                    entity_count += 1
                    
                    for pattern_type in opt.get("related_patterns", []):
                        if pattern_type in pattern_map:
                            head_info = pattern_map[pattern_type]
                            self._save_relation(head_info["uid"], strategy_uid, "OPTIMIZES_PATTERN",
                                              head_name=head_info["name"], tail_name=strategy_entity["name"])
                            relation_count += 1
                    
                    hw_name = opt.get("target_hardware_feature_name")
                    if hw_name:
                        hw_entity = {
                            "name": hw_name, 
                            "architecture": architecture, 
                            "description": opt.get("target_hardware_feature", ""),
                            "source_algorithm": source_algorithm,
                            "source_file": operator_name
                        }
                        hw_uid = self._generate_uid_from_dict(hw_entity)
                        hw_entity["uid"] = hw_uid
                        self._save_entity("hardware_feature", hw_entity)
                        entity_count += 1
                        self._save_relation(strategy_uid, hw_uid, "TARGETS",
                                          head_name=strategy_entity["name"], tail_name=hw_name)
                        relation_count += 1
                    
                    code_examples = []
                    if 'code_example' in opt and isinstance(opt['code_example'], dict) and opt['code_example']:
                        code_examples.append(opt['code_example'])
                    elif 'code_examples' in opt and isinstance(opt['code_examples'], list) and opt['code_examples']:
                        code_examples.extend(opt['code_examples'])

                    for code_obj in code_examples:
                        code_entity = {
                            "name": f"code{self.code_counter}",
                            "snippet": code_obj.get("snippet", "") if isinstance(code_obj, dict) else str(code_obj),
                            "explanation": code_obj.get("explanation", "") if isinstance(code_obj, dict) else "",
                            "source_file": operator_name,
                            "source_algorithm": source_algorithm,
                            "architecture": architecture
                        }
                        code_uid = self._generate_uid_from_dict(code_entity)
                        code_entity["uid"] = code_uid
                        self._save_entity("code_example", code_entity)
                        entity_count += 1
                        self._save_relation(strategy_uid, code_uid, "IS_ILLUSTRATED_BY",
                                          head_name=strategy_entity["name"], tail_name=code_entity["name"])
                        relation_count += 1
                        self.code_counter += 1
                    
                    for param in opt.get("tunable_parameters", []):
                        param_name = param.get("parameter_name") if isinstance(param, dict) else str(param)
                        if not param_name: continue
                        
                        typical_range = param.get("typical_range", []) if isinstance(param, dict) else []
                        param_entity = {
                            "name": param_name,
                            "description": param.get("description", "") if isinstance(param, dict) else f"Tunable parameter: {param_name}",
                            "impact": param.get("impact", "") if isinstance(param, dict) else "",
                            "value_in_code": str(param.get("value_in_code", "")) if isinstance(param, dict) else "",
                            "typical_range": ",".join(map(str, typical_range)),
                            "source_algorithm": source_algorithm,
                            "source_file": operator_name,
                            "architecture": architecture
                        }
                        param_uid = self._generate_uid_from_dict(param_entity)
                        param_entity["uid"] = param_uid
                        self._save_entity("tunable_parameter", param_entity)
                        entity_count += 1
                        self._save_relation(strategy_uid, param_uid, "HAS_PARAMETER",
                                          head_name=strategy_entity["name"], tail_name=param_entity["name"])
                        relation_count += 1
            
            print(f"  📊 完成: 新增实体={entity_count}, 新增关系={relation_count}")
        
        self.processed_files.add(file_path)
        self._save_checkpoint()
        print("💾 断点已保存")

    def _write_relation_txt(self, output_directory: str):
        output_path = os.path.join(output_directory, "relation.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            for head, rel_type, tail in self.all_relations_for_txt:
                f.write(f"{head}\t{rel_type}\t{tail}\n")
        print(f"✅ 关系文本文件已保存到: {output_path}")

    def _write_relation_entity_json(self, output_directory: str):
        output_path = os.path.join(output_directory, "relation_entity.json")
        grouped_relations = {}
        for relation in self.all_relations_for_json:
            rel_type = relation["relation_type"]
            if rel_type not in grouped_relations:
                grouped_relations[rel_type] = []
            grouped_relations[rel_type].append(relation)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(grouped_relations, f, ensure_ascii=False, indent=2)
        print(f"✅ 关系JSON文件已保存到: {output_path}")

    def extract_from_directory(self, json_input_dir: str, base_output_dir: str):
        json_files = sorted(list(Path(json_input_dir).glob("*.json")))
        print(f"📁 发现 {len(json_files)} 个JSON文件")
        
        for i, file_path in enumerate(json_files, 1):
            print(f"\n{'='*60}\n进度: {i}/{len(json_files)}\n{'='*60}")
            self.extract_from_file(str(file_path))
        
        print("\n🔧 数据插入完成，正在刷新和索引集合...")
        self._build_indexes_for_all_collections()
        
        print("\n💾 正在写入关系文件...")
        self._write_relation_txt(base_output_dir)
        self._write_relation_entity_json(base_output_dir)
        
        print(f"\n{'='*60}\n📊 最终统计:")
        total_entities, total_relations = 0, 0
        all_collections = utility.list_collections()
        for name in all_collections:
            try:
                count = Collection(name).num_entities
                if name != "relation": total_entities += count
                else: total_relations = count
                print(f"  ✅ {name}: {count} 个")
            except Exception as e:
                print(f"  ⚠️ {name}: 统计失败 - {e}")
        
        print(f"\n📊 总计: 实体={total_entities}, 关系={total_relations}")
        print(f"{'='*60}\n🎉 完成！")


def main():
    parser = argparse.ArgumentParser(description="OpenBLAS知识图谱实体抽取器")
    parser.add_argument("--config", type=str, default="kg_config.json", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, help="分析结果的基准目录")
    parser.add_argument("--fresh", action="store_true", help="忽略断点文件，从头开始处理所有文件")
    args = parser.parse_args()
    
    config = KnowledgeGraphExtractor._load_config(args.config)
    
    base_dir = args.data_dir or config.get("data_source", {}).get("analysis_results_dir")
    if not base_dir:
        print("❌ 错误：未在配置或命令行中指定基准目录 (analysis_results_dir)")
        return

    if not os.path.isabs(base_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        
        resolved_path = os.path.join(project_root, base_dir)
        
        if not os.path.exists(resolved_path):
            project_folder_name = os.path.basename(project_root)
            if project_folder_name in base_dir:
                try:
                    idx = base_dir.index(project_folder_name)
                    suffix = base_dir[idx:]
                    root_parent = os.path.dirname(project_root)
                    resolved_path = os.path.join(root_parent, suffix)
                except ValueError:
                    pass
        base_dir = os.path.abspath(resolved_path)

    json_input_dir = os.path.join(base_dir, "analysis_results")

    if not os.path.exists(json_input_dir):
        print(f"❌ 错误：JSON输入目录不存在: {json_input_dir}")
        return
    
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_file_path = os.path.join(checkpoints_dir, "extraction_checkpoint.json")

    if args.fresh and os.path.exists(checkpoint_file_path):
        os.remove(checkpoint_file_path)
        print(f"🗑️ 已删除旧的断点文件 '{checkpoint_file_path}'，将从头开始处理。")
    
    extractor = KnowledgeGraphExtractor(config=config, checkpoint_path=checkpoint_file_path)
    
    print(f"📁 基准目录: {base_dir}")
    print(f"📂 JSON输入目录: {json_input_dir}")
    extractor.extract_from_directory(json_input_dir=json_input_dir, base_output_dir=base_dir)


if __name__ == "__main__":
    main()