#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS知识图谱实体抽取器v1
支持断点续传的实体和关系抽取
"""

import os
import json
import uuid
import hashlib
from typing import Dict, List, Any, Optional
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from pymilvus import Index, IndexType, MetricType
from langchain_community.embeddings import DashScopeEmbeddings
import argparse
from dotenv import load_dotenv

load_dotenv()


class KnowledgeGraphExtractor:
    """知识图谱实体抽取器"""
    
    def __init__(self, config_path: str = "kg_config.json"):
        """初始化抽取器"""
        self.config = self._load_config(config_path)
        self.milvus_config = self.config.get("milvus", {})
        self.embedding_config = self.config.get("dashscope_embeddings", {})
        self.data_source_config = self.config.get("data_source", {})
        
        # 初始化向量化模型（LangChain DashScopeEmbeddings）
        self.embedding_model_name = self.embedding_config.get("name", "text-embedding-v3")
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("❌ 错误：未提供 DashScope API Key。请设置环境变量 DASHSCOPE_API_KEY 或在 kg_config.json 中添加 dashscope_api_key 字段。")
            raise RuntimeError("DashScope API key missing")
        try:
            self.embedding_model = DashScopeEmbeddings(model=self.embedding_model_name, dashscope_api_key=api_key)
        except Exception as e:
            print(f"❌ 无法初始化 DashScopeEmbeddings: {e}")
            raise
        
        # 连接Milvus
        self._connect_milvus()
        
        # 创建集合
        self._create_collections()
        
        # 断点续传状态
        self.checkpoint_file = "extraction_checkpoint.json"
        self.processed_files = self._load_checkpoint()
        
        print("✅ 知识图谱抽取器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "dashscope_embeddings": {"name": "text-embedding-v3", "dimension": 1024}
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _connect_milvus(self):
        """连接Milvus数据库"""
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        database = self.milvus_config.get("database", "code_op")
        
        connections.connect(
            alias="default",
            host=host,
            port=port,
            db_name=database
        )
        
        print(f"✅ 已连接到Milvus: {host}:{port}/{database}")
    
    def _create_collections(self):
        """创建Milvus集合"""
        dimension = self.embedding_config.get("dimension", 1024)
        
        # 定义集合schema
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
        
        # 创建集合
        for collection_name, fields in collections_schema.items():
            if not utility.has_collection(collection_name):
                schema = CollectionSchema(fields, f"{collection_name} collection")
                collection = Collection(collection_name, schema)
                print(f"✅ 创建集合: {collection_name}")
            else:
                print(f"✅ 集合已存在: {collection_name}")
        
        # 为所有集合创建索引并加载
        self._create_indexes_and_load()
    
    def _create_indexes_and_load(self):
        """为所有集合的向量列创建索引并加载集合到内存"""
        collection_names = ["optimization_strategy", "computational_pattern", "hardware_feature", 
                           "tunable_parameter", "code_example", "relation"]
        
        for collection_name in collection_names:
            if not utility.has_collection(collection_name):
                continue
            
            try:
                collection = Collection(collection_name)
                
                # 检查是否已有索引
                indexes = collection.indexes
                has_embedding_index = False
                for index in indexes:
                    if index.field_name == "embedding":
                        has_embedding_index = True
                        break
                
                # 如果没有索引，创建索引
                if not has_embedding_index:
                    # 先flush确保数据已写入
                    collection.flush()
                    
                    # 根据集合大小选择索引类型
                    num_entities = collection.num_entities
                    if num_entities > 0:
                        if num_entities < 1000:
                            # 小数据集使用FLAT索引
                            index_params = {
                                "index_type": IndexType.FLAT,
                                "metric_type": MetricType.L2
                            }
                        else:
                            # 大数据集使用IVF_FLAT索引
                            index_params = {
                                "index_type": IndexType.IVF_FLAT,
                                "metric_type": MetricType.L2,
                                "params": {"nlist": min(1024, num_entities // 10)}
                            }
                        
                        collection.create_index("embedding", index_params)
                        print(f"✅ 为 {collection_name} 创建向量索引 (实体数: {num_entities})")
                
                # 加载集合到内存（只有在有数据时才加载）
                if num_entities > 0:
                    try:
                        collection.load()
                        print(f"✅ 加载集合到内存: {collection_name}")
                    except Exception as e:
                        # 如果加载失败，可能是因为没有索引，尝试创建默认索引后再加载
                        if "index" in str(e).lower() or "Index" in str(e):
                            try:
                                index_params = {
                                    "index_type": IndexType.FLAT,
                                    "metric_type": MetricType.L2
                                }
                                collection.create_index("embedding", index_params)
                                collection.load()
                                print(f"✅ 为 {collection_name} 创建默认索引并加载")
                            except Exception as e2:
                                print(f"⚠️ 为 {collection_name} 创建索引并加载失败: {e2}")
                        else:
                            print(f"⚠️ 加载集合 {collection_name} 失败: {e}")
                else:
                    print(f"ℹ️ 集合 {collection_name} 暂无数据，跳过加载")
            except Exception as e:
                print(f"⚠️ 处理集合 {collection_name} 失败: {e}")
    
    def _ensure_collection_loaded(self, collection_name: str):
        """确保集合已加载到内存（在数据插入后调用）"""
        try:
            if not utility.has_collection(collection_name):
                return
            
            collection = Collection(collection_name)
            
            # 检查集合是否已加载
            if collection.has_index():
                # 如果集合有索引，尝试加载
                try:
                    collection.load()
                except:
                    pass  # 如果已经加载，会抛出异常，忽略即可
        except:
            pass  # 忽略错误，避免影响主流程
    
    def _load_checkpoint(self) -> set:
        """加载断点续传状态"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                processed = set(data.get("processed_files", []))
                print(f"✅ 断点续传: 已处理 {len(processed)} 个文件")
                return processed
        return set()
    
    def _save_checkpoint(self):
        """保存断点续传状态"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({"processed_files": list(self.processed_files)}, f, ensure_ascii=False, indent=2)
        print("💾 断点已保存")
    
    def _generate_uid(self, content: str) -> str:
        """生成唯一ID"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        try:
            if self.embedding_model is None:
                raise RuntimeError("Embedding model not initialized")
            # LangChain DashScopeEmbeddings provides embed_query / embed_documents
            try:
                emb = self.embedding_model.embed_query(text)
            except TypeError:
                # some versions may expose embed rather than embed_query
                emb = self.embedding_model.embed(text)
            # emb expected to be a list[float]
            if isinstance(emb, (list, tuple)):
                return list(emb)
            # fallback if the returned structure is nested
            if isinstance(emb, dict) and "data" in emb:
                # try to extract vector
                first = emb["data"][0]
                if isinstance(first, dict) and "embedding" in first:
                    return first["embedding"]
            # unknown format
            print("⚠️ 向量化返回未知格式，使用零向量作为回退")
            return [0.0] * self.embedding_config.get("dimension", 1024)
        except Exception as e:
            print(f"⚠️ 向量化异常: {e}")
            return [0.0] * self.embedding_config.get("dimension", 1024)
    
    def _process_optimization(self, opt: Dict[str, Any]) -> Dict[str, Any]:
        """处理优化策略实体"""
        # 获取 related_patterns，优先从顶层，然后从 description 中
        related_patterns = opt.get('related_patterns', [])
        desc = opt.get('description', {})
        if not related_patterns and isinstance(desc, dict):
            related_patterns = desc.get('related_patterns', [])
        
        strategy = {
            "uid": self._generate_uid(opt.get("optimization_name", "")),
            "name": opt.get("optimization_name", ""),
            "level": opt.get("level", ""),
            "rationale": desc.get("strategy_rationale", "") if isinstance(desc, dict) else "",
            "implementation": desc.get("implementation_pattern", "") if isinstance(desc, dict) else "",
            "impact": desc.get("performance_impact", "") if isinstance(desc, dict) else "",
            "trade_offs": desc.get("trade_offs", "") if isinstance(desc, dict) else "",
            "related_patterns": related_patterns,
            "applicability_conditions": opt.get("applicability_conditions", []),
            "tunable_parameters": opt.get("tunable_parameters", []),
            "target_hardware_feature_name": opt.get("target_hardware_feature_name") or opt.get("target_hardware_feature_name".lower(), ""),
            "target_hardware_feature": opt.get("target_hardware_feature", "")
        }
        
        return strategy
    
    def _save_entity(self, collection_name: str, entity_data: Dict[str, Any]) -> str:
        """保存实体到Milvus"""
        collection = Collection(collection_name)
        
        # 准备数据
        uid = entity_data["uid"]
        # 根据集合选择更有信息量的嵌入文本
        if collection_name == "computational_pattern":
            embedding_text = f"{entity_data.get('type','')} {entity_data.get('name','')} {entity_data.get('description','')} {entity_data.get('code','')}"
        elif collection_name == "optimization_strategy":
            embedding_text = f"{entity_data.get('name','')} {entity_data.get('level','')} {entity_data.get('rationale','')} {entity_data.get('implementation','')}"
        elif collection_name == "hardware_feature":
            embedding_text = f"{entity_data.get('name','')} {entity_data.get('architecture','')} {entity_data.get('description','')}"
        elif collection_name == "tunable_parameter":
            embedding_text = f"{entity_data.get('name','')} {entity_data.get('description','')} {entity_data.get('impact','')}"
        elif collection_name == "code_example":
            embedding_text = f"{entity_data.get('name','')} {entity_data.get('snippet','')} {entity_data.get('explanation','')}"
        else:
            embedding_text = entity_data.get("name", "") + " " + str(entity_data)
        embedding = self._get_embedding(embedding_text)
        
        # 构建插入数据
        if collection_name == "optimization_strategy":
            insert_data = [
                [uid],
                [entity_data["name"]],
                [entity_data["level"]],
                [entity_data.get("rationale", "")],
                [entity_data.get("implementation", "")],
                [entity_data.get("impact", "")],
                [entity_data.get("trade_offs", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        elif collection_name == "computational_pattern":
            insert_data = [
                [uid],
                [entity_data.get("name", "")],
                [entity_data.get("type", "")],
                [entity_data.get("description", "")],
                [entity_data.get("code", "")],
                [entity_data.get("numeric_kind", "")],
                [entity_data.get("numeric_precision", "")],
                [entity_data.get("structural_properties", "")],
                [entity_data.get("storage_layout", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        elif collection_name == "hardware_feature":
            insert_data = [
                [uid],
                [entity_data.get("name", "")],
                [entity_data.get("architecture", "")],
                [entity_data.get("description", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        elif collection_name == "tunable_parameter":
            insert_data = [
                [uid],
                [entity_data.get("name", "")],
                [entity_data.get("description", "")],
                [entity_data.get("impact", "")],
                [entity_data.get("value_in_code", "")],
                [entity_data.get("typical_range", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        elif collection_name == "code_example":
            insert_data = [
                [uid],
                [entity_data.get("name", "")],
                [entity_data.get("snippet", "")],
                [entity_data.get("explanation", "")],
                [entity_data.get("source_file", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        else:
            # 兜底
            insert_data = [
                [uid],
                [entity_data.get("name", "")],
                [json.dumps(entity_data, ensure_ascii=False)],
                [embedding]
            ]
        
        # 插入数据
        collection.insert(insert_data)
        collection.flush()
        
        return uid
    
    def _save_relation(self, head_uid: str, tail_uid: str, relation_type: str,
                       head_name: str = "", tail_name: str = "", description: str = ""):
        """保存关系到Milvus"""
        collection = Collection("relation")
        
        # 清理字符串中的NUL字符，避免输出问题
        def clean_str(s: str) -> str:
            if not s:
                return ""
            return s.replace('\x00', '').strip()
        
        head_name = clean_str(head_name)
        tail_name = clean_str(tail_name)
        description = clean_str(description)
        
        relation_uid = self._generate_uid(f"{head_uid}_{tail_uid}_{relation_type}")
        if not description:
            if relation_type == "OPTIMIZES_PATTERN":
                description = f"{tail_name or tail_uid} 优化了计算流程 {head_name or head_uid}"
            elif relation_type == "IS_ILLUSTRATED_BY":
                description = f"{head_name or head_uid} 由代码示例 {tail_name or tail_uid} 说明"
            elif relation_type == "TARGETS":
                description = f"{head_name or head_uid} 面向硬件特性 {tail_name or tail_uid}"
            elif relation_type == "HAS_PARAMETER":
                description = f"{head_name or head_uid} 具有可调参数 {tail_name or tail_uid}"
            else:
                description = relation_type
        embedding_text = f"{relation_type} {head_name} {tail_name} {description}"
        embedding = self._get_embedding(embedding_text)
        
        insert_data = [
            [relation_uid],
            [relation_type],
            [head_uid],
            [tail_uid],
            [head_name],
            [tail_name],
            [description],
            [embedding]
        ]
        
        collection.insert(insert_data)
        collection.flush()
        
        print(f"✓ 保存关系: {relation_type} ({head_name or head_uid} -> {tail_name or tail_uid})")
    
    def extract_from_file(self, file_path: str):
        """从单个文件中抽取实体和关系"""
        if file_path in self.processed_files:
            print(f"⏭️ 跳过已处理文件: {file_path}")
            return
        
        print(f"📄 处理: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理每个算子分析
            for analysis in data.get("individual_analyses", []):
                operator_name = analysis.get("file_path", "").split("/")[-1]
                print(f"  🔍 算子: {operator_name}")
                
                entity_count = 0
                relation_count = 0
                code_counter = 1
                # 建立当前分析中已保存的计算流程类型到UID的映射，确保关系指向已存在的头实体
                pattern_uid_by_type = {}
                
                # 处理计算流程
                for pattern in analysis.get("computational_patterns", []):
                    # 输入结构参考 agent23 产物
                    ptype = pattern.get("pattern_type", "") or pattern.get("type", "")
                    dof = pattern.get("data_object_features") or {}
                    pattern_entity = {
                        "uid": self._generate_uid(ptype or pattern.get("name", "")),
                        "name": pattern.get("name", ""),
                        "type": ptype,
                        "description": pattern.get("description", ""),
                        "code": pattern.get("code", ""),
                        "numeric_kind": dof.get("numeric_kind", ""),
                        "numeric_precision": dof.get("numeric_precision", ""),
                        "structural_properties": dof.get("structural_properties", ""),
                        "storage_layout": dof.get("storage_layout", "")
                    }
                    saved_uid = self._save_entity("computational_pattern", pattern_entity)
                    if pattern_entity.get("type"):
                        pattern_uid_by_type[pattern_entity["type"]] = saved_uid
                    entity_count += 1
                    print(f"    ✓ 保存computational_pattern: {pattern_entity.get('type', '')} -> {pattern_entity['uid']}")
                
                # 处理优化策略
                for level in ["algorithm_level_optimizations", "code_level_optimizations", "instruction_level_optimizations"]:
                    for opt in analysis.get(level, []):
                        strategy = self._process_optimization(opt)
                        strategy_uid = self._save_entity("optimization_strategy", strategy)
                        entity_count += 1
                        print(f"    ✓ 保存optimization_strategy: {strategy['name']} -> {strategy_uid}")
                        
                        # 创建OPTIMIZES_PATTERN关系
                        for pattern_type in strategy.get("related_patterns", []):
                            # 优先使用当前文件已保存的计算流程UID，避免不一致
                            pattern_uid = pattern_uid_by_type.get(pattern_type, self._generate_uid(pattern_type))
                            self._save_relation(
                                pattern_uid, strategy_uid, "OPTIMIZES_PATTERN",
                                head_name=pattern_type, tail_name=strategy.get("name", "")
                            )
                            relation_count += 1
                        
                        # 处理硬件特征
                        hardware_names: List[str] = []
                        if isinstance(opt.get("hardware_features"), list):
                            hardware_names = [str(x) for x in opt.get("hardware_features") if x]
                        else:
                            if strategy.get("target_hardware_feature_name"):
                                hardware_names = [strategy.get("target_hardware_feature_name")]
                        for hw_name in hardware_names:
                            if not hw_name:
                                continue
                            hw_entity = {
                                "uid": self._generate_uid(hw_name),
                                "name": hw_name,
                                "architecture": "",
                                "description": opt.get("target_hardware_feature", "") or f"Hardware feature: {hw_name}"
                            }
                            hw_uid = self._save_entity("hardware_feature", hw_entity)
                            self._save_relation(
                                strategy_uid, hw_uid, "TARGETS",
                                head_name=strategy.get("name", ""), tail_name=hw_name
                            )
                            entity_count += 1
                            relation_count += 1
                            print(f"    ✓ 保存hardware_feature: {hw_name} -> {hw_uid}")
                        
                        # 处理代码示例
                        code_examples: List[Dict[str, Any]] = []
                        if isinstance(opt.get("code_example"), dict):
                            code_examples = [opt.get("code_example")]
                        elif isinstance(opt.get("code_examples"), list):
                            code_examples = opt.get("code_examples")
                        for i, code_obj in enumerate(code_examples):
                            snippet = code_obj.get("snippet", "") if isinstance(code_obj, dict) else str(code_obj)
                            explanation = code_obj.get("explanation", "") if isinstance(code_obj, dict) else ""
                            code_entity = {
                                "uid": self._generate_uid(f"{strategy['name']}_code_{code_counter}"),
                                "name": f"code{code_counter}",
                                "snippet": snippet,
                                "explanation": explanation,
                                "source_file": operator_name
                            }
                            code_uid = self._save_entity("code_example", code_entity)
                            self._save_relation(
                                strategy_uid, code_uid, "IS_ILLUSTRATED_BY",
                                head_name=strategy.get("name", ""), tail_name=code_entity["name"]
                            )
                            entity_count += 1
                            relation_count += 1
                            print(f"    ✓ 保存code_example: {code_entity['name']} -> {code_uid}")
                            code_counter += 1
                        
                        # 处理可调参数
                        for param in strategy.get("tunable_parameters", []):
                            if not param:
                                continue
                            if isinstance(param, str):
                                param_name = param
                                param_entity = {
                                    "uid": self._generate_uid(param_name),
                                    "name": param_name,
                                    "description": f"Tunable parameter: {param_name}",
                                    "impact": "",
                                    "value_in_code": "",
                                    "typical_range": ""
                                }
                            else:
                                param_name = param.get("parameter_name") or param.get("name") or ""
                                if not param_name:
                                    continue
                                # 处理 typical_range：如果是列表，将每个元素转为字符串后连接
                                typical_range = param.get("typical_range", [])
                                if isinstance(typical_range, list):
                                    typical_range_str = ",".join(str(x) for x in typical_range)
                                else:
                                    typical_range_str = str(typical_range) if typical_range else ""
                                
                                param_entity = {
                                    "uid": self._generate_uid(param_name),
                                    "name": param_name,
                                    "description": param.get("description", ""),
                                    "impact": param.get("impact", ""),
                                    "value_in_code": str(param.get("value_in_code", "")),
                                    "typical_range": typical_range_str
                                }
                            param_uid = self._save_entity("tunable_parameter", param_entity)
                            self._save_relation(
                                strategy_uid, param_uid, "HAS_PARAMETER",
                                head_name=strategy.get("name", ""), tail_name=param_entity["name"]
                            )
                            entity_count += 1
                            relation_count += 1
                            print(f"    ✓ 保存tunable_parameter: {param_entity['name']} -> {param_uid}")
                
                print(f"  📊 完成: 实体={entity_count}, 关系={relation_count}")
            
            # 标记文件已处理
            self.processed_files.add(file_path)
            self._save_checkpoint()
            
        except Exception as e:
            print(f"❌ 处理文件失败 {file_path}: {e}")
    
    def extract_from_directory(self, directory_path: str):
        """从目录中抽取所有JSON文件"""
        json_files = list(Path(directory_path).glob("*.json"))
        print(f"📁 发现 {len(json_files)} 个JSON文件")
        
        total_entities = 0
        total_relations = 0
        
        for i, file_path in enumerate(json_files, 1):
            print(f"\n{'='*60}")
            print(f"进度: {i}/{len(json_files)}")
            print(f"{'='*60}")
            
            self.extract_from_file(str(file_path))
        
        # 统计总数
        for collection_name in ["optimization_strategy", "computational_pattern", "hardware_feature", "tunable_parameter", "code_example"]:
            collection = Collection(collection_name)
            count = collection.num_entities
            total_entities += count
            print(f"  ✅ {collection_name}: {count} 个")
        
        relation_collection = Collection("relation")
        total_relations = relation_collection.num_entities
        
        print(f"\n{'='*60}")
        print(f"📊 总计: 实体={total_entities}, 关系={total_relations}")
        print(f"{'='*60}")
        
        # 为所有集合创建索引并加载到内存
        print("\n🔧 正在为集合创建索引并加载到内存...")
        self._create_indexes_and_load()
        
        print("🎉 完成！")
        print("✅ 已保存断点并关闭Milvus连接")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="OpenBLAS知识图谱实体抽取器")
    parser.add_argument("--config", type=str, default="kg_config.json", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, help="分析结果JSON文件目录（可选，优先使用配置文件中的路径）")
    
    args = parser.parse_args()
    
    print("知识图谱抽取器（带断点续传）")
    print()
    
    extractor = KnowledgeGraphExtractor(args.config)
    
    # 确定输入目录：优先使用命令行参数，否则使用配置文件
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = extractor.data_source_config.get("analysis_results_dir")
        if not data_dir:
            print("❌ 错误：未指定输入目录。请在配置文件中设置 data_source.analysis_results_dir 或使用 --data_dir 参数")
            return
    
    # 根据 extractor1.py 的位置和配置路径构造输入目录
    # extractor1.py 在 /home/dgc/mjs/project/analyze_OB/KG/extractor1.py
    # 项目根目录是 /home/dgc/mjs/project/analyze_OB/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # /home/dgc/mjs/project/analyze_OB/
    
    cfg_path = data_dir
    if os.path.isabs(cfg_path):
        # 已经是绝对路径，直接使用
        data_dir = cfg_path
    else:
        # 相对路径：如果包含 analyze_OB，取其后缀拼接到项目根
        if "analyze_OB" in cfg_path:
            idx = cfg_path.find("analyze_OB")
            suffix = cfg_path[idx + len("analyze_OB"):].lstrip("/\\")
            data_dir = os.path.join(project_root, suffix)
        else:
            # 不包含 analyze_OB，直接拼接到项目根
            data_dir = os.path.join(project_root, cfg_path.lstrip("/\\"))
    
    if not os.path.exists(data_dir):
        print(f"❌ 错误：输入目录不存在: {data_dir}")
        return
    
    print(f"📁 输入目录: {data_dir}")
    extractor.extract_from_directory(data_dir)


if __name__ == "__main__":
    main()
