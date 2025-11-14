#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus到Neo4j数据导出工具
将知识图谱数据从Milvus导出到Neo4j图数据库
"""

import os
import json
from typing import Dict, List, Any
from pymilvus import connections, Collection
from neo4j import GraphDatabase
import argparse


class MilvusToNeo4jExporter:
    """Milvus到Neo4j导出器"""
    
    def __init__(self, config_path: str = "kg_config.json"):
        """初始化导出器"""
        self.config = self._load_config(config_path)
        self.milvus_config = self.config.get("milvus", {})
        self.neo4j_config = self.config.get("neo4j", {})
        
        # 连接Milvus
        self._connect_milvus()
        
        # 连接Neo4j
        self.neo4j_driver = GraphDatabase.driver(
            self.neo4j_config.get("uri", "bolt://localhost:7687"),
            auth=(
                self.neo4j_config.get("username", "neo4j"),
                self.neo4j_config.get("password", "password")
            )
        )
        
        print("✅ Milvus到Neo4j导出器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "neo4j": {"uri": "bolt://localhost:7687", "username": "neo4j", "password": "password"}
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
    
    def _clear_neo4j_database(self):
        """清空Neo4j数据库"""
        with self.neo4j_driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        print("🗑️ 已清空Neo4j数据库")
    
    def _export_entities(self, collection_name: str, node_label: str) -> int:
        """导出实体到Neo4j"""
        collection = Collection(collection_name)
        collection.load()
        
        # 获取所有实体
        entities = collection.query(
            expr="",
            output_fields=["*"],
            limit=16384
        )
        
        exported_count = 0
        
        with self.neo4j_driver.session() as session:
            for entity in entities:
                try:
                    # 解析entity_data
                    entity_data = json.loads(entity.get("entity_data", "{}"))
                    
                    # 构建节点属性
                    properties = {
                        "uid": entity["uid"],
                        "name": entity.get("name", entity.get("type", "")),
                    }
                    
                    # 添加特定字段
                    if collection_name == "optimization_strategy":
                        properties.update({
                            "level": entity.get("level", ""),
                            "rationale": entity_data.get("rationale", ""),
                            "implementation": entity_data.get("implementation", ""),
                            "impact": entity_data.get("impact", ""),
                            "trade_offs": entity_data.get("trade_offs", ""),
                            "related_patterns": json.dumps(entity_data.get("related_patterns", [])),
                            "optimization_context": json.dumps(entity_data.get("optimization_context", {}))
                        })
                    elif collection_name == "computational_pattern":
                        properties.update({
                            "type": entity.get("type", ""),
                            "description": entity_data.get("description", "")
                        })
                    else:
                        properties.update({
                            "description": entity_data.get("description", ""),
                            "content": entity_data.get("content", "")
                        })
                    
                    # 创建节点
                    cypher = f"""
                    CREATE (n:{node_label} {{
                        uid: $uid,
                        name: $name,
                        {', '.join([f'{k}: ${k}' for k in properties.keys() if k not in ['uid', 'name']])}
                    }})
                    """
                    
                    session.run(cypher, **properties)
                    exported_count += 1
                
                except Exception as e:
                    print(f"⚠️ 导出实体失败 {entity.get('uid', 'unknown')}: {e}")
        
        print(f"✅ 导出 {node_label}: {exported_count} 个节点")
        return exported_count
    
    def _export_relations(self) -> int:
        """导出关系到Neo4j"""
        collection = Collection("relation")
        collection.load()
        
        # 获取所有关系
        relations = collection.query(
            expr="",
            output_fields=["*"],
            limit=16384
        )
        
        exported_count = 0
        
        with self.neo4j_driver.session() as session:
            for relation in relations:
                try:
                    head_uid = relation["head_entity_uid"]
                    tail_uid = relation["tail_entity_uid"]
                    relation_type = relation["relation_type"]
                    
                    # 创建关系
                    cypher = f"""
                    MATCH (head {{uid: $head_uid}})
                    MATCH (tail {{uid: $tail_uid}})
                    CREATE (head)-[r:{relation_type}]->(tail)
                    SET r.uid = $relation_uid
                    """
                    
                    session.run(cypher, {
                        "head_uid": head_uid,
                        "tail_uid": tail_uid,
                        "relation_uid": relation["uid"]
                    })
                    
                    exported_count += 1
                
                except Exception as e:
                    print(f"⚠️ 导出关系失败 {relation.get('uid', 'unknown')}: {e}")
        
        print(f"✅ 导出关系: {exported_count} 条")
        return exported_count
    
    def _create_indexes(self):
        """创建Neo4j索引"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (n:OptimizationStrategy) ON (n.uid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:ComputationalPattern) ON (n.uid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:HardwareFeature) ON (n.uid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:TunableParameter) ON (n.uid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:CodeExample) ON (n.uid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:OptimizationStrategy) ON (n.level)",
            "CREATE INDEX IF NOT EXISTS FOR (n:ComputationalPattern) ON (n.type)"
        ]
        
        with self.neo4j_driver.session() as session:
            for index_cypher in indexes:
                session.run(index_cypher)
        
        print("🔍 已创建Neo4j索引")
    
    def export_knowledge_graph(self, clear_existing: bool = True) -> Dict[str, int]:
        """导出完整知识图谱"""
        print("🚀 开始导出知识图谱到Neo4j")
        
        # 清空现有数据
        if clear_existing:
            self._clear_neo4j_database()
        
        # 导出实体
        entity_mappings = [
            ("optimization_strategy", "OptimizationStrategy"),
            ("computational_pattern", "ComputationalPattern"),
            ("hardware_feature", "HardwareFeature"),
            ("tunable_parameter", "TunableParameter"),
            ("code_example", "CodeExample")
        ]
        
        export_stats = {}
        
        for collection_name, node_label in entity_mappings:
            try:
                count = self._export_entities(collection_name, node_label)
                export_stats[node_label] = count
            except Exception as e:
                print(f"❌ 导出 {collection_name} 失败: {e}")
                export_stats[node_label] = 0
        
        # 导出关系
        try:
            relation_count = self._export_relations()
            export_stats["Relations"] = relation_count
        except Exception as e:
            print(f"❌ 导出关系失败: {e}")
            export_stats["Relations"] = 0
        
        # 创建索引
        self._create_indexes()
        
        print(f"\n🎉 知识图谱导出完成:")
        for entity_type, count in export_stats.items():
            print(f"  - {entity_type}: {count}")
        
        return export_stats
    
    def close(self):
        """关闭连接"""
        self.neo4j_driver.close()
        print("👋 已关闭Neo4j连接")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Milvus到Neo4j数据导出工具")
    parser.add_argument("--config", type=str, default="kg_config.json", help="配置文件路径")
    parser.add_argument("--no-clear", action="store_true", help="不清空现有Neo4j数据")
    
    args = parser.parse_args()
    
    print("📤 Milvus到Neo4j数据导出工具")
    print("=" * 50)
    
    exporter = MilvusToNeo4jExporter(args.config)
    
    try:
        stats = exporter.export_knowledge_graph(clear_existing=not args.no_clear)
        
        # 保存导出统计
        with open("neo4j_export_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 导出统计已保存: neo4j_export_stats.json")
        
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
