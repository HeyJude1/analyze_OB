#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体聚类检索器v4 (Milvus查询修复)
- 对 hardware_feature, optimization_strategy, tunable_parameter 进行聚类。
- 严格遵循“一个实体只属于一个簇”的规则。
- 簇中心按顺序从未聚类的实体中选取。
- 输出文件 clusters_retrieved.json 保存到 analysis_results_dir。
- 修复了 Milvus collection.query() 的 expr 语法问题。
"""

import os
import json
from typing import Dict, List, Any
from pymilvus import connections, Collection, utility
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

class EntityClusterRetriever:
    """实体聚类检索器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化聚类检索器"""
        self.config = config
        self.milvus_config = self.config.get("milvus", {})
        self.clustering_config = self.config.get("clustering", {})
        
        self.similarity_threshold = self.clustering_config.get("similarity_threshold", 0.85)
        
        self._connect_milvus()
        
        print("✅ 实体聚类检索器初始化完成")
        self.entity_types_to_export = [
            "hardware_feature",
            "optimization_strategy",
            "tunable_parameter",
        ]
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "clustering": {"similarity_threshold": 0.85}
            }
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _connect_milvus(self):
        """连接Milvus数据库"""
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        database = self.milvus_config.get("database", "code_op")
        
        connections.connect(alias="default", host=host, port=port, db_name=database)
        print(f"✅ 已连接到Milvus: {host}:{port}/{database}")
    
    # <<< MODIFIED: Corrected the query expression
    def _get_all_entities(self, collection_name: str) -> List[Dict[str, Any]]:
        """获取集合中的所有实体"""
        if not utility.has_collection(collection_name):
            print(f"  ⚠️ 集合 {collection_name} 不存在，跳过。")
            return []
            
        collection = Collection(collection_name)
        collection.load()
        
        num_entities = collection.num_entities
        if num_entities == 0:
            return []

        # 动态获取主键字段名
        primary_key_field = next((f.name for f in collection.schema.fields if f.is_primary), None)
        if not primary_key_field:
            raise ValueError(f"集合 {collection_name} 中未找到主键字段。")

        # 使用一个始终为真的表达式，例如 "pk_field != ''"
        query_expr = f'{primary_key_field} != ""'

        results = collection.query(
            expr=query_expr,
            output_fields=["*"],
            limit=16384 # Milvus's max limit per query
        )
        print(f"📊 {collection_name}: 加载 {len(results)} 个实体")
        return results
    
    def _calculate_similarity_matrix(self, embeddings: List[List[float]]) -> np.ndarray:
        """计算相似度矩阵"""
        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        return similarity_matrix
    
    def _cluster_entities(self, entities: List[Dict[str, Any]], entity_type: str) -> List[List[int]]:
        """
        对实体进行聚类。
        严格遵循规则：一个实体（无论是中心还是成员）只能属于一个簇。
        """
        if not entities:
            return []
        
        embeddings = [entity["embedding"] for entity in entities]
        similarity_matrix = self._calculate_similarity_matrix(embeddings)
        
        num_entities = len(entities)
        unclustered_indices = set(range(num_entities))
        clusters = []

        while unclustered_indices:
            center_idx = sorted(list(unclustered_indices))[0]
            new_cluster = [center_idx]
            unclustered_indices.remove(center_idx)
            
            potential_members = list(unclustered_indices)
            for member_idx in potential_members:
                if similarity_matrix[center_idx][member_idx] >= self.similarity_threshold:
                    new_cluster.append(member_idx)
                    unclustered_indices.remove(member_idx)
            
            clusters.append(new_cluster)
        
        print(f"  🔍 {entity_type}: {num_entities} 个实体 -> {len(clusters)} 个簇")
        return clusters

    def _format_clusters_map(self, entities: List[Dict[str, Any]], clusters: List[List[int]]) -> Dict[str, Dict[str, Any]]:
        """格式化为 { cluster_k: { center_uid, center_name, members: [ {uid,name,score} ] } }"""
        if not entities:
            return {}
            
        embeddings = np.array([e["embedding"] for e in entities])
        clusters_map: Dict[str, Dict[str, Any]] = {}
        
        for cluster_idx, idx_list in enumerate(clusters):
            if not idx_list:
                continue
            
            center_idx = idx_list[0]
            center_entity = entities[center_idx]
            center_vec = embeddings[center_idx].reshape(1, -1)
            
            members = []
            for member_idx in idx_list[1:]:
                sim = float(cosine_similarity(center_vec, embeddings[member_idx].reshape(1, -1))[0][0])
                member_entity = entities[member_idx]
                members.append({
                    "uid": member_entity["uid"],
                    "name": member_entity.get("name", member_entity.get("type", "")),
                    "score": sim
                })
            
            members.sort(key=lambda x: x["score"], reverse=True)

            clusters_map[f"cluster_{cluster_idx}"] = {
                "center_uid": center_entity["uid"],
                "center_name": center_entity.get("name", center_entity.get("type", "")),
                "members": members
            }
        return clusters_map
    
    def retrieve_and_cluster(self) -> Dict[str, Any]:
        """检索所有指定实体类型的聚类"""
        print("🚀 开始实体聚类检索")
        
        final_clusters: Dict[str, Any] = {}
        
        for entity_type in self.entity_types_to_export:
            print(f"\n📋 处理实体类型: {entity_type}")
            
            try:
                entities = self._get_all_entities(entity_type)
                
                if not entities:
                    print(f"  ⚠️ {entity_type}: 无实体数据，跳过。")
                    final_clusters[entity_type] = {}
                    continue
                
                clusters = self._cluster_entities(entities, entity_type)
                
                clusters_map = self._format_clusters_map(entities, clusters)
                final_clusters[entity_type] = clusters_map
                
            except Exception as e:
                print(f"  ❌ 在处理 {entity_type} 时发生错误: {e}")
                import traceback
                traceback.print_exc()
                final_clusters[entity_type] = {}
        
        print(f"\n🎉 聚类检索完成")
        return final_clusters
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存聚类结果"""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 聚类结果已保存: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实体聚类检索器v4")
    parser.add_argument("--config", type=str, default="config/kg_config.json", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, help="分析结果的基准目录，用于确定输出位置")
    parser.add_argument("--output", type=str, default="clusters_retrieved.json", help="输出文件名")
    
    args = parser.parse_args()
    
    print("🔍 实体聚类检索器v4")
    print("=" * 50)

    config = EntityClusterRetriever._load_config(args.config)
    
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

    if not os.path.exists(base_dir):
        print(f"❌ 错误：基准目录不存在: {base_dir}")
        return

    output_file_path = os.path.join(base_dir, args.output)
    
    retriever = EntityClusterRetriever(config)
    results = retriever.retrieve_and_cluster()
    retriever.save_results(results, output_file_path)


if __name__ == "__main__":
    main()