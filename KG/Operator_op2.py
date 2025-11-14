#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略检索与评分系统v3 (高级评分版)
- 严格遵循 agent23 的四阶段计算流程识别。
- 增加 Milvus 相似度检索与最高分筛选。
- 根据关联关系查找优化策略。
- 实现新的、基于上下文的评分与筛选逻辑。
"""

import os
import json
from typing import Dict, List, Any
from pymilvus import connections, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.embeddings import DashScopeEmbeddings
import argparse
from dotenv import load_dotenv

# 动态导入 agent23 中的 AgentFactory
from agent23 import AgentFactory

load_dotenv()


class OptimizationStrategyOperator:
    """优化策略操作器"""
    
    def __init__(self, config_path: str = "kg_config.json"):
        """初始化操作器"""
        self.config = self._load_config(config_path)
        self.milvus_config = self.config.get("milvus", {})
        self.model_config = self.config.get("model", {})
        self.embedding_config = self.config.get("dashscope_embeddings", {})
        
        self._connect_milvus()
        self._init_llm()
        self._init_embedding_model()
        
        # AgentFactory for pattern detection
        self.agent_factory = AgentFactory()
        
        print("✅ 优化策略操作器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "model": {
                    "name": "qwen-max",
                    "temperature": 0.0,
                    "max_tokens": 8192,
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                },
                "dashscope_embeddings": {"name": "text-embedding-v3"}
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

    def _init_llm(self):
        """初始化 ChatOpenAI 模型"""
        self.llm = ChatOpenAI(
            model=self.model_config.get("name"),
            temperature=float(self.model_config.get("temperature", 0.0)),
            max_tokens=int(self.model_config.get("max_tokens", 8192)),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.model_config.get("base_url"),
        )
        
    def _init_embedding_model(self):
        """初始化 Embedding 模型"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for embedding model")
        
        self.embedding_model = DashScopeEmbeddings(
            model=self.embedding_config.get("name", "text-embedding-v3"), 
            dashscope_api_key=api_key
        )

    def _detect_computational_patterns(self, source_code: str) -> List[Dict[str, Any]]:
        """使用AgentFactory按四个阶段检测计算流程模式"""
        all_patterns = []
        stages = ["prep", "transform", "core", "post"]
        
        for stage in stages:
            print(f"  -> 正在识别 {stage} 阶段的计算流程...")
            try:
                patterns = self.agent_factory.analyze_patterns_stage(source_code, "unknown", stage)
                if patterns:
                    all_patterns.extend(patterns)
                    print(f"    ✅ {stage} 阶段识别到 {len(patterns)} 个模式")
            except Exception as e:
                print(f"    ❌ {stage} 阶段识别失败: {e}")
        
        return all_patterns

    def _search_similar_patterns(self, detected_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在Milvus中检索与检测到的计算流程相似的实体"""
        if not detected_patterns:
            return []

        collection = Collection("computational_pattern")
        collection.load()
        
        # 为检测到的模式生成向量
        embedding_texts = [json.dumps(p, ensure_ascii=False) for p in detected_patterns]
        vectors_to_search = self.embedding_model.embed_documents(embedding_texts)
        
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        all_hits = []

        results = collection.search(
            data=vectors_to_search,
            anns_field="embedding",
            param=search_params,
            limit=10, # 每个检测到的模式检索10个最相似的
            output_fields=["uid", "name", "type"]
        )
        
        for i, hits in enumerate(results):
            for hit in hits:
                if hit.distance <= 0.2: # 相似度阈值 (1 - L2距离)，0.2表示非常相似
                    all_hits.append({
                        "uid": hit.entity.get("uid"),
                        "name": hit.entity.get("name"),
                        "type": hit.entity.get("type"),
                        "similarity": 1 - hit.distance,
                        "query_pattern": detected_patterns[i]['name']
                    })
        return all_hits

    def _filter_top_patterns(self, similar_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从相似结果中为每种类型筛选出得分最高的实体"""
        top_patterns = {}
        for pattern in similar_patterns:
            ptype = pattern['type']
            if ptype not in top_patterns or pattern['similarity'] > top_patterns[ptype]['similarity']:
                top_patterns[ptype] = pattern
        return list(top_patterns.values())

    def _find_related_strategies(self, top_patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """根据计算流程查找关联的优化策略"""
        if not top_patterns:
            return []
            
        pattern_uids = [p['uid'] for p in top_patterns]
        relation_col = Collection("relation")
        strategy_col = Collection("optimization_strategy")
        
        expr = f'head_entity_uid in {json.dumps(pattern_uids)} and relation_type == "OPTIMIZES_PATTERN"'
        relations = relation_col.query(expr, output_fields=["tail_entity_uid"])
        
        strategy_uids = list({rel['tail_entity_uid'] for rel in relations})
        if not strategy_uids:
            return []
            
        strategies = strategy_col.query(f'uid in {json.dumps(strategy_uids)}', output_fields=["*"])
        return strategies

    def _score_and_filter_strategies(self, strategies: List[Dict[str, Any]], detected_pattern_types: List[str]) -> List[Dict[str, Any]]:
        """根据自定义评分公式筛选策略"""
        scored_strategies = []
        w_context = 0.5
        
        for strategy in strategies:
            try:
                entity_data = json.loads(strategy.get("entity_data", "{}"))
                context = entity_data.get("optimization_context", {})
                core_patterns = context.get("core_patterns", [])
                contextual_patterns = context.get("contextual_patterns", {})
                
                # 条件1: 核心模式必须是检测到模式的子集
                if not set(core_patterns).issubset(set(detected_pattern_types)):
                    continue

                # 计算Score_core
                score_core = len(set(core_patterns) & set(detected_pattern_types)) / len(detected_pattern_types) if detected_pattern_types else 0

                # 计算Score_context
                score_context = 0.0
                for pattern, freq in contextual_patterns.items():
                    if pattern in detected_pattern_types:
                        score_context += freq
                
                # 计算总分
                score_total = score_core + w_context * score_context
                
                # 条件2: 总分 >= 0.5
                if score_total >= 0.5:
                    strategy_info = {
                        "strategy_uid": strategy['uid'],
                        "strategy_name": strategy['name'],
                        "level": strategy['level'],
                        "overview": entity_data.get('rationale', ''),
                        "when_to_use": entity_data.get('applicability_conditions', ''),
                        "hardware": entity_data.get('target_hardware_feature', ''),
                        "key_actions": (entity_data.get('implementation_pattern', '') or '').split('\n'),
                        "code_examples": [], # This would require another query if needed
                        "parameters": entity_data.get('tunable_parameters', []),
                        "cautions": entity_data.get('trade_offs', ''),
                        "related_patterns": entity_data.get('related_patterns', []),
                        "optimization_context": context,
                        "score": score_total
                    }
                    scored_strategies.append(strategy_info)
            except Exception as e:
                print(f"  ⚠️ 评分策略 {strategy.get('uid')} 失败: {e}")
                
        # 按分数降序排列
        scored_strategies.sort(key=lambda x: x['score'], reverse=True)
        return scored_strategies

    def process_source_code(self, source_file: str) -> Dict[str, Any]:
        """处理源代码文件，执行完整的检索和评分流程"""
        print(f"🚀 开始处理源代码: {source_file}")
        
        if not os.path.exists(source_file):
            return {"error": f"源文件不存在: {source_file}"}
        
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # 1. 识别计算流程
        patterns_detected_full = self._detect_computational_patterns(source_code)
        patterns_detected_types = [p['pattern_type'] for p in patterns_detected_full]
        print(f"✅ 步骤1完成: 检测到 {len(patterns_detected_types)} 个计算流程: {patterns_detected_types}")
        
        # 2. 检索相似计算流程
        similar_patterns = self._search_similar_patterns(patterns_detected_full)
        print(f"✅ 步骤2完成: 检索到 {len(similar_patterns)} 个相似计算流程 (相似度 > 0.8)")

        # 3. 筛选每种类型的最高分
        top_patterns = self._filter_top_patterns(similar_patterns)
        print(f"✅ 步骤3完成: 筛选出 {len(top_patterns)} 个最高分计算流程")

        # 4. 查找关联的优化策略
        search_strategies = self._find_related_strategies(top_patterns)
        print(f"✅ 步骤4完成: 找到 {len(search_strategies)} 个关联的优化策略")

        # 5. 评分和筛选
        scored_strategies = self._score_and_filter_strategies(search_strategies, patterns_detected_types)
        print(f"✅ 步骤5完成: 最终筛选出 {len(scored_strategies)} 个高分策略")

        result = {
            "source_file": source_file,
            "patterns_detected": patterns_detected_full,
            "similar_patterns_found": similar_patterns,
            "top_patterns_per_type": top_patterns,
            "search_strategies": [s['name'] for s in search_strategies],
            "scored_strategies": scored_strategies
        }
        return result
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存处理结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化策略检索与评分系统v3")
    parser.add_argument("--source", type=str, required=True, help="源代码文件路径")
    parser.add_argument("--output", type=str, default="opinfo2.json", help="输出文件路径")
    parser.add_argument("--config", type=str, default="kg_config.json", help="配置文件路径")
    
    args = parser.parse_args()
    
    print("⚖️ 优化策略检索与评分系统v3")
    print("=" * 50)
    
    config = OptimizationStrategyOperator._load_config(args.config)
    operator = OptimizationStrategyOperator(config)
    results = operator.process_source_code(args.source)
    operator.save_results(results, args.output)


if __name__ == "__main__":
    main()