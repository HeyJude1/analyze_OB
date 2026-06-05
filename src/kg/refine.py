#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体聚类精炼器v10 (最终健壮版)
- 输入: clusters_retrieved.json
- 输出: clusters_retrieved_refined.json
- 功能:
  1. 对每个簇，调用LLM进行语义分组并命名。
  2. 程序根据LLM的分组结果，按照明确规则设置is_primary标志。
  3. 严格遵循输出JSON格式。
  4. 智能跳过只含单个实体的簇，节省API调用。
  5. 在每次LLM调用前增加固定延时，主动避免API速率限制。
  6. 修复了LLM调用失败导致的程序崩溃问题。
  7. 新增：通过Prompt引导和代码后处理，确保similar_groups中每组至少有2个实体。
"""

import os
import json
import time
from typing import Dict, List, Any
from pathlib import Path
from pymilvus import connections, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import argparse
from dotenv import load_dotenv

try:
    from ..utils.prompt_loader import get_prompt_loader
except ImportError:
    import sys, os as _os
    _src_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader

load_dotenv("config/.env")

_prompts = get_prompt_loader()


class EntityClusterRefiner:
    """实体聚类精炼器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化聚类精炼器"""
        self.config = config
        self.milvus_config = self.config.get("milvus", {})
        self.model_config = self.config.get("model", {})
        self.entity_types = ["hardware_feature", "optimization_strategy", "tunable_parameter"]
        
        self._connect_milvus()
        self._init_llm()
        
        print("✅ 实体聚类精炼器初始化完成")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "model": {
                    "name": "qwen-max",
                    "temperature": 0.0,
                    "max_tokens": 8192,
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                }
            }
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_clusters_data(self, input_file: str) -> Dict[str, Any]:
        """加载由 retrieve_clusters.py 生成的聚类结果"""
        print(f"📂 正在加载聚类文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _connect_milvus(self):
        """连接 Milvus"""
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        db_name = self.milvus_config.get("database", "code_op")
        connections.connect(alias="default", host=host, port=port, db_name=db_name)
        print(f"✅ 已连接到Milvus: {host}:{port}/{db_name}")
    
    def _init_llm(self):
        """初始化 ChatOpenAI 模型"""
        self.llm = ChatOpenAI(
            model=self.model_config.get("name"),
            temperature=float(self.model_config.get("temperature", 0.0)),
            max_tokens=int(self.model_config.get("max_tokens", 8192)),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.model_config.get("base_url"),
        )

    def _fetch_entity_details(self, entity_type: str, uids: List[str]) -> Dict[str, Dict[str, Any]]:
        """从 Milvus 高效查询一批实体的详细信息"""
        if not uids: return {}
        try:
            col = Collection(entity_type)
            col.load()
            expr = f'uid in {json.dumps(uids)}'
            output_fields = ["uid", "name"]
            if entity_type == "optimization_strategy":
                output_fields.extend(["level", "rationale", "implementation", "impact", "trade_offs"])
            elif entity_type == "hardware_feature":
                output_fields.extend(["architecture", "description"])
            elif entity_type == "tunable_parameter":
                output_fields.extend(["description", "impact", "value_in_code", "typical_range"])
            results = col.query(expr=expr, output_fields=output_fields, limit=len(uids))
            return {res['uid']: res for res in results}
        except Exception as e:
            print(f"  ⚠️ 查询实体详情失败 ({entity_type}): {e}")
            return {}

    def _build_entity_summary(self, entity_type: str, entity_details: Dict[str, Any]) -> str:
        """根据实体详情构建用于LLM判断的摘要"""
        name = entity_details.get("name", "未知名称")
        
        if entity_type == "optimization_strategy":
            parts = [f"策略名称: {name}", f"原理: {entity_details.get('rationale', 'N/A')}", f"实现: {entity_details.get('implementation', 'N/A')}", f"影响: {entity_details.get('impact', 'N/A')}", f"权衡: {entity_details.get('trade_offs', 'N/A')}"]
            return "；".join(p for p in parts if p.split(': ')[-1] not in ['N/A', ''])
        elif entity_type == "hardware_feature":
            return f"硬件特性: {name}；描述: {entity_details.get('description', 'N/A')}"
        elif entity_type == "tunable_parameter":
            parts = [f"可调参数: {name}", f"描述: {entity_details.get('description', 'N/A')}", f"影响: {entity_details.get('impact', 'N/A')}"]
            return "；".join(p for p in parts if p.split(': ')[-1] not in ['N/A', ''])
        return f"实体名称: {name}"

    def _create_refine_parser(self) -> StructuredOutputParser:
        """创建用于解析LLM响应的结构化解析器"""
        response_schemas = [
            ResponseSchema(name="similar_groups", description="一个列表，每个元素代表一个应合并的实体组。每个组是一个字典，包含 'canonical_name' (字符串) 和 'entities' (一个实体临时名称的列表)。"),
            ResponseSchema(name="remaining_entities", description="一个列表，包含那些不属于任何组的独立实体的临时名称。"),
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)

    def _invoke_llm_with_retry(self, messages: List[Any], parser: StructuredOutputParser, retries: int = 3) -> Dict[str, Any]:
        """带重试和结构化解析的LLM调用，确保始终返回有效字典"""
        for attempt in range(retries):
            try:
                time.sleep(1) 
                response = self.llm.invoke(messages)
                content = response.content
                if not content:
                    raise ValueError("LLM返回空内容")
                
                parsed_output = parser.parse(content)
                if not isinstance(parsed_output, dict):
                    raise ValueError(f"解析器返回了非字典类型: {type(parsed_output)}")
                    
                return parsed_output
            except Exception as e:
                print(f"  - LLM调用或解析失败 (尝试 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"    将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("  - 达到最大重试次数，该簇精炼失败。")
                    return {"similar_groups": [], "remaining_entities": []}
        return {"similar_groups": [], "remaining_entities": []}

    def _refine_one_cluster(self, entity_type: str, cluster_obj: Dict[str, Any]) -> Dict[str, Any]:
        """对单个簇进行精炼，返回符合最终格式的 analysis 字典"""
        center_uid = cluster_obj.get("center_uid")
        center_name = cluster_obj.get("center_name")
        members = cluster_obj.get("members", [])

        if not members:
            print("    -> 簇只包含中心实体，跳过LLM调用。")
            return {
                "status": "success",
                "analysis": {
                    "similar_groups": [],
                    "remaining_entities": [
                        {"name": center_name, "uid": center_uid, "is_primary": True}
                    ]
                }
            }

        all_uids = [center_uid] + [m['uid'] for m in members]
        details_map = self._fetch_entity_details(entity_type, all_uids)

        temp_id_map = {}
        llm_input_items = []
        name_counts = {}

        all_cluster_entities = [{"uid": center_uid, "name": cluster_obj.get("center_name")}] + members
        
        for entity in all_cluster_entities:
            uid = entity.get('uid')
            name = entity.get('name')
            if not uid or not name: continue
            
            details = details_map.get(uid)
            if not details: continue

            count = name_counts.get(name, 0)
            temp_name = f"{name}_{count}"
            name_counts[name] = count + 1
            temp_id_map[temp_name] = {"uid": uid, "name": name}
            
            is_center = (uid == center_uid)
            llm_input_items.append({
                "temp_name": temp_name,
                "summary": self._build_entity_summary(entity_type, details),
                "is_center": is_center
            })

        parser = self._create_refine_parser()
        system_prompt = _prompts.load_system_with_format("kg/entity_refine.yaml")
        
        llm_input_payload = {"entities_to_group": llm_input_items}

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        messages = prompt_template.format_messages(
            format_instructions=parser.get_format_instructions(),
            input=json.dumps(llm_input_payload, ensure_ascii=False)
        )
        
        llm_result = self._invoke_llm_with_retry(messages, parser)

        final_analysis = {"similar_groups": [], "remaining_entities": []}
        
        for group in llm_result.get("similar_groups", []):
            # <<< MODIFIED: Post-processing to enforce group size rule
            if len(group.get("entities", [])) < 2:
                for temp_name in group.get("entities", []):
                    llm_result.setdefault("remaining_entities", []).append(temp_name)
                continue

            final_group = {"canonical_name": group.get("canonical_name"), "entities": []}
            contains_center = False
            
            for temp_name in group.get("entities", []):
                original_info = temp_id_map.get(temp_name)
                if original_info:
                    final_group["entities"].append({"name": original_info["name"], "uid": original_info["uid"]})
                    if original_info["uid"] == center_uid:
                        contains_center = True
            
            if final_group["entities"]:
                if contains_center:
                    for entity in final_group["entities"]:
                        entity["is_primary"] = (entity["uid"] == center_uid)
                else:
                    final_group["entities"][0]["is_primary"] = True
                    for entity in final_group["entities"][1:]:
                        entity["is_primary"] = False
            final_analysis["similar_groups"].append(final_group)

        for temp_name in llm_result.get("remaining_entities", []):
            original_info = temp_id_map.get(temp_name)
            if original_info:
                final_analysis["remaining_entities"].append({
                    "name": original_info["name"],
                    "uid": original_info["uid"],
                    "is_primary": True
                })

        return {"status": "success", "analysis": final_analysis}

    def refine_all_clusters(self, input_file: str) -> Dict[str, Any]:
        """精炼所有聚类"""
        print(f"🚀 开始精炼聚类文件: {input_file}")
        clusters_data = self._load_clusters_data(input_file)
        refined_results = {}

        for entity_type in self.entity_types:
            if entity_type not in clusters_data:
                continue

            print(f"\n📋 开始处理实体类型: {entity_type}")
            refined_results[entity_type] = []
            
            clusters = clusters_data.get(entity_type, {})
            cluster_count = len(clusters)
            
            for i, (cluster_name, cluster_obj) in enumerate(clusters.items()):
                print(f"  -> 正在精炼簇 {i + 1}/{cluster_count} ({cluster_name})...")
                analysis_result = self._refine_one_cluster(entity_type, cluster_obj)
                refined_results[entity_type].append(analysis_result)

        print("\n🎉 聚类精炼完成！")
        return refined_results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存精炼结果"""
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"💾 精炼结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实体聚类精炼器v10")
    parser.add_argument("--config", type=str, default="config/kg_config.json", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default=None, help="分析结果的基准目录，用于确定输入输出位置")
    parser.add_argument("--input", type=str, default="clusters_retrieved.json", help="输入文件名")
    parser.add_argument("--output", type=str, default="clusters_retrieved_refined.json", help="输出文件名")
    
    args = parser.parse_args()
    
    print("🔧 实体聚类精炼器v10")
    print("=" * 50)

    config = EntityClusterRefiner._load_config(args.config)
    
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

    input_file_path = os.path.join(base_dir, args.input)
    output_file_path = os.path.join(base_dir, args.output)

    if not os.path.exists(input_file_path):
        print(f"❌ 错误：输入文件不存在: {input_file_path}")
        return

    refiner = EntityClusterRefiner(config)
    results = refiner.refine_all_clusters(input_file_path)
    refiner.save_results(results, output_file_path)


if __name__ == "__main__":
    main()