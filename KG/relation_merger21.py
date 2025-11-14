#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实体关系合并器v10 (信息持久化修复版)
- 根据 refine 后的聚类结果，合并实体并重定向关系。
- 支持多轮次迭代，通过 --round 参数控制。
- 状态化管理 optimization_strategy 的上下文信息，并在多轮合并中累积。
- 调用大模型融合实体字段内容。
- 修复了在合并实体时 related_patterns 等关键信息丢失的问题。
- 在写入数据库前，对超长的 entity_data 字段进行智能截断。
- 输出文件均带有轮次号。
"""

import os
import json
import hashlib
import csv
from typing import Dict, List, Any
from collections import Counter
from pymilvus import connections, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
import argparse
import time
from dotenv import load_dotenv

load_dotenv()


class EntityRelationMerger:
    """实体关系合并器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化合并器"""
        self.config = config
        self.milvus_config = self.config.get("milvus", {})
        self.entity_alignment_config = self.config.get("entity_alignment", {})
        self.model_config = self.config.get("model", {})
        self.embedding_config = self.config.get("dashscope_embeddings", {})

        self.t_core = self.entity_alignment_config.get("t_core", 0.6)
        
        self._connect_milvus()
        self._init_llm()
        self._init_embedding_model()
        
        print("✅ 实体关系合并器初始化完成")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(config_path):
            return {
                "milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                "entity_alignment": {"t_core": 0.6},
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
        host = self.milvus_config.get("host", "localhost")
        port = self.milvus_config.get("port", 19530)
        database = self.milvus_config.get("database", "code_op")
        
        connections.connect(alias="default", host=host, port=port, db_name=database)
        print(f"✅ 已连接到Milvus: {host}:{port}/{database}")

    def _init_llm(self):
        self.llm = ChatOpenAI(
            model=self.model_config.get("name"),
            temperature=float(self.model_config.get("temperature", 0.0)),
            max_tokens=int(self.model_config.get("max_tokens", 4096)),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=self.model_config.get("base_url"),
        )

    def _init_embedding_model(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is required for embedding model")
        
        self.embedding_model = DashScopeEmbeddings(
            model=self.embedding_config.get("name", "text-embedding-v3"), 
            dashscope_api_key=api_key
        )

    def _load_json_file(self, file_path: str) -> Any:
        if not os.path.exists(file_path):
            print(f"ℹ️ 文件不存在: {file_path}，将返回空结构。")
            if 'context' in os.path.basename(file_path):
                return []
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

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
            return []

    def _get_entity_details(self, uid: str, collection_name: str) -> Dict[str, Any]:
        try:
            collection = Collection(collection_name)
            res = collection.query(f'uid == "{uid}"', output_fields=["*"], limit=1)
            return res[0] if res else {}
        except Exception as e:
            print(f"⚠️ 查询实体 {uid} 失败: {e}")
            return {}

    def _invoke_llm_with_retry(self, messages: List[Any], retries: int = 3) -> str:
        for attempt in range(retries):
            try:
                time.sleep(1) 
                response = self.llm.invoke(messages)
                return response.content
            except Exception as e:
                print(f"  - LLM调用失败 (尝试 {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"    将在 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("  - 达到最大重试次数，内容合并失败。")
                    return ""
        return ""

    def _merge_entity_content_with_llm(self, primary_entity: Dict, member_entity: Dict, entity_type: str) -> Dict[str, Any]:
        print(f"    🧠 调用LLM合并实体内容: {primary_entity['name']} <- {member_entity['name']}")
        
        fields_to_merge = []
        if entity_type == "optimization_strategy":
            fields_to_merge = ["rationale", "implementation", "impact", "trade_offs"]
        elif entity_type == "hardware_feature":
            fields_to_merge = ["description"]
        elif entity_type == "tunable_parameter":
            fields_to_merge = ["description", "impact"]
        
        updated_data = {}
        for field in fields_to_merge:
            primary_content = primary_entity.get(field, "")
            member_content = member_entity.get(field, "")
            
            if not member_content or member_content == primary_content:
                updated_data[field] = primary_content
                continue

            system_prompt = (
                "你是一个文本融合专家。你的任务是概括和融合两个关于同一主题的描述性文本。"
                "请以'主体文本'为基础，吸收'辅助文本'中的新信息或不同视角，生成一段更全面、更通用的新描述。"
                "保持专业、简洁的风格。只输出融合后的文本，不要任何额外的解释或标题。"
            )
            human_prompt = (
                f"请融合以下关于 '{field}' 的描述：\n\n"
                f"主体文本 (以此为准):\n---\n{primary_content}\n---\n\n"
                f"辅助文本 (吸收其中的新信息):\n---\n{member_content}\n---\n\n"
                "融合后的新文本："
            )
            
            messages = [("system", system_prompt), ("human", human_prompt)]
            merged_content = self._invoke_llm_with_retry(messages)
            
            updated_data[field] = merged_content.strip() if merged_content else primary_content

        return updated_data

    def _truncate_entity_data(self, entity_data: Dict[str, Any], max_len: int = 65535) -> Dict[str, Any]:
        """如果序列化后的entity_data超长，则迭代截断最长的文本字段"""
        data_copy = entity_data.copy()
        truncatable_fields = ["rationale", "implementation", "impact", "trade_offs", "description"]
        safe_max_len = max_len - 256

        while len(json.dumps(data_copy, ensure_ascii=False)) > safe_max_len:
            longest_field = ""
            max_field_len = -1
            for field in truncatable_fields:
                if field in data_copy and isinstance(data_copy[field], str) and len(data_copy[field]) > max_field_len:
                    max_field_len = len(data_copy[field])
                    longest_field = field
            
            if not longest_field or max_field_len < 200:
                print(f"  ⚠️ 无法进一步截断 entity_data，可能仍会超长。")
                break

            new_len = int(max_field_len * 0.9)
            print(f"    - 截断字段 '{longest_field}' 从 {max_field_len} 到 {new_len} 以满足长度限制...")
            data_copy[longest_field] = data_copy[longest_field][:new_len] + "..."
        
        return data_copy

    def _save_relation(self, head_uid: str, tail_uid: str, relation_type: str, head_name: str, tail_name: str):
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
        
        Collection("relation").insert([[relation_uid], [relation_type], [head_uid], [tail_uid], [head_name], [tail_name], [description], [embedding]])
        print(f"      + 创建新关系: {relation_type} ({head_name} -> {tail_name})")

    def _redirect_relations(self, primary_entity: Dict, member_uid: str):
        primary_uid = primary_entity['uid']
        primary_name = primary_entity['name']
        print(f"    🔗 重定向关系: {member_uid[:8]} -> {primary_uid[:8]} ({primary_name})")
        relation_col = Collection("relation")
        
        for direction in ["head", "tail"]:
            expr = f'{direction}_entity_uid == "{member_uid}"'
            relations = relation_col.query(expr, output_fields=["*"])
            
            if not relations: continue

            ids_to_delete = [r['relation_id'] for r in relations]
            relation_col.delete(f'relation_id in {json.dumps(ids_to_delete)}')
            
            for rel in relations:
                new_head_uid = primary_uid if direction == "head" else rel["head_entity_uid"]
                new_tail_uid = primary_uid if direction == "tail" else rel["tail_entity_uid"]
                
                new_head_name = primary_name if direction == "head" else rel["head_name"]
                new_tail_name = primary_name if direction == "tail" else rel["tail_name"]

                check_expr = f'head_entity_uid == "{new_head_uid}" and tail_entity_uid == "{new_tail_uid}" and relation_type == "{rel["relation_type"]}"'
                if relation_col.query(check_expr):
                    print(f"      - 关系已存在，跳过: {rel['relation_type']}")
                    continue
                
                self._save_relation(new_head_uid, new_tail_uid, rel["relation_type"], new_head_name, new_tail_name)

    def merge_entities_and_relations(self, input_file: str, base_output_dir: str, round_num: int):
        print(f"🚀 开始第 {round_num} 轮实体和关系合并: {input_file}")
        
        refined_data = self._load_json_file(input_file)
        
        prev_round = round_num - 1
        prev_context_file = os.path.join(base_output_dir, "relation_refine", f"optimization_strategy_context_{prev_round}.json")
        prev_context_data = self._load_json_file(prev_context_file)
        prev_context_map = {item['strategy_uid']: item for item in prev_context_data}

        current_context_list = []

        for entity_type, analyses in refined_data.items():
            if entity_type not in ["hardware_feature", "optimization_strategy", "tunable_parameter"]:
                continue
            
            print(f"\n🔗 开始合并实体类型: {entity_type}")
            
            all_primary_entities = []
            for analysis_item in analyses:
                if analysis_item.get("status") != "success": continue
                analysis = analysis_item.get("analysis", {})
                
                for group in analysis.get("similar_groups", []):
                    primary = next((e for e in group["entities"] if e.get("is_primary")), None)
                    if primary:
                        all_primary_entities.append({"group": group, "primary_info": primary, "is_group": True})
                
                for entity in analysis.get("remaining_entities", []):
                    if entity.get("is_primary"):
                        all_primary_entities.append({"group": [entity], "primary_info": entity, "is_group": False})

            for item in all_primary_entities:
                primary_info = item["primary_info"]
                primary_details = self._get_entity_details(primary_info['uid'], entity_type)
                if not primary_details: continue
                
                members_to_merge = []
                if item["is_group"]:
                    members_to_merge = [m for m in item["group"]["entities"] if not m.get("is_primary")]
                
                if entity_type == "optimization_strategy":
                    cluster_size = 0
                    pattern_counts = Counter()
                    all_member_uids = set()

                    if primary_info['uid'] in prev_context_map:
                        context = prev_context_map[primary_info['uid']]
                        cluster_size = context.get("cluster_size", 1)
                        pattern_counts = Counter(context.get("pattern_counts", {}))
                        all_member_uids.update(context.get("members", []))
                    else:
                        cluster_size = 1
                        primary_entity_data = json.loads(primary_details.get("entity_data", "{}"))
                        pattern_counts = Counter(primary_entity_data.get("related_patterns", []))
                
                for member in members_to_merge:
                    member_details = self._get_entity_details(member['uid'], entity_type)
                    if not member_details: continue
                    
                    updated_fields = self._merge_entity_content_with_llm(primary_details, member_details, entity_type)
                    primary_details.update(updated_fields)
                    
                    self._redirect_relations(primary_details, member['uid'])
                    
                    if entity_type == "optimization_strategy":
                        if member['uid'] in prev_context_map:
                            member_context = prev_context_map[member['uid']]
                            cluster_size += member_context.get("cluster_size", 1)
                            pattern_counts.update(member_context.get("pattern_counts", {}))
                            all_member_uids.update(member_context.get("members", []))
                            all_member_uids.add(member['uid'])
                        else:
                            cluster_size += 1
                            member_entity_data = json.loads(member_details.get("entity_data", "{}"))
                            pattern_counts.update(member_entity_data.get("related_patterns", []))
                            all_member_uids.add(member['uid'])
                    
                    try:
                        Collection(entity_type).delete(f'uid == "{member["uid"]}"')
                        print(f"    🗑️ 已删除被合并实体: {member['uid'][:8]}")
                    except Exception as e:
                        print(f"    ❌ 删除实体 {member['uid']} 失败: {e}")
                
                try:
                    # <<< MODIFIED: Correctly build the new entity_data for re-insertion
                    # 1. Start with the original entity_data from the database
                    new_entity_data = json.loads(primary_details.get("entity_data", "{}"))

                    # 2. Update it with the LLM-merged fields
                    for key, value in primary_details.items():
                        if key in ["rationale", "implementation", "impact", "trade_offs", "description"]:
                            new_entity_data[key] = value
                    
                    if entity_type == "optimization_strategy":
                        core_patterns = [p for p, c in pattern_counts.items() if (c / cluster_size) >= self.t_core]
                        contextual_patterns = {p: f"{c / cluster_size:.6f}" for p, c in pattern_counts.items() if (c / cluster_size) < self.t_core}
                        
                        final_members = list(all_member_uids - {primary_info['uid']})

                        optimization_context = {
                            "cluster_size": cluster_size,
                            "pattern_counts": dict(pattern_counts),
                            "core_patterns": core_patterns,
                            "contextual_patterns": contextual_patterns,
                            "members": final_members
                        }
                        
                        context_to_save = {
                            "strategy_uid": primary_info['uid'],
                            "canonical_name": item["group"].get("canonical_name") if item["is_group"] else primary_info['name'],
                            **optimization_context
                        }
                        current_context_list.append(context_to_save)
                    
                    # 3. Truncate if necessary (this dictionary does NOT contain uid or embedding)
                    truncated_data = self._truncate_entity_data(new_entity_data)
                    
                    # 4. Prepare the final record for insertion
                    record_to_insert = primary_details.copy()
                    record_to_insert["entity_data"] = json.dumps(truncated_data, ensure_ascii=False)

                    # 5. Delete old and insert new
                    Collection(entity_type).delete(f'uid == "{primary_info["uid"]}"')
                    Collection(entity_type).insert([record_to_insert])
                    print(f"  ✅ 主实体已更新: {primary_info['uid'][:8]}")
                
                except Exception as e:
                    print(f"  ❌ 更新主实体 {primary_info['uid']} 失败: {e}")
        
        utility.flush_all()
        
        output_dir = os.path.join(base_output_dir, "relation_refine")
        context_output_file = os.path.join(output_dir, f"optimization_strategy_context_{round_num}.json")
        self._save_json(current_context_list, context_output_file)
        
        self._export_pattern_frequencies_csv(current_context_list, output_dir, round_num)
        self._export_final_relations(output_dir, round_num)
        
        print("\n🎉 实体关系合并完成！")

    def _save_json(self, data: Any, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 数据已保存: {output_file}")

    def _export_pattern_frequencies_csv(self, context_list: List[Dict[str, Any]], output_dir: str, round_num: int):
        if not context_list: return
        csv_data = []
        for context in context_list:
            cluster_size = context.get("cluster_size", 1)
            for pattern, count in context.get("pattern_counts", {}).items():
                frequency = count / cluster_size if cluster_size > 0 else 0
                csv_data.append({
                    "strategy_uid": context["strategy_uid"],
                    "canonical_name": context["canonical_name"],
                    "pattern_type": pattern,
                    "count": count,
                    "cluster_size": cluster_size,
                    "frequency": f"{frequency:.6f}",
                    "is_core": frequency >= self.t_core
                })
        
        if not csv_data: return
        
        output_file = os.path.join(output_dir, f"optimization_strategy_pattern_frequencies_{round_num}.csv")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"📊 已导出模式频率CSV: {output_file}")

    def _export_final_relations(self, output_dir: str, round_num: int):
        print("\n导出最终关系文件中...")
        relation_col = Collection("relation")
        
        primary_key_field = next((f.name for f in relation_col.schema.fields if f.is_primary), "relation_id")
        
        all_relations = relation_col.query(f'{primary_key_field} != ""', output_fields=["*"], limit=16384)
        
        txt_data = [(r['head_name'], r['relation_type'], r['tail_name']) for r in all_relations]
        
        json_data_grouped = {}
        for r in all_relations:
            rel_type = r['relation_type']
            if rel_type not in json_data_grouped:
                json_data_grouped[rel_type] = []
            json_data_grouped[rel_type].append({
                "relation_type": rel_type,
                "relation_id": r['relation_id'],
                "head": {"name": r['head_name'], "uid": r['head_entity_uid']},
                "tail": {"name": r['tail_name'], "uid": r['tail_entity_uid']},
                "description": r.get('description', '')
            })

        txt_path = os.path.join(output_dir, f"relation_refine_{round_num}.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for head, rel_type, tail in txt_data:
                f.write(f"{head}\t{rel_type}\t{tail}\n")
        print(f"✅ 关系文本文件已保存到: {txt_path}")
        
        json_path = os.path.join(output_dir, f"relation_entity_refine_{round_num}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data_grouped, f, ensure_ascii=False, indent=2)
        print(f"✅ 关系JSON文件已保存到: {json_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="实体关系合并器v9")
    parser.add_argument("--config", type=str, default="kg_config.json", help="配置文件路径")
    parser.add_argument("--round", type=int, required=True, help="当前的合并轮次 (例如: 1, 2, ...)")
    
    args = parser.parse_args()
    
    print("🔗 实体关系合并器v9")
    print("=" * 50)
    
    config = EntityRelationMerger._load_config(args.config)
    
    base_dir = config.get("data_source", {}).get("analysis_results_dir")
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

    input_file_path = os.path.join(base_dir, "clusters_retrieved_refined.json")

    if not os.path.exists(input_file_path):
        print(f"❌ 错误：输入文件不存在: {input_file_path}")
        return

    merger = EntityRelationMerger(config)
    merger.merge_entities_and_relations(input_file_path, base_dir, args.round)


if __name__ == "__main__":
    main()