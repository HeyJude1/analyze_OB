#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略检索与评分系统v5 (TypeError修复版)
- 将 agent23 的四阶段计算流程识别逻辑完全集成到本文件中。
- 移除对 agent23.py 的外部依赖。
- 保持相似度检索、关联策略查找和高级评分逻辑不变。
- 输入文件硬编码为同目录下的 gemm.txt。
- 输出文件路径根据配置文件自动确定。
- 修复了因错误调用实例方法导致的TypeError。
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from pymilvus import connections, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
import argparse
from dotenv import load_dotenv

try:
    from ..utils.prompt_loader import get_prompt_loader
except ImportError:
    import sys
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader

load_dotenv("config/.env")

_prompts = get_prompt_loader()


# ===== 基础工具 (为Agent提供) =====
@tool
def read_source_file(file_path: str) -> str:
    """(此工具仅为Agent内部使用) 读取源代码文件。"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(15000)
        return f"文件路径: {file_path}\n内容:\n{content}\n..."
    except Exception as e:
        return f"读取失败: {str(e)}"


class OptimizationStrategyOperator:
    """优化策略操作器"""
    
    # <<< MODIFIED: __init__ now accepts the config dictionary directly
    def __init__(self, config: Dict[str, Any]):
        """初始化操作器"""
        self.config = config
        self.milvus_config = self.config.get("milvus", {})
        self.model_config = self.config.get("model", {})
        self.embedding_config = self.config.get("dashscope_embeddings", {})
        
        self._connect_milvus()
        self._init_llm()
        self._init_embedding_model()
        
        print("✅ 优化策略操作器初始化完成")
    
    # <<< MODIFIED: Changed to a staticmethod
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

    # ===== START: AgentFactory Logic Integration =====
    
    def _create_pattern_parser(self) -> StructuredOutputParser:
        schemas = [
            ResponseSchema(name="computational_patterns", description=(
                "计算流程列表。每项包含: pattern_type(流程类型标签), name(流程中文名称), "
                "description(对流程的简要说明), code(该流程最相关的完整代码片段), "
                "data_object_features(对象，含 numeric_kind, numeric_precision, structural_properties, storage_layout 四键)"
            )),
        ]
        return StructuredOutputParser.from_response_schemas(schemas)

    def create_prep_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _prompts.load_system_with_format("kg/pattern_recognition/stage1_prep.yaml")),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_transform_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _prompts.load_system_with_format("kg/pattern_recognition/stage2_transform.yaml")),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_core_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _prompts.load_system_with_format("kg/pattern_recognition/stage3_core.yaml")),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def create_post_pattern_agent(self) -> AgentExecutor:
        tools = [read_source_file]
        parser = self._create_pattern_parser()
        prompt = ChatPromptTemplate.from_messages([
            ("system", _prompts.load_system_with_format("kg/pattern_recognition/stage4_post.yaml")),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)

    def _extract_json_from_output(self, output: str) -> Optional[Dict]:
        if not output: return None
        try:
            return json.loads(output)
        except json.JSONDecodeError: pass
        if "```json" in output:
            s = output.find("```json") + 7
            e = output.find("```", s)
            if e > s:
                try: return json.loads(output[s:e].strip())
                except json.JSONDecodeError: return None
        if "```" in output:
            s = output.find("```") + 3
            e = output.find("```", s)
            if e > s:
                try: return json.loads(output[s:e].strip())
                except json.JSONDecodeError: return None
        return None

    def _invoke_with_retry(self, agent: AgentExecutor, payload: Dict[str, Any], label: str, retries: int = 3) -> Dict[str, Any]:
        attempt = 0
        delay_seq = [3, 6, 12]
        while True:
            try:
                return agent.invoke(payload)
            except Exception as e:
                if attempt >= retries: raise e
                wait = delay_seq[attempt] if attempt < len(delay_seq) else delay_seq[-1]
                print(f"  - {label} 失败，第 {attempt+1} 次重试前等待 {wait}s：{e}")
                time.sleep(wait)
                attempt += 1

    # ===== END: AgentFactory Logic Integration =====
    
    def _detect_computational_patterns(self, source_code: str) -> List[Dict[str, Any]]:
        """使用集成的Agent按四个阶段检测计算流程模式"""
        all_patterns = []
        stages = ["prep", "transform", "core", "post"]
        
        agent_map = {
            "prep": self.create_prep_pattern_agent(),
            "transform": self.create_transform_pattern_agent(),
            "core": self.create_core_pattern_agent(),
            "post": self.create_post_pattern_agent()
        }
        
        for stage in stages:
            print(f"  -> 正在识别 {stage} 阶段的计算流程...")
            try:
                agent = agent_map[stage]
                stage_input = f"请分析以下源码，识别‘{stage}’阶段的细粒度计算流程。\n\n源码:\n{source_code}"
                result = self._invoke_with_retry(agent, {"input": stage_input}, f"计算流程({stage})")
                output_raw = self._extract_json_from_output(result.get("output", "")) or {}
                
                if isinstance(output_raw, list):
                    patterns = output_raw
                elif isinstance(output_raw, dict):
                    patterns = output_raw.get("computational_patterns", [])
                else:
                    patterns = []

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

        # 归一化搜索向量，配合 COSINE 检索
        def _normalize(v: List[float]) -> List[float]:
            try:
                s = sum(x * x for x in v)
                if s <= 0:
                    return v
                inv = 1.0 / (s ** 0.5)
                return [x * inv for x in v]
            except Exception:
                return v

        embedding_texts = [json.dumps(p, ensure_ascii=False, sort_keys=True) for p in detected_patterns]
        vectors_to_search_raw = self.embedding_model.embed_documents(embedding_texts)
        vectors_to_search = [_normalize(v) for v in vectors_to_search_raw]

        # 使用 COSINE 度量；这里直接把返回的 score(distance 字段)作为相似度使用
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        all_hits = []

        results = collection.search(
            data=vectors_to_search,
            anns_field="embedding",
            param=search_params,
            limit=50,
            output_fields=["uid", "name", "type"]
        )
        
        for i, hits in enumerate(results):
            for rank, hit in enumerate(hits):
                # 直接使用 Milvus 返回的 score（pymilvus 暴露为 distance 字段）
                similarity = float(hit.distance)
                # Top-2 模式（保留注释）：
                # if rank < 2:
                #     all_hits.append({...})
                # 当前采用：相似度阈值模式（>= 0.8）
                if similarity >= 0.8:
                    all_hits.append({
                        "uid": hit.entity.get("uid"),
                        "name": hit.entity.get("name"),
                        "type": hit.entity.get("type"),
                        "similarity": similarity,
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

    def _find_related_strategy_uids(self, top_patterns: List[Dict[str, Any]]) -> List[str]:
        """根据 top_patterns（计算流程）通过关系集合找到关联的优化策略UID"""
        if not top_patterns:
            return []
            
        pattern_uids = [p['uid'] for p in top_patterns]
        relation_col = Collection("relation")
        
        expr = f'head_entity_uid in {json.dumps(pattern_uids)} and relation_type == "OPTIMIZES_PATTERN"'
        relations = relation_col.query(expr, output_fields=["tail_entity_uid"])
        strategy_uids = list({rel['tail_entity_uid'] for rel in relations})
        return strategy_uids
    
    def _load_strategy_context(self, base_dir: Path) -> List[Dict[str, Any]]:
        """加载 relation_refine/optimization_strategy_context_3.json（位于 analysis_results_dir 下）"""
        ctx_path = base_dir / "relation_refine" / "optimization_strategy_context_3.json"
        if not ctx_path.exists():
            print(f"⚠️ 未找到优化策略上下文文件: {ctx_path}")
            return []
        try:
            with open(ctx_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                print(f"⚠️ 优化策略上下文文件结构异常（期望list）: {ctx_path}")
                return []
        except Exception as e:
            print(f"⚠️ 加载优化策略上下文失败: {e}")
            return []
    
    def _score_and_select_final(self, context_by_uid: Dict[str, Dict[str, Any]], detected_pattern_types: List[str], candidate_uids: List[str], w_context: float = 0.5) -> List[Dict[str, Any]]:
        """按给定公式计算得分并筛选（得分>=0.5）为 final_strategies"""
        detected_set = set(detected_pattern_types)
        denom = float(len(detected_pattern_types)) if detected_pattern_types else 1.0
        finals: List[Dict[str, Any]] = []
        
        for uid in candidate_uids:
            entry = context_by_uid.get(uid)
            if not entry:
                continue
            core_patterns = entry.get("core_patterns", []) or []
            contextual_patterns = entry.get("contextual_patterns", {}) or {}
            
            # 约束：core_patterns 必须完全包含于所给代码的计算流程（patterns_detected）
            if core_patterns:
                if not set(core_patterns).issubset(detected_set):
                    continue
            
            # Score_core = len(S_core ∩ P_code) / len(P_code)
            s_core = set(core_patterns) & detected_set
            score_core = (len(s_core) / denom) if denom > 0 else 0.0
            
            # Score_context = sum(freq for matched contextual patterns)
            score_context = 0.0
            for pattern, freq_str in contextual_patterns.items():
                if pattern in detected_set:
                    try:
                        score_context += float(freq_str)
                    except Exception:
                        # 忽略不可解析的频率
                        pass
            
            score_total = score_core + w_context * score_context
            # 最终阈值：大于等于 0.5
            if score_total >= 0.5:
                # 输出条目基于上下文数据，附加 score
                out = {k: v for k, v in entry.items() if k != "members"}
                out["score"] = score_total
                finals.append(out)
        
        finals.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return finals

    def process_source_code(self, source_file: str) -> Dict[str, Any]:
        """处理源代码文件，执行完整的检索和评分流程"""
        print(f"🚀 开始处理源代码: {source_file}")
        
        if not os.path.exists(source_file):
            return {"error": f"源文件不存在: {source_file}"}
        
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        patterns_detected_full = self._detect_computational_patterns(source_code)
        patterns_detected_types = [p['pattern_type'] for p in patterns_detected_full]
        print(f"✅ 步骤1完成: 检测到 {len(patterns_detected_types)} 个计算流程: {patterns_detected_types}")
        
        similar_patterns = self._search_similar_patterns(patterns_detected_full)
        print(f"✅ 步骤2完成: 检索到 {len(similar_patterns)} 个相似计算流程 (相似度 >= 0.8)")

        top_patterns = self._filter_top_patterns(similar_patterns)
        print(f"✅ 步骤3完成: 筛选出 {len(top_patterns)} 个最高分计算流程")

        # 步骤4：通过关系查找关联策略UID，并从优化上下文文件中构建 search_strategies
        related_strategy_uids = self._find_related_strategy_uids(top_patterns)
        print(f"✅ 步骤4完成: 找到 {len(related_strategy_uids)} 个关联的优化策略UID")
        
        # 重新解析 base_dir（与 main 中逻辑保持一致）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_data_source = self.config.get("data_source", {})
        base_dir_str = cfg_data_source.get("analysis_results_dir", "")
        # 与 main 中解析一致
        base_dir = Path(base_dir_str)
        if not base_dir.is_absolute():
            project_root = Path(script_dir).parent
            resolved_path = project_root / base_dir
            if not resolved_path.exists():
                project_folder_name = project_root.name
                if project_folder_name in base_dir_str:
                    try:
                        idx = base_dir_str.index(project_folder_name)
                        suffix = base_dir_str[idx:]
                        root_parent = project_root.parent
                        resolved_path = root_parent / suffix
                    except ValueError:
                        pass
            base_dir = resolved_path.resolve()
        
        context_list = self._load_strategy_context(base_dir)
        context_by_uid = {e.get("strategy_uid"): e for e in context_list if isinstance(e, dict) and e.get("strategy_uid")}
        
        # search_strategies: 从上下文中挑出关联UID的条目，移除 members 字段
        search_strategies = []
        for uid in related_strategy_uids:
            entry = context_by_uid.get(uid)
            if not entry:
                continue
            filtered = {k: v for k, v in entry.items() if k != "members"}
            # 为 search_strategies 计算并添加 score（不进行 core 子集约束，仅评分）
            try:
                detected_set = set(patterns_detected_types)
                denom = float(len(patterns_detected_types)) if patterns_detected_types else 1.0
                core_patterns = entry.get("core_patterns", []) or []
                contextual_patterns = entry.get("contextual_patterns", {}) or {}
                
                s_core = set(core_patterns) & detected_set
                score_core = (len(s_core) / denom) if denom > 0 else 0.0
                
                score_context = 0.0
                for pattern, freq_str in contextual_patterns.items():
                    if pattern in detected_set:
                        try:
                            score_context += float(freq_str)
                        except Exception:
                            pass
                filtered["score"] = score_core + 0.5 * score_context
            except Exception:
                filtered["score"] = 0.0
            search_strategies.append(filtered)
        print(f"✅ 步骤4.1完成: 组装 {len(search_strategies)} 个上下文策略（去除 members）")
        
        # 步骤5：计算得分并筛选 final_strategies
        final_strategies = self._score_and_select_final(context_by_uid, patterns_detected_types, related_strategy_uids, w_context=0.5)
        print(f"✅ 步骤5完成: 最终筛选出 {len(final_strategies)} 个高分策略")

        result = {
            "source_file": source_file,
            "patterns_detected": patterns_detected_full,
            "similar_patterns_found": similar_patterns,
            "top_patterns_per_type": top_patterns,
            "search_strategies": search_strategies,
            "final_strategies": final_strategies
        }
        return result
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存处理结果"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 结果已保存: {output_file}")


def process_single_file(operator, source_file, base_dir):
    """处理单个源文件"""
    if not os.path.exists(source_file):
        print(f"❌ 错误：源文件不存在: {source_file}")
        return False
    
    # 从源文件名提取算子名称
    source_filename = os.path.basename(source_file)
    if source_filename.endswith('.c'):
        operator_name = source_filename[:-2]  # 去掉.c后缀
    elif source_filename.endswith('.txt'):
        operator_name = source_filename[:-4]  # 去掉.txt后缀
    else:
        operator_name = os.path.splitext(source_filename)[0]
    
    # 创建算子专用目录
    operator_dir = os.path.join(base_dir, operator_name)
    os.makedirs(operator_dir, exist_ok=True)
    
    # 输出文件名基于源文件名
    output_file = os.path.join(operator_dir, f"{operator_name}.json")
    
    print(f"🔄 处理算子: {operator_name}")
    print(f"📁 输出目录: {operator_dir}")
    print(f"📄 输出文件: {output_file}")
    
    try:
        results = operator.process_source_code(source_file)
        operator.save_results(results, output_file)
        print(f"✅ 完成: {operator_name}")
        return True
    except Exception as e:
        print(f"❌ 错误: 处理 {operator_name} 时出错: {e}")
        return False

def process_batch_files(operator, openblas_dir, base_dir):
    """批量处理openblas_output目录中的所有.c文件"""
    if not os.path.exists(openblas_dir):
        print(f"❌ 错误：OpenBLAS输出目录不存在: {openblas_dir}")
        return
    
    # 查找所有.c文件
    c_files = []
    for file in os.listdir(openblas_dir):
        if file.endswith('.c'):
            c_files.append(os.path.join(openblas_dir, file))
    
    if not c_files:
        print(f"⚠️ 警告：在 {openblas_dir} 中未找到.c文件")
        return
    
    print(f"📋 找到 {len(c_files)} 个算子文件:")
    for file in c_files:
        print(f"   - {os.path.basename(file)}")
    
    print(f"\n🚀 开始批量处理...")
    
    success_count = 0
    total_count = len(c_files)
    
    for i, source_file in enumerate(c_files, 1):
        print(f"\n[{i}/{total_count}] " + "="*50)
        if process_single_file(operator, source_file, base_dir):
            success_count += 1
    
    print(f"\n🎉 批量处理完成!")
    print(f"📊 处理结果: {success_count}/{total_count} 成功")
    print(f"📁 结果保存在: {base_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化策略检索与评分系统v5")
    parser.add_argument("--config", type=str, default="config/kg_config.json", help="配置文件路径")
    parser.add_argument("--source", type=str, help="源代码文件路径")
    parser.add_argument("--output_dir", type=str, help="输出目录路径（可选）")
    parser.add_argument("--batch", action="store_true", help="批量处理openblas_output目录中的所有.c文件")
    parser.add_argument("--openblas_dir", type=str, help="OpenBLAS输出目录路径（用于批量处理）")
    
    args = parser.parse_args()
    
    print("⚖️ 优化策略检索与评分系统v5")
    print("=" * 50)
    
    # <<< MODIFIED: Pass config dictionary instead of path
    config = OptimizationStrategyOperator._load_config(args.config)
    operator = OptimizationStrategyOperator(config=config)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定输出目录
    if args.output_dir:
        base_dir_str = args.output_dir
    else:
        # 优先使用optimization_results配置
        base_dir_str = config.get("optimization_results", {}).get("output_dir")
        if not base_dir_str:
            # 回退到原来的analysis_results_dir
            base_dir_str = config.get("data_source", {}).get("analysis_results_dir")
        
        if not base_dir_str:
            print("❌ 错误: 未能在 kg_config.json 中找到输出目录配置。")
            return

    base_dir = Path(base_dir_str)
    if not base_dir.is_absolute():
        project_root = Path(script_dir).parent
        resolved_path = project_root / base_dir
        if not resolved_path.exists():
             project_folder_name = project_root.name
             if project_folder_name in base_dir_str:
                 try:
                     idx = base_dir_str.index(project_folder_name)
                     suffix = base_dir_str[idx:]
                     root_parent = project_root.parent
                     resolved_path = root_parent / suffix
                 except ValueError: pass
        base_dir = resolved_path.resolve()

    # 创建输出目录（如果不存在）
    os.makedirs(base_dir, exist_ok=True)

    # 判断是批量处理还是单文件处理
    if args.batch:
        # 批量处理模式
        if args.openblas_dir:
            openblas_dir = args.openblas_dir
        else:
            # 默认使用相对路径
            openblas_dir = os.path.join(script_dir, "..", "Morph", "openblas_output")
        
        print(f"🔄 批量处理模式")
        print(f"📂 OpenBLAS目录: {openblas_dir}")
        print(f"📁 输出目录: {base_dir}")
        
        process_batch_files(operator, openblas_dir, str(base_dir))
        
    else:
        # 单文件处理模式
        if args.source:
            source_file = args.source
        else:
            source_file = os.path.join(script_dir, "gemm.txt")
        
        print(f"🔄 单文件处理模式")
        print(f"📄 源文件: {source_file}")
        print(f"📁 输出目录: {base_dir}")
        
        process_single_file(operator, source_file, str(base_dir))


if __name__ == "__main__":
    main()