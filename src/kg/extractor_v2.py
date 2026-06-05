#!/usr/bin/env python3
"""
KG v2 实体抽取器
从 LLM 分析结果中提取:
  - SourcePattern (开放标签, 不限15种)
  - CodeCharacteristic (抽象代码特征)
  - OptimizationPrinciple (可泛化的优化原则)
  - 13 种关系类型
"""

import os, json, time, argparse
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config/.env")

from .vector_store import VectorStore
from .characteristic_extractor import CharacteristicExtractor
from .schemas import RelType

try:
    from ..utils.prompt_loader import get_prompt_loader
except ImportError:
    import sys
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader

_prompts = get_prompt_loader()


class ExtractorV2:
    """KG v2 实体抽取器"""

    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        mc = config.get("milvus", {})
        ec = config.get("dashscope_embeddings", {})
        self.store = VectorStore(
            host=mc.get("host", "localhost"), port=mc.get("port", 19530),
            database=mc.get("database", "code_op"),
            dimension=ec.get("dimension", 1024)
        )
        model_cfg = config.get("model", {})
        self.char_extractor = CharacteristicExtractor(self.store, model_cfg)

        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(
            model=model_cfg.get("name", "qwen-plus-2025-09-11"),
            temperature=0.1, max_tokens=8192,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=model_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )

        self.checkpoint_file = checkpoint_path
        self.processed_files = self._load_checkpoint()
        self.relation_buffer = []  # (head, rtype, tail, head_name, tail_name, desc)
        self.code_counter = 1
        print("✅ ExtractorV2 初始化完成")

    def _load_checkpoint(self) -> set:
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file) as f:
                return set(json.load(f).get("processed_files", []))
        return set()

    def _save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump({"processed_files": list(self.processed_files)}, f, ensure_ascii=False)

    def _extract_principle(self, strategy: Dict[str, Any], source_algorithm: str) -> Optional[Dict]:
        """LLM 从具体优化策略中提取抽象原则"""
        desc = strategy.get("description", {})
        analysis_text = json.dumps({
            "optimization_name": strategy.get("optimization_name", ""),
            "level": strategy.get("level", ""),
            "rationale": desc.get("strategy_rationale", ""),
            "implementation": desc.get("implementation_pattern", ""),
            "impact": desc.get("performance_impact", ""),
            "trade_offs": desc.get("trade_offs", ""),
            "conditions": strategy.get("applicability_conditions", ""),
            "algorithm": source_algorithm
        }, ensure_ascii=False)

        system_prompt = _prompts.load_system_prompt("kg/kg_v2/principle_extraction.yaml")
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "请从以下优化策略中提取抽象原则:\n\n{input}")
        ])

        for attempt in range(3):
            try:
                messages = prompt.format_messages(input=analysis_text[:6000])
                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                result = self._parse_json(content)
                if result and isinstance(result, dict) and "principle" in result:
                    return result
            except Exception as e:
                if attempt == 2: print(f"   ⚠️ 原则提取失败: {e}")
            time.sleep(1)
        return None

    def _save_principle(self, principle_data: Dict[str, Any], source_info: Dict) -> str:
        entity = {
            "name": f"{principle_data.get('scope','')}: {principle_data.get('principle','')[:80]}",
            "principle": principle_data.get("principle", ""),
            "scope": principle_data.get("scope", ""),
            "level": principle_data.get("level", ""),
            "constraints": json.dumps(principle_data.get("constraints", {}), ensure_ascii=False),
            "evidence_strength": 0.5,
            "source_count": 1,
            **source_info
        }
        uid = VectorStore.generate_uid(entity)
        entity["uid"] = uid

        existing = self.store.query_by_uid("optimization_principle", uid)
        if existing:
            # 更新 source_count
            current_count = existing.get("source_count", 1)
            self._update_principle_field(uid, "source_count", current_count + 1)
            return uid

        cleaned = {k: v for k, v in entity.items() if k != "uid"}
        embed_text = json.dumps({k: v for k, v in cleaned.items()
                                  if k != "constraints"}, ensure_ascii=False, sort_keys=True)
        embedding = self.store.embed(embed_text)
        self.store.insert("optimization_principle", [
            [uid], [entity["name"]], [entity["principle"]], [entity["scope"]],
            [entity["level"]], [entity["constraints"]], [0.5], [1],
            [json.dumps(cleaned, ensure_ascii=False)], [embedding]
        ])
        return uid

    def _update_principle_field(self, uid: str, field: str, value):
        """简单更新 - 通过 delete + re-insert 实现"""
        pass

    def _save_source_pattern(self, pattern: Dict[str, Any], source_info: Dict) -> str:
        entity = {
            "name": pattern.get("name", ""),
            "code_snippet": pattern.get("code", "")[:10000],
            "pattern_type": pattern.get("pattern_type", "") or pattern.get("type", ""),
            "description": pattern.get("description", ""),
            "data_object_features": json.dumps(pattern.get("data_object_features", {}), ensure_ascii=False),
            "source_algorithm": source_info.get("source_algorithm", ""),
            "source_file": source_info.get("source_file", ""),
            "extracted_characteristics": json.dumps([])
        }
        uid = VectorStore.generate_uid(entity)
        entity["uid"] = uid

        existing = self.store.query_by_uid("source_pattern", uid)
        if existing: return uid

        cleaned = {k: v for k, v in entity.items() if k != "uid"}
        embed_text = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        embedding = self.store.embed(embed_text)
        self.store.insert("source_pattern", [
            [uid], [entity["name"]], [entity["code_snippet"]], [entity["pattern_type"]],
            [entity["description"]], [entity["data_object_features"]],
            [entity["source_algorithm"]], [entity["source_file"]],
            [entity["extracted_characteristics"]], [json.dumps(cleaned, ensure_ascii=False)], [embedding]
        ])
        return uid

    def _save_relation(self, head: str, rtype: str, tail: str,
                       hname: str = "", tname: str = "", desc: str = ""):
        self.relation_buffer.append((head, rtype, tail, hname, tname, desc))

    def extract_from_file(self, file_path: str):
        if file_path in self.processed_files:
            print(f"⏭️ 跳过: {os.path.basename(file_path)}")
            return

        print(f"📄 处理: {os.path.basename(file_path)}")
        with open(file_path) as f:
            data = json.load(f)

        source_algorithm = data.get("algorithm", "unknown")
        principle_count = 0
        pattern_count = 0
        relation_count = 0

        for analysis in data.get("individual_analyses", []):
            operator_name = analysis.get("file_path", "").split("/")[-1]
            architecture = analysis.get("architecture", "通用")
            source_info = {"source_algorithm": source_algorithm,
                           "source_file": operator_name, "architecture": architecture}

            # 1. 提取 SourcePattern (开放标签)
            for pattern in analysis.get("computational_patterns", []):
                pattern_uid = self._save_source_pattern(pattern, source_info)
                pattern_count += 1

                # 提取 CodeCharacteristics
                dof = pattern.get("data_object_features", {})
                char_uids = self.char_extractor.extract_and_save(
                    pattern.get("code", ""), pattern_uid, dof
                )
                for cuid in char_uids:
                    self._save_relation(pattern_uid, RelType.HAS_CHARACTERISTIC, cuid,
                                        pattern.get("name", ""), "", "")
                    relation_count += 1

            # 2. 提取 OptimizationPrinciples
            for level_key in ["algorithm_level_optimizations", "code_level_optimizations",
                              "instruction_level_optimizations"]:
                for opt in analysis.get(level_key, []):
                    # 传统策略实体 (向后兼容)
                    self._save_strategy_v1(opt, source_info)

                    # 提取抽象原则
                    principle_data = self._extract_principle(opt, source_algorithm)
                    if not principle_data:
                        continue

                    principle_uid = self._save_principle(principle_data, source_info)
                    principle_count += 1

                    # 关系: strategy → INSTANCE_OF → principle
                    strategy_name = opt.get("optimization_name", "")
                    self._save_relation(f"strategy:{strategy_name}", RelType.INSTANCE_OF,
                                        principle_uid, strategy_name, principle_data.get("principle", "")[:50],
                                        json.dumps({"level": principle_data.get("abstraction_level", 3)}))
                    relation_count += 1

                    # 关系: GENERALIZES (原则的抽象层级)
                    for gen_name in principle_data.get("generalizes", []):
                        self._save_relation(principle_uid, RelType.GENERALIZES,
                                            f"__to_be_resolved__:{gen_name}",
                                            principle_data.get("principle", "")[:50], gen_name,
                                            json.dumps({"type": "generalization"}))
                        relation_count += 1

                    # 关系: COMPOSES_WITH
                    for comp_name in principle_data.get("composes_with", []):
                        self._save_relation(principle_uid, RelType.COMPOSES_WITH,
                                            f"__to_be_resolved__:{comp_name}",
                                            principle_data.get("principle", "")[:50], comp_name,
                                            json.dumps({"type": "composition"}))
                        relation_count += 1

                    # 关系: REQUIRES (硬件要求)
                    constraints = principle_data.get("constraints", {})
                    for hw_req in constraints.get("hardware_requirements", []):
                        self._save_relation(principle_uid, RelType.REQUIRES,
                                            f"__hw__:{hw_req}",
                                            principle_data.get("principle", "")[:50], hw_req,
                                            json.dumps({"requirement": hw_req}))
                        relation_count += 1

            print(f"  📊 模式={pattern_count}, 原则={principle_count}, 关系={relation_count}")

        self.processed_files.add(file_path)
        self._save_checkpoint()

    def _save_strategy_v1(self, opt: Dict[str, Any], source_info: Dict):
        """保持 v1 的 optimization_strategy 实体兼容"""
        desc = opt.get("description", {})
        entity = {
            "name": opt.get("optimization_name", ""),
            "level": opt.get("level", ""),
            "rationale": desc.get("strategy_rationale", ""),
            "implementation": desc.get("implementation_pattern", ""),
            "impact": desc.get("performance_impact", ""),
            "trade_offs": desc.get("trade_offs", ""),
            "related_patterns": json.dumps(opt.get("related_patterns", [])),
            "principle_links": json.dumps([]),
            "applicability_conditions": json.dumps(opt.get("applicability_conditions", {})),
            **source_info
        }
        uid = VectorStore.generate_uid(entity)
        entity["uid"] = uid
        if self.store.query_by_uid("optimization_strategy", uid):
            return uid
        cleaned = {k: v for k, v in entity.items() if k != "uid"}
        embed_text = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        embedding = self.store.embed(embed_text)
        self.store.insert("optimization_strategy", [
            [uid], [entity["name"]], [entity["level"]], [entity["rationale"]],
            [entity["implementation"]], [entity["impact"]], [entity["trade_offs"]],
            [entity["related_patterns"]], [entity["principle_links"]],
            [entity["applicability_conditions"]], [json.dumps(cleaned, ensure_ascii=False)], [embedding]
        ])
        return uid

    def flush_relations(self):
        """批量写入所有缓存的 relation"""
        if not self.relation_buffer:
            return
        batch_size = 100
        for i in range(0, len(self.relation_buffer), batch_size):
            batch = self.relation_buffer[i:i + batch_size]
            for head, rtype, tail, hname, tname, desc in batch:
                rel_data = {"type": rtype, "head": head, "tail": tail, "desc": desc}
                rel_uid = VectorStore.generate_uid(rel_data)
                embed_text = f"{rtype} from {hname} to {tname}: {desc}"
                embedding = self.store.embed(embed_text)
                self.store.insert("relation", [
                    [rel_uid], [rtype], [head], [tail], [hname], [tname], [desc], [embedding]
                ])
        print(f"💾 写入 {len(self.relation_buffer)} 条关系")
        self.relation_buffer.clear()

    def extract_from_directory(self, json_input_dir: str, base_output_dir: str):
        json_files = sorted(Path(json_input_dir).glob("*.json"))
        print(f"📁 {len(json_files)} 个分析文件")
        for i, fp in enumerate(json_files, 1):
            print(f"\n{'='*50}\n[{i}/{len(json_files)}]")
            self.extract_from_file(str(fp))
        self.flush_relations()
        self.store.build_all_indexes()
        print(f"\n🎉 完成: {self.store.count('optimization_principle')} 原则, "
              f"{self.store.count('source_pattern')} 模式, "
              f"{self.store.count('relation')} 关系")

    @staticmethod
    def _parse_json(content: str):
        for fmt in ['```json', '```', '']:
            try:
                if fmt:
                    s = content.find(fmt) + len(fmt); e = content.rfind('```')
                    return json.loads(content[s:e].strip())
                return json.loads(content.strip())
            except (json.JSONDecodeError, ValueError): continue
        return None

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {"milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                    "dashscope_embeddings": {"name": "text-embedding-v3", "dimension": 1024}}
        with open(config_path) as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="KG v2 实体抽取器")
    parser.add_argument("--config", type=str, default="config/kg_config.json")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    config = ExtractorV2._load_config(args.config)
    base_dir = args.data_dir or config.get("data_source", {}).get("analysis_results_dir")
    if not base_dir:
        print("❌ 需要 --data_dir 或 config 中的 analysis_results_dir")
        return
    if not os.path.isabs(base_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(os.path.dirname(script_dir), base_dir))

    json_input_dir = os.path.join(base_dir, "analysis_results")
    if not os.path.exists(json_input_dir):
        print(f"❌ {json_input_dir} 不存在")
        return

    checkpoints_dir = os.path.join(base_dir, "checkpoints_v2")
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoints_dir, "extraction_checkpoint.json")
    if args.fresh and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    extractor = ExtractorV2(config=config, checkpoint_path=checkpoint_file)
    extractor.extract_from_directory(json_input_dir=json_input_dir, base_output_dir=base_dir)


if __name__ == "__main__":
    main()
