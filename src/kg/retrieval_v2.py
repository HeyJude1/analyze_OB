#!/usr/bin/env python3
"""
KG v2 优化策略检索
集成: 特征提取 → 直接匹配 → 泛化搜索 → 约束过滤 → 排序推荐
"""

import os, json, argparse
from typing import Dict, List, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config/.env")

from .vector_store import VectorStore
from .characteristic_extractor import CharacteristicExtractor
from .generalization_engine import GeneralizationEngine

try:
    from ..utils.prompt_loader import get_prompt_loader
except ImportError:
    import sys
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader


class RetrievalV2:
    """KG v2 检索推荐器"""

    def __init__(self, config: Dict[str, Any]):
        mc = config.get("milvus", {})
        ec = config.get("dashscope_embeddings", {})
        self.store = VectorStore(
            host=mc.get("host", "localhost"), port=mc.get("port", 19530),
            database=mc.get("database", "code_op"),
            dimension=ec.get("dimension", 1024)
        )
        self.char_extractor = CharacteristicExtractor(self.store, config.get("model", {}))
        self.engine = GeneralizationEngine(self.store)
        self.config = config

    def recommend(self, source_file: str,
                  hardware_context: Dict[str, List] = None) -> Dict[str, Any]:
        """对新代码推荐优化策略"""
        if not os.path.exists(source_file):
            return {"error": f"文件不存在: {source_file}"}

        with open(source_file) as f:
            code = f.read()

        print(f"📄 分析: {os.path.basename(source_file)} ({len(code)} 字符)")

        # Step 1: 提取抽象代码特征
        print("  步骤1: 提取代码特征...")
        chars = self.char_extractor.from_code(code)
        print(f"    → {len(chars)} 个抽象特征: {[c.get('characteristic_type','') for c in chars]}")

        if not chars:
            # 回退: 从 data_object_features 映射
            print("  步骤1.1: LLM特征为空, 尝试规则映射...")
            chars = self._fallback_characteristics(code)

        # Step 2: 泛化搜索
        print("  步骤2: 泛化搜索...")
        recommendations = self.engine.search(chars, hardware_context)

        # Step 3: 分类结果
        direct = [r for r in recommendations if not r.get("is_generalized")]
        generalized = [r for r in recommendations if r.get("is_generalized")]

        print(f"    → 直接匹配: {len(direct)}, 泛化推荐: {len(generalized)}")

        return {
            "source_file": source_file,
            "extracted_characteristics": chars,
            "recommendations": recommendations,
            "direct_matches": len(direct),
            "generalized_matches": len(generalized),
            "total": len(recommendations)
        }

    def _fallback_characteristics(self, code: str) -> List[Dict]:
        """规则兜底: 从代码结构推断基本特征"""
        chars = []
        lines = code.split('\n')
        code_clean = '\n'.join(l for l in lines if l.strip() and not l.strip().startswith('//'))

        # 计算强度: 统计算术操作 vs 内存访问
        arith_ops = sum(1 for c in code_clean if c in '+-*/')
        mem_ops = sum(1 for c in code_clean if c in '[]*&')

        if 'for' in code_clean:
            nest_count = code_clean.count('for')
            if nest_count >= 3:
                chars.append({"characteristic_type": "compute_intensity",
                              "value_descriptor": "high", "metric_range": "ops_per_byte > 8",
                              "description": f"code_with_{nest_count}_nested_loops"})
                chars.append({"characteristic_type": "loop_structure",
                              "value_descriptor": f"depth_{nest_count}",
                              "metric_range": f"nesting_depth={nest_count}",
                              "description": f"{nest_count}_nested_loops"})
            elif nest_count == 2:
                chars.append({"characteristic_type": "compute_intensity",
                              "value_descriptor": "medium", "metric_range": "ops_per_byte 2-8",
                              "description": f"code_with_{nest_count}_nested_loops"})
            else:
                chars.append({"characteristic_type": "compute_intensity",
                              "value_descriptor": "low", "metric_range": "ops_per_byte < 2",
                              "description": f"streaming_code_with_{nest_count}_loop"})

        # 归约检测
        if any(kw in code_clean for kw in ['+=', '-=', 'max', 'min']):
            chars.append({"characteristic_type": "data_dependency",
                          "value_descriptor": "reduction",
                          "metric_range": "N/A",
                          "description": "code_contains_accumulation_pattern"})

        # 访问模式
        if 'inc_x' in code or 'inc_y' in code or 'stride' in code.lower():
            chars.append({"characteristic_type": "access_pattern",
                          "value_descriptor": "contiguous_strided",
                          "metric_range": "variable_stride",
                          "description": "code_uses_variable_stride_access"})
        else:
            chars.append({"characteristic_type": "access_pattern",
                          "value_descriptor": "contiguous_unit_stride",
                          "metric_range": "stride=1",
                          "description": "code_uses_unit_stride_access"})

        return chars

    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            return {"milvus": {"host": "localhost", "port": 19530, "database": "code_op"},
                    "dashscope_embeddings": {"name": "text-embedding-v3", "dimension": 1024}}
        with open(config_path) as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="KG v2 优化策略检索 (支持泛化)")
    parser.add_argument("--config", type=str, default="config/kg_config.json")
    parser.add_argument("--source", type=str, required=True, help="待分析代码文件")
    parser.add_argument("--output", type=str, help="输出 JSON 文件路径")
    args = parser.parse_args()

    config = RetrievalV2._load_config(args.config)
    retriever = RetrievalV2(config)
    results = retriever.recommend(args.source)

    output_file = args.output or f"{os.path.splitext(args.source)[0]}_v2_recs.json"
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存: {output_file}")

    # 摘要
    recs = results.get("recommendations", [])
    print(f"\n📊 推荐摘要:")
    print(f"   直接匹配: {results.get('direct_matches', 0)}")
    print(f"   泛化推荐: {results.get('generalized_matches', 0)}")
    for r in recs[:5]:
        tag = f"[{r.get('generalization_type', '?')}]" if r.get("is_generalized") else "[direct]"
        print(f"   {tag} {r['principle_text'][:80]}... (评分: {r['final_score']:.3f})")


if __name__ == "__main__":
    main()
