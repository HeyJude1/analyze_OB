"""
代码特征提取器
从源代码中提取与算子无关的抽象特征 (compute_intensity, access_pattern, ...)
支撑跨模式优化策略泛化
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv("config/.env")

try:
    from ..utils.prompt_loader import get_prompt_loader
except ImportError:
    import sys
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader

from .vector_store import VectorStore
from .schemas import RelType

_prompts = get_prompt_loader()

# 从 data_object_features 到抽象特征的映射规则（不依赖LLM）
DOF_TO_CHAR = {
    "numeric_kind": {
        "实数": ("numeric_kind", "real", "N/A", "数据类型为实数"),
        "复数": ("numeric_kind", "complex", "N/A", "数据类型为复数"),
    },
    "numeric_precision": {
        "单精度": ("precision", "single", "N/A", "单精度浮点"),
        "双精度": ("precision", "double", "N/A", "双精度浮点"),
    },
    "storage_layout": {
        "连续": ("access_pattern", "contiguous_unit_stride", "N/A", "连续内存访问"),
        "跨步": ("access_pattern", "contiguous_strided", "inc_x > 1", "跨步内存访问"),
        "跨步 -> 连续": ("access_pattern", "blocked", "跨步→连续转换", "分块打包访问"),
    },
    "structural_properties": {
        "对称": ("data_dependency", "recurrence", "对称结构", "对称矩阵结构"),
        "三角": ("data_dependency", "recurrence", "三角结构", "三角矩阵结构"),
        "厄米特": ("data_dependency", "recurrence", "厄米特结构", "厄米特矩阵结构"),
    },
}


class CharacteristicExtractor:
    """从代码中提取抽象特征"""

    def __init__(self, store: VectorStore, model_config: Dict[str, Any] = None):
        self.store = store
        mc = model_config or {}
        self.llm = ChatOpenAI(
            model=mc.get("name", "qwen-plus-2025-09-11"),
            temperature=0.1, max_tokens=4096,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=mc.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )

    def from_data_object_features(self, dof: Dict[str, str], source_uid: str) -> List[str]:
        """从已有的 data_object_features 映射到 CodeCharacteristic UIDs（零LLM成本）"""
        char_uids = []
        for field, value in dof.items():
            if not value or value == "N/A":
                continue
            mapping = DOF_TO_CHAR.get(field, {})
            entry = mapping.get(value)
            if not entry:
                continue
            ctype, vdesc, mrange, desc = entry
            char_entity = {
                "name": f"{ctype}:{vdesc}", "characteristic_type": ctype,
                "value_descriptor": vdesc, "metric_range": mrange,
                "description": desc, "source": "data_object_features"
            }
            uid = VectorStore.generate_uid(char_entity)
            char_entity["uid"] = uid
            self._save_characteristic(char_entity)
            char_uids.append(uid)
        return char_uids

    def from_code(self, code_text: str) -> List[Dict[str, Any]]:
        """LLM 从原始代码中提取抽象特征"""

        system_prompt = _prompts.load_system_prompt("kg/kg_v2/characteristic_extraction.yaml")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "请分析以下代码并提取抽象特征:\n\n```c\n{code}\n```")
        ])
        messages = prompt.format_messages(code=code_text[:8000])

        for attempt in range(3):
            try:
                response = self.llm.invoke(messages)
                content = response.content if hasattr(response, 'content') else str(response)
                chars = self._parse_json(content)
                if chars:
                    return [c for c in chars if isinstance(c, dict) and "characteristic_type" in c]
            except Exception as e:
                if attempt == 2: print(f"⚠️ LLM特征提取失败: {e}")
            time.sleep(1)
        return []

    def extract_and_save(self, code_text: str, pattern_uid: str,
                         dof: Dict[str, str] = None) -> List[str]:
        """综合特征提取并保存"""
        all_chars = []

        # 1. 从 data_object_features 映射
        if dof:
            dof_uids = self.from_data_object_features(dof, pattern_uid)
            all_chars.extend(dof_uids)

        # 2. LLM 提取
        llm_chars = self.from_code(code_text)
        for c in llm_chars:
            c.setdefault("name", f"{c['characteristic_type']}:{c.get('value_descriptor','')}")
            uid = VectorStore.generate_uid(c)
            c["uid"] = uid
            self._save_characteristic(c)
            all_chars.append(uid)

        return list(set(all_chars))

    def _save_characteristic(self, entity: Dict[str, Any]):
        uid = entity["uid"]
        existing = self.store.query_by_uid("code_characteristic", uid)
        if existing:
            return
        cleaned = {k: v for k, v in entity.items() if k != "uid"}
        embed_text = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        embedding = self.store.embed(embed_text)
        insert_data = [
            [uid], [entity.get("name", "")], [entity.get("characteristic_type", "")],
            [entity.get("value_descriptor", "")], [entity.get("metric_range", "")],
            [entity.get("description", "")], [json.dumps(cleaned, ensure_ascii=False)],
            [embedding]
        ]
        self.store.insert("code_characteristic", insert_data)

    @staticmethod
    def _parse_json(content: str) -> Optional[List]:
        for fmt in ['```json', '```', '']:
            try:
                if fmt:
                    start = content.find(fmt) + len(fmt)
                    end = content.rfind('```')
                    return json.loads(content[start:end].strip())
                return json.loads(content.strip())
            except (json.JSONDecodeError, ValueError):
                continue
        return None
