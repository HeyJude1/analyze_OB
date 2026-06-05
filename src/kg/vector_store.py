"""
统一向量存储操作
从 extractor.py 中提取通用逻辑: embed, normalize, insert, search, index
"""

import os
import json
from typing import Dict, List, Any, Optional
from pymilvus import connections, Collection, utility
from langchain_community.embeddings import DashScopeEmbeddings


class VectorStore:
    def __init__(self, host: str = "localhost", port: int = 19530, database: str = "code_op",
                 embedding_model: str = "text-embedding-v3", dimension: int = 1024,
                 connection_alias: str = "default"):
        self.host = host; self.port = port; self.database = database
        self.dimension = dimension; self.connection_alias = connection_alias
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY required")
        self.embedder = DashScopeEmbeddings(model=embedding_model, dashscope_api_key=api_key)
        self._connect()
        self._collections_loaded = set()

    def _connect(self):
        try:
            connections.connect(alias=self.connection_alias, host=self.host, port=self.port, db_name=self.database)
        except Exception:
            pass

    def embed(self, text: str) -> List[float]:
        try:
            v = self.embedder.embed_query(text)
            return self._normalize(v)
        except Exception:
            return [0.0] * self.dimension

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            vecs = self.embedder.embed_documents(texts)
            return [self._normalize(v) for v in vecs]
        except Exception:
            return [[0.0] * self.dimension] * len(texts)

    def _normalize(self, vec: List[float]) -> List[float]:
        s = sum(v * v for v in vec)
        return [v / (s ** 0.5) for v in vec] if s > 0 else vec

    def insert(self, collection_name: str, data: List[List]) -> bool:
        try:
            Collection(collection_name).insert(data)
            return True
        except Exception as e:
            print(f"⚠️ insert {collection_name}: {e}")
            return False

    def search(self, collection_name: str, query_vectors: List[List[float]],
               limit: int = 20, threshold: float = 0.75,
               output_fields: Optional[List[str]] = None) -> List[List[Dict]]:
        try:
            col = Collection(collection_name)
            self._ensure_loaded(collection_name)
            if output_fields is None:
                output_fields = ["uid", "name"]
            results = col.search(
                data=query_vectors, anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                limit=limit, output_fields=output_fields
            )
            filtered = []
            for hits in results:
                filtered.append([h for h in hits if h.distance >= threshold])
            return filtered
        except Exception as e:
            print(f"⚠️ search {collection_name}: {e}")
            return [[] for _ in query_vectors]

    def query(self, collection_name: str, expr: str, output_fields: List[str] = None,
              limit: int = 1000, offset: int = 0) -> List[Dict]:
        try:
            col = Collection(collection_name)
            self._ensure_loaded(collection_name)
            if output_fields is None:
                output_fields = ["*"]
            return col.query(expr=expr, output_fields=output_fields, limit=limit, offset=offset)
        except Exception as e:
            print(f"⚠️ query {collection_name}: {e}")
            return []

    def query_by_uid(self, collection_name: str, uid: str, output_fields: List[str] = None) -> Optional[Dict]:
        results = self.query(collection_name, f'uid == "{uid}"', output_fields=output_fields, limit=1)
        return results[0] if results else None

    def count(self, collection_name: str) -> int:
        try:
            return Collection(collection_name).num_entities
        except Exception:
            return 0

    def build_index(self, collection_name: str):
        try:
            col = Collection(collection_name)
            col.flush()
            n = col.num_entities
            if n == 0: return
            try:
                if col.has_index(): col.drop_index()
            except Exception: pass
            if n < 1000:
                params = {"index_type": "FLAT", "metric_type": "COSINE"}
            else:
                nlist = max(128, min(1024, int((n ** 0.5) * 2)))
                params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": nlist}}
            col.create_index(field_name="embedding", index_params=params)
            col.load()
        except Exception as e:
            print(f"⚠️ index {collection_name}: {e}")

    def build_all_indexes(self, collection_names: List[str] = None):
        names = collection_names or ["optimization_principle", "code_characteristic", "source_pattern",
                                      "architecture_capability", "optimization_strategy",
                                      "tunable_parameter", "code_example", "relation"]
        for n in names: self.build_index(n)

    def _ensure_loaded(self, collection_name: str):
        if collection_name not in self._collections_loaded:
            try:
                Collection(collection_name).load()
                self._collections_loaded.add(collection_name)
            except Exception: pass

    @staticmethod
    def generate_uid(data: Dict[str, Any]) -> str:
        import hashlib
        dhash = hashlib.md5()
        dhash.update(json.dumps(data, sort_keys=True).encode('utf-8'))
        return dhash.hexdigest()
