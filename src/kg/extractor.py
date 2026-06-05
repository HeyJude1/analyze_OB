#!/usr/bin/env python3
"""
OpenBLAS 知识图谱实体抽取器 (v2)
支持: 优化原则抽象 + 代码特征提取 + 开放模式标签
7 实体类型, 13 关系类型
"""

import os, json, hashlib, time, argparse
from typing import Dict, List, Any, Optional
from pathlib import Path
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
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

DIM = 1024

# ============================================================
# 关系类型
# ============================================================
class Rel:
    OPTIMIZES_PATTERN  = "OPTIMIZES_PATTERN"
    HAS_PARAMETER      = "HAS_PARAMETER"
    IS_ILLUSTRATED_BY  = "IS_ILLUSTRATED_BY"
    TARGETS            = "TARGETS"
    APPLIES_WHEN       = "APPLIES_WHEN"
    REQUIRES           = "REQUIRES"
    GENERALIZES        = "GENERALIZES"
    COMPOSES_WITH      = "COMPOSES_WITH"
    INSTANCE_OF        = "INSTANCE_OF"
    CONFLICTS_WITH     = "CONFLICTS_WITH"
    ANALOGOUS_TO       = "ANALOGOUS_TO"
    HAS_CHARACTERISTIC = "HAS_CHARACTERISTIC"

    @classmethod
    def all(cls): return [v for k, v in vars(cls).items() if not k.startswith("_") and isinstance(v, str)]


# ============================================================
# 向量存储
# ============================================================
class VectorStore:
    def __init__(self, host="localhost", port=19530, database="code_op", dim=1024):
        self.dim = dim
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key: raise RuntimeError("DASHSCOPE_API_KEY required")
        self.embedder = DashScopeEmbeddings(model="text-embedding-v3", dashscope_api_key=api_key)
        try: connections.connect(alias="default", host=host, port=port, db_name=database)
        except Exception: pass
        self._loaded = set()

    def embed(self, text: str) -> List[float]:
        try:
            v = self.embedder.embed_query(text)
            s = sum(x*x for x in v)
            return [x/(s**0.5) for x in v] if s > 0 else v
        except Exception: return [0.0]*self.dim

    def insert(self, col: str, data: List[List]): Collection(col).insert(data)

    def search(self, col: str, vecs: List[List[float]], limit=20, threshold=0.75,
               fields=None) -> List[List]:
        if fields is None: fields = ["uid", "name"]
        self._ensure_loaded(col)
        c = Collection(col)
        results = c.search(data=vecs, anns_field="embedding",
                           param={"metric_type": "COSINE", "params": {"nprobe": 16}},
                           limit=limit, output_fields=fields)
        return [[h for h in hits if h.distance >= threshold] for hits in results]

    def query(self, col: str, expr: str, fields=None, limit=1000, offset=0) -> List[Dict]:
        self._ensure_loaded(col)
        return Collection(col).query(expr=expr, output_fields=fields or ["*"], limit=limit, offset=offset)

    def query_one(self, col: str, uid: str, fields=None) -> Optional[Dict]:
        r = self.query(col, f'uid == "{uid}"', fields=fields, limit=1)
        return r[0] if r else None

    def count(self, col: str) -> int:
        try: return Collection(col).num_entities
        except Exception: return 0

    def build_index(self, col: str):
        try:
            c = Collection(col); c.flush(); n = c.num_entities
            if n == 0: return
            try:
                if c.has_index(): c.drop_index()
            except Exception: pass
            p = {"index_type": "FLAT", "metric_type": "COSINE"} if n < 1000 else \
                {"index_type": "IVF_FLAT", "metric_type": "COSINE",
                 "params": {"nlist": max(128, min(1024, int((n**0.5)*2)))}}
            c.create_index(field_name="embedding", index_params=p); c.load()
        except Exception as e: print(f"  ⚠️ index {col}: {e}")

    def build_all(self, names=None):
        for n in (names or ["optimization_principle","code_characteristic","source_pattern",
                            "architecture_capability","optimization_strategy",
                            "tunable_parameter","code_example","relation"]):
            self.build_index(n)

    def _ensure_loaded(self, col: str):
        if col not in self._loaded:
            try: Collection(col).load(); self._loaded.add(col)
            except Exception: pass

    @staticmethod
    def generate_uid(data: Dict) -> str:
        h = hashlib.md5(); h.update(json.dumps(data, sort_keys=True).encode()); return h.hexdigest()

    @staticmethod
    def parse_json(content: str):
        for fmt in ['```json', '```', '']:
            try:
                if fmt: s=content.find(fmt)+len(fmt); e=content.rfind('```'); return json.loads(content[s:e].strip())
                return json.loads(content.strip())
            except (json.JSONDecodeError,ValueError): continue
        return None


# ============================================================
# 代码特征提取器
# ============================================================
DOF_TO_CHAR = {
    "numeric_kind": {"实数":("numeric_kind","real","N/A","实数"), "复数":("numeric_kind","complex","N/A","复数")},
    "numeric_precision": {"单精度":("precision","single","N/A","单精度"), "双精度":("precision","double","N/A","双精度")},
    "storage_layout": {"连续":("access_pattern","contiguous_unit_stride","N/A","连续访问"),
                       "跨步":("access_pattern","contiguous_strided","inc_x>1","跨步访问"),
                       "跨步 -> 连续":("access_pattern","blocked","跨步→连续","分块打包")},
    "structural_properties": {"对称":("data_dependency","recurrence","对称","对称结构"),
                              "三角":("data_dependency","recurrence","三角","三角结构"),
                              "厄米特":("data_dependency","recurrence","厄米特","厄米特结构")},
}


class CharacteristicExtractor:
    def __init__(self, store: VectorStore, model_cfg: Dict = None):
        self.store = store
        mc = model_cfg or {}
        self.llm = ChatOpenAI(model=mc.get("name","qwen-plus-2025-09-11"),
                              temperature=0.1, max_tokens=4096,
                              api_key=os.getenv("DASHSCOPE_API_KEY"),
                              base_url=mc.get("base_url","https://dashscope.aliyuncs.com/compatible-mode/v1"))

    def from_dof(self, dof: Dict[str,str]) -> List[str]:
        uids = []
        for field, val in dof.items():
            if not val or val == "N/A": continue
            m = DOF_TO_CHAR.get(field,{}); e = m.get(val)
            if not e: continue
            ctype, vdesc, mrange, desc = e
            ent = {"name":f"{ctype}:{vdesc}","characteristic_type":ctype,
                   "value_descriptor":vdesc,"metric_range":mrange,"description":desc}
            uid = VectorStore.generate_uid(ent); ent["uid"] = uid
            self._save(ent); uids.append(uid)
        return uids

    def from_code(self, code: str) -> List[Dict]:
        sp = _prompts.load_system_prompt("kg/kg_v2/characteristic_extraction.yaml")
        prompt = ChatPromptTemplate.from_messages([("system",sp),("human","分析代码:\n```c\n{code}\n```")])
        for attempt in range(3):
            try:
                resp = self.llm.invoke(prompt.format_messages(code=code[:8000]))
                content = resp.content if hasattr(resp,'content') else str(resp)
                r = VectorStore.parse_json(content)
                if r: return [c for c in r if isinstance(c,dict) and "characteristic_type" in c]
            except Exception: pass
            time.sleep(1)
        return []

    def extract_and_save(self, code: str, dof: Dict[str,str]=None) -> List[str]:
        all_uids = list(set(self.from_dof(dof or {})))
        for c in self.from_code(code):
            c.setdefault("name",f"{c.get('characteristic_type','')}:{c.get('value_descriptor','')}")
            uid = VectorStore.generate_uid(c); c["uid"] = uid; self._save(c); all_uids.append(uid)
        return list(set(all_uids))

    def _save(self, ent: Dict):
        uid = ent["uid"]
        if self.store.query_one("code_characteristic", uid): return
        cleaned = {k:v for k,v in ent.items() if k!="uid"}
        emb = self.store.embed(json.dumps(cleaned, ensure_ascii=False, sort_keys=True))
        self.store.insert("code_characteristic", [
            [uid],[ent.get("name","")],[ent.get("characteristic_type","")],
            [ent.get("value_descriptor","")],[ent.get("metric_range","")],
            [ent.get("description","")],[json.dumps(cleaned,ensure_ascii=False)],[emb]
        ])


# ============================================================
# 实体抽取器
# ============================================================
class KnowledgeGraphExtractor:
    def __init__(self, config: Dict, checkpoint_path: str):
        mc = config.get("milvus",{})
        self.store = VectorStore(mc.get("host","localhost"), mc.get("port",19530),
                                 mc.get("database","code_op"),
                                 config.get("dashscope_embeddings",{}).get("dimension",1024))
        self.char_ext = CharacteristicExtractor(self.store, config.get("model",{}))
        self.llm = ChatOpenAI(model=config.get("model",{}).get("name","qwen-plus-2025-09-11"),
                              temperature=0.1, max_tokens=8192,
                              api_key=os.getenv("DASHSCOPE_API_KEY"),
                              base_url=config.get("model",{}).get("base_url","https://dashscope.aliyuncs.com/compatible-mode/v1"))
        self._create_collections()
        self.checkpoint_file = checkpoint_path
        self.processed_files = set(json.load(open(checkpoint_path)).get("processed_files",[])) if os.path.exists(checkpoint_path) else set()
        self.relations = []
        self.code_counter = 1
        print("✅ 抽取器初始化完成")

    def _create_collections(self):
        uid = lambda n=100: FieldSchema(name="uid",dtype=DataType.VARCHAR,max_length=n,is_primary=True)
        emb = lambda: FieldSchema(name="embedding",dtype=DataType.FLOAT_VECTOR,dim=DIM)
        edata = lambda: FieldSchema(name="entity_data",dtype=DataType.VARCHAR,max_length=65535)
        nm = lambda n=500: FieldSchema(name="name",dtype=DataType.VARCHAR,max_length=n)
        ds = lambda n=5000: FieldSchema(name="description",dtype=DataType.VARCHAR,max_length=n)
        schemas = {
            "optimization_principle": [uid(),nm(),FieldSchema(name="principle",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="scope",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="level",dtype=DataType.VARCHAR,max_length=50),
                FieldSchema(name="constraints",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="evidence_strength",dtype=DataType.FLOAT),
                FieldSchema(name="source_count",dtype=DataType.INT64),edata(),emb()],
            "code_characteristic": [uid(),nm(),
                FieldSchema(name="characteristic_type",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="value_descriptor",dtype=DataType.VARCHAR,max_length=500),
                FieldSchema(name="metric_range",dtype=DataType.VARCHAR,max_length=500),ds(),edata(),emb()],
            "source_pattern": [uid(),nm(),
                FieldSchema(name="code_snippet",dtype=DataType.VARCHAR,max_length=10000),
                FieldSchema(name="pattern_type",dtype=DataType.VARCHAR,max_length=100),ds(5000),
                FieldSchema(name="data_object_features",dtype=DataType.VARCHAR,max_length=2000),
                FieldSchema(name="source_algorithm",dtype=DataType.VARCHAR,max_length=500),
                FieldSchema(name="source_file",dtype=DataType.VARCHAR,max_length=500),
                FieldSchema(name="extracted_characteristics",dtype=DataType.VARCHAR,max_length=2000),edata(),emb()],
            "architecture_capability": [uid(),nm(),
                FieldSchema(name="architecture",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="capability_type",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="params",dtype=DataType.VARCHAR,max_length=500),ds(),edata(),emb()],
            "optimization_strategy": [uid(),nm(),
                FieldSchema(name="level",dtype=DataType.VARCHAR,max_length=50),
                FieldSchema(name="rationale",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="implementation",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="impact",dtype=DataType.VARCHAR,max_length=2000),
                FieldSchema(name="trade_offs",dtype=DataType.VARCHAR,max_length=2000),
                FieldSchema(name="related_patterns",dtype=DataType.VARCHAR,max_length=2000),
                FieldSchema(name="principle_links",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="applicability_conditions",dtype=DataType.VARCHAR,max_length=5000),edata(),emb()],
            "tunable_parameter": [uid(),nm(),ds(),FieldSchema(name="impact",dtype=DataType.VARCHAR,max_length=2000),
                FieldSchema(name="value_in_code",dtype=DataType.VARCHAR,max_length=500),
                FieldSchema(name="typical_range",dtype=DataType.VARCHAR,max_length=500),edata(),emb()],
            "code_example": [uid(),nm(),
                FieldSchema(name="snippet",dtype=DataType.VARCHAR,max_length=10000),
                FieldSchema(name="explanation",dtype=DataType.VARCHAR,max_length=5000),
                FieldSchema(name="source_file",dtype=DataType.VARCHAR,max_length=500),edata(),emb()],
            "relation": [FieldSchema(name="relation_id",dtype=DataType.VARCHAR,max_length=100,is_primary=True),
                FieldSchema(name="relation_type",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="head_entity_uid",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="tail_entity_uid",dtype=DataType.VARCHAR,max_length=100),
                FieldSchema(name="head_name",dtype=DataType.VARCHAR,max_length=500),
                FieldSchema(name="tail_name",dtype=DataType.VARCHAR,max_length=500),ds(),emb()],
        }
        for name, fields in schemas.items():
            if not utility.has_collection(name):
                Collection(name, CollectionSchema(fields, f"{name} collection"))
                print(f"  ✓ 创建 {name}")

    def _save_entity(self, col: str, ent: Dict) -> str:
        uid = ent["uid"]
        cleaned = {k:v for k,v in ent.items() if k!="uid"}
        if col == "optimization_principle":
            cleaned.pop("related_patterns",None)
        emb_text = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
        emb = self.store.embed(emb_text)
        schema = Collection(col).schema; field_names = [f.name for f in schema.fields]
        row = []
        for fn in field_names:
            if fn == "uid": row.append([uid])
            elif fn == "embedding": row.append([emb])
            elif fn == "entity_data": row.append([json.dumps(cleaned,ensure_ascii=False)])
            elif fn in ["related_patterns","principle_links","extracted_characteristics","data_object_features","constraints"]:
                v = ent.get(fn,""); row.append([json.dumps(v,ensure_ascii=False) if not isinstance(v,str) else [v]])
            elif fn in ["numeric_kind","numeric_precision","structural_properties","storage_layout"]:
                row.append([ent.get("data_object_features",{}).get(fn,"")])
            elif fn in ["evidence_strength","source_count"]:
                row.append([ent.get(fn, 0.5 if fn=="evidence_strength" else 1)])
            else:
                row.append([str(ent.get(fn,""))[:10000]])
        self.store.insert(col, row)
        return uid

    def _save_relation(self, head, rtype, tail, hname="", tname="", desc=""):
        self.relations.append((head, rtype, tail, hname, tname, desc))

    def _extract_principle(self, strategy: Dict, src_alg: str) -> Optional[Dict]:
        desc = strategy.get("description",{})
        text = json.dumps({"name":strategy.get("optimization_name",""),
            "level":strategy.get("level",""),"rationale":desc.get("strategy_rationale",""),
            "implementation":desc.get("implementation_pattern",""),
            "impact":desc.get("performance_impact",""),"trade_offs":desc.get("trade_offs",""),
            "conditions":strategy.get("applicability_conditions",""),"algorithm":src_alg},ensure_ascii=False)
        sp = _prompts.load_system_prompt("kg/kg_v2/principle_extraction.yaml")
        prompt = ChatPromptTemplate.from_messages([("system",sp),("human","从优化策略中提取抽象原则:\n{input}")])
        for attempt in range(3):
            try:
                resp = self.llm.invoke(prompt.format_messages(input=text[:6000]))
                content = resp.content if hasattr(resp,'content') else str(resp)
                r = VectorStore.parse_json(content)
                if r and isinstance(r,dict) and "principle" in r: return r
            except Exception: pass
            time.sleep(1)
        return None

    def extract_from_file(self, file_path: str):
        if file_path in self.processed_files:
            print(f"⏭️ 跳过: {os.path.basename(file_path)}"); return
        print(f"📄 {os.path.basename(file_path)}")
        with open(file_path) as f: data = json.load(f)
        src_alg = data.get("algorithm","unknown")
        pc = pp = pr = 0
        for ana in data.get("individual_analyses",[]):
            op = ana.get("file_path","").split("/")[-1]
            arch = ana.get("architecture","通用")
            info = {"source_algorithm":src_alg,"source_file":op,"architecture":arch}
            # SourcePatterns
            for pat in ana.get("computational_patterns",[]):
                ent = {"name":pat.get("name",""),"code_snippet":(pat.get("code","") or "")[:10000],
                    "pattern_type":pat.get("pattern_type","") or pat.get("type",""),
                    "description":pat.get("description",""),
                    "data_object_features":pat.get("data_object_features",{}),**info}
                uid = VectorStore.generate_uid(ent)
                if not self.store.query_one("source_pattern",uid):
                    ent["uid"]=uid; self._save_entity("source_pattern",ent); pc+=1
                    cuids = self.char_ext.extract_and_save(pat.get("code",""), pat.get("data_object_features",{}))
                    for cid in cuids: self._save_relation(uid,Rel.HAS_CHARACTERISTIC,cid,pat.get("name",""),"","")
            # Strategies + Principles
            for lv in ["algorithm_level_optimizations","code_level_optimizations","instruction_level_optimizations"]:
                for opt in ana.get(lv,[]):
                    d = opt.get("description",{})
                    # v1 strategy
                    sent = {"name":opt.get("optimization_name",""),"level":opt.get("level",""),
                        "rationale":d.get("strategy_rationale",""),"implementation":d.get("implementation_pattern",""),
                        "impact":d.get("performance_impact",""),"trade_offs":d.get("trade_offs",""),
                        "related_patterns":opt.get("related_patterns",[]),"principle_links":[],
                        "applicability_conditions":opt.get("applicability_conditions",{}),**info}
                    suid = VectorStore.generate_uid(sent)
                    if not self.store.query_one("optimization_strategy",suid):
                        sent["uid"]=suid; self._save_entity("optimization_strategy",sent)
                    # Principle
                    pdata = self._extract_principle(opt, src_alg)
                    if pdata:
                        pent = {"name":f"{pdata.get('scope','')}: {(pdata.get('principle','') or '')[:80]}",
                            "principle":pdata.get("principle",""),"scope":pdata.get("scope",""),
                            "level":pdata.get("level",""),"constraints":pdata.get("constraints",{}),
                            "evidence_strength":0.5,"source_count":1,**info}
                        puid = VectorStore.generate_uid(pent)
                        if not self.store.query_one("optimization_principle",puid):
                            pent["uid"]=puid; self._save_entity("optimization_principle",pent); pp+=1
                            self._save_relation(suid,Rel.INSTANCE_OF,puid,opt.get("optimization_name",""),pdata.get("principle","")[:50],json.dumps({"abstraction_level":pdata.get("abstraction_level",3)}))
                            for g in pdata.get("generalizes",[]):
                                self._save_relation(puid,Rel.GENERALIZES,f"__reserved__:{g}",pdata.get("principle","")[:50],g,json.dumps({"type":"generalization"}))
                            for c in pdata.get("composes_with",[]):
                                self._save_relation(puid,Rel.COMPOSES_WITH,f"__reserved__:{c}",pdata.get("principle","")[:50],c,json.dumps({"type":"composition"}))
                            for hw in pdata.get("constraints",{}).get("hardware_requirements",[]):
                                self._save_relation(puid,Rel.REQUIRES,f"__hw__:{hw}",pdata.get("principle","")[:50],hw,json.dumps({"requirement":hw}))
        print(f"  📊 patterns={pc}, principles={pp}")
        self.processed_files.add(file_path)
        os.makedirs(os.path.dirname(self.checkpoint_file),exist_ok=True)
        json.dump({"processed_files":list(self.processed_files)},open(self.checkpoint_file,'w'),ensure_ascii=False)

    def flush_relations(self):
        if not self.relations: return
        for h,r,t,hn,tn,desc in self.relations:
            rd = {"type":r,"head":h,"tail":t,"desc":desc}
            rid = VectorStore.generate_uid(rd)
            emb = self.store.embed(f"{r} from {hn} to {tn}: {desc}")
            self.store.insert("relation",[[rid],[r],[h],[t],[hn],[tn],[desc],[emb]])
        print(f"💾 {len(self.relations)} 条关系已写入")
        self.relations.clear()

    def extract_from_directory(self, json_dir: str, output_dir: str):
        files = sorted(Path(json_dir).glob("*.json"))
        print(f"📁 {len(files)} 个分析文件")
        for i, fp in enumerate(files, 1):
            print(f"\n{'='*50}\n[{i}/{len(files)}]")
            self.extract_from_file(str(fp))
        self.flush_relations()
        self.store.build_all()
        print(f"\n🎉 完成: {self.store.count('optimization_principle')} 原则, "
              f"{self.store.count('source_pattern')} 模式, {self.store.count('relation')} 关系")

    @staticmethod
    def _load_config(path: str) -> Dict:
        if not os.path.exists(path): return {"milvus":{"host":"localhost","port":19530,"database":"code_op"},"dashscope_embeddings":{"dimension":1024}}
        return json.load(open(path))


def main():
    p = argparse.ArgumentParser(description="KG 实体抽取器")
    p.add_argument("--config",type=str,default="config/kg_config.json")
    p.add_argument("--data_dir",type=str,default=None)
    p.add_argument("--fresh",action="store_true")
    args = p.parse_args()
    config = KnowledgeGraphExtractor._load_config(args.config)
    base = args.data_dir or config.get("data_source",{}).get("analysis_results_dir","")
    if not base: print("❌ 需要 --data_dir"); return
    if not os.path.isabs(base):
        sd = os.path.dirname(os.path.abspath(__file__))
        base = os.path.abspath(os.path.join(os.path.dirname(sd), base))
    jin = os.path.join(base,"analysis_results")
    if not os.path.exists(jin): print(f"❌ {jin} 不存在"); return
    ckpt_dir = os.path.join(base,"checkpoints"); os.makedirs(ckpt_dir,exist_ok=True)
    ckpt = os.path.join(ckpt_dir,"extraction_checkpoint.json")
    if args.fresh and os.path.exists(ckpt): os.remove(ckpt)
    ext = KnowledgeGraphExtractor(config=config, checkpoint_path=ckpt)
    ext.extract_from_directory(jin, base)


if __name__ == "__main__":
    main()
