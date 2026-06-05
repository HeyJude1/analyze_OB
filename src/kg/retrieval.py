#!/usr/bin/env python3
"""
优化策略检索与推荐系统
支持: 四阶段模式检测 + 向量相似度 + 图引擎多跳遍历 + 泛化搜索 + 反馈闭环
"""

import os, json, time, argparse
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from collections import deque
from pymilvus import connections, Collection, utility
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

try:
    from ..utils.prompt_loader import get_prompt_loader
    from .graph import KnowledgeGraph
except ImportError:
    import sys
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, _src_dir)
    from utils.prompt_loader import get_prompt_loader
    from kg.graph import KnowledgeGraph

load_dotenv("config/.env")
_prompts = get_prompt_loader()


# ============================================================
# 图引擎增强
# ============================================================
class Graph:
    """轻量图引擎 (基于 Milvus relation collection)"""
    def __init__(self):
        self._adj_out: Dict[str,List[Tuple[str,str,str]]] = {}
        self._adj_in: Dict[str,List[Tuple[str,str,str]]] = {}
        self._entities: Dict[str,Dict] = {}
        self._loaded = False

    def load(self):
        try:
            col = Collection("relation"); col.load()
            offset = 0
            while True:
                rows = col.query(expr="relation_id != ''",
                    output_fields=["relation_type","head_entity_uid","tail_entity_uid","head_name","tail_name","description"],
                    limit=50000, offset=offset)
                if not rows: break
                for r in rows:
                    h,t,rt = r["head_entity_uid"],r["tail_entity_uid"],r["relation_type"]
                    d = r.get("description","")
                    self._adj_out.setdefault(h,[]).append((rt,t,d))
                    self._adj_in.setdefault(t,[]).append((rt,h,d))
                offset += 50000
            for ec in ["optimization_principle","source_pattern","code_characteristic","optimization_strategy","architecture_capability"]:
                try:
                    c = Collection(ec); c.load(); offset=0
                    while True:
                        rows=c.query(expr="uid != ''",output_fields=["uid","name"],limit=50000,offset=offset)
                        if not rows: break
                        for r in rows: self._entities[r["uid"]]={"name":r.get("name",""),"type":ec}
                        offset+=50000
                except Exception: pass
            self._loaded = True
            print(f"✅ 图谱: {len(self._entities)} 实体, {sum(len(v) for v in self._adj_out.values())} 边")
        except Exception as e: print(f"⚠️ 图谱加载失败: {e}")

    def loaded(self): return self._loaded

    def neighbors(self, uid, direction="both") -> List[Dict]:
        ns = []
        if direction in ("out","both"):
            for rt,t,d in self._adj_out.get(uid,[]):
                ns.append({"direction":"out","relation_type":rt,"target_uid":t,"description":d,
                           "target_name":self._entities.get(t,{}).get("name",""),
                           "target_type":self._entities.get(t,{}).get("type","")})
        if direction in ("in","both"):
            for rt,s,d in self._adj_in.get(uid,[]):
                ns.append({"direction":"in","relation_type":rt,"source_uid":s,"description":d,
                           "source_name":self._entities.get(s,{}).get("name",""),
                           "source_type":self._entities.get(s,{}).get("type","")})
        return ns

    def neighbors_of_type(self, uid, rel_type, direction="out") -> List[Dict]:
        return [n for n in self.neighbors(uid,direction) if n.get("relation_type")==rel_type]

    def traverse_typed(self, start, rel_types, direction="out", max_depth=3) -> List[tuple]:
        visited = {start}; results = []; q = deque([(start,0,[start])])
        while q:
            cur,d,path = q.popleft()
            if d >= max_depth: continue
            for rt,nb,desc in self._adj_out.get(cur,[]):
                if rt in rel_types and nb not in visited:
                    visited.add(nb); np = path+[nb]; results.append((nb,d+1,np)); q.append((nb,d+1,np))
            for rt,nb,desc in self._adj_in.get(cur,[]):
                if rt in rel_types and nb not in visited:
                    visited.add(nb); np = path+[nb]; results.append((nb,d+1,np)); q.append((nb,d+1,np))
        return results

    def has_edge(self, a, b, rel_type=None):
        for rt,t,_ in self._adj_out.get(a,[]):
            if t==b and (rel_type is None or rt==rel_type): return True
        return False

    def co_occurrence_rank(self, strategy_uids, query_patterns) -> List[Tuple[str,float]]:
        scores = {}
        for uid in strategy_uids:
            ns = self.neighbors(uid)
            pm = sum(1 for n in ns if n.get("target_type")=="computational_pattern" or n.get("source_type")=="source_pattern")
            ntypes = len(set(n.get("target_type","") or n.get("source_type","") for n in ns))
            deg = len(self._adj_out.get(uid,[]))+len(self._adj_in.get(uid,[]))
            scores[uid] = pm*2.0 + ntypes*0.5 + min(deg/10.0,1.0)
        return sorted(scores.items(), key=lambda x:x[1], reverse=True)

    def get_context(self, uid) -> Dict:
        ns = self.neighbors(uid)
        pats = [n for n in ns if n.get("target_type")=="source_pattern" or n.get("source_type")=="source_pattern"]
        params = [n for n in ns if n.get("target_type")=="tunable_parameter"]
        hw = [n for n in ns if n.get("target_type")=="architecture_capability"]
        return {"uid":uid,"patterns":pats,"parameters":params,"hardware":hw,"total_connections":len(ns)}


# ============================================================
# 基础工具
# ============================================================
@tool
def read_source_file(file_path: str) -> str:
    try:
        with open(file_path,'r',encoding='utf-8',errors='ignore') as f: return f"文件: {file_path}\n{f.read(15000)}..."
    except Exception as e: return f"读取失败: {e}"


# ============================================================
# 优化策略检索器
# ============================================================
class OptimizationStrategyOperator:
    def __init__(self, config: Dict):
        mc = config.get("milvus",{})
        self.host = mc.get("host","localhost"); self.port = mc.get("port",19530)
        self.db = mc.get("database","code_op")
        ec = config.get("dashscope_embeddings",{})
        self.embedder = DashScopeEmbeddings(model=ec.get("name","text-embedding-v3"),
                                             dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"))
        self.llm = ChatOpenAI(model=config.get("model",{}).get("name","qwen-plus-2025-09-11"),
                              temperature=float(config.get("model",{}).get("temperature",0.0)),
                              max_tokens=int(config.get("model",{}).get("max_tokens",8192)),
                              api_key=os.getenv("DASHSCOPE_API_KEY"),
                              base_url=config.get("model",{}).get("base_url"))
        self.config = config
        self.graph = Graph()
        try:
            connections.connect(alias="default",host=self.host,port=self.port,db_name=self.db)
            self.graph.load()
        except Exception: pass
        print("✅ 检索器初始化完成")

    def _embed(self, text: str) -> List[float]:
        try:
            v = self.embedder.embed_query(text)
            s = sum(x*x for x in v); return [x/(s**0.5) for x in v] if s>0 else v
        except Exception: return [0.0]*1024

    def _search_similar(self, col: str, vecs: List[List], limit=20, thresh=0.8) -> List[List]:
        try:
            c = Collection(col); c.load()
            results = c.search(data=vecs, anns_field="embedding",
                param={"metric_type":"COSINE","params":{"nprobe":16}},
                limit=limit, output_fields=["uid","name","description","pattern_type","code_snippet","level","rationale","implementation","impact","trade_offs"])
            return [[h for h in hits if h.distance>=thresh] for hits in results]
        except Exception: return [[] for _ in vecs]

    # ====== 四阶段模式检测 ======
    def _create_pattern_parser(self):
        schemas = [ResponseSchema(name="computational_patterns", description="计算流程列表")]
        return StructuredOutputParser.from_response_schemas(schemas)

    def _make_agent(self, prompt_path: str) -> AgentExecutor:
        parser = self._create_pattern_parser()
        sp = _prompts.load_system_with_format(prompt_path)
        prompt = ChatPromptTemplate.from_messages([
            ("system", sp), ("human","{input}"), ("placeholder","{agent_scratchpad}")
        ])
        formatted = prompt.partial(format_instructions=parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, [read_source_file], formatted)
        return AgentExecutor(agent=agent, tools=[read_source_file], verbose=False, max_iterations=10)

    def _detect_patterns(self, code: str) -> List[Dict]:
        agents = {
            "stage1_prep.yaml": self._make_agent("kg/pattern_recognition/stage1_prep.yaml"),
            "stage2_transform.yaml": self._make_agent("kg/pattern_recognition/stage2_transform.yaml"),
            "stage3_core.yaml": self._make_agent("kg/pattern_recognition/stage3_core.yaml"),
            "stage4_post.yaml": self._make_agent("kg/pattern_recognition/stage4_post.yaml"),
        }
        all_patterns = []
        for _, agent in agents.items():
            try:
                result = agent.invoke({"input": f"分析以下源码:\n{code[:8000]}"})
                output = result.get("output","")
                data = self._parse_json(output)
                if isinstance(data, list):
                    all_patterns.extend(data)
            except Exception as e: print(f"  ⚠️ 模式检测: {e}")
        return all_patterns

    # ====== 泛化搜索 ======
    def _generalize(self, chars: List[Dict], max_depth=3,
                    alpha=0.5, beta=0.3, gamma=0.2) -> List[Dict]:
        """泛化搜索核心: DIRECT_MATCH → ABSTRACTION_EXPAND → FILTER → RANK"""
        if not self.graph.loaded(): return []

        candidates = {}
        char_set = {c.get("characteristic_type","") for c in chars}

        # A. DIRECT_MATCH
        for c in chars:
            vec = self._embed(json.dumps(c, ensure_ascii=False))
            hits = self._search_similar("code_characteristic", [vec], limit=10, thresh=0.7)
            for h in hits[0]:
                # HAS_CHARACTERISTIC → INSTANCE_OF
                for n in self.graph.neighbors_of_type(h.id, "HAS_CHARACTERISTIC", "in"):
                    puid = n.get("source_uid","")
                    for pn in self.graph.neighbors_of_type(puid, "INSTANCE_OF", "out"):
                        prid = pn.get("target_uid","")
                        if prid not in candidates: candidates[prid]={"uid":prid,"score":0,"paths":[],"chars":[],"dist":0}
                        candidates[prid]["score"]=max(candidates[prid]["score"],h.distance)
                        candidates[prid]["paths"].append("direct:CHAR→INSTANCE_OF")
                        candidates[prid]["chars"].append(c.get("characteristic_type",""))
                # APPLIES_WHEN
                for pn in self.graph.neighbors_of_type(h.id, "APPLIES_WHEN", "in"):
                    prid = pn.get("source_uid","")
                    if prid not in candidates: candidates[prid]={"uid":prid,"score":0,"paths":[],"chars":[],"dist":0}
                    candidates[prid]["score"]=max(candidates[prid]["score"],h.distance)
                    candidates[prid]["paths"].append("direct:APPLIES_WHEN")
                    candidates[prid]["chars"].append(c.get("characteristic_type",""))

        # B. ABSTRACTION_EXPAND
        expanded = dict(candidates)
        for uid in list(candidates.keys()):
            base = candidates[uid]["score"]
            # GENERALIZES 上行 (0.7^d)
            for anc,d,path in self.graph.traverse_typed(uid,["GENERALIZES"],"out",max_depth):
                sc = base*(0.7**d); k=anc
                if k not in expanded or expanded[k]["score"]<sc:
                    expanded[k]={"uid":anc,"score":sc,"paths":candidates[uid]["paths"]+[f"gen:d{d}"],
                                 "chars":candidates[uid]["chars"],"dist":d,"from_gen":True}
            # SPECIALIZATION 下行 (0.85^d)
            for spec,d,path in self.graph.traverse_typed(uid,["GENERALIZES"],"in",max_depth):
                sc = base*(0.85**d); k=spec
                if k not in expanded or expanded[k]["score"]<sc:
                    expanded[k]={"uid":spec,"score":sc,"paths":candidates[uid]["paths"]+[f"spec:d{d}"],
                                 "chars":candidates[uid]["chars"],"dist":d,"from_spec":True}
            # COMPOSES_WITH (0.85)
            for cn in self.graph.neighbors_of_type(uid,"COMPOSES_WITH","both"):
                k = cn.get("target_uid","") or cn.get("source_uid","")
                if k not in expanded or expanded[k]["score"]<base*0.85:
                    expanded[k]={"uid":k,"score":base*0.85,"paths":candidates[uid]["paths"]+["compose"],
                                 "chars":candidates[uid]["chars"],"dist":1,"from_comp":True}
            # ANALOGOUS_TO (0.6)
            for an in self.graph.neighbors_of_type(uid,"ANALOGOUS_TO","both"):
                k = an.get("target_uid","") or an.get("source_uid","")
                if k not in expanded or expanded[k]["score"]<base*0.6:
                    expanded[k]={"uid":k,"score":base*0.6,"paths":candidates[uid]["paths"]+["analogy"],
                                 "chars":candidates[uid]["chars"],"dist":1,"from_analogy":True}
        candidates = expanded

        # C. CONSTRAINT_FILTER
        filtered = {}
        for uid, info in candidates.items():
            try:
                c = Collection("optimization_principle"); c.load()
                rows = c.query(expr=f'uid == "{uid}"', output_fields=["constraints","principle","name","scope","level","evidence_strength"], limit=1)
                if not rows: continue
                p = rows[0]
                try: constraints = json.loads(p.get("constraints","{}"))
                except Exception: constraints = {}
                status, reasons = "pass", []
                for req in self.graph.neighbors_of_type(uid,"REQUIRES","out"):
                    hw = req.get("target_name","")
                    if hw:
                        status="degraded"; reasons.append(f"missing_hw:{hw}")
                for app in self.graph.neighbors_of_type(uid,"APPLIES_WHEN","out"):
                    try: cond = json.loads(app.get("description","{}"))
                    except Exception: cond = {}
                    cn = app.get("target_name","")
                    if cond.get("condition_type")=="requires" and cn not in char_set:
                        status="fail"; reasons.append(f"missing:{cn}"); break
                    if cond.get("condition_type")=="conflicts" and cn in char_set:
                        status="fail"; reasons.append(f"conflict:{cn}"); break
                if status=="fail": continue
                if status=="degraded": info["score"]*=0.5
                info["constraint_status"]=status; info["constraint_reasons"]=reasons
                info["principle"]=p.get("principle",""); info["name"]=p.get("name","")
                info["scope"]=p.get("scope",""); info["level"]=p.get("level","")
                info["evidence"]=float(p.get("evidence_strength",0.5))
                filtered[uid]=info
            except Exception: continue

        # D. RANK
        ranked = []
        for uid, info in filtered.items():
            gs = 1.0/(1.0+info.get("dist",0))
            fs = alpha*info["score"] + beta*gs + gamma*info.get("evidence",0.5)
            t = "direct"
            if info.get("from_gen"): t="generalization"
            elif info.get("from_spec"): t="specialization"
            elif info.get("from_comp"): t="composition"
            elif info.get("from_analogy"): t="analogy"
            ranked.append({"principle_uid":uid,"principle_name":info.get("name",""),
                "principle_text":info.get("principle",""),"scope":info.get("scope",""),
                "level":info.get("level",""),"final_score":round(fs,4),
                "score_breakdown":{"char_sim":round(info["score"],3),"graph_prox":round(gs,3),"evidence":round(info.get("evidence",0.5),3)},
                "paths":info.get("paths",[]),"generalization_type":t,
                "is_generalized":t!="direct","constraint_status":info.get("constraint_status","pass"),
                "graph_context":self.graph.get_context(uid)})
        ranked.sort(key=lambda x:x["final_score"], reverse=True)
        return ranked

    # ====== 主入口 ======
    def process_source_code(self, source_file: str) -> Dict:
        print(f"🚀 处理: {source_file}")
        if not os.path.exists(source_file): return {"error":f"不存在: {source_file}"}
        with open(source_file) as f: code = f.read()

        # 1. 四阶段模式检测
        patterns = self._detect_patterns(code)
        print(f"✅ 步骤1: {len(patterns)} 个计算流程")

        # 2. 向量相似度检索
        pattern_types = [p.get("pattern_type","") for p in patterns]
        similar = []
        for p in patterns:
            vec = self._embed(json.dumps(p, ensure_ascii=False))
            hits = self._search_similar("source_pattern", [vec], 10, 0.8)
            similar.extend(hits[0])
        similar_dedup = {}
        for h in similar:
            if h.id not in similar_dedup or h.distance > similar_dedup[h.id].distance:
                similar_dedup[h.id] = h
        print(f"✅ 步骤2: {len(similar_dedup)} 个相似模式")

        # 3. 关联网格查找策略UID
        strategy_uids = set()
        for uid in similar_dedup:
            for n in self.graph.neighbors_of_type(uid, "INSTANCE_OF", "out"):
                strategy_uids.add(n.get("target_uid",""))
        print(f"✅ 步骤3: {len(strategy_uids)} 个关联策略")

        # 4. 图引擎增强评分
        graph_scores = {}
        if self.graph.loaded() and strategy_uids:
            ranked = self.graph.co_occurrence_rank(list(strategy_uids), set(pattern_types))
            graph_scores = {uid:s for uid,s in ranked}
        print(f"✅ 步骤4: 图引擎增强 ({len(graph_scores)} 个评分)")

        # 5. 泛化搜索 (新!)
        chars_from_llm = self._extract_characteristics(code)
        gen_recs = self._generalize(chars_from_llm)
        print(f"✅ 步骤5: 泛化搜索 → {len(gen_recs)} 个推荐 ({sum(1 for r in gen_recs if r['is_generalized'])} 泛化)")

        # 6. 汇总
        final = []
        seen = set()
        for uid in strategy_uids:
            try:
                c = Collection("optimization_strategy"); c.load()
                rows = c.query(expr=f'uid == "{uid}"', output_fields=["name","level","rationale","implementation","impact"], limit=1)
                if rows:
                    s = rows[0]; s["uid"]=uid; s["score"]=graph_scores.get(uid,0)
                    s["source"]="direct"; final.append(s); seen.add(uid)
            except Exception: pass
        for r in gen_recs:
            if r["principle_uid"] not in seen:
                final.append({"uid":r["principle_uid"],"name":r.get("principle_name",""),
                    "principle_text":r.get("principle_text",""),"score":r["final_score"],
                    "source":f"generalized:{r['generalization_type']}",
                    "generalization_type":r["generalization_type"],"paths":r.get("paths",[]),
                    "graph_context":r.get("graph_context",{})})
        final.sort(key=lambda x: x.get("score",0), reverse=True)

        result = {
            "source_file": source_file,
            "patterns_detected": patterns,
            "similar_patterns": len(similar_dedup),
            "final_strategies": final,
            "direct_matches": sum(1 for s in final if s.get("source")=="direct"),
            "generalized_matches": sum(1 for s in final if s.get("source","").startswith("generalized")),
        }

        # 7. 反馈记录
        self._record_feedback(code, chars_from_llm, final)

        return result

    def _extract_characteristics(self, code: str) -> List[Dict]:
        """从代码中提取抽象特征 (LLM + 规则兜底)"""
        from .extractor import CharacteristicExtractor
        # 内存store复用现有embedder
        ce = CharacteristicExtractor.__new__(CharacteristicExtractor)
        # 简化: 使用规则兜底
        chars = []
        clean = '\n'.join(l for l in code.split('\n') if l.strip() and not l.strip().startswith('//'))
        nest = clean.count('for')
        if nest >= 3: chars.append({"characteristic_type":"compute_intensity","value_descriptor":"high","metric_range":"ops_per_byte>8"})
        elif nest == 2: chars.append({"characteristic_type":"compute_intensity","value_descriptor":"medium","metric_range":"ops_per_byte_2-8"})
        else: chars.append({"characteristic_type":"compute_intensity","value_descriptor":"low","metric_range":"ops_per_byte<2"})
        if any(kw in clean for kw in ['+=','-=','max','min']):
            chars.append({"characteristic_type":"data_dependency","value_descriptor":"reduction"})
        if 'inc_x' in clean or 'stride' in clean.lower():
            chars.append({"characteristic_type":"access_pattern","value_descriptor":"contiguous_strided"})
        else:
            chars.append({"characteristic_type":"access_pattern","value_descriptor":"contiguous_unit_stride"})
        chars.append({"characteristic_type":"loop_structure","value_descriptor":f"depth_{nest}","metric_range":f"nesting={nest}"})
        return chars

    def _record_feedback(self, code: str, chars: List[Dict], results: List[Dict]):
        try:
            log_file = "output/feedback_log.jsonl"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            event = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "code_hash": str(hash(code))[:16],
                     "extracted_chars": [c.get("characteristic_type","") for c in chars],
                     "recommended_uids": [r.get("uid","") for r in results[:10]],
                     "recommended_scores": [r.get("score",0) for r in results[:10]]}
            with open(log_file, 'a') as f: f.write(json.dumps(event, ensure_ascii=False)+'\n')
        except Exception: pass

    def save_results(self, results: Dict, output_file: str):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file,'w') as f: json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 {output_file}")

    @staticmethod
    def _load_config(path: str) -> Dict:
        if not os.path.exists(path):
            return {"milvus":{"host":"localhost","port":19530,"database":"code_op"},
                    "model":{"name":"qwen-max","temperature":0.0,"max_tokens":8192,
                             "base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"},
                    "dashscope_embeddings":{"name":"text-embedding-v3"}}
        return json.load(open(path))

    @staticmethod
    def _parse_json(content: str):
        for fmt in ['```json','```','']:
            try:
                if fmt: s=content.find(fmt)+len(fmt); e=content.rfind('```'); return json.loads(content[s:e].strip())
                return json.loads(content.strip())
            except (json.JSONDecodeError,ValueError): continue
        return None


# ============================================================
# 命令行入口
# ============================================================
def process_single_file(operator, source_file, base_dir):
    if not os.path.exists(source_file): print(f"❌ {source_file}"); return False
    name = os.path.splitext(os.path.basename(source_file))[0]
    out_dir = os.path.join(base_dir, name); os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{name}.json")
    try:
        results = operator.process_source_code(source_file)
        operator.save_results(results, out_file)
        return True
    except Exception as e: print(f"❌ {name}: {e}"); return False

def process_batch_files(operator, openblas_dir, base_dir):
    files = sorted([os.path.join(openblas_dir,f) for f in os.listdir(openblas_dir) if f.endswith('.c')])
    print(f"📋 {len(files)} 个文件"); ok = 0
    for i, f in enumerate(files,1):
        print(f"\n[{i}/{len(files)}]")
        if process_single_file(operator, f, base_dir): ok += 1
    print(f"\n🎉 {ok}/{len(files)} 成功")

def main():
    p = argparse.ArgumentParser(description="优化策略检索与推荐 (含泛化)")
    p.add_argument("--config",type=str,default="config/kg_config.json")
    p.add_argument("--source",type=str,help="源代码文件")
    p.add_argument("--batch",action="store_true")
    p.add_argument("--openblas_dir",type=str)
    p.add_argument("--output_dir",type=str)
    args = p.parse_args()
    config = OptimizationStrategyOperator._load_config(args.config)
    operator = OptimizationStrategyOperator(config=config)
    base = args.output_dir or config.get("optimization_results",{}).get("output_dir","output/op_results")
    if args.batch and args.openblas_dir:
        process_batch_files(operator, args.openblas_dir, base)
    elif args.source:
        process_single_file(operator, args.source, base)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
