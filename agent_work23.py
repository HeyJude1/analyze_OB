#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - LangGraph工作流（agent_work23.py）

说明：
- 基于 agent_work22.py 演进：将“计算流程识别”拆分为四个阶段（prep/transform/core/post）串行执行
- 每个阶段独立提问与保存返回结果；完成四阶段后再进行三层优化策略分析
- 继续支持：断点续跑、按算子/文件限制、失败重试、analysis_only/full 两种模式
"""

import os
import time
import json
import argparse
from typing import Dict, List, Literal, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

from agent23 import AgentFactory, FileManager


load_dotenv()


class WorkState(TypedDict, total=False):
    mode: Literal['analysis_only', 'full']
    report_folder: str
    algorithms: List[str]
    files_per_algorithm: int
    current_algorithm: str
    completed_analysis: List[str]
    completed_individual_summaries: List[str]
    final_summary_done: bool
    errors: List[str]


class Workflow23:
    def __init__(self, files_per_algorithm: int | None = None):
        self.factory = AgentFactory()
        self.file_mgr = FileManager()
        self.files_per_algorithm = files_per_algorithm if (isinstance(files_per_algorithm, int) and files_per_algorithm > 0) else None

        # 构建LangGraph工作流
        self.graph = self._build_graph()

        # 预创建总结器（与 21/22 一致的职责）
        self.individual_summarizer = self.factory.create_individual_summarizer()
        self.final_summarizer = self.factory.create_final_summarizer()

    def _build_graph(self):
        g = StateGraph(WorkState)
        g.add_node("start_analysis", self.start_analysis_node)
        g.add_node("analyzer_work", self.analyzer_work_node)
        g.add_node("individual_summary_work", self.individual_summary_work_node)
        g.add_node("final_summary_work", self.final_summary_work_node)

        g.set_entry_point("start_analysis")
        g.add_conditional_edges(
            "start_analysis", self._route_after_start,
            {"continue": "analyzer_work", END: END},
        )
        g.add_conditional_edges(
            "analyzer_work", self._route_after_analyzer,
            {"continue_summary": "individual_summary_work", "next_algo": "analyzer_work", END: END},
        )
        g.add_conditional_edges(
            "individual_summary_work", self._route_after_individual_summary,
            {"next_algo": "analyzer_work", "do_final_summary": "final_summary_work"},
        )
        g.add_edge("final_summary_work", END)
        return g.compile()

    # --- 路由逻辑 ---
    def _route_after_start(self, state: WorkState) -> str:
        if state.get("algorithms"):
            return "continue"
        return END

    def _route_after_analyzer(self, state: WorkState) -> str:
        completed = state.get("completed_analysis", [])
        if state.get("mode") == 'analysis_only':
            return "next_algo" if len(completed) < len(state.get("algorithms", [])) else END
        return "continue_summary"

    def _route_after_individual_summary(self, state: WorkState) -> str:
        completed = state.get("completed_individual_summaries", [])
        return "next_algo" if len(completed) < len(state.get("algorithms", [])) else "do_final_summary"

    # --- 节点 ---
    def start_analysis_node(self, state: WorkState) -> WorkState:
        print(f"▶️  工作流启动，模式: {state.get('mode')}")
        state["completed_analysis"] = []
        state["completed_individual_summaries"] = []
        state["final_summary_done"] = False

        # 生成 discovery 文件（本地扫描，不依赖 LLM）
        report_folder = state["report_folder"]
        all_algorithms_map = self._scan_and_classify_files_locally()
        discovery_path = self.file_mgr.get_discovery_output_path(report_folder, "all_algorithms")
        final_discovery = {
            "algorithms": list(all_algorithms_map.values()),
            "total_algorithms": len(all_algorithms_map),
            "total_files": sum(len(info["files"]) for info in all_algorithms_map.values()),
        }
        FileManager.save_content(discovery_path, json.dumps(final_discovery, ensure_ascii=False, indent=2))
        print(f"✅ 发现 {len(all_algorithms_map)} 种算子，discovery 已写入。")
        return state

    def analyzer_work_node(self, state: WorkState) -> WorkState:
        completed = state.get("completed_analysis", [])
        algorithms = state.get("algorithms", [])
        idx = len(completed)
        if idx >= len(algorithms):
            return state
        current_algo = algorithms[idx]
        state["current_algorithm"] = current_algo
        report_folder = state["report_folder"]
        print(f"\n🔬 开始分析算子: {current_algo}")

        try:
            # 读取 discovery 获取该算子的文件列表
            discovery_path = self.file_mgr.get_discovery_output_path(report_folder, "all_algorithms")
            with open(discovery_path, 'r', encoding='utf-8') as f:
                all_algos = json.load(f).get("algorithms", [])
            input_files = next((a.get("files", []) for a in all_algos if a.get("algorithm") == current_algo), [])
            if not input_files:
                raise ValueError(f"未在discovery中找到 {current_algo} 的文件列表")

            analysis_path = self.file_mgr.get_analysis_output_path(report_folder, current_algo)
            existing_analyses: List[Dict] = []
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, 'r', encoding='utf-8') as rf:
                        existing = json.load(rf)
                        if isinstance(existing, dict) and isinstance(existing.get("individual_analyses"), list):
                            existing_analyses = existing["individual_analyses"]
                except Exception:
                    existing_analyses = []

            # 跳过已完全分析的文件，支持断点续跑
            processed_names = set()
            for ea in existing_analyses:
                if isinstance(ea, dict):
                    name = ea.get("file_path") or ea.get("file") or ea.get("filename")
                    # 认为计算流程与三层优化都已有则视为处理完成
                    if isinstance(name, str) and name and all(
                        k in ea for k in [
                            "computational_patterns",
                            "algorithm_level_optimizations",
                            "code_level_optimizations",
                            "instruction_level_optimizations",
                        ]
                    ):
                        processed_names.add(name)

            if processed_names:
                input_files = [fi for fi in input_files if fi.get("name") not in processed_names]

            # 限制每个算子文件数量
            if self.files_per_algorithm:
                input_files = input_files[: self.files_per_algorithm]

            # 逐文件串行分析并分阶段增量保存
            for i, file_info in enumerate(input_files, 1):
                file_name = file_info.get("name", "")
                if not file_name:
                    continue
                print(f"  📄 分析文件 {i}/{len(input_files)}: {file_name}")

                # 在工作流中读取文件
                source_code = self._read_source(file_name)

                # 找到或创建该文件的分析条目
                entry = self._find_or_create_entry(existing_analyses, current_algo, file_name)
                
                # 如果该文件已完全分析，跳过
                if entry.get("computational_patterns") and entry.get("algorithm_level_optimizations"):
                    continue
                
                # 使用 analyze_file 进行完整分析（内部会分阶段处理并合并）
                attempts = 0
                while True:
                    try:
                        result = self.factory.analyze_file(
                            source_code=source_code,
                            file_path=file_name,
                            algorithm=current_algo,
                            architecture="通用"
                        )
                        # 更新条目
                        entry.update(result)
                        self._save_analysis(analysis_path, existing_analyses, current_algo, len(input_files))
                        break
                    except Exception as fe:
                        if attempts >= 3:
                            err = f"文件分析失败(重试已达上限): {fe}"
                            print(f"    ❌ {err}")
                            state.setdefault("errors", []).append(err)
                            break
                        wait = [3, 6, 12][attempts] if attempts < 3 else 12
                        print(f"    - 文件分析失败，第 {attempts+1} 次重试前等待 {wait}s: {fe}")
                        time.sleep(wait)
                        attempts += 1

                # 限流保护
                if i < len(input_files):
                    time.sleep(8)

            print(f"  ✅ {current_algo} 分析完成 → {os.path.basename(analysis_path)}")
            state["completed_analysis"].append(current_algo)
        except Exception as e:
            err = f"分析算子 '{current_algo}' 失败: {e}"
            print(f"  ❌ {err}")
            state.setdefault("errors", []).append(err)
            state.setdefault("completed_analysis", []).append(current_algo)
        return state

    def individual_summary_work_node(self, state: WorkState) -> WorkState:
        completed_summaries = state.get("completed_individual_summaries", [])
        algorithms = state.get("algorithms", [])
        idx = len(completed_summaries)
        if idx >= len(algorithms):
            return state
        current_algo = algorithms[idx]
        state["current_algorithm"] = current_algo
        report_folder = state["report_folder"]
        print(f"📝 开始总结算子: {current_algo}")

        try:
            analysis_path = self.file_mgr.get_analysis_output_path(report_folder, current_algo)
            summary_path = self.file_mgr.get_individual_summary_path(report_folder, current_algo)
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)

            first = (analysis_data.get("individual_analyses") or [ {} ])[0]
            current_summary = {
                "algorithm": current_algo,
                "algorithm_level_optimizations": first.get("algorithm_level_optimizations", []),
                "code_level_optimizations": first.get("code_level_optimizations", []),
                "instruction_level_optimizations": first.get("instruction_level_optimizations", []),
            }
            FileManager.save_content(summary_path, json.dumps(current_summary, ensure_ascii=False, indent=2))

            # 迭代整合后续文件结果
            for analysis in (analysis_data.get("individual_analyses", []))[1:]:
                summary_input = (
                    f"增量整合算子 {current_algo} 的优化策略。\n\n"
                    f"已有总结:\n算法层: {json.dumps(current_summary.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"代码层: {json.dumps(current_summary.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"指令层: {json.dumps(current_summary.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}\n\n"
                    f"新增分析:\n算法层: {json.dumps(analysis.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"代码层: {json.dumps(analysis.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"指令层: {json.dumps(analysis.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    "请输出整合后的JSON，仅包含 algorithm_level_optimizations, code_level_optimizations, instruction_level_optimizations。"
                )
                attempts = 0
                while True:
                    try:
                        result = self.individual_summarizer.invoke({"input": summary_input})
                        break
                    except Exception as fe:
                        if attempts >= 3:
                            print(f"  ❌ 个体总结调用失败(重试已达上限): {fe}")
                            result = {"output": "{}"}
                            break
                        wait = [3, 6, 12][attempts] if attempts < 3 else 12
                        print(f"  - 个体总结失败，第 {attempts+1} 次重试前等待 {wait}s: {fe}")
                        time.sleep(wait)
                        attempts += 1
                updated = self._extract_json_from_result(result)
                if isinstance(updated, dict):
                    for key in [
                        "algorithm_level_optimizations",
                        "code_level_optimizations",
                        "instruction_level_optimizations",
                    ]:
                        current_summary[key] = updated.get(key, current_summary.get(key, []))
                    FileManager.save_content(summary_path, json.dumps(current_summary, ensure_ascii=False, indent=2))

            print(f"  ✅ {current_algo} 总结完成 → {os.path.basename(summary_path)}")
            state["completed_individual_summaries"].append(current_algo)
        except Exception as e:
            err = f"总结算子 '{current_algo}' 失败: {e}"
            print(f"  ❌ {err}")
            state.setdefault("errors", []).append(err)
        return state

    def final_summary_work_node(self, state: WorkState) -> WorkState:
        report_folder = state["report_folder"]
        algorithms = state.get("algorithms", [])
        print("\n🔗 汇总所有算子的总结，生成最终报告…")
        try:
            if not algorithms:
                raise ValueError("没有算子可汇总")
            first_summary_path = self.file_mgr.get_individual_summary_path(report_folder, algorithms[0])
            with open(first_summary_path, 'r', encoding='utf-8') as f:
                first_summary = json.load(f)
            current_final = {
                "analyzed_algorithms": [algorithms[0]],
                "algorithm_level_optimizations": first_summary.get("algorithm_level_optimizations", []),
                "code_level_optimizations": first_summary.get("code_level_optimizations", []),
                "instruction_level_optimizations": first_summary.get("instruction_level_optimizations", []),
            }
            final_path = self.file_mgr.get_final_summary_path(report_folder)
            FileManager.save_content(final_path, json.dumps(current_final, ensure_ascii=False, indent=2))

            for algorithm in algorithms[1:]:
                path = self.file_mgr.get_individual_summary_path(report_folder, algorithm)
                if not os.path.exists(path):
                    continue
                with open(path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                prompt = (
                    "跨算子整合优化策略。\n"
                    f"已有:\n算法层: {json.dumps(current_final.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"代码层: {json.dumps(current_final.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"指令层: {json.dumps(current_final.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"新算子 {algorithm}:\n算法层: {json.dumps(summary.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"代码层: {json.dumps(summary.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    f"指令层: {json.dumps(summary.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}\n"
                    "请输出包含 algorithm_level_optimizations, code_level_optimizations, instruction_level_optimizations 的JSON。"
                )
                attempts = 0
                while True:
                    try:
                        result = self.final_summarizer.invoke({"input": prompt})
                        break
                    except Exception as fe:
                        if attempts >= 3:
                            print(f"  ❌ 最终总结调用失败(重试已达上限): {fe}")
                            result = {"output": "{}"}
                            break
                        wait = [3, 6, 12][attempts] if attempts < 3 else 12
                        print(f"  - 最终总结失败，第 {attempts+1} 次重试前等待 {wait}s: {fe}")
                        time.sleep(wait)
                        attempts += 1
                updated = self._extract_json_from_result(result)
                if isinstance(updated, dict):
                    current_final["algorithm_level_optimizations"] = updated.get(
                        "algorithm_level_optimizations", current_final["algorithm_level_optimizations"])
                    current_final["code_level_optimizations"] = updated.get(
                        "code_level_optimizations", current_final["code_level_optimizations"])
                    current_final["instruction_level_optimizations"] = updated.get(
                        "instruction_level_optimizations", current_final["instruction_level_optimizations"])
                    current_final["analyzed_algorithms"].append(algorithm)
                    FileManager.save_content(final_path, json.dumps(current_final, ensure_ascii=False, indent=2))

            print(f"  ✅ 最终总结完成 → {os.path.basename(final_path)}")
            state["final_summary_done"] = True
        except Exception as e:
            err = f"最终总结失败: {e}"
            print(f"  ❌ {err}")
            state.setdefault("errors", []).append(err)
        return state

    # --- 工具 ---
    def _read_source(self, file_path: str, limit: int = 15000) -> str:
        """本地文件读取函数。"""
        try:
            full_path = os.path.join("openblas-output/GENERIC/kernel", file_path)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(limit)
        except Exception as e:
            return f"读取失败: {e}"

    def _find_or_create_entry(self, analyses: List[Dict[str, Any]], algorithm: str, file_path: str) -> Dict[str, Any]:
        for ea in analyses:
            if isinstance(ea, dict) and ea.get("file_path") == file_path:
                return ea
        entry = {
            "algorithm": algorithm,
            "file_path": file_path,
            "architecture": "通用",
            "computational_patterns": [],
            "algorithm_level_optimizations": [],
            "code_level_optimizations": [],
            "instruction_level_optimizations": [],
            "implementation_details": "",
            "performance_insights": "",
        }
        analyses.append(entry)
        return entry

    def _save_analysis(self, analysis_path: str, existing_analyses: List[Dict[str, Any]], algorithm: str, total_files: int):
        payload = {
            "algorithm": algorithm,
            "total_files": total_files,
            "analyzed_files": len(existing_analyses),
            "individual_analyses": existing_analyses,
        }
        FileManager.save_content(analysis_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _extract_json_from_result(self, result):
        def wrap_if_list(obj):
            if isinstance(obj, list):
                return {
                    "algorithm_level_optimizations": obj,
                    "code_level_optimizations": [],
                    "instruction_level_optimizations": [],
                }
            return obj

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if "```json" in output:
                s = output.find("```json") + 7
                e = output.find("```", s)
                parsed = self._parse_json(output[s:e])
                return wrap_if_list(parsed)
            if "```" in output:
                s = output.find("```") + 3
                e = output.find("```", s)
                parsed = self._parse_json(output[s:e])
                return wrap_if_list(parsed)
            parsed = self._parse_json(output)
            return wrap_if_list(parsed)
        elif isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return wrap_if_list(result)
        return None

    @staticmethod
    def _parse_json(text: str):
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def _scan_and_classify_files_locally(self) -> Dict[str, Dict]:
        """直接扫描并分类文件（与 agent_work21/22 保持一致逻辑）。"""
        kernel_path = "openblas-output/GENERIC/kernel"
        if not os.path.exists(kernel_path):
            return {}

        all_files = sorted([f for f in os.listdir(kernel_path) if f.endswith('.c') and 'clean' in f])
        import re

        algorithm_patterns = {
            'axpy': r'.*axpy.*', 'gemm': r'.*gemm.*', 'dot': r'.*(dot|dotu|dotc).*',
            'asum': r'.*asum.*', 'nrm2': r'.*nrm2.*', 'scal': r'.*scal.*', 'copy': r'.*copy.*',
            'swap': r'.*swap.*', 'amax': r'.*amax.*', 'amin': r'.*amin.*', 'ger': r'.*ger.*',
            'gemv': r'.*gemv.*', 'symv': r'.*symv.*', 'hemv': r'.*hemv.*', 'trmm': r'.*trmm.*',
            'trsm': r'.*trsm.*', 'symm': r'.*symm.*', 'hemm': r'.*hemm.*', 'rot': r'.*rot.*',
            'rotm': r'.*rotm.*', 'geadd': r'.*geadd.*', 'imatcopy': r'.*imatcopy.*',
            'omatcopy': r'.*omatcopy.*', 'laswp': r'.*laswp.*', 'max': r'.*max.*',
            'min': r'.*min.*', 'sum': r'.*sum.*', 'neg': r'.*neg.*'
        }

        algorithms: Dict[str, Dict] = {}
        for filename in all_files:
            classified = False
            for algo_name, pattern in algorithm_patterns.items():
                if re.match(pattern, filename, re.IGNORECASE):
                    algorithms.setdefault(algo_name, {"algorithm": algo_name, "files": []})
                    algorithms[algo_name]["files"].append({"name": filename})
                    classified = True
                    break

            if not classified:
                base_name = filename.replace('.clean.c', '')
                if len(base_name) > 1 and base_name[0] in 'sdcz':
                    potential_algo = base_name[1:]
                else:
                    potential_algo = base_name

                potential_algo = re.sub(r'_.*', '', potential_algo)

                if len(potential_algo) > 2:
                    algorithms.setdefault(potential_algo, {"algorithm": potential_algo, "files": []})
                    algorithms[potential_algo]["files"].append({"name": filename})
        return algorithms


def _scan_algorithms_default() -> List[str]:
    wf_for_scan = Workflow23()
    algos_map = wf_for_scan._scan_and_classify_files_locally()
    return sorted(list(algos_map.keys()))


def _save_run_state(report_folder: str, mode: str, algorithms: List[str], files_per_algorithm: Optional[int]):
    """保存运行状态到文件夹，用于续跑。"""
    state_file = os.path.join(report_folder, "run_state.json")
    state = {
        "mode": mode,
        "algorithms": algorithms,
        "files_per_algorithm": files_per_algorithm,
        "created_time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存运行状态失败: {e}")

def _load_run_state(report_folder: str) -> Optional[Dict]:
    """从文件夹加载运行状态。"""
    state_file = os.path.join(report_folder, "run_state.json")
    if not os.path.exists(state_file):
        return None
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载运行状态失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="OpenBLAS优化分析工作流 (agent_work23 - LangGraph)")
    parser.add_argument("mode", choices=['analysis_only', 'full'], help="执行模式")
    parser.add_argument("--algorithms", nargs='+', help="指定要分析的算子列表；未提供则自动扫描内置集合。")
    parser.add_argument("--files-per-algorithm", type=int, help="限制每个算子要分析的文件数量（正整数）。")
    parser.add_argument("--resume", help="恢复指定文件夹的分析（如：results/20251103_173600）")
    args = parser.parse_args()

    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请设置 DASHSCOPE_API_KEY")
        return
    if not os.path.exists("./openblas-output/GENERIC/kernel"):
        print("❌ 错误: 未找到 openblas-output/GENERIC/kernel 目录")
        return

    # 处理续跑逻辑
    if args.resume:
        if not os.path.exists(args.resume):
            print(f"❌ 错误: 指定的恢复路径不存在: {args.resume}")
            return
        report_folder = args.resume
        print(f"📁 恢复分析，使用文件夹: {report_folder}")
        
        # 加载之前的运行状态
        saved_state = _load_run_state(report_folder)
        if saved_state:
            # 使用保存的参数，但允许命令行参数覆盖
            mode = args.mode if hasattr(args, 'mode') and args.mode else saved_state.get("mode", "analysis_only")
            algorithms = args.algorithms if args.algorithms else saved_state.get("algorithms", [])
            files_per_algorithm = args.files_per_algorithm if args.files_per_algorithm else saved_state.get("files_per_algorithm")
            print(f"📋 恢复状态: 模式={mode}, 算子={len(algorithms)}个, 文件限制={files_per_algorithm}")
        else:
            print("⚠️ 未找到运行状态文件，使用当前命令行参数")
            mode = args.mode
            algorithms = args.algorithms or _scan_algorithms_default()
            files_per_algorithm = args.files_per_algorithm
    else:
        # 新建分析
        report_folder = f"results/{time.strftime('%Y%m%d_%H%M%S')}"
        FileManager.ensure_directories(report_folder)
        print(f"📁 报告将保存在: {report_folder}")
        
        mode = args.mode
        algorithms = args.algorithms or _scan_algorithms_default()
        files_per_algorithm = args.files_per_algorithm

    if not algorithms:
        print("❌ 未发现可分析的算子")
        return
    print(f"🎯 将分析以下算子: {', '.join(algorithms)}")

    # 保存/更新运行状态
    _save_run_state(report_folder, mode, algorithms, files_per_algorithm)

    wf = Workflow23(files_per_algorithm=files_per_algorithm)
    initial_state: WorkState = {
        "mode": mode,
        "report_folder": report_folder,
        "algorithms": algorithms,
        "errors": [],
    }
    print("\n🚀 开始执行 LangGraph 工作流...")
    final_state = wf.graph.invoke(initial_state)
    print("\n🏁 工作流执行完毕。")
    if final_state.get("errors"):
        print("\n⚠️ 期间出现错误:")
        for i, err in enumerate(final_state["errors"], 1):
            print(f"  {i}. {err}")


if __name__ == "__main__":
    main()


