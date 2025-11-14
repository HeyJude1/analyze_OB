#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - 硬编码工作流编排器
关键改进：路径由代码控制，Agent只生成内容
"""

import os
import time
import json
from typing import Dict, List
from typing_extensions import TypedDict
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from agent import (
    AgentFactory,
    FileManager,
    AnalysisTask
)

load_dotenv()


# ===== 工作流状态 =====
class WorkState(TypedDict):
    """工作流状态"""
    report_folder: str
    algorithms: List[str]
    current_algorithm_index: int
    completed_tasks: List[str]
    errors: List[str]


# ===== 工作流 =====
class Workflow:
    """工作流"""
    
    def __init__(self):
        self.factory = AgentFactory()
        self.file_mgr = FileManager()
        
        # 创建专家Agents
        self.scout = self.factory.create_scout_specialist()
        self.analyzer = self.factory.create_analyzer_specialist()
        self.individual_summarizer = self.factory.create_individual_summarizer()
        self.final_summarizer = self.factory.create_final_summarizer()
        
        # 构建工作流
        self.workflow = self._build_workflow()
    
    
    def _build_workflow(self) -> StateGraph:
        """构建硬编码的顺序工作流"""
        workflow = StateGraph(WorkState)
        
        # 添加节点
        workflow.add_node("orchestrator", self.orchestrator_node)
        workflow.add_node("scout_work", self.scout_work)
        workflow.add_node("analyzer_work", self.analyzer_work)
        workflow.add_node("individual_summary_work", self.individual_summary_work)
        workflow.add_node("final_summary_work", self.final_summary_work)
        
        # 设置入口
        workflow.add_edge(START, "orchestrator")
        
        # 编排器决策路由
        workflow.add_conditional_edges(
            "orchestrator",
            self._orchestrator_route,
            {
                "scout": "scout_work",
                "analyze": "analyzer_work",
                "individual_summary": "individual_summary_work",
                "final_summary": "final_summary_work",
                "complete": END
            }
        )
        
        # 各节点完成后返回编排器
        for node in ["scout_work", "analyzer_work", 
                     "individual_summary_work", "final_summary_work"]:
            workflow.add_edge(node, "orchestrator")
        
        return workflow.compile()
    
    def _orchestrator_route(self, state: WorkState) -> str:
        """编排器决策下一步行动 - 基于已完成任务和算子索引"""
        algorithms = state["algorithms"]
        current_idx = state["current_algorithm_index"]
        completed = state["completed_tasks"]
        
        # 首先检查当前算子是否完成，如果完成则移动到下一个
        if current_idx < len(algorithms):
            current_algo = algorithms[current_idx]
            if (f"scout_{current_algo}" in completed and 
                f"analyze_{current_algo}" in completed and 
                f"individual_summary_{current_algo}" in completed):
                # 当前算子完成，移动到下一个
                print(f"✅ {current_algo} 算子完成！移动到下一个算子...")
                state["current_algorithm_index"] += 1
                current_idx = state["current_algorithm_index"]
        
        # 检查是否所有算子都完成
        if current_idx >= len(algorithms):
            # 检查是否需要final summary
            if "final_summary" not in completed:
                return "final_summary"
            return "complete"
        
        current_algo = algorithms[current_idx]
        
        # 按固定顺序：scout → analyze → individual_summary
        if f"scout_{current_algo}" not in completed:
            return "scout"
        elif f"analyze_{current_algo}" not in completed:
            return "analyze"
        elif f"individual_summary_{current_algo}" not in completed:
            return "individual_summary"
        else:
            # 这个分支不应该到达，因为上面已经处理了算子完成的情况
            return "complete"
    
    def orchestrator_node(self, state: WorkState) -> WorkState:
        """编排器节点 - 显示状态但不做决策"""
        print(f"\n🎯 [Orchestrator] 分析工作流状态...")
        
        algorithms = state["algorithms"]
        current_idx = state["current_algorithm_index"]
        completed = state["completed_tasks"]
        
        if current_idx < len(algorithms):
            current_algo = algorithms[current_idx]
            
            # 显示进度
            total_tasks = len(algorithms) * 3 + 1  # 每个算子3个任务 + 1个final
            print(f"📊 进度: {len(completed)}/{total_tasks} 任务完成")
            print(f"📍 当前算子: {current_algo} ({current_idx + 1}/{len(algorithms)})")
        else:
            print(f"📊 所有算子完成，准备最终总结")
        
        return state
    
    def scout_work(self, state: WorkState) -> WorkState:
        """Scout工作 - 直接扫描kernel目录，按算子种类分组所有文件"""
        report_folder = state["report_folder"]
        
        try:
            print(f"🔍 扫描kernel目录...")
            
            discovery_path = self.file_mgr.get_discovery_output_path(report_folder, "all_algorithms")
            
            # 直接在Python中处理文件扫描，避免大模型调用
            all_algorithms = self._scan_and_classify_files()
            
            # 构建最终结果
            final_discovery = {
                "algorithms": list(all_algorithms.values()),
                "total_algorithms": len(all_algorithms),
                "total_files": sum(len(algo["files"]) for algo in all_algorithms.values()),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 保存发现结果
            success = FileManager.save_content(discovery_path, json.dumps(final_discovery, ensure_ascii=False, indent=2))
            if success:
                # 更新state中的algorithms列表
                algorithm_names = list(all_algorithms.keys())
                state["algorithms"] = algorithm_names
                print(f"✅ 发现 {len(algorithm_names)} 种算子，共 {final_discovery['total_files']} 个文件")
                
                state["completed_tasks"].append("scout_all")
            else:
                print(f"❌ 保存失败")
                state["errors"].append("Scout保存失败")
            
        except Exception as e:
            error_msg = f"Scout失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def _scan_and_classify_files(self) -> Dict[str, Dict]:
        """直接扫描并分类文件"""
        import os
        import re
        
        kernel_path = "openblas-output/GENERIC/kernel"
        if not os.path.exists(kernel_path):
            return {}
        
        # 获取所有.c文件
        all_files = []
        for file in os.listdir(kernel_path):
            if file.endswith('.c') and 'clean' in file:
                all_files.append(file)
        
        all_files.sort()
        print(f"  找到 {len(all_files)} 个文件")
        
        # 算子分类规则
        algorithm_patterns = {
            'axpy': r'.*axpy.*',
            'gemm': r'.*gemm.*',
            'dot': r'.*(dot|dotu|dotc).*',
            'asum': r'.*asum.*',
            'nrm2': r'.*nrm2.*',
            'scal': r'.*scal.*',
            'copy': r'.*copy.*',
            'swap': r'.*swap.*',
            'amax': r'.*amax.*',
            'amin': r'.*amin.*',
            'ger': r'.*ger.*',
            'gemv': r'.*gemv.*',
            'symv': r'.*symv.*',
            'hemv': r'.*hemv.*',
            'trmm': r'.*trmm.*',
            'trsm': r'.*trsm.*',
            'symm': r'.*symm.*',
            'hemm': r'.*hemm.*',
            'rot': r'.*rot.*',
            'rotm': r'.*rotm.*',
            'geadd': r'.*geadd.*',
            'imatcopy': r'.*imatcopy.*',
            'omatcopy': r'.*omatcopy.*',
            'laswp': r'.*laswp.*',
            'max': r'.*max.*',
            'min': r'.*min.*',
            'sum': r'.*sum.*',
            'neg': r'.*neg.*'
        }
        
        # 分类文件
        algorithms = {}
        for filename in all_files:
            classified = False
            for algo_name, pattern in algorithm_patterns.items():
                if re.match(pattern, filename, re.IGNORECASE):
                    if algo_name not in algorithms:
                        algorithms[algo_name] = {"algorithm": algo_name, "files": []}
                    algorithms[algo_name]["files"].append({
                        "name": filename
                    })
                    classified = True
                    break
            
            # 如果没有匹配到已知模式，尝试提取算子名
            if not classified:
                # 简单的启发式：取文件名的第一个单词部分
                base_name = filename.replace('.clean.c', '')
                # 移除前缀字母（如s, d, c, z）
                if len(base_name) > 1 and base_name[0] in 'sdcz':
                    potential_algo = base_name[1:]
                else:
                    potential_algo = base_name
                
                # 进一步清理
                potential_algo = re.sub(r'_.*', '', potential_algo)  # 移除下划线后的部分
                
                if len(potential_algo) > 2:  # 只考虑长度大于2的算子名
                    if potential_algo not in algorithms:
                        algorithms[potential_algo] = {"algorithm": potential_algo, "files": []}
                    algorithms[potential_algo]["files"].append({
                        "name": filename
                    })
        
        return algorithms
    
    def _discover_algorithm_files(self, algorithm: str) -> List[Dict[str, str]]:
        """动态发现算子相关文件"""
        import glob
        import re
        
        base_dir = "openblas-output/GENERIC/kernel"
        if not os.path.exists(base_dir):
            return []
        
        # 搜索模式：算子名相关的文件
        patterns = [
            f"*{algorithm}*.c",
            f"*{algorithm.upper()}*.c",
            f"{algorithm}_*.c",
            f"{algorithm.upper()}_*.c",
            f"*_{algorithm}.c",
            f"*_{algorithm.upper()}.c"
        ]
        
        found_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(base_dir, pattern))
            found_files.extend(files)
        
        # 去重并限制数量（选择前5个最相关的）
        unique_files = list(set(found_files))
        
        # 按相关性排序（文件名包含算子名的优先）
        def relevance_score(filepath):
            filename = os.path.basename(filepath).lower()
            algo_lower = algorithm.lower()
            
            if filename.startswith(algo_lower):
                return 3
            elif algo_lower in filename:
                return 2
            elif algorithm.upper() in os.path.basename(filepath):
                return 1
            else:
                return 0
        
        unique_files.sort(key=relevance_score, reverse=True)
        
        # 选择前5个最相关的文件
        selected_files = unique_files[:5]
        
        # 转换为所需格式
        result = []
        for filepath in selected_files:
            filename = os.path.basename(filepath)
            result.append({
                "path": filename,
                "type": "discovered",
                "description": f"动态发现的{algorithm}相关文件"
            })
        
        return result
    
    def analyzer_work(self, state: WorkState) -> WorkState:
        """Analyzer工作 - 分析算子文件"""
        current_algo = state["algorithms"][state["current_algorithm_index"]]
        report_folder = state["report_folder"]
        
        try:
            # 从all_algorithms discovery文件中获取当前算子的文件列表
            discovery_path = self.file_mgr.get_discovery_output_path(report_folder, "all_algorithms")
            with open(discovery_path, 'r', encoding='utf-8') as f:
                discovery_data = json.load(f)
            
            # 找到当前算子的文件列表
            input_files = []
            if "algorithms" in discovery_data:
                for algo_info in discovery_data["algorithms"]:
                    if algo_info.get("algorithm") == current_algo:
                        input_files = algo_info.get("files", [])
                        break
            
            if not input_files:
                raise ValueError(f"未找到{current_algo}算子的文件列表")
            # 获取分析结果文件路径
            analysis_path = self.file_mgr.get_analysis_output_path(report_folder, current_algo)
            
            # 读取已有的分析结果（如果存在）
            existing_analyses = []
            if os.path.exists(analysis_path):
                try:
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        existing_analyses = existing_data.get("individual_analyses", [])
                except:
                    existing_analyses = []
            
            # 逐个分析每个文件并增量保存
            for i, file_info in enumerate(input_files):
                file_name = file_info.get("name", "")
                if not file_name:
                    continue
                
                print(f"  📄 分析文件 {i+1}/{len(input_files)}: {file_name}")
                
                analyzer_input = f"""分析{current_algo}算子文件: {file_name}

按三层优化策略框架进行结构化分析，每个层次输出多个优化策略。

**分析要求**：
- 完全基于代码内容发现优化策略
- 每个策略必须包含详细的结构化信息
- 提供足够的代码上下文来支撑优化策略的解释

**三层优化框架**：
**算法设计层次**：识别算法层优化策略（如"复数运算展开"、"分块计算"、"预计算优化"等）
**代码优化层次**：识别代码层优化策略（如"循环展开"、"指针递增"、"条件分支优化"等）  
**特有指令层次**：识别指令层优化策略（如"SIMD向量化"、"自动向量化适配"、"内联汇编"等）

**输出格式要求**：
每个策略包含：
- name: 规范化策略名称
- description_details: 包含strategy_rationale、implementation_pattern、performance_impact、trade_offs四个字段
- code_context: 包含snippet（完整代码块）、highlighted_code（核心语句）、explanation三个字段

**策略命名要求**：使用规范简练的技术术语，避免口语化表达

使用read_source_file工具读取文件内容，然后输出结构化的JSON格式分析结果。"""
                
                result = self.analyzer.invoke({"input": analyzer_input})
                time.sleep(2)
                
                file_analysis = self._extract_json_from_result(result)
                existing_analyses.append(file_analysis)
                
                # 每分析一个文件就保存一次（增量保存）
                updated_analysis = {
                    "algorithm": current_algo,
                    "total_files": len(input_files),
                    "analyzed_files": len(existing_analyses),
                    "individual_analyses": existing_analyses,
                    "timestamp": datetime.now().isoformat()
                }
                
                success = FileManager.save_content(analysis_path, json.dumps(updated_analysis, ensure_ascii=False, indent=2))
                if not success:
                    print(f"    ❌ 保存失败: {analysis_path}")
                print(f"    ✅ 增量保存: {os.path.basename(analysis_path)} (已分析 {len(existing_analyses)}/{len(input_files)} 个文件)")
            state["completed_tasks"].append(f"analyze_{current_algo}")
            print(f"✅ Analyzer完成: {analysis_path}")
            
        except Exception as e:
            print(f"❌ Analyzer失败 ({current_algo}): {str(e)}")
            state["errors"].append(str(e))
        
        return state
    
    
    def individual_summary_work(self, state: WorkState) -> WorkState:
        """Individual Summary工作 - 单算子增量整合"""
        current_algo = state["algorithms"][state["current_algorithm_index"]]
        report_folder = state["report_folder"]
        
        try:
            analysis_path = self.file_mgr.get_analysis_output_path(report_folder, current_algo)
            summary_path = self.file_mgr.get_individual_summary_path(report_folder, current_algo)
            
            # 读取分析结果
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            individual_analyses = analysis_data.get("individual_analyses", [])
            if not individual_analyses:
                raise ValueError(f"{current_algo}算子没有分析结果")
            
            # 初始化：将第一个结构化分析结果转换为summary格式
            print(f"  🔄 初始化summary (基于第1个文件)")
            first_analysis = individual_analyses[0]
            
            # 辅助函数：将结构化策略转换为summary格式
            def convert_to_summary_format(strategies):
                summary_strategies = []
                for strategy in strategies:
                    if isinstance(strategy, dict) and "name" in strategy:
                        if "description_details" in strategy:
                            # 从结构化格式提取核心内容
                            details = strategy["description_details"]
                            unified_desc = f"{details.get('strategy_rationale', '')} {details.get('implementation_pattern', '')} {details.get('performance_impact', '')}".strip()
                        else:
                            # 兼容旧格式
                            unified_desc = strategy.get("description", "")
                        
                        summary_strategies.append({
                            "name": strategy["name"],
                            "unified_description": unified_desc
                        })
                return summary_strategies
            
            current_summary = {
                "algorithm": current_algo,
                "algorithm_level_optimizations": convert_to_summary_format(first_analysis.get("algorithm_level_optimizations", [])),
                "code_level_optimizations": convert_to_summary_format(first_analysis.get("code_level_optimizations", [])),
                "instruction_level_optimizations": convert_to_summary_format(first_analysis.get("instruction_level_optimizations", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存初始summary
            FileManager.save_content(summary_path, json.dumps(current_summary, ensure_ascii=False, indent=2))
            print(f"    ✅ 初始summary已保存")
            
            # 逐个整合后续的分析结果
            for i, analysis in enumerate(individual_analyses[1:], start=2):
                print(f"  🔄 整合第{i}个文件的分析结果...")
                
                summary_input = f"""增量整合{current_algo}算子的优化策略：

**已有的优化策略**：
算法设计层次: {json.dumps(current_summary.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}
代码优化层次: {json.dumps(current_summary.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}
特有指令层次: {json.dumps(current_summary.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}

**新的结构化分析结果**：
算法设计层次: {json.dumps(analysis.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}
代码优化层次: {json.dumps(analysis.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}
特有指令层次: {json.dumps(analysis.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}

**任务**：将结构化分析结果整合到已有策略中，提炼相近策略并统一命名。

**结构化数据处理说明**：
- 分析结果中每个策略包含name、description_details和code_context
- 你需要提取description_details的核心内容，形成unified_description
- 合并相似策略时，综合多个description_details的要点

**整合规则**：
- 如果新策略与已有策略相似，合并为统一命名的策略（保持已有名称）
- 如果新策略是全新的，直接添加到策略列表中
- 保持策略名称的规范化和一致性
- 从详细的description_details中提取核心内容，形成简洁的unified_description

**输出格式**：每个层次输出策略列表，格式为：
[{{"name": "统一的策略名称", "unified_description": "合并多个相似策略后的统一描述"}}]

输出JSON格式结果，只包含三个字段：algorithm_level_optimizations, code_level_optimizations, instruction_level_optimizations"""
                
                result = self.individual_summarizer.invoke({"input": summary_input})
                time.sleep(2)
                
                # 提取JSON结果
                updated_summary = self._extract_json_from_result(result)
                
                # 更新current_summary
                current_summary["algorithm_level_optimizations"] = updated_summary.get("algorithm_level_optimizations", current_summary["algorithm_level_optimizations"])
                current_summary["code_level_optimizations"] = updated_summary.get("code_level_optimizations", current_summary["code_level_optimizations"])
                current_summary["instruction_level_optimizations"] = updated_summary.get("instruction_level_optimizations", current_summary["instruction_level_optimizations"])
                current_summary["timestamp"] = datetime.now().isoformat()
                
                # 增量保存
                FileManager.save_content(summary_path, json.dumps(current_summary, ensure_ascii=False, indent=2))
                print(f"    ✅ 已整合并保存 (进度: {i}/{len(individual_analyses)})")
            
            state["completed_tasks"].append(f"individual_summary_{current_algo}")
            print(f"✅ Individual Summary完成: {summary_path}")
            
        except Exception as e:
            print(f"❌ Individual Summary失败 ({current_algo}): {str(e)}")
            state["errors"].append(str(e))
        
        return state
    
    def final_summary_work(self, state: WorkState) -> WorkState:
        """Final Summary工作 - 跨算子增量整合"""
        algorithms = state["algorithms"]
        report_folder = state["report_folder"]
        
        try:
            final_path = self.file_mgr.get_final_summary_path(report_folder)
            
            if not algorithms:
                raise ValueError("没有已完成的算子")
            
            # 初始化：将第一个算子的summary转换为final格式
            print(f"  🔄 初始化final summary (基于第1个算子: {algorithms[0]})")
            first_summary_path = self.file_mgr.get_individual_summary_path(report_folder, algorithms[0])
            with open(first_summary_path, 'r', encoding='utf-8') as f:
                first_summary = json.load(f)
            
            # 辅助函数：将summary格式转换为final格式
            def convert_to_final_format(strategies):
                final_strategies = []
                for strategy in strategies:
                    if isinstance(strategy, dict) and "name" in strategy:
                        universal_desc = strategy.get("unified_description", strategy.get("description", ""))
                        final_strategies.append({
                            "name": strategy["name"],
                            "universal_description": universal_desc
                        })
                return final_strategies
            
            current_final = {
                "analyzed_algorithms": [algorithms[0]],
                "algorithm_level_optimizations": convert_to_final_format(first_summary.get("algorithm_level_optimizations", [])),
                "code_level_optimizations": convert_to_final_format(first_summary.get("code_level_optimizations", [])),
                "instruction_level_optimizations": convert_to_final_format(first_summary.get("instruction_level_optimizations", [])),
                "timestamp": datetime.now().isoformat()
            }
            
            # 保存初始final summary
            FileManager.save_content(final_path, json.dumps(current_final, ensure_ascii=False, indent=2))
            print(f"    ✅ 初始final summary已保存")
            
            # 逐个整合后续算子的summary
            for i, algo in enumerate(algorithms[1:], start=2):
                print(f"  🔄 整合第{i}个算子: {algo}...")
                
                # 读取当前算子的summary
                algo_summary_path = self.file_mgr.get_individual_summary_path(report_folder, algo)
                with open(algo_summary_path, 'r', encoding='utf-8') as f:
                    algo_summary = json.load(f)
                
                final_input = f"""增量整合OpenBLAS优化策略库：

**已有的优化策略库**：
算法设计层次: {json.dumps(current_final.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}
代码优化层次: {json.dumps(current_final.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}
特有指令层次: {json.dumps(current_final.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}

**新算子({algo})的优化策略**：
算法设计层次: {json.dumps(algo_summary.get('algorithm_level_optimizations', []), ensure_ascii=False, indent=2)}
代码优化层次: {json.dumps(algo_summary.get('code_level_optimizations', []), ensure_ascii=False, indent=2)}
特有指令层次: {json.dumps(algo_summary.get('instruction_level_optimizations', []), ensure_ascii=False, indent=2)}

**任务**：将新算子的优化策略整合到已有策略库中，提炼跨算子的通用优化模式并统一命名。

**跨算子整合说明**：
- 输入的算子策略格式：name（统一名称）和unified_description（统一描述）
- 输出的策略库格式：name（通用策略名称）和universal_description（通用描述和应用场景）
- 重点识别在多个算子中都出现的优化模式
- 提炼适用于整个OpenBLAS库的通用优化策略

**整合规则**：
- 如果新策略与已有策略相似，合并为统一命名的策略（保持已有名称）
- 如果新策略是全新的，直接添加到策略列表中
- 保持策略名称的规范化和一致性
- 从多个算子的共性中提炼通用的优化规律

**策略命名规范**：
- 使用标准技术术语（如"SIMD向量化"、"分块计算"、"并行计算"）
- 避免口语化表达
- 保持命名简练准确

**输出格式**：每个层次输出策略列表，格式为：
[{{"name": "通用的策略名称", "universal_description": "通用的策略描述和跨算子应用场景"}}]

输出JSON格式结果，只包含三个字段：algorithm_level_optimizations, code_level_optimizations, instruction_level_optimizations"""
                
                result = self.final_summarizer.invoke({"input": final_input})
                time.sleep(2)
                
                # 提取JSON结果
                updated_final = self._extract_json_from_result(result)
                
                # 更新current_final
                current_final["algorithm_level_optimizations"] = updated_final.get("algorithm_level_optimizations", current_final["algorithm_level_optimizations"])
                current_final["code_level_optimizations"] = updated_final.get("code_level_optimizations", current_final["code_level_optimizations"])
                current_final["instruction_level_optimizations"] = updated_final.get("instruction_level_optimizations", current_final["instruction_level_optimizations"])
                current_final["analyzed_algorithms"].append(algo)
                current_final["timestamp"] = datetime.now().isoformat()
                
                # 增量保存
                FileManager.save_content(final_path, json.dumps(current_final, ensure_ascii=False, indent=2))
                print(f"    ✅ 已整合并保存 (进度: {i}/{len(algorithms)})")
            
            state["completed_tasks"].append("final_summary")
            print(f"✅ Final Summary完成: {final_path}")
            
        except Exception as e:
            print(f"❌ Final Summary失败: {str(e)}")
            state["errors"].append(str(e))
        
        return state
    
    def _extract_json_from_result(self, result):
        """从Agent结果中提取JSON"""
        if isinstance(result, dict) and "output" in result:
            output_content = result["output"]
            if "```json" in output_content:
                json_start = output_content.find("```json") + 7
                json_end = output_content.find("```", json_start)
                json_str = output_content[json_start:json_end].strip()
                try:
                    return json.loads(json_str)
                except:
                    return {"error": "JSON解析失败", "raw": json_str}
            elif "```" in output_content:
                json_start = output_content.find("```") + 3
                json_end = output_content.find("```", json_start)
                json_str = output_content[json_start:json_end].strip()
                try:
                    return json.loads(json_str)
                except:
                    return {"error": "JSON解析失败", "raw": json_str}
        elif isinstance(result, dict):
            return result
        
        return {"error": "无法解析结果", "raw": str(result)}
    
    def _limit_files_for_quick_analysis(self, report_folder: str, algorithms: List[str]):
        """限制快速分析模式下每个算子只分析前5个文件"""
        try:
            discovery_path = self.file_mgr.get_discovery_output_path(report_folder, "all_algorithms")
            with open(discovery_path, 'r', encoding='utf-8') as f:
                discovery_data = json.load(f)
            
            # 修改每个目标算子的文件列表，只保留前5个
            modified = False
            for algo_info in discovery_data["algorithms"]:
                if algo_info["algorithm"] in algorithms:
                    original_count = len(algo_info["files"])
                    if original_count > 5:
                        algo_info["files"] = algo_info["files"][:5]
                        print(f"   {algo_info['algorithm']}: 限制为前5个文件 (原有{original_count}个)")
                        modified = True
                    else:
                        print(f"   {algo_info['algorithm']}: 保持{original_count}个文件")
            
            # 如果有修改，保存更新后的发现结果
            if modified:
                FileManager.save_content(discovery_path, json.dumps(discovery_data, ensure_ascii=False, indent=2))
                print("✅ 已更新算子文件列表以适配快速分析模式")
                
        except Exception as e:
            print(f"⚠️ 限制文件数量时出错: {str(e)}")
    
    def run(self, algorithms = None) -> dict:
        """运行工作流 - 支持快速分析和全部分析模式"""
        # 创建报告文件夹
        report_folder = f"results/{time.strftime('%Y%m%d_%H%M%S')}"
        self.file_mgr.ensure_directories(report_folder)
        
        print(f"📁 报告文件夹: {report_folder}")
        
        completed_algorithms = []
        all_errors = []
        
        # 第一步：Scout扫描所有算子
        if algorithms is None or algorithms == "quick_analysis":
            scout_state = {
                "algorithms": [], 
                "current_algorithm_index": 0,
                "completed_tasks": [], 
                "report_folder": report_folder, 
                "errors": []
            }
            
            scout_result = self.scout_work(scout_state)
            if "scout_all" not in scout_result["completed_tasks"]:
                return {"success": False, "errors": scout_result["errors"]}
            
            # 获取Scout发现的算子列表
            discovered_algorithms = scout_result["algorithms"]
            
            # 如果是快速分析模式，只选择指定的算子
            if algorithms == "quick_analysis":
                target_algorithms = ['axpy', 'hemv', 'gemm']
                algorithms = [algo for algo in discovered_algorithms if algo in target_algorithms]
                print(f"🚀 快速分析模式：从 {len(discovered_algorithms)} 种算子中选择 {len(algorithms)} 种进行分析")
                print(f"   选中的算子: {algorithms}")
                
                # 限制每个算子只分析前5个文件
                self._limit_files_for_quick_analysis(report_folder, algorithms)
            else:
                # 全部分析模式
                algorithms = discovered_algorithms
        
        # 第二阶段：逐个处理每个算法 (Analyzer -> Individual Summary)
        for i, algorithm in enumerate(algorithms):
            print(f"\n🔄 分析算子 {i+1}/{len(algorithms)}: {algorithm}")
            
            try:
                # 为每个算法运行两个阶段（Analyzer -> Individual Summary）
                single_result = self.run_single_algorithm_phases(algorithm, report_folder, algorithms)
                
                if single_result["success"]:
                    completed_algorithms.append(algorithm)
                    print(f"✅ {algorithm} 分析完成")
                else:
                    print(f"⚠️ {algorithm} 分析未完全完成")
                    all_errors.extend(single_result["errors"])
                    
            except Exception as e:
                error_msg = f"{algorithm} 分析失败: {str(e)}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)
        
        # 第二阶段：如果有算法成功完成，进行最终总结
        final_summary_completed = False
        if completed_algorithms:
            print(f"\n📝 最终总结: 整合 {len(completed_algorithms)} 个算法")
            
            try:
                # 运行最终总结
                final_summary_result = self.run_final_summary(completed_algorithms, report_folder)
                final_summary_completed = final_summary_result["success"]
                
                if final_summary_completed:
                    print(f"✅ 最终总结完成")
                else:
                    print(f"❌ 最终总结失败")
                    all_errors.extend(final_summary_result["errors"])
                    
            except Exception as e:
                error_msg = f"最终总结失败: {str(e)}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)
        
        return {
            "success": len(completed_algorithms) > 0,
            "completed_algorithms": completed_algorithms,
            "final_summary_completed": final_summary_completed,
            "report_folder": report_folder,
            "errors": all_errors
        }
    
    def run_single_algorithm_phases(self, algorithm: str, report_folder: str, all_algorithms: List[str] = None) -> dict:
        """运行单个算法的三个阶段：Scout -> Analyzer -> Individual Summary"""
        errors = []
        
        try:
            # 使用传入的算子列表，跳过Scout阶段（已在run方法中完成）
            if all_algorithms is None:
                all_algorithms = [algorithm]
            
            # 阶段1：Analyzer分析代码
            print(f"  📊 分析代码...")
            analyzer_state = {
                "algorithms": all_algorithms,
                "current_algorithm_index": all_algorithms.index(algorithm),
                "completed_tasks": ["scout_all"],
                "report_folder": report_folder,
                "errors": []
            }
            analyzer_result = self.analyzer_work(analyzer_state)
            
            if f"analyze_{algorithm}" not in analyzer_result["completed_tasks"]:
                return {"success": False, "errors": analyzer_result["errors"]}
            
            # 阶段2：Individual Summary总结
            print(f"  📝 策略总结...")
            summary_result = self.individual_summary_work(analyzer_result)
            
            if f"individual_summary_{algorithm}" not in summary_result["completed_tasks"]:
                return {"success": False, "errors": summary_result["errors"]}
            
            return {"success": True, "errors": []}
            
        except Exception as e:
            error_msg = f"{algorithm} 阶段执行失败: {str(e)}"
            errors.append(error_msg)
            return {"success": False, "errors": errors}
    
    def run_final_summary(self, completed_algorithms: List[str], report_folder: str) -> dict:
        """运行最终总结阶段"""
        try:
            print(f"📝 [Final Summary] 整合所有算子的优化策略...")
            
            # 创建最终总结状态
            final_state = {
                "algorithms": completed_algorithms,
                "current_algorithm_index": len(completed_algorithms),  # 表示所有算法都完成了
                "completed_tasks": [f"scout_{algo}" for algo in completed_algorithms] + 
                                 [f"analyze_{algo}" for algo in completed_algorithms] + 
                                 [f"individual_summary_{algo}" for algo in completed_algorithms],
                "report_folder": report_folder,
                "errors": []
            }
            
            final_result = self.final_summary_work(final_state)
            
            if "final_summary" in final_result["completed_tasks"]:
                return {"success": True, "errors": []}
            else:
                return {"success": False, "errors": final_result["errors"]}
                
        except Exception as e:
            error_msg = f"最终总结失败: {str(e)}"
            return {"success": False, "errors": [error_msg]}


def main():
    """主函数"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    if not os.path.exists("./openblas-output/GENERIC/kernel"):
        print("❌ 错误: 未找到openblas-output/GENERIC/kernel目录")
        return
    
    workflow = Workflow()
    
    print("🎯 OpenBLAS优化分析")
    print("1. 快速分析 (扫描后只分析axpy、hemv、gemm前5个文件)")
    print("2. 全部分析 (扫描kernel目录下所有算子)")
    
    choice = input("请选择 (1-2): ").strip()
    
    if choice == "1":
        print("选择快速分析模式，将扫描kernel目录后只分析axpy、hemv、gemm三种算子的前5个文件")
        algorithms = "quick_analysis"  # 特殊标记，表示快速分析
    elif choice == "2":
        print("选择全部分析模式，将扫描kernel目录下的所有算子种类")
        algorithms = None  # 让Scout自动发现
    else:
        print("❌ 无效选择")
        return
    
    try:
        result = workflow.run(algorithms)
        
        print("\n📊 分析完成")
        
        completed_algorithms = result["completed_algorithms"]
        final_summary_completed = result["final_summary_completed"]
        report_folder = result["report_folder"]
        errors = result["errors"]
        
        if isinstance(algorithms, list):
            total_algorithms = len(algorithms)
        else:
            total_algorithms = len(completed_algorithms)
        
        print(f"\n✅ 完成算法: {len(completed_algorithms)}/{total_algorithms} 个")
        print(f"🎯 最终总结: {'✅' if final_summary_completed else '❌'}")
        print(f"📁 报告位置: {report_folder}")
        
        if len(completed_algorithms) > 0:
            print(f"📋 已分析算子: {', '.join(completed_algorithms)}")
        
        if errors:
            print(f"\n⚠️ 错误: {len(errors)} 个")
        
        if final_summary_completed:
            final_path = FileManager.get_final_summary_path(report_folder)
            print(f"\n🎉 分析完成！查看 {final_path}")
        
    except Exception as e:
        print(f"\n❌ 工作流执行失败: {str(e)}")


if __name__ == "__main__":
    main()

