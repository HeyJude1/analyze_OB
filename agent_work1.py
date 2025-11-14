#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - 真正的LangGraph Supervisor工作流
基于官方Supervisor模式实现智能决策的多Agent协作系统
"""

import os
import time
import json
from typing import Dict, List, Literal
from typing_extensions import TypedDict
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from agent1 import (
    AgentFactory,
    FileManager,
    supervisor_router,
    create_supervisor_agent
)

load_dotenv()


# ===== 工作流状态 =====
class WorkState(TypedDict):
    """工作流状态"""
    # 基础任务信息
    report_folder: str
    algorithms: List[str]
    current_algorithm: str
    current_phase: str
    
    # 执行状态跟踪
    completed_algorithms: List[str]
    completed_tasks: List[str]
    skipped_algorithms: List[str]
    
    # 错误和重试管理
    errors: List[str]
    retry_count: int
    last_error: str
    
    # 智能决策支持
    execution_history: List[Dict]
    performance_metrics: Dict
    resource_status: Dict
    
    # 任务依赖状态
    scout_completed: bool
    available_algorithms: List[str]
    pending_files_count: int
    pending_summary_count: int
    
    # 质量控制
    quality_scores: Dict[str, float]
    confidence_levels: Dict[str, float]


# ===== 工作流 =====
class Workflow:
    """工作流"""
    
    def __init__(self):
        self.factory = AgentFactory()
        self.file_mgr = FileManager()
        
        # 创建智能专家Agents
        self.scout = self.factory.create_scout_agent()
        self.analyzer = self.factory.create_analyzer_agent()
        self.individual_summarizer = self.factory.create_individual_summarizer_agent()
        self.final_summarizer = self.factory.create_final_summarizer_agent()
        
        # 构建智能工作流
        self.workflow = self._build_intelligent_workflow()
        
        # 性能监控
        self.start_time = None
        self.decision_count = 0
    
    def _build_intelligent_workflow(self) -> StateGraph:
        """构建真正的Supervisor智能工作流"""
        workflow = StateGraph(WorkState)
        
        # 添加核心节点
        workflow.add_node("supervisor", self.supervisor_node)
        workflow.add_node("scout_agent", self.scout_agent_node)
        workflow.add_node("analyzer_agent", self.analyzer_agent_node)
        workflow.add_node("individual_summarizer_agent", self.individual_summarizer_agent_node)
        workflow.add_node("final_summarizer_agent", self.final_summarizer_agent_node)
        
        # 设置入口 - 直接进入Supervisor进行智能决策
        workflow.add_edge(START, "supervisor")
        
        # 🧠 核心：Supervisor智能路由决策
        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,  # 使用LLM进行智能决策
            {
                "scout": "scout_agent",
                "analyzer": "analyzer_agent",
                "individual_summarizer": "individual_summarizer_agent",
                "final_summarizer": "final_summarizer_agent",
                "FINISH": END
            }
        )
        
        # 所有Agent完成后都回到Supervisor重新评估
        for agent_node in ["scout_agent", "analyzer_agent", 
                          "individual_summarizer_agent", "final_summarizer_agent"]:
            workflow.add_edge(agent_node, "supervisor")
        
        return workflow.compile()
    
    def supervisor_node(self, state: WorkState) -> WorkState:
        """智能Supervisor节点 - 状态分析和决策准备"""
        self.decision_count += 1
        
        print(f"\n🧠 [Supervisor #{self.decision_count}] 智能分析当前状态...")
        
        # 更新执行时长
        if self.start_time:
            execution_time = time.time() - self.start_time
            state["performance_metrics"] = state.get("performance_metrics", {})
            state["performance_metrics"]["execution_time"] = execution_time
        
        # 智能状态分析
        completed_count = len(state.get("completed_algorithms", []))
        total_count = len(state.get("algorithms", []))
        error_count = len(state.get("errors", []))
        
        print(f"📊 进度分析: {completed_count}/{total_count} 算子完成")
        print(f"⚠️ 错误统计: {error_count} 个错误")
        print(f"🔄 当前算子: {state.get('current_algorithm', 'None')}")
        print(f"📍 当前阶段: {state.get('current_phase', 'None')}")
        
        # 记录决策上下文
        decision_context = {
            "decision_id": self.decision_count,
            "state_snapshot": {
                "completed_algorithms": state.get("completed_algorithms", []),
                "current_algorithm": state.get("current_algorithm"),
                "current_phase": state.get("current_phase"),
                "retry_count": state.get("retry_count", 0),
                "error_count": error_count
            }
        }
        
        # 记录Supervisor决策日志
        self.file_mgr.log_supervisor_decision(
            state["report_folder"], 
            decision_context
        )
        
        return state
    
    def scout_agent_node(self, state: WorkState) -> WorkState:
        """Scout Agent节点 - 智能文件发现"""
        print(f"🔍 [Scout Agent] 开始智能算子发现...")
        
        try:
            state["current_phase"] = "scout"
            
            # 调用Scout Agent
            scout_input = """执行智能算子发现任务：
            
🎯 任务目标：
- 扫描 /home/dgc/mjs/project/analyze_OB/openblas-output/GENERIC/kernel 目录
- 智能识别和分类所有算子种类
- 生成高质量的算子分类报告
- 评估分类准确度和置信度

🧠 智能要求：
- 根据目录大小自动调整扫描策略
- 使用模式匹配和启发式规则
- 处理边界情况和异常文件
- 提供分类置信度评分

请开始智能扫描和分类。"""
            
            result = self.scout.invoke({"input": scout_input})
            time.sleep(2)  # API限制缓解
            
            # 解析结果并保存
            discovery_data = self._extract_json_from_result(result)
            
            if "algorithms" in discovery_data:
                # 更新状态
                algorithms = [algo["algorithm"] for algo in discovery_data["algorithms"]]
                state["algorithms"] = algorithms
                state["available_algorithms"] = algorithms.copy()
                state["scout_completed"] = True
                state["completed_tasks"].append("scout_discovery")
                
                # 保存发现结果
                discovery_path = self.file_mgr.get_discovery_output_path(
                    state["report_folder"], "all_algorithms"
                )
                success = self.file_mgr.save_content(
                    discovery_path, 
                    json.dumps(discovery_data, ensure_ascii=False, indent=2)
                )
                
                if success:
                    print(f"✅ Scout完成: 发现 {len(algorithms)} 种算子")
                    
                    # 更新质量评分
                    confidence = discovery_data.get("confidence_score", 0.8)
                    state["confidence_levels"] = state.get("confidence_levels", {})
                    state["confidence_levels"]["scout"] = confidence
                else:
                    raise Exception("保存发现结果失败")
            else:
                raise Exception("Scout结果格式错误")
                
        except Exception as e:
            error_msg = f"Scout Agent失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["last_error"] = error_msg
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def analyzer_agent_node(self, state: WorkState) -> WorkState:
        """Analyzer Agent节点 - 智能代码分析"""
        current_algo = state.get("current_algorithm")
        if not current_algo:
            # 智能选择下一个待分析的算子
            available = state.get("available_algorithms", [])
            completed = state.get("completed_algorithms", [])
            remaining = [algo for algo in available if algo not in completed]
            
            if remaining:
                current_algo = remaining[0]
                state["current_algorithm"] = current_algo
            else:
                state["errors"].append("没有可分析的算子")
                return state
        
        print(f"📊 [Analyzer Agent] 智能分析算子: {current_algo}")
        
        try:
            state["current_phase"] = "analyzer"
            
            # 获取算子文件列表
            discovery_path = self.file_mgr.get_discovery_output_path(
                state["report_folder"], "all_algorithms"
            )
            
            with open(discovery_path, 'r', encoding='utf-8') as f:
                discovery_data = json.load(f)
            
            # 找到当前算子的文件
            target_files = []
            for algo_info in discovery_data["algorithms"]:
                if algo_info["algorithm"] == current_algo:
                    target_files = algo_info["files"]
                    break
            
            if not target_files:
                raise Exception(f"未找到{current_algo}的文件列表")
            
            # 智能分析每个文件
            analysis_path = self.file_mgr.get_analysis_output_path(
                state["report_folder"], current_algo
            )
            
            all_analyses = []
            total_files = len(target_files)
            
            for i, file_info in enumerate(target_files):
                file_name = file_info["name"]
                print(f"  📄 智能分析 {i+1}/{total_files}: {file_name}")
                
                analyzer_input = f"""执行{current_algo}算子文件的智能深度分析：

📁 目标文件: {file_name}

🧠 智能分析要求：
- 根据代码复杂度自动调整分析深度
- 识别所有优化策略并评估置信度
- 提供代码复杂度和优化潜力评估
- 生成高质量的分析报告

🎯 分析框架：
- 算法层：计算逻辑、数据结构、算法设计优化
- 代码层：循环、分支、内存访问、编译器优化  
- 指令层：SIMD、向量化、特殊指令、汇编优化

请开始智能深度分析。"""
                
                result = self.analyzer.invoke({"input": analyzer_input})
                time.sleep(2)
                
                file_analysis = self._extract_json_from_result(result)
                all_analyses.append(file_analysis)
                
                # 增量保存
                analysis_data = {
                    "algorithm": current_algo,
                    "total_files": total_files,
                    "analyzed_files": len(all_analyses),
                    "individual_analyses": all_analyses,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.file_mgr.save_content(
                    analysis_path, 
                    json.dumps(analysis_data, ensure_ascii=False, indent=2)
                )
            
            # 更新状态
            state["completed_tasks"].append(f"analyze_{current_algo}")
            state["pending_files_count"] = state.get("pending_files_count", 0) - total_files
            
            # 计算平均质量分数
            complexity_scores = [a.get("complexity_score", 5) for a in all_analyses if "complexity_score" in a]
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 5
            
            state["quality_scores"] = state.get("quality_scores", {})
            state["quality_scores"][f"analyzer_{current_algo}"] = min(avg_complexity / 10, 1.0)
            
            print(f"✅ Analyzer完成: {current_algo} ({total_files} 个文件)")
            
        except Exception as e:
            error_msg = f"Analyzer Agent失败 ({current_algo}): {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["last_error"] = error_msg
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def individual_summarizer_agent_node(self, state: WorkState) -> WorkState:
        """Individual Summarizer Agent节点 - 智能策略整合"""
        current_algo = state.get("current_algorithm")
        if not current_algo:
            state["errors"].append("Individual Summarizer: 没有指定算子")
            return state
        
        print(f"📝 [Individual Summarizer] 智能整合算子: {current_algo}")
        
        try:
            state["current_phase"] = "individual_summary"
            
            # 读取分析结果
            analysis_path = self.file_mgr.get_analysis_output_path(
                state["report_folder"], current_algo
            )
            
            with open(analysis_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            summarizer_input = f"""执行{current_algo}算子的智能策略整合：

📊 输入数据：
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}

🧠 智能整合要求：
- 自动识别相似和重复的优化策略
- 智能合并策略，保持最佳描述和命名
- 评估整合质量并提供改进建议
- 消除冗余，提升策略库的简洁性

🎯 整合目标：
- 生成高质量的算子优化策略总结
- 统一命名规范，提升可读性
- 保留关键差异，避免过度简化

请开始智能策略整合。"""
            
            result = self.individual_summarizer.invoke({"input": summarizer_input})
            time.sleep(2)
            
            summary_data = self._extract_json_from_result(result)
            
            # 保存总结结果
            summary_path = self.file_mgr.get_individual_summary_path(
                state["report_folder"], current_algo
            )
            
            success = self.file_mgr.save_content(
                summary_path,
                json.dumps(summary_data, ensure_ascii=False, indent=2)
            )
            
            if success:
                # 更新状态
                state["completed_tasks"].append(f"individual_summary_{current_algo}")
                state["completed_algorithms"].append(current_algo)
                state["pending_summary_count"] = state.get("pending_summary_count", 0) - 1
                
                # 记录质量分数
                quality_score = summary_data.get("quality_score", 0.8)
                state["quality_scores"] = state.get("quality_scores", {})
                state["quality_scores"][f"summary_{current_algo}"] = quality_score
                
                print(f"✅ Individual Summary完成: {current_algo}")
                
                # 重置当前算子，让Supervisor选择下一个
                state["current_algorithm"] = None
                state["retry_count"] = 0
            else:
                raise Exception("保存总结结果失败")
                
        except Exception as e:
            error_msg = f"Individual Summarizer失败 ({current_algo}): {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["last_error"] = error_msg
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def final_summarizer_agent_node(self, state: WorkState) -> WorkState:
        """Final Summarizer Agent节点 - 智能跨算子总结"""
        print(f"🎯 [Final Summarizer] 智能跨算子总结...")
        
        try:
            state["current_phase"] = "final_summary"
            
            completed_algorithms = state.get("completed_algorithms", [])
            if not completed_algorithms:
                raise Exception("没有已完成的算子可供总结")
            
            # 收集所有算子的总结数据
            all_summaries = {}
            for algo in completed_algorithms:
                summary_path = self.file_mgr.get_individual_summary_path(
                    state["report_folder"], algo
                )
                
                with open(summary_path, 'r', encoding='utf-8') as f:
                    all_summaries[algo] = json.load(f)
            
            final_input = f"""执行OpenBLAS优化策略的智能跨算子总结：

📊 输入数据 - 所有算子总结：
{json.dumps(all_summaries, ensure_ascii=False, indent=2)}

🧠 智能总结要求：
- 识别跨算子的通用优化模式和规律
- 构建完整的优化策略分类体系
- 提供策略覆盖度分析和质量评估
- 生成实用的最佳实践建议

🎯 总结目标：
- 构建OpenBLAS优化策略知识库
- 发现通用优化规律和最佳实践
- 提供策略应用指导和建议
- 评估优化策略的完整性和实用性

请开始智能跨算子总结。"""
            
            result = self.final_summarizer.invoke({"input": final_input})
            time.sleep(2)
            
            final_data = self._extract_json_from_result(result)
            
            # 保存最终总结
            final_path = self.file_mgr.get_final_summary_path(state["report_folder"])
            success = self.file_mgr.save_content(
                final_path,
                json.dumps(final_data, ensure_ascii=False, indent=2)
            )
            
            if success:
                state["completed_tasks"].append("final_summary")
                
                # 记录最终质量分数
                coverage_score = len(completed_algorithms) / len(state.get("algorithms", [1]))
                state["quality_scores"] = state.get("quality_scores", {})
                state["quality_scores"]["final_summary"] = coverage_score
                
                print(f"✅ Final Summary完成: 整合了 {len(completed_algorithms)} 个算子")
            else:
                raise Exception("保存最终总结失败")
                
        except Exception as e:
            error_msg = f"Final Summarizer失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["last_error"] = error_msg
            state["retry_count"] = state.get("retry_count", 0) + 1
        
        return state
    
    def _extract_json_from_result(self, result):
        """从Agent结果中提取JSON - 增强版"""
        try:
            if isinstance(result, dict) and "output" in result:
                output_content = result["output"]
                
                # 尝试多种JSON提取方式
                if "```json" in output_content:
                    json_start = output_content.find("```json") + 7
                    json_end = output_content.find("```", json_start)
                    json_str = output_content[json_start:json_end].strip()
                elif "```" in output_content:
                    json_start = output_content.find("```") + 3
                    json_end = output_content.find("```", json_start)
                    json_str = output_content[json_start:json_end].strip()
                else:
                    # 尝试直接解析整个输出
                    json_str = output_content.strip()
                
                return json.loads(json_str)
                
            elif isinstance(result, dict):
                return result
            else:
                return {"error": "无法解析结果", "raw": str(result)}
                
        except json.JSONDecodeError as e:
            return {"error": f"JSON解析失败: {str(e)}", "raw": str(result)}
        except Exception as e:
            return {"error": f"结果提取失败: {str(e)}", "raw": str(result)}
    
    def run(self, algorithms: List[str] = None) -> dict:
        """运行智能Supervisor工作流"""
        self.start_time = time.time()
        
        # 创建报告文件夹
        report_folder = f"results/{time.strftime('%Y%m%d_%H%M%S')}_supervisor"
        self.file_mgr.ensure_directories(report_folder)
        
        print(f"🧠 启动真正的Supervisor工作流")
        print(f"📁 报告文件夹: {report_folder}")
        
        # 初始化智能状态
        initial_state = {
            "report_folder": report_folder,
            "algorithms": algorithms or [],
            "current_algorithm": None,
            "current_phase": "initialization",
            
            "completed_algorithms": [],
            "completed_tasks": [],
            "skipped_algorithms": [],
            
            "errors": [],
            "retry_count": 0,
            "last_error": "",
            
            "execution_history": [],
            "performance_metrics": {},
            "resource_status": {"api_status": "正常", "file_system_status": "正常"},
            
            "scout_completed": False,
            "available_algorithms": [],
            "pending_files_count": 0,
            "pending_summary_count": 0,
            
            "quality_scores": {},
            "confidence_levels": {}
        }
        
        try:
            # 🧠 启动智能工作流 - Supervisor将智能决策每一步
            final_state = self.workflow.invoke(initial_state)
            
            # 计算最终统计
            execution_time = time.time() - self.start_time
            completed_count = len(final_state.get("completed_algorithms", []))
            total_algorithms = len(final_state.get("algorithms", []))
            error_count = len(final_state.get("errors", []))
            
            # 生成智能分析报告
            performance_report = {
                "execution_time": execution_time,
                "decision_count": self.decision_count,
                "completed_algorithms": completed_count,
                "total_algorithms": total_algorithms,
                "success_rate": completed_count / max(total_algorithms, 1),
                "error_count": error_count,
                "quality_scores": final_state.get("quality_scores", {}),
                "confidence_levels": final_state.get("confidence_levels", {}),
                "avg_quality": sum(final_state.get("quality_scores", {}).values()) / max(len(final_state.get("quality_scores", {})), 1)
            }
            
            print(f"\n🎯 Supervisor工作流完成")
            print(f"⏱️ 执行时间: {execution_time:.1f}秒")
            print(f"🧠 智能决策次数: {self.decision_count}")
            print(f"✅ 成功率: {performance_report['success_rate']:.1%}")
            print(f"📊 平均质量分数: {performance_report['avg_quality']:.2f}")
            
            return {
                "success": completed_count > 0,
                "completed_algorithms": final_state.get("completed_algorithms", []),
                "final_summary_completed": "final_summary" in final_state.get("completed_tasks", []),
                "report_folder": report_folder,
                "errors": final_state.get("errors", []),
                "performance_report": performance_report,
                "supervisor_decisions": self.decision_count
            }
            
        except Exception as e:
            error_msg = f"Supervisor工作流执行失败: {str(e)}"
            print(f"❌ {error_msg}")
            
            return {
                "success": False,
                "completed_algorithms": [],
                "final_summary_completed": False,
                "report_folder": report_folder,
                "errors": [error_msg],
                "performance_report": {"execution_time": time.time() - self.start_time},
                "supervisor_decisions": self.decision_count
            }


def main():
    """主函数 - 真正的Supervisor模式演示"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    if not os.path.exists("/home/dgc/mjs/project/analyze_OB/openblas-output/GENERIC/kernel"):
        print("❌ 错误: 未找到openblas-output/GENERIC/kernel目录")
        return
    
    workflow = Workflow()
    
    print("🧠 OpenBLAS优化分析 - 真正的Supervisor模式")
    print("1. 快速分析 (gemm, axpy, dot) - Supervisor智能调度")
    print("2. 全部分析 (智能发现所有算子) - 完全自主决策")
    print("3. 自定义算子列表 - Supervisor优化执行")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        algorithms = ['gemm', 'axpy', 'dot']
        print("🧠 Supervisor将智能调度快速分析...")
    elif choice == "2":
        print("🧠 Supervisor将自主发现并分析所有算子...")
        algorithms = None  # 让Supervisor自主发现
    elif choice == "3":
        algo_input = input("请输入算子列表 (用逗号分隔): ").strip()
        algorithms = [algo.strip() for algo in algo_input.split(",") if algo.strip()]
        print(f"🧠 Supervisor将智能分析: {algorithms}")
    else:
        print("❌ 无效选择")
        return
    
    try:
        print(f"\n🚀 启动真正的Supervisor工作流...")
        result = workflow.run(algorithms)
        
        print(f"\n📊 Supervisor分析完成")
        print(f"✅ 成功: {result['success']}")
        print(f"📁 报告位置: {result['report_folder']}")
        print(f"🧠 智能决策次数: {result['supervisor_decisions']}")
        
        if result["performance_report"]:
            perf = result["performance_report"]
            print(f"⏱️ 执行时间: {perf.get('execution_time', 0):.1f}秒")
            print(f"📈 成功率: {perf.get('success_rate', 0):.1%}")
            print(f"📊 平均质量: {perf.get('avg_quality', 0):.2f}")
        
        if result["errors"]:
            print(f"\n⚠️ 错误: {len(result['errors'])} 个")
            for error in result["errors"][-3:]:  # 显示最后3个错误
                print(f"  - {error}")
        
        if result["final_summary_completed"]:
            final_path = FileManager.get_final_summary_path(result["report_folder"])
            print(f"\n🎉 智能分析完成！查看最终报告: {final_path}")
        
    except Exception as e:
        print(f"\n❌ Supervisor工作流执行失败: {str(e)}")


if __name__ == "__main__":
    main()
