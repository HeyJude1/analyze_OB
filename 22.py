#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析 - 标准LangChain Agent + Tools架构版本
符合LangChain官方Agent定义的标准实现

架构特点:
- 使用标准LangChain Agent (create_openai_tools_agent)
- 工具使用@tool装饰器定义，不包含内部LLM
- Agent通过LLM推理决定工具调用序列
- 符合官方Agent工作模式：提示→推理→工具调用→观察→下一步
"""

import os
import time
import json
from typing import Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

# LangGraph imports (标准Agent架构)
from langgraph.graph import StateGraph, START, END

# 本地imports
from analyze_agent_tools import StandardAgentFactory

# 加载环境变量
load_dotenv()

# ===== 标准Agent架构状态定义 =====
class StandardAgentState(TypedDict):
    """标准Agent架构的工作流状态"""
    # 用户输入和配置
    user_request: str
    analysis_config: Dict[str, Any]
    
    # Agent执行结果
    agent_results: Dict[str, Any]
    
    # 工作流控制
    current_stage: str  # planning, scout, analyze, strategize, summarize, complete
    current_algorithm: str
    current_algorithm_index: int
    completed_tasks: list
    
    # 运行时状态
    iteration_count: int
    max_iterations: int
    workflow_complete: bool
    
    # 错误和历史
    errors: list
    execution_history: list
    messages: list

# ===== 标准Agent架构工作流 =====
class StandardAgentWorkflow:
    """标准LangChain Agent工作流 - 符合官方Agent定义"""
    
    def __init__(self):
        self.factory = StandardAgentFactory()
        
        # 创建标准LangChain Agents
        self.master_coordinator = self.factory.create_master_coordinator_agent()
        self.scout_specialist = self.factory.create_scout_specialist_agent()
        self.analyzer_specialist = self.factory.create_analyzer_specialist_agent()
        self.strategist_specialist = self.factory.create_strategist_specialist_agent()
        self.summarizer_specialist = self.factory.create_summarizer_specialist_agent()
        
        # 构建标准Agent工作流
        self.workflow = self._build_standard_agent_workflow()
    
    def _build_standard_agent_workflow(self) -> StateGraph:
        """构建标准Agent架构的LangGraph工作流"""
        workflow = StateGraph(StandardAgentState)
        
        # 添加节点 - 每个节点对应一个标准Agent
        workflow.add_node("master_planning", self.master_planning_node)
        workflow.add_node("scout_work", self.scout_work_node)
        workflow.add_node("analyzer_work", self.analyzer_work_node)
        workflow.add_node("strategist_work", self.strategist_work_node)
        workflow.add_node("summarizer_work", self.summarizer_work_node)
        workflow.add_node("coordination_check", self.coordination_check_node)
        
        # 设置入口点
        workflow.add_edge(START, "master_planning")
        
        # 主要工作流路径
        workflow.add_edge("master_planning", "coordination_check")
        
        # 协调检查后的路由
        workflow.add_conditional_edges(
            "coordination_check",
            self._route_by_coordination,
            {
                "scout": "scout_work",
                "analyze": "analyzer_work",
                "strategize": "strategist_work",
                "summarize": "summarizer_work",
                "complete": END
            }
        )
        
        # 专家工作完成后返回协调检查
        workflow.add_edge("scout_work", "coordination_check")
        workflow.add_edge("analyzer_work", "coordination_check")
        workflow.add_edge("strategist_work", "coordination_check")
        workflow.add_edge("summarizer_work", "coordination_check")
        
        return workflow.compile()
    
    def _route_by_coordination(self, state: StandardAgentState) -> str:
        """基于协调器的建议进行路由"""
        try:
            # 从Agent结果中获取路由建议
            agent_results = state.get("agent_results", {})
            coordination_result = agent_results.get("coordination", {})
            
            next_action = coordination_result.get("next_action", "complete")
            
            # 映射到具体的工作节点
            action_mapping = {
                "scout": "scout",
                "analyze": "analyze",
                "strategize": "strategize", 
                "summarize": "summarize",
                "complete": "complete"
            }
            
            return action_mapping.get(next_action, "complete")
            
        except Exception as e:
            print(f"⚠️ 路由决策失败: {e}")
            return "complete"
    
    def master_planning_node(self, state: StandardAgentState) -> StandardAgentState:
        """Master协调器规划节点 - 使用标准Agent"""
        print(f"🎯 [Master Planning] 标准Agent协调规划...")
        
        try:
            user_request = state["user_request"]
            
            # 使用Master Agent进行初始规划
            planning_input = f"""
            用户请求: "{user_request}"
            
            作为Master协调器Agent，请完成以下任务：
            
            1. 使用get_algorithm_list工具确定要分析的算子列表
            2. 使用create_output_directory工具确保输出目录存在
            3. 使用analyze_workflow_state工具分析当前状态
            4. 制定初始的执行计划
            
            请开始执行并调用相应的工具。
            """
            
            result = self.master_coordinator.invoke({"input": planning_input})
            
            # 解析算子列表
            if "快速" in user_request or "quick" in user_request.lower():
                algorithms = ['gemm', 'axpy', 'dot']
                analysis_type = "quick"
            elif "全面" in user_request or "comprehensive" in user_request.lower():
                algorithms = ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger']
                analysis_type = "comprehensive"
            else:
                algorithms = ['gemm', 'axpy', 'dot']
                analysis_type = "custom"
            
            # 初始化分析配置
            state["analysis_config"] = {
                "algorithms": algorithms,
                "analysis_type": analysis_type,
                "report_folder": f"{int(time.time())}",
                "total_algorithms": len(algorithms)
            }
            
            # 初始化状态
            state["current_stage"] = "scout"
            state["current_algorithm"] = algorithms[0] if algorithms else ""
            state["current_algorithm_index"] = 0
            state["completed_tasks"] = []
            state["iteration_count"] = 0
            state["max_iterations"] = 50
            state["workflow_complete"] = False
            
            # 记录规划结果
            state["agent_results"]["planning"] = {
                "status": "completed",
                "algorithms": algorithms,
                "analysis_type": analysis_type,
                "result": result.get("output", "") if hasattr(result, "get") else str(result)
            }
            
            print(f"✅ Master规划完成: {analysis_type}分析，算子: {algorithms}")
            
        except Exception as e:
            error_msg = f"Master规划失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["workflow_complete"] = True
        
        return state
    
    def coordination_check_node(self, state: StandardAgentState) -> StandardAgentState:
        """协调检查节点 - 使用Master Agent决定下一步"""
        print(f"🧠 [Coordination Check] 标准Agent协调决策...")
        
        try:
            # 准备状态数据
            current_state_data = {
                "current_stage": state.get("current_stage", "planning"),
                "completed_tasks": state.get("completed_tasks", []),
                "algorithms": state.get("analysis_config", {}).get("algorithms", []),
                "current_algorithm_index": state.get("current_algorithm_index", 0),
                "iteration_count": state.get("iteration_count", 0)
            }
            
            # 使用Master Agent分析状态并决定下一步
            coordination_input = f"""
            当前工作流状态分析和决策任务：
            
            请使用analyze_workflow_state工具分析当前状态：
            状态数据: {json.dumps(current_state_data, ensure_ascii=False)}
            
            然后使用schedule_next_tasks工具制定下一步计划。
            
            请调用这些工具并基于结果决定下一步行动。
            """
            
            result = self.master_coordinator.invoke({"input": coordination_input})
            
            # 从Agent的工具调用结果中提取决策
            # 简单的状态分析逻辑作为备用
            algorithms = current_state_data["algorithms"]
            current_index = current_state_data["current_algorithm_index"]
            completed_tasks = current_state_data["completed_tasks"]
            
            if current_index < len(algorithms):
                current_alg = algorithms[current_index]
                
                if not any(f"scout_{current_alg}" in task for task in completed_tasks):
                    next_action = "scout"
                elif not any(f"analyze_{current_alg}" in task for task in completed_tasks):
                    next_action = "analyze"
                elif not any(f"strategize_{current_alg}" in task for task in completed_tasks):
                    next_action = "strategize"
                else:
                    # 移动到下一个算子
                    state["current_algorithm_index"] = current_index + 1
                    if current_index + 1 < len(algorithms):
                        state["current_algorithm"] = algorithms[current_index + 1]
                        next_action = "scout"
                    else:
                        next_action = "summarize"
            else:
                # 所有算子完成，检查是否需要总结
                if not any("summarize" in task for task in completed_tasks):
                    next_action = "summarize"
                else:
                    next_action = "complete"
                    state["workflow_complete"] = True
            
            # 更新状态
            state["iteration_count"] = current_state_data["iteration_count"] + 1
            
            # 记录协调结果
            state["agent_results"]["coordination"] = {
                "next_action": next_action,
                "reasoning": f"基于状态分析决定: {next_action}",
                "current_algorithm": state.get("current_algorithm", ""),
                "result": result.get("output", "") if hasattr(result, "get") else str(result)
            }
            
            print(f"🎯 协调决策: {next_action} (算子: {state.get('current_algorithm', 'N/A')})")
            
            # 检查是否应该完成
            if next_action == "complete" or state["iteration_count"] >= state["max_iterations"]:
                state["workflow_complete"] = True
                print("✅ 协调器决定完成工作流")
                
        except Exception as e:
            error_msg = f"协调检查失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            # 安全的默认决策
            state["agent_results"]["coordination"] = {"next_action": "complete"}
        
        return state
    
    def scout_work_node(self, state: StandardAgentState) -> StandardAgentState:
        """Scout专家工作节点 - 使用标准Agent"""
        print(f"🔍 [Scout Work] 发现 {state['current_algorithm']} 算子文件...")
        
        try:
            config = state["analysis_config"]
            current_algorithm = state["current_algorithm"]
            report_folder = config["report_folder"]
            
            scout_input = f"""
            专家任务：为 {current_algorithm} 算子进行专业文件发现
            
            请按以下步骤执行：
            
            1. 使用create_output_directory工具确保 ../discovery_results 目录存在
            2. 使用list_directory工具探索 kernel/ 目录，找到 {current_algorithm} 相关目录
            3. 使用file_search工具搜索 {current_algorithm} 相关实现文件
            4. 使用read_file工具分析关键文件类型（只分析文件头部信息，不要输出完整文件内容）
            5. 整理发现结果为JSON格式，包含：
               - 发现的文件列表
               - 架构类型分类 (generic, x86_64, arm64等)
               - 实现类型识别 (simd_optimized, microkernel等)
            6. 使用write_file工具保存发现结果到：
               ../discovery_results/{current_algorithm}_discovered_{report_folder}.json
            7. 使用check_output_quality工具验证保存的JSON文件
            
            重要：确保文件保存到正确的 ../discovery_results 目录，而不是其他目录。
            """
            
            result = self.scout_specialist.invoke({"input": scout_input})
            
            # 记录Scout工作结果
            task_key = f"scout_{current_algorithm}"
            state["agent_results"][task_key] = {
                "agent": "scout_specialist",
                "algorithm": current_algorithm,
                "result": result.get("output", "") if hasattr(result, "get") else str(result),
                "timestamp": int(time.time()),
                "status": "completed"
            }
            
            state["completed_tasks"].append(task_key)
            print(f"✅ Scout Agent完成 {current_algorithm} 算子文件发现")
            
        except Exception as e:
            error_msg = f"Scout Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def analyzer_work_node(self, state: StandardAgentState) -> StandardAgentState:
        """Analyzer专家工作节点 - 使用标准Agent"""
        print(f"📊 [Analyzer Work] 分析 {state['current_algorithm']} 算子代码...")
        
        try:
            config = state["analysis_config"]
            current_algorithm = state["current_algorithm"]
            report_folder = config["report_folder"]
            
            analyzer_input = f"""
            专家任务：为 {current_algorithm} 算子进行专业代码分析
            
            请按以下步骤执行：
            
            1. 使用read_file工具读取发现结果（只获取文件列表，不输出具体内容）：
               ../discovery_results/{current_algorithm}_discovered_{report_folder}.json
            2. 使用create_output_directory工具创建算子专用目录：
               ../analysis_results/{current_algorithm}/
            3. 基于发现的文件列表，使用read_file工具分析源代码（只分析优化技术，不输出完整代码）
            4. 按三层框架分类优化技术：
               - 算法层：循环展开、分块、数据重用
               - 代码层：缓存优化、内存对齐、预取
               - 指令层：SIMD向量化、FMA、指令并行
            5. 整理分析结果为JSON格式，包含：
               - 优化技术分类
               - 性能影响评估
               - 适用场景分析
            6. 使用write_file工具保存分析结果到：
               ../analysis_results/{current_algorithm}/analysis_{current_algorithm}_{report_folder}.json
            7. 使用check_output_quality工具验证保存的JSON文件
            
            重要：确保文件保存到正确的 ../analysis_results/{current_algorithm}/ 目录。
            """
            
            result = self.analyzer_specialist.invoke({"input": analyzer_input})
            
            # 记录Analyzer工作结果
            task_key = f"analyze_{current_algorithm}"
            state["agent_results"][task_key] = {
                "agent": "analyzer_specialist",
                "algorithm": current_algorithm,
                "result": result.get("output", "") if hasattr(result, "get") else str(result),
                "timestamp": int(time.time()),
                "status": "completed"
            }
            
            state["completed_tasks"].append(task_key)
            print(f"✅ Analyzer Agent完成 {current_algorithm} 算子代码分析")
            
        except Exception as e:
            error_msg = f"Analyzer Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def strategist_work_node(self, state: StandardAgentState) -> StandardAgentState:
        """Strategist专家工作节点 - 使用标准Agent"""
        print(f"🎯 [Strategist Work] 提炼 {state['current_algorithm']} 算子策略...")
        
        try:
            config = state["analysis_config"]
            current_algorithm = state["current_algorithm"]
            report_folder = config["report_folder"]
            
            strategist_input = f"""
            专家任务：为 {current_algorithm} 算子进行专业策略提炼
            
            请按以下步骤执行：
            
            1. 使用read_file工具读取分析结果（只获取优化技术信息，不输出完整内容）：
               ../analysis_results/{current_algorithm}/analysis_{current_algorithm}_{report_folder}.json
            2. 使用get_current_timestamp工具获取当前时间戳用于创建报告文件夹
            3. 使用create_output_directory工具创建带时间戳的策略报告目录：
               ../strategy_reports/report_时间戳/
            4. 按三层策略框架提炼优化策略：
               - 算法设计层：计算逻辑优化、时空权衡
               - 代码优化层：性能加速、循环优化、代码顺序
               - 特有指令层：专有指令使用和优化设计
            5. 生成Markdown格式的策略报告，包含：
               - 优化策略分析
               - 实施建议
               - 性能预期
            6. 使用write_file工具保存策略报告到：
               ../strategy_reports/report_时间戳/{current_algorithm}_optimization_analysis.md
            7. 使用check_output_quality工具验证策略报告质量
            
            重要：
            - 必须先获取时间戳，然后创建带时间戳的报告目录
            - 确保文件保存到正确的时间戳目录中
            - 保存报告文件夹名称供后续总结使用
            """
            
            result = self.strategist_specialist.invoke({"input": strategist_input})
            
            # 记录Strategist工作结果
            task_key = f"strategize_{current_algorithm}"
            state["agent_results"][task_key] = {
                "agent": "strategist_specialist",
                "algorithm": current_algorithm,
                "result": result.get("output", "") if hasattr(result, "get") else str(result),
                "timestamp": int(time.time()),
                "status": "completed"
            }
            
            state["completed_tasks"].append(task_key)
            print(f"✅ Strategist Agent完成 {current_algorithm} 算子策略提炼")
            
        except Exception as e:
            error_msg = f"Strategist Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def summarizer_work_node(self, state: StandardAgentState) -> StandardAgentState:
        """Summarizer专家工作节点 - 使用标准Agent"""
        print(f"📝 [Summarizer Work] 生成跨算子总结报告...")
        
        try:
            config = state["analysis_config"]
            algorithms = config["algorithms"]
            
            summarizer_input = f"""
            专家任务：生成跨算子优化策略总结
            
            请按以下步骤执行：
            
            1. 使用list_directory工具查找策略报告目录：
               ../strategy_reports/
               找到最新创建的 report_时间戳 目录
            2. 使用read_file工具逐个读取算子策略报告（只获取策略信息，不输出完整内容）：
               ../strategy_reports/report_时间戳/算子名_optimization_analysis.md
            3. 进行跨算子分析：
               - 跨算子共性分析：相同优化技术、通用设计模式
               - 架构特化对比：不同架构的优化差异
               - 性能提升模式：优化技术收益和适用场景
            4. 生成综合性Markdown总结报告，包含：
               - 通用优化模式总结
               - 算子特化策略对比
               - 实用优化建议
            5. 使用write_file工具保存总结报告到：
               ../strategy_reports/report_时间戳/optimization_summary_report.md
            6. 使用check_output_quality工具验证总结报告质量
            
            目标：生成高价值的跨算子优化洞察和指导。
            """
            
            result = self.summarizer_specialist.invoke({"input": summarizer_input})
            
            # 记录Summarizer工作结果
            task_key = "summarize_all"
            state["agent_results"][task_key] = {
                "agent": "summarizer_specialist",
                "algorithms": algorithms,
                "result": result.get("output", "") if hasattr(result, "get") else str(result),
                "timestamp": int(time.time()),
                "status": "completed"
            }
            
            state["completed_tasks"].append(task_key)
            state["workflow_complete"] = True
            print(f"✅ Summarizer Agent完成跨算子总结")
            
        except Exception as e:
            error_msg = f"Summarizer Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def run_standard_agent_workflow(self, user_request: str) -> StandardAgentState:
        """运行标准Agent架构的工作流"""
        # 初始化状态
        initial_state = StandardAgentState(
            user_request=user_request,
            analysis_config={},
            agent_results={},
            current_stage="planning",
            current_algorithm="",
            current_algorithm_index=0,
            completed_tasks=[],
            iteration_count=0,
            max_iterations=50,
            workflow_complete=False,
            errors=[],
            execution_history=[],
            messages=[HumanMessage(content=user_request)]
        )
        
        # 工作流配置
        config = {
            "recursion_limit": 60,
            "max_iterations": 50
        }
        
        print(f"🚀 启动标准LangChain Agent工作流")
        print(f"🎯 理念: 符合官方Agent定义，LLM推理+工具调用序列")
        print(f"📝 用户请求: {user_request}")
        print(f"⚙️ 配置: 递归限制={config['recursion_limit']}")
        print()
        
        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            return final_state
            
        except Exception as e:
            print(f"❌ 标准Agent工作流执行失败: {str(e)}")
            initial_state["errors"].append(str(e))
            initial_state["workflow_complete"] = True
            return initial_state

def main():
    """主函数 - 标准LangChain Agent系统入口"""
    # 环境检查
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    if not os.path.exists("./OpenBLAS-develop"):
        print("❌ 错误: 未找到OpenBLAS-develop目录")
        return
    
    # 创建标准Agent工作流
    standard_workflow = StandardAgentWorkflow()
    
    # 用户交互
    print("🎯 OpenBLAS优化策略分析 - 标准LangChain Agent架构")
    print("=" * 60)
    print("🎯 设计理念: 符合LangChain官方Agent定义")
    print("🤖 Agent使用LLM推理决定工具调用序列")
    print("🛠️ 工具使用@tool装饰器，执行具体业务逻辑")
    print("🔄 工作模式: 提示→推理→工具调用→观察→下一步决策")
    print("=" * 60)
    print()
    
    print("分析选项:")
    print("1. 快速分析 - 标准Agent分析核心算子")
    print("2. 全面分析 - 标准Agent分析完整算子集")
    print("3. 自定义分析 - 指定算子，标准Agent执行")
    print("4. 直接输入 - 自然语言描述需求")
    print()
    
    choice = input("请选择 (1-4) 或直接输入需求: ").strip()
    
    # 解析用户输入
    if choice == "1":
        user_request = "请进行快速分析，使用标准Agent分析核心BLAS算子的优化策略"
    elif choice == "2":
        user_request = "请进行全面分析，使用标准Agent完整分析BLAS算子的优化策略"
    elif choice == "3":
        algorithms = input("请输入要分析的算子: ").strip()
        user_request = f"请使用标准Agent分析以下算子的优化策略: {algorithms}"
    elif choice == "4":
        user_request = input("请输入分析需求: ").strip()
    else:
        user_request = choice
    
    if not user_request:
        print("❌ 未提供有效需求")
        return
    
    print(f"\n🎯 系统接收需求: {user_request}")
    print("🤖 启动标准LangChain Agent协作系统...")
    print()
    
    try:
        # 运行标准Agent工作流
        final_state = standard_workflow.run_standard_agent_workflow(user_request)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("📊 标准LangChain Agent工作流完成")
        print("=" * 60)
        
        # 分析执行情况
        iteration_count = final_state.get("iteration_count", 0)
        errors = final_state.get("errors", [])
        is_complete = final_state.get("workflow_complete", False)
        agent_results = final_state.get("agent_results", {})
        
        print(f"\n🔄 执行统计:")
        print(f"  - Master协调决策次数: {iteration_count}")
        print(f"  - Agent任务完成: {len([k for k in agent_results.keys() if k not in ['planning', 'coordination']])} 个")
        print(f"  - 工作流状态: {'✅ 完成' if is_complete else '⚠️ 未完成'}")
        
        # 显示Agent工作结果
        if agent_results:
            print(f"\n🤖 标准Agent工作成果:")
            for task_key, result in agent_results.items():
                if task_key in ['planning', 'coordination']:
                    continue
                agent = result.get("agent", "unknown").replace("_specialist", "")
                algorithm = result.get("algorithm", result.get("algorithms", ""))
                status = result.get("status", "unknown")
                print(f"  - {agent.upper()}: {algorithm} ({status})")
        
        # 显示错误
        if errors:
            print(f"\n⚠️ 执行过程中的问题 ({len(errors)} 个):")
            for i, error in enumerate(errors[:3], 1):
                print(f"  {i}. {error}")
            if len(errors) > 3:
                print(f"  ... 还有 {len(errors) - 3} 个问题")
        
        # 显示生成的文件
        print(f"\n📁 标准Agent生成的文件:")
        
        dirs_to_check = [
            ("discovery_results", "🔍 Scout Agent发现结果"),
            ("analysis_results", "📊 Analyzer Agent分析结果"),
            ("strategy_reports", "🎯 Strategist Agent策略报告")
        ]
        
        total_files = 0
        for dir_name, desc in dirs_to_check:
            if os.path.exists(dir_name):
                files = []
                if dir_name == "analysis_results":
                    # 检查算子子文件夹
                    for item in os.listdir(dir_name):
                        item_path = os.path.join(dir_name, item)
                        if os.path.isdir(item_path):
                            sub_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                            files.extend([f"{item}/{f}" for f in sub_files])
                elif dir_name == "strategy_reports":
                    # 检查时间戳子文件夹
                    for item in os.listdir(dir_name):
                        item_path = os.path.join(dir_name, item)
                        if os.path.isdir(item_path):
                            sub_files = [f for f in os.listdir(item_path) if f.endswith('.md')]
                            files.extend([f"{item}/{f}" for f in sub_files])
                else:
                    files = [f for f in os.listdir(dir_name) if f.endswith(('.json', '.md'))]
                
                print(f"  {desc}: {len(files)} 个")
                for file in sorted(files)[:3]:
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... 还有 {len(files) - 3} 个文件")
                
                total_files += len(files)
        
        # 最终总结
        if is_complete and total_files > 0:
            print(f"\n🎉 标准LangChain Agent分析成功完成!")
            print(f"🎯 符合官方Agent定义的标准实现")
            print(f"🤖 Agent通过LLM推理决定工具调用序列")
            print(f"📄 共生成 {total_files} 个专业分析文件")
            print(f"💡 真正的LangChain Agent + Tools架构!")
        else:
            print(f"\n⚠️ 分析未完全完成")
            print(f"🔍 请检查Agent执行过程和工具调用结果")
            
    except Exception as e:
        print(f"\n❌ 标准Agent系统执行失败: {str(e)}")
        print("🔧 请检查环境配置和Agent状态")

if __name__ == "__main__":
    main() 