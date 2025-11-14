#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析 - 纯Agent架构版本
真正实现"让AI思考一切"，消除所有硬编码逻辑

架构特点:
- 每个Node = 一个完整的Agent调用
- 所有结果解析和状态管理都由Agent的工具完成
- 消除游离在Agent边界外的硬编码逻辑
- 真正的智能体自主决策系统
"""

import os
import time
from typing import Dict, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

# LangGraph imports (纯Agent架构专用)
from langgraph.graph import StateGraph, START, END

# 本地imports
from analyze_pure_agent import PureAgentFactory

# 加载环境变量
load_dotenv()

# ===== 纯Agent架构状态定义 =====
class PureAgentState(TypedDict):
    """纯Agent架构的简化状态 - 主要由Agent自主管理"""
    # 用户输入
    user_request: str
    
    # Master Agent的决策结果 (通过结构化LLM链自动解析)
    master_decision: Dict[str, Any]
    
    # Worker Agent的状态容器 (Agent通过工具自主更新)
    worker_states: Dict[str, Dict[str, Any]]
    
    # 简单的运行状态追踪
    is_workflow_complete: bool
    iteration_count: int
    max_iterations: int
    
    # 错误处理
    errors: list
    messages: list

# ===== 纯Agent架构工作流 =====
class PureAgentWorkflow:
    """纯Agent架构工作流 - 真正让AI决策一切"""
    
    def __init__(self):
        self.factory = PureAgentFactory()
        
        # 创建Master Agent
        self.master_agent = self.factory.create_master_agent()
        
        # Worker Agents将在运行时根据需要创建，避免预先创建
        self.worker_agents = {}
        
        # 构建纯Agent工作流
        self.workflow = self._build_pure_agent_workflow()
    
    def _get_or_create_worker_agent(self, agent_type: str, state: PureAgentState):
        """按需创建Worker Agent，每个Agent都有独立的状态容器"""
        if agent_type not in self.worker_agents:
            # 为每个Worker Agent创建独立的状态容器
            state_container = {}
            self.worker_agents[agent_type] = {
                "agent": self.factory.create_worker_agent_with_tools(agent_type, state_container),
                "state_container": state_container
            }
            
            # 在全局状态中记录这个Agent的状态容器
            if "worker_states" not in state:
                state["worker_states"] = {}
            state["worker_states"][agent_type] = state_container
        
        return self.worker_agents[agent_type]
    
    def _build_pure_agent_workflow(self) -> StateGraph:
        """构建纯Agent架构的LangGraph工作流"""
        workflow = StateGraph(PureAgentState)
        
        # 添加节点 - 每个Node都是纯Agent调用
        workflow.add_node("master_control", self.master_control_node)
        workflow.add_node("scout_work", self.scout_work_node)
        workflow.add_node("analyzer_work", self.analyzer_work_node)
        workflow.add_node("strategist_work", self.strategist_work_node)
        workflow.add_node("summarizer_work", self.summarizer_work_node)
        
        # 设置入口点
        workflow.add_edge(START, "master_control")
        
        # 关键：使用Master Agent的推理结果进行路由
        workflow.add_conditional_edges(
            "master_control",
            self._route_by_master_decision,
            {
                "route_to_scout": "scout_work",
                "route_to_analyzer": "analyzer_work",
                "route_to_strategist": "strategist_work", 
                "route_to_summarizer": "summarizer_work",
                "complete": END
            }
        )
        
        # Worker完成后返回Master控制
        workflow.add_edge("scout_work", "master_control")
        workflow.add_edge("analyzer_work", "master_control")
        workflow.add_edge("strategist_work", "master_control")
        workflow.add_edge("summarizer_work", "master_control")
        
        return workflow.compile()
    
    def _route_by_master_decision(self, state: PureAgentState) -> str:
        """基于Master Agent的推理结果进行路由（最小化硬编码）"""
        try:
            master_decision = state.get("master_decision", {})
            next_action = master_decision.get("next_action", "complete")
            
            # 这是唯一保留的硬编码部分，但逻辑来源于Master Agent的推理
            if next_action in ["route_to_scout", "route_to_analyzer", "route_to_strategist", "route_to_summarizer"]:
                return next_action
            else:
                return "complete"
                
        except Exception as e:
            print(f"⚠️ 路由解析失败: {e}")
            return "complete"
    
    def master_control_node(self, state: PureAgentState) -> PureAgentState:
        """Master Agent控制节点 - 使用结构化LLM链自动解析"""
        print(f"🧠 [Master Control] 智能决策中...")
        
        try:
            # 构建给Master Agent的完整上下文
            user_request = state["user_request"]
            current_iteration = state.get("iteration_count", 0)
            worker_states = state.get("worker_states", {})
            previous_decision = state.get("master_decision", {})
            
            # 让Master Agent基于全部上下文进行完整决策
            master_input = f"""
            **工作流控制请求:**
            
            **用户原始需求:** {user_request}
            
            **当前状态:**
            - 迭代次数: {current_iteration}
            - 最大迭代: {state.get('max_iterations', 50)}
            - 上次决策: {previous_decision}
            - Worker状态: {worker_states}
            
            **Master Agent任务:**
            请基于以上信息进行完整的工作流控制决策，包括：
            1. 分析当前工作流进展状态
            2. 决定下一步具体行动
            3. 设定相关参数和指令
            4. 评估质量和进度
            5. 输出结构化决策结果
            
            **重要:** 你拥有完全的决策权，请通过推理自主决定一切。
            你的输出将自动解析为结构化格式，请包含所有必需的字段。
            """
            
            # 使用结构化LLM链进行决策和自动解析
            try:
                master_decision = self.factory.master_llm_chain.invoke({"content": master_input})
                state["master_decision"] = master_decision
                
                # 显示Master Agent的推理过程
                reasoning = master_decision.get("reasoning", "未提供推理过程")
                next_action = master_decision.get("next_action", "complete")
                current_stage = master_decision.get("current_stage", "unknown")
                current_algorithm = master_decision.get("current_algorithm", "unknown")
                
                print(f"🎯 Master决策: {next_action}")
                print(f"📊 当前阶段: {current_stage}")
                print(f"🔧 当前算子: {current_algorithm}")
                print(f"💭 推理过程: {reasoning[:100]}...")
                
                # 更新迭代计数
                state["iteration_count"] = current_iteration + 1
                
                # 检查是否应该完成
                if (next_action == "complete" or 
                    master_decision.get("workflow_status") == "completed" or
                    current_iteration >= state.get("max_iterations", 50)):
                    state["is_workflow_complete"] = True
                    print("✅ Master Agent决定完成工作流")
                
            except Exception as parse_error:
                print(f"⚠️ Master决策结构化解析失败: {parse_error}")
                # 回退到直接调用Master Agent
                result = self.master_agent.invoke({"input": master_input})
                try:
                    master_decision = self.factory.master_parser.parse(result["output"])
                    state["master_decision"] = master_decision
                    print(f"🔄 回退解析成功: {master_decision.get('next_action', 'complete')}")
                except:
                    print("❌ 完全解析失败，终止流程")
                    state["master_decision"] = {"next_action": "complete", "reasoning": "解析失败，安全终止"}
                    state["is_workflow_complete"] = True
                
        except Exception as e:
            error_msg = f"Master控制失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["is_workflow_complete"] = True
        
        return state
    
    def scout_work_node(self, state: PureAgentState) -> PureAgentState:
        """Scout Agent工作节点 - Agent自主管理状态"""
        print(f"🔍 [Scout Work] Agent自主工作中...")
        
        try:
            master_decision = state["master_decision"]
            instructions = master_decision.get("instructions_for_worker", "")
            current_algorithm = master_decision.get("current_algorithm", "")
            
            # 获取或创建Scout Agent（带独立状态容器）
            scout_info = self._get_or_create_worker_agent("scout", state)
            scout_agent = scout_info["agent"]
            
            # 构建给Scout Agent的指令
            scout_input = f"""
            **Master Agent指令:** {instructions}
            
            **任务目标:** 为 {current_algorithm} 算子自主发现实现文件
            
            **完整自主工作要求:**
            你拥有完全的工作自主权和状态管理权，请：
            1. 自主设计搜索策略和执行计划
            2. 自主探索文件系统，发现相关实现
            3. 自主分类和评估文件重要性
            4. 自主生成JSON格式结果并保存
            5. **使用state_update工具更新你的工作状态**
            6. **使用result_verification工具验证工作成果**
            7. **最终输出结构化的工作总结**
            
            **重要**: 
            - 你必须主动使用tools管理状态和验证结果
            - 你的最终输出将自动解析为结构化格式
            - 请确保输出包含work_completed, work_summary等字段
            """
            
            # Scout Agent自主执行工作（包含状态管理）
            result = scout_agent.invoke({"input": scout_input})
            
            # 使用结构化LLM链解析Worker结果
            try:
                worker_result = self.factory.worker_llm_chain.invoke({"content": result["output"]})
                
                work_completed = worker_result.get("work_completed", "false").lower() == "true"
                files_count = worker_result.get("found_files_count", "0")
                work_summary = worker_result.get("work_summary", "")
                
                if work_completed:
                    print(f"✅ Scout Agent自主完成: 发现 {files_count} 个文件")
                    print(f"📝 工作总结: {work_summary}")
                else:
                    print(f"⚠️ Scout Agent工作未完成: {work_summary}")
                    
            except Exception as parse_error:
                print(f"⚠️ Scout结果结构化解析失败: {parse_error}")
                # 回退到简单状态记录
                scout_info["state_container"]["work_completed"] = "unknown"
                scout_info["state_container"]["work_summary"] = "解析失败但Agent可能已完成工作"
                
        except Exception as e:
            error_msg = f"Scout工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def analyzer_work_node(self, state: PureAgentState) -> PureAgentState:
        """Analyzer Agent工作节点 - Agent自主管理状态"""
        print(f"📊 [Analyzer Work] Agent自主分析中...")
        
        try:
            master_decision = state["master_decision"]
            instructions = master_decision.get("instructions_for_worker", "")
            current_algorithm = master_decision.get("current_algorithm", "")
            
            # 获取或创建Analyzer Agent
            analyzer_info = self._get_or_create_worker_agent("analyzer", state)
            analyzer_agent = analyzer_info["agent"]
            
            analyzer_input = f"""
            **Master Agent指令:** {instructions}
            
            **任务目标:** 为 {current_algorithm} 算子自主分析代码实现
            
            **完整自主工作要求:**
            你拥有完全的分析自主权和状态管理权，请：
            1. 自主读取相关发现结果和源代码
            2. 自主设计分析框架和深度
            3. 自主分类优化技术（算法/代码/指令层）
            4. 自主生成JSON格式分析结果并保存
            5. **使用state_update工具更新你的工作状态**
            6. **使用result_verification工具验证分析成果**
            7. **最终输出结构化的工作总结**
            
            **重要**: 
            - 你必须主动使用tools管理状态和验证结果
            - 你的最终输出将自动解析为结构化格式
            - 请确保输出包含work_completed, optimization_layers等字段
            """
            
            result = analyzer_agent.invoke({"input": analyzer_input})
            
            # 使用结构化LLM链解析Worker结果
            try:
                worker_result = self.factory.worker_llm_chain.invoke({"content": result["output"]})
                
                work_completed = worker_result.get("work_completed", "false").lower() == "true"
                optimization_layers = worker_result.get("optimization_layers", "0")
                work_summary = worker_result.get("work_summary", "")
                
                if work_completed:
                    print(f"✅ Analyzer Agent自主完成: 发现 {optimization_layers} 层优化技术")
                    print(f"📝 工作总结: {work_summary}")
                else:
                    print(f"⚠️ Analyzer Agent工作未完成: {work_summary}")
                    
            except Exception as parse_error:
                print(f"⚠️ Analyzer结果结构化解析失败: {parse_error}")
                analyzer_info["state_container"]["work_completed"] = "unknown"
                analyzer_info["state_container"]["work_summary"] = "解析失败但Agent可能已完成工作"
                
        except Exception as e:
            error_msg = f"Analyzer工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def strategist_work_node(self, state: PureAgentState) -> PureAgentState:
        """Strategist Agent工作节点 - Agent自主管理状态"""
        print(f"🎯 [Strategist Work] Agent自主策略制定中...")
        
        try:
            master_decision = state["master_decision"]
            instructions = master_decision.get("instructions_for_worker", "")
            current_algorithm = master_decision.get("current_algorithm", "")
            
            # 获取或创建Strategist Agent
            strategist_info = self._get_or_create_worker_agent("strategist", state)
            strategist_agent = strategist_info["agent"]
            
            strategist_input = f"""
            **Master Agent指令:** {instructions}
            
            **任务目标:** 为 {current_algorithm} 算子自主提炼优化策略
            
            **完整自主工作要求:**
            你拥有完全的策略制定自主权和状态管理权，请：
            1. 自主读取分析结果和相关数据
            2. 自主设计策略框架和深度
            3. 自主提炼优化原则和最佳实践
            4. 自主生成Markdown格式策略报告并保存
            5. **使用state_update工具更新你的工作状态**
            6. **使用result_verification工具验证策略成果**
            7. **最终输出结构化的工作总结**
            
            **重要**: 
            - 你必须主动使用tools管理状态和验证结果
            - 你的最终输出将自动解析为结构化格式
            - 请确保输出包含work_completed, work_summary等字段
            """
            
            result = strategist_agent.invoke({"input": strategist_input})
            
            # 使用结构化LLM链解析Worker结果
            try:
                worker_result = self.factory.worker_llm_chain.invoke({"content": result["output"]})
                
                work_completed = worker_result.get("work_completed", "false").lower() == "true"
                work_summary = worker_result.get("work_summary", "")
                
                if work_completed:
                    print(f"✅ Strategist Agent自主完成: {work_summary}")
                else:
                    print(f"⚠️ Strategist Agent工作未完成: {work_summary}")
                    
            except Exception as parse_error:
                print(f"⚠️ Strategist结果结构化解析失败: {parse_error}")
                strategist_info["state_container"]["work_completed"] = "unknown"
                strategist_info["state_container"]["work_summary"] = "解析失败但Agent可能已完成工作"
                
        except Exception as e:
            error_msg = f"Strategist工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def summarizer_work_node(self, state: PureAgentState) -> PureAgentState:
        """Summarizer Agent工作节点 - Agent自主管理状态"""
        print(f"📝 [Summarizer Work] Agent自主跨算子总结中...")
        
        try:
            master_decision = state["master_decision"]
            instructions = master_decision.get("instructions_for_worker", "")
            
            # 获取或创建Summarizer Agent
            summarizer_info = self._get_or_create_worker_agent("summarizer", state)
            summarizer_agent = summarizer_info["agent"]
            
            summarizer_input = f"""
            **Master Agent指令:** {instructions}
            
            **任务目标:** 自主生成跨算子优化策略总结
            
            **完整自主工作要求:**
            你拥有完全的总结分析自主权和状态管理权，请：
            1. 自主收集和读取所有策略报告
            2. 自主发现跨算子的通用规律和差异
            3. 自主设计总结框架和分析维度
            4. 自主生成高质量总结报告并保存
            5. **使用state_update工具更新你的工作状态**
            6. **使用result_verification工具验证总结成果**
            7. **最终输出结构化的工作总结**
            
            **重要**: 
            - 你必须主动使用tools管理状态和验证结果
            - 你的最终输出将自动解析为结构化格式
            - 请确保输出包含work_completed, work_summary等字段
            """
            
            result = summarizer_agent.invoke({"input": summarizer_input})
            
            # 使用结构化LLM链解析Worker结果
            try:
                worker_result = self.factory.worker_llm_chain.invoke({"content": result["output"]})
                
                work_completed = worker_result.get("work_completed", "false").lower() == "true"
                work_summary = worker_result.get("work_summary", "")
                
                if work_completed:
                    print(f"✅ Summarizer Agent自主完成: {work_summary}")
                    state["is_workflow_complete"] = True  # 总结完成即工作流完成
                else:
                    print(f"⚠️ Summarizer Agent工作未完成: {work_summary}")
                    
            except Exception as parse_error:
                print(f"⚠️ Summarizer结果结构化解析失败: {parse_error}")
                summarizer_info["state_container"]["work_completed"] = "unknown"
                summarizer_info["state_container"]["work_summary"] = "解析失败但Agent可能已完成工作"
                state["is_workflow_complete"] = True  # 容错处理，避免无限循环
                
        except Exception as e:
            error_msg = f"Summarizer工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["is_workflow_complete"] = True  # 容错处理
        
        return state
    
    def run_pure_agent_workflow(self, user_request: str) -> PureAgentState:
        """运行纯Agent架构的工作流"""
        # 初始化状态
        initial_state = PureAgentState(
            user_request=user_request,
            master_decision={},
            worker_states={},
            is_workflow_complete=False,
            iteration_count=0,
            max_iterations=30,
            errors=[],
            messages=[HumanMessage(content=user_request)]
        )
        
        # 工作流配置
        config = {
            "recursion_limit": 50,
            "max_iterations": 30
        }
        
        print(f"🚀 启动纯Agent架构工作流")
        print(f"🧠 理念: 让AI思考一切，Agent自主管理状态")
        print(f"🔧 特点: 结构化LLM链 + Agent工具自主状态管理")
        print(f"📝 用户请求: {user_request}")
        print(f"⚙️ 配置: 递归限制={config['recursion_limit']}")
        print()
        
        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            return final_state
            
        except Exception as e:
            print(f"❌ 纯Agent工作流执行失败: {str(e)}")
            initial_state["errors"].append(str(e))
            initial_state["is_workflow_complete"] = True
            return initial_state

def main():
    """主函数 - 纯Agent架构系统入口"""
    # 环境检查
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请设置DASHSCOPE_API_KEY环境变量")
        return
    
    if not os.path.exists("./OpenBLAS-develop"):
        print("❌ 错误: 未找到OpenBLAS-develop目录")
        return
    
    # 创建纯Agent工作流
    pure_workflow = PureAgentWorkflow()
    
    # 用户交互
    print("🧠 OpenBLAS优化策略分析 - 纯Agent架构版本")
    print("=" * 60)
    print("🎯 设计理念: 让AI思考一切，Agent自主管理状态")
    print("🔧 技术特点: 结构化LLM链 + Agent工具自主状态管理")
    print("📊 架构优势: 消除所有硬编码逻辑，真正的智能体系统")
    print("=" * 60)
    print()
    
    print("分析选项:")
    print("1. 快速分析 - AI自主分析核心算子")
    print("2. 全面分析 - AI自主分析完整算子集")
    print("3. 自定义分析 - 指定算子让AI自主分析")
    print("4. 直接输入 - 自然语言描述需求")
    print()
    
    choice = input("请选择 (1-4) 或直接输入需求: ").strip()
    
    # 解析用户输入
    if choice == "1":
        user_request = "请进行快速分析，分析核心BLAS算子的优化策略"
    elif choice == "2":
        user_request = "请进行全面分析，完整分析BLAS算子的优化策略" 
    elif choice == "3":
        algorithms = input("请输入要分析的算子: ").strip()
        user_request = f"请自主分析以下算子的优化策略: {algorithms}"
    elif choice == "4":
        user_request = input("请输入分析需求: ").strip()
    else:
        user_request = choice
    
    if not user_request:
        print("❌ 未提供有效需求")
        return
    
    print(f"\n🎯 AI接收需求: {user_request}")
    print("🧠 启动真正的纯Agent智能分析系统...")
    print("🔧 所有逻辑都由Agent自主决策和管理...")
    print()
    
    try:
        # 运行纯Agent工作流
        final_state = pure_workflow.run_pure_agent_workflow(user_request)
        
        # 显示结果
        print("\n" + "=" * 60)
        print("📊 纯Agent架构工作流完成")
        print("=" * 60)
        
        # 分析执行情况
        iteration_count = final_state.get("iteration_count", 0)
        errors = final_state.get("errors", [])
        is_complete = final_state.get("is_workflow_complete", False)
        master_decision = final_state.get("master_decision", {})
        worker_states = final_state.get("worker_states", {})
        
        print(f"\n🔄 执行统计:")
        print(f"  - Master Agent决策次数: {iteration_count}")
        print(f"  - 工作流状态: {'✅ 完成' if is_complete else '⚠️ 未完成'}")
        print(f"  - 最终决策状态: {master_decision.get('workflow_status', '未知')}")
        print(f"  - Worker Agent数量: {len(worker_states)}")
        
        if master_decision:
            target_algorithms = master_decision.get("target_algorithms", "未指定")
            analysis_type = master_decision.get("analysis_type", "未知")
            print(f"  - 分析类型: {analysis_type}")
            print(f"  - 目标算子: {target_algorithms}")
        
        # 显示Worker Agent状态 (由Agent自主管理)
        if worker_states:
            print(f"\n🤖 Worker Agent自主状态:")
            for agent_type, agent_state in worker_states.items():
                work_completed = agent_state.get("work_completed", "unknown")
                work_summary = agent_state.get("work_summary", "未提供")
                print(f"  - {agent_type.upper()}: {work_completed}")
                if work_summary != "未提供":
                    print(f"    总结: {work_summary[:80]}...")
        
        # 显示错误
        if errors:
            print(f"\n⚠️ 执行过程中的问题 ({len(errors)} 个):")
            for i, error in enumerate(errors[:3], 1):
                print(f"  {i}. {error}")
            if len(errors) > 3:
                print(f"  ... 还有 {len(errors) - 3} 个问题")
        
        # 显示生成的文件
        print(f"\n📁 Agent自主生成的文件:")
        
        dirs_to_check = [
            ("discovery_results", "🔍 发现结果"),
            ("analysis_results", "📊 分析结果"), 
            ("strategy_reports", "🎯 策略报告")
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
            print(f"\n🎉 纯Agent架构分析成功完成!")
            print(f"🧠 AI自主决策了整个分析流程")
            print(f"🔧 Agent自主管理了所有状态")
            print(f"📄 共生成 {total_files} 个分析文件")
            print(f"💡 真正实现了\"让AI思考一切\"的理念!")
        else:
            print(f"\n⚠️ 分析未完全完成")
            print(f"🔍 请检查AI决策过程和Agent状态管理")
            
    except Exception as e:
        print(f"\n❌ 纯Agent系统执行失败: {str(e)}")
        print("🔧 请检查环境配置和Agent工具状态")

if __name__ == "__main__":
    main() 