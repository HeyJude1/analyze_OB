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

# ===== 简化的状态定义 =====
class ConfigBasedState(TypedDict):
    """基于config.json的简化状态"""
    # 工作流只需要基本的运行时状态
    iteration_count: int
    max_iterations: int
    errors: list

# ===== 简化的Agent工作流 =====
class ConfigBasedWorkflow:
    """基于config.json的简化Agent工作流"""
    
    def __init__(self):
        self.factory = StandardAgentFactory()
        
        # 创建标准LangChain Agents
        self.master_coordinator = self.factory.create_master_coordinator_agent()
        self.scout_specialist = self.factory.create_scout_specialist_agent()
        self.analyzer_specialist = self.factory.create_analyzer_specialist_agent()
        self.strategist_specialist = self.factory.create_strategist_specialist_agent()
        self.individual_summarizer = self.factory.create_individual_summarizer_agent()
        self.final_summarizer = self.factory.create_final_summarizer_agent()
        
        # 构建简化的工作流
        self.workflow = self._build_config_based_workflow()
    
    def _build_config_based_workflow(self) -> StateGraph:
        """构建基于config.json的简化工作流"""
        workflow = StateGraph(ConfigBasedState)
        
        # 添加节点
        workflow.add_node("master_planning", self.master_planning_node)
        workflow.add_node("scout_work", self.scout_work_node)
        workflow.add_node("analyzer_work", self.analyzer_work_node)
        workflow.add_node("strategist_work", self.strategist_work_node)
        workflow.add_node("individual_summarizer_work", self.individual_summarizer_work_node)
        workflow.add_node("final_summarizer_work", self.final_summarizer_work_node)
        workflow.add_node("coordination_check", self.coordination_check_node)
        
        # 设置入口点
        workflow.add_edge(START, "master_planning")
        
        # 主要工作流路径
        workflow.add_edge("master_planning", "coordination_check")
        
        # 协调检查后的路由
        workflow.add_conditional_edges(
            "coordination_check",
            self._route_by_config,
            {
                "scout": "scout_work",
                "analyze": "analyzer_work",
                "strategize": "strategist_work",
                "individual_summarize": "individual_summarizer_work",
                "final_summarize": "final_summarizer_work",
                "complete": END
            }
        )
        
        # 专家工作完成后返回协调检查
        workflow.add_edge("scout_work", "coordination_check")
        workflow.add_edge("analyzer_work", "coordination_check")
        workflow.add_edge("strategist_work", "coordination_check")
        workflow.add_edge("individual_summarizer_work", "coordination_check")
        workflow.add_edge("final_summarizer_work", "coordination_check")
        
        return workflow.compile()
    
    def _route_by_config(self, state: ConfigBasedState) -> str:
        """基于config.json中的next_action进行路由"""
        try:
            # 直接从config.json读取next_action
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            next_action = config.get("workflow", {}).get("next_action", "complete")
            workflow_complete = config.get("workflow", {}).get("workflow_complete", False)
            current_algorithm = config.get("workflow", {}).get("current_algorithm", "")
            completed_tasks = config.get("workflow", {}).get("completed_tasks", [])
            
            if workflow_complete:
                next_action = "complete"
            
            # 显示详细的执行状态
            print(f"🎯 路由决策: {next_action}")
            if current_algorithm:
                print(f"📍 当前算子: {current_algorithm}")
            if completed_tasks:
                print(f"✅ 已完成任务: {', '.join(completed_tasks[-3:])}")  # 显示最近3个任务
            
            return next_action
            
        except Exception as e:
            print(f"⚠️ 路由读取失败: {e}")
            return "complete"
    
    def master_planning_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Master规划节点 - 简化版"""
        print(f"🎯 [Master Planning] 使用config.json管理状态...")
        
        try:
            # 让Master Agent进行初始规划
            planning_input = """
            请作为Master协调器进行工作流规划：
            
            根据用户需求确定分析类型和算子列表，初始化工作流状态，
            并决定第一步应该执行的任务。
            
            请开始执行规划任务。
            """
            
            result = self.master_coordinator.invoke({"input": planning_input})
            print(f"✅ Master规划完成")
            
        except Exception as e:
            error_msg = f"Master规划失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def coordination_check_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """协调检查节点 - 让Agent通过config.json自主决策"""
        print(f"🧠 [Coordination Check] Agent通过config.json自主决策...")
        
        try:
            # 让Master Agent分析状态并决策
            coordination_input = """
            请作为Master协调器进行状态分析和决策：
            
            分析当前工作流进度，判断已完成的任务，确定下一步应该执行的操作。
            如果当前算子的所有阶段都已完成，则切换到下一个算子。
            
            请分析当前状态并决定下一步行动。
            """
            
            result = self.master_coordinator.invoke({"input": coordination_input})
            
            # 更新迭代计数
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            
            print(f"✅ Agent完成状态分析和决策")
            
            # 检查完成条件
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            if (config.get("workflow", {}).get("workflow_complete", False) or 
                state["iteration_count"] >= state["max_iterations"]):
                print("✅ 工作流准备完成")
                
        except Exception as e:
            error_msg = f"协调决策失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def scout_work_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Scout工作节点 - 简化版"""
        print(f"🔍 [Scout Work] Agent执行文件发现...")
        
        try:
            scout_input = """
            请作为Scout专家执行文件发现任务：
            
            任务目标：发现当前算子的代表性实现文件（3-5个），分析其架构特征，
            将结果保存为JSON格式，并标记任务完成状态。
            
            请开始执行文件发现任务。
            """
            
            result = self.scout_specialist.invoke({"input": scout_input})
            print(f"✅ Scout Agent完成工作")
            
        except Exception as e:
            error_msg = f"Scout Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def analyzer_work_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Analyzer工作节点 - 简化版"""
        print(f"📊 [Analyzer Work] Agent执行代码分析...")
        
        try:
            analyzer_input = """
            请作为Analyzer专家执行代码分析任务：
            
            任务目标：深度分析当前算子的优化技术，按三层框架（算法层、代码层、指令层）
            分类识别优化策略，生成结构化分析报告，并标记任务完成状态。
            
            请开始执行代码分析任务。
            """
            
            result = self.analyzer_specialist.invoke({"input": analyzer_input})
            print(f"✅ Analyzer Agent完成工作")
            
        except Exception as e:
            error_msg = f"Analyzer Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def strategist_work_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Strategist工作节点 - 简化版"""
        print(f"🎯 [Strategist Work] Agent执行策略提炼...")
        
        try:
            strategist_input = """
            请作为Strategist专家执行策略提炼任务：
            
            任务目标：将技术分析结果转化为可实施的优化策略，生成包含代码示例和
            实施步骤的Markdown报告，并标记任务完成状态。
            
            请开始执行策略提炼任务。
            """
            
            result = self.strategist_specialist.invoke({"input": strategist_input})
            print(f"✅ Strategist Agent完成工作")
            
        except Exception as e:
            error_msg = f"Strategist Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def individual_summarizer_work_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Individual Summarizer Agent工作节点 - 单算子总结"""
        try:
            print("🔍 [Individual Summarizer Work] Agent生成单算子总结...")
            
            # 获取输入：当前算子名称和任务描述
            config = self._load_config()
            current_algorithm = config["workflow"]["current_algorithm"]
            
            input_prompt = f"""
请为{current_algorithm}算子生成专门的优化总结报告：

🎯 **任务：** 整合该算子的发现、分析、策略结果，生成完整的单算子优化总结

📋 **要求：**
1. 读取该算子的discovery、analysis、strategy三个文件
2. 整合信息生成单算子总结报告
3. 按照final_optimization_summary.md的格式生成
4. 保存为{current_algorithm}_summary.md
5. 完成后标记任务为已完成

💡 现在开始为{current_algorithm}算子生成优化总结！
"""
            
            result = self.individual_summarizer.invoke({
                "input": input_prompt
            })
            
            print(f"✅ Individual Summarizer Agent完成工作")
            return state
            
        except Exception as e:
            print(f"❌ Individual Summarizer Agent工作失败: {e}")
            state["errors"].append(f"individual_summarizer_error: {str(e)}")
            return state

    def final_summarizer_work_node(self, state: ConfigBasedState) -> ConfigBasedState:
        """Final Summarizer工作节点 - 跨算子最终总结"""
        print(f"📝 [Final Summarizer Work] Agent执行跨算子最终总结...")
        
        try:
            final_summarizer_input = """
            请作为Final Summarizer专家执行跨算子最终总结任务：
            
            任务目标：分析所有算子的个人总结报告（_summary.md文件），识别通用优化模式，
            生成综合性的最终总结报告，并标记整个工作流完成。
            
            请开始执行最终总结任务。
            """
            
            result = self.final_summarizer.invoke({"input": final_summarizer_input})
            print(f"✅ Final Summarizer Agent完成工作")
            
        except Exception as e:
            error_msg = f"Final Summarizer Agent工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def run_config_based_workflow(self, user_request: str) -> ConfigBasedState:
        """运行基于config.json的简化工作流"""
        # 初始化简化状态
        initial_state = ConfigBasedState(
            iteration_count=0,
            max_iterations=50,
            errors=[]
        )
        
        # 将用户请求更新到config.json
        try:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            # 分析用户请求类型
            if "快速" in user_request or "quick" in user_request.lower():
                analysis_type = "quick"
            elif "全面" in user_request or "comprehensive" in user_request.lower():
                analysis_type = "comprehensive"
            else:
                analysis_type = "custom"
            
            # 初始化workflow状态到config.json
            config["workflow"] = {
                "user_request": user_request,
                "analysis_type": analysis_type,
                "current_algorithm": "",
                "current_algorithm_index": 0,
                "completed_tasks": [],
                "algorithms": [],
                "workflow_complete": False,
                "report_folder": f"results/{time.strftime('%Y%m%d_%H%M%S')}",
                "iteration_count": 0,
                "errors": [],
                "next_action": "planning"
            }
            
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"❌ 配置初始化失败: {e}")
            initial_state["errors"].append(str(e))
        
        # 工作流配置
        config = {
            "recursion_limit": 60,
            "max_iterations": 50
        }
        
        print(f"🚀 启动基于config.json的简化Agent工作流")
        print(f"📝 用户请求: {user_request}")
        print(f"💾 状态管理: config.json")
        print()
        
        try:
            final_state = self.workflow.invoke(initial_state, config=config)
            return final_state
            
        except Exception as e:
            print(f"❌ 工作流执行失败: {str(e)}")
            initial_state["errors"].append(str(e))
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
    standard_workflow = ConfigBasedWorkflow()
    
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
        final_state = standard_workflow.run_config_based_workflow(user_request)
        
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
        
        # 检查results目录下的最新时间戳文件夹
        total_files = 0
        if not os.path.exists("results"):
            print("  📁 输出目录 results/ 不存在")
        else:
            # 找到最新的时间戳文件夹
            timestamp_folders = [d for d in os.listdir("results") if os.path.isdir(os.path.join("results", d))]
            if not timestamp_folders:
                print("  📁 results/ 目录下没有时间戳文件夹")
            else:
                latest_folder = sorted(timestamp_folders)[-1]  # 取最新的
                base_path = os.path.join("results", latest_folder)
                print(f"  📁 检查最新输出: results/{latest_folder}/")
                
                for dir_name, desc in dirs_to_check:
                    full_path = os.path.join(base_path, dir_name)
                    if os.path.exists(full_path):
                        files = [f for f in os.listdir(full_path) if f.endswith(('.json', '.md'))]
                        print(f"    {desc}: {len(files)} 个")
                        for file in sorted(files)[:3]:
                            print(f"      - {file}")
                        if len(files) > 3:
                            print(f"      ... 还有 {len(files) - 3} 个文件")
                        total_files += len(files)
                    else:
                        print(f"    {desc}: 未找到")
        
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