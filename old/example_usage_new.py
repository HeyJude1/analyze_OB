#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析 - Master Agent调度系统
基于LangGraph的智能多Agent协作工作流
"""

import os
import time
from typing import List, Literal, Dict, Any
from typing_extensions import TypedDict  # 官方推荐使用typing_extensions
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# LangGraph imports (官方推荐的导入方式)
from langgraph.graph import StateGraph, START, END

# 本地imports
from analyze_new import OpenBLASMasterAgentFactory

# 加载环境变量
load_dotenv()

# ===== Master工作流状态定义 =====
class MasterWorkflowState(TypedDict):
    """Master Agent调度系统的工作流状态"""
    # 用户需求相关
    user_request: str                    # 用户原始请求
    analysis_type: str                   # "quick", "comprehensive", "custom"
    target_algorithms: List[str]         # 要分析的算子列表
    
    # 工作流进度相关
    current_algorithm: str               # 当前处理的算子
    current_stage: str                   # 当前阶段: "planning", "scout", "analyze", "strategize", "summarize"
    completed_algorithms: List[str]      # 已完成的算子列表
    algorithm_progress: Dict[str, Dict]  # 每个算子的详细进度
    
    # 文件路径和结果相关
    report_folder: str                   # 报告文件夹时间戳
    stage_results: Dict[str, Any]        # 各阶段的结果数据
    quality_checks: Dict[str, bool]      # 质量检查结果
    
    # 控制流相关
    master_decision: str                 # Master Agent的决策: "continue", "retry", "skip", "complete"
    retry_count: int                     # 重试次数
    max_retries: int                     # 最大重试次数
    
    # 错误和消息
    errors: List[str]                    # 错误记录
    messages: List[BaseMessage]          # Agent间消息历史

# ===== Master Agent调度工作流 =====
class OpenBLASMasterWorkflow:
    """基于LangGraph的Master Agent智能调度系统"""
    
    def __init__(self):
        self.factory = OpenBLASMasterAgentFactory()
        
        # 创建所有Agent
        self.master_agent = self.factory.create_master_agent()
        self.scout_agent = self.factory.create_scout_agent()
        self.analyzer_agent = self.factory.create_analyzer_agent()
        self.strategist_agent = self.factory.create_strategist_agent()
        self.summarizer_agent = self.factory.create_summarizer_agent()
        
        # 创建Master Agent专用的质量检查和决策Agent
        self.quality_check_agent = self.factory.create_quality_check_agent()
        self.decision_agent = self.factory.create_decision_agent()
        
        # 构建工作流
        self.workflow = self._build_master_workflow()
    
    def _build_master_workflow(self) -> StateGraph:
        """构建Master Agent调度的LangGraph工作流"""
        workflow = StateGraph(MasterWorkflowState)
        
        # 添加节点
        workflow.add_node("master_planning", self.master_planning_node)
        workflow.add_node("master_dispatch", self.master_dispatch_node)
        workflow.add_node("scout_work", self.scout_work_node)
        workflow.add_node("analyzer_work", self.analyzer_work_node)
        workflow.add_node("strategist_work", self.strategist_work_node)
        workflow.add_node("master_quality_check", self.master_quality_check_node)
        workflow.add_node("master_next_decision", self.master_next_decision_node)
        workflow.add_node("summarizer_work", self.summarizer_work_node)
        
        # 设置入口点 (使用官方推荐的START常量)
        workflow.add_edge(START, "master_planning")
        
        # 添加边
        workflow.add_edge("master_planning", "master_dispatch")
        
        # 条件边：Master Agent调度Worker Agents
        workflow.add_conditional_edges(
            "master_dispatch",
            self._route_to_worker,
            {
                "scout": "scout_work",
                "analyzer": "analyzer_work", 
                "strategist": "strategist_work",
                "summarizer": "summarizer_work",
                "complete": END
            }
        )
        
        # Worker工作完成后返回Master质量检查
        workflow.add_edge("scout_work", "master_quality_check")
        workflow.add_edge("analyzer_work", "master_quality_check")
        workflow.add_edge("strategist_work", "master_quality_check")
        
        # 质量检查后Master决策下一步
        workflow.add_edge("master_quality_check", "master_next_decision")
        
        # 条件边：Master决策流程控制
        workflow.add_conditional_edges(
            "master_next_decision",
            self._route_master_decision,
            {
                "continue": "master_dispatch",
                "retry": "master_dispatch", 
                "summarize": "summarizer_work",
                "complete": END
            }
        )
        
        # 总结完成后结束
        workflow.add_edge("summarizer_work", END)
        
        return workflow.compile()
    
    def _route_to_worker(self, state: MasterWorkflowState) -> str:
        """Master Agent路由到合适的Worker Agent"""
        current_stage = state["current_stage"]
        
        if current_stage == "scout":
            return "scout"
        elif current_stage == "analyze":
            return "analyzer"
        elif current_stage == "strategize":
            return "strategist"
        elif current_stage == "summarize":
            return "summarizer"
        else:
            return "complete"
    
    def _route_master_decision(self, state: MasterWorkflowState) -> str:
        """根据Master Agent的决策路由下一步"""
        decision = state["master_decision"]
        
        if decision == "continue":
            return "continue"
        elif decision == "retry":
            return "retry"
        elif decision == "summarize":
            return "summarize"
        else:
            return "complete"
    
    def master_planning_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Master Agent规划节点 - 解析用户需求，制定分析计划"""
        print(f"🎯 [Master规划] 分析用户需求...")
        
        try:
            user_request = state["user_request"]
            
            planning_input = f"""
            用户请求: "{user_request}"
            
            请作为Master Agent分析此请求并制定分析计划：
            
            1. **需求识别**: 
               - 如果是"快速分析"，算子列表为: ['gemm', 'axpy', 'dot']
               - 如果是"全面分析"，算子列表为: ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger']
               - 如果是自定义分析，请从用户请求中提取算子名称
            
            2. **制定计划**:
               - 确定要分析的算子类型和数量
               - 设定工作流程: scout -> analyze -> strategize -> summarize
               - 评估预计完成时间
            
            3. **输出格式**:
               明确回答要分析的算子列表，如: "决定分析算子: ['gemm', 'axpy', 'dot']"
            
            请简洁明确地给出分析计划。
            """
            
            result = self.master_agent.invoke({"input": planning_input})
            
            # 解析Master Agent的回复，提取算子列表
            response = result["output"]
            
            # 简单的算子列表解析逻辑
            if "快速" in user_request or "quick" in user_request.lower():
                algorithms = ['gemm', 'axpy', 'dot']
                analysis_type = "quick"
            elif "全面" in user_request or "comprehensive" in user_request.lower():
                algorithms = ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger'] 
                analysis_type = "comprehensive"
            else:
                # 默认快速分析
                algorithms = ['gemm', 'axpy', 'dot']
                analysis_type = "quick"
            
            # 生成报告文件夹时间戳
            report_timestamp = f"{int(time.time())}"
            
            # 更新状态
            state["analysis_type"] = analysis_type
            state["target_algorithms"] = algorithms
            state["current_algorithm"] = algorithms[0] if algorithms else ""
            state["current_stage"] = "scout"
            state["report_folder"] = report_timestamp
            state["algorithm_progress"] = {algo: {"scout": False, "analyze": False, "strategize": False} for algo in algorithms}
            state["stage_results"] = {}
            state["quality_checks"] = {}
            state["master_decision"] = "continue"
            state["retry_count"] = 0
            state["max_retries"] = 3
            
            # 添加消息记录
            state["messages"].append(AIMessage(content=f"Master规划完成: 分析 {len(algorithms)} 个算子 {algorithms}"))
            
            print(f"✅ Master规划完成: {analysis_type}分析，算子列表: {algorithms}")
            print(f"📁 报告将保存到: strategy_reports/report_{report_timestamp}/")
            
        except Exception as e:
            error_msg = f"Master规划失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["master_decision"] = "complete"
        
        return state
    
    def master_dispatch_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Master Agent调度节点 - 分配任务给Worker Agents"""
        current_algorithm = state["current_algorithm"]
        current_stage = state["current_stage"]
        
        print(f"📋 [Master调度] 算子: {current_algorithm}, 阶段: {current_stage}")
        
        try:
            # 根据当前阶段准备调度指令
            if current_stage == "scout":
                dispatch_message = f"开始对 {current_algorithm} 算子进行文件发现工作"
            elif current_stage == "analyze":
                dispatch_message = f"开始对 {current_algorithm} 算子进行代码分析工作"
            elif current_stage == "strategize":
                dispatch_message = f"开始对 {current_algorithm} 算子进行策略提炼工作"
            elif current_stage == "summarize":
                completed_algos = state["completed_algorithms"]
                dispatch_message = f"开始对 {completed_algos} 算子进行跨算子总结工作"
            else:
                dispatch_message = "调度完成"
            
            # 记录调度消息
            state["messages"].append(AIMessage(content=f"Master调度: {dispatch_message}"))
            
            print(f"🔄 {dispatch_message}")
            
        except Exception as e:
            error_msg = f"Master调度失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
        
        return state
    
    def scout_work_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Scout Agent工作节点 - 执行算子文件发现任务"""
        current_algorithm = state["current_algorithm"]
        report_folder = state["report_folder"]
        
        print(f"🔍 [Scout工作] 发现 {current_algorithm} 算子文件...")
        
        try:
            scout_input = f"""
            Master Agent调度任务: 发现 {current_algorithm} 算子的实现文件
            
            请执行以下工作:
            1. 在kernel/目录中搜索 {current_algorithm} 相关文件
            2. 至少发现3种不同架构的实现 (generic, x86_64, arm64等)
            3. 识别每个文件的实现类型 (generic, simd_optimized, microkernel等)
            4. 生成JSON格式的发现结果
            5. 保存到: ../discovery_results/{current_algorithm}_discovered_{report_folder}.json
            
            请使用工具完成工作，并汇报发现的文件数量和架构类型。
            """
            
            result = self.scout_agent.invoke({"input": scout_input})
            
            # 记录结果
            state["stage_results"][f"{current_algorithm}_scout"] = {
                "status": "completed",
                "result": result["output"],
                "timestamp": int(time.time())
            }
            
            # 更新算子进度
            state["algorithm_progress"][current_algorithm]["scout"] = True
            
            print(f"✅ {current_algorithm} Scout工作完成")
            
        except Exception as e:
            error_msg = f"{current_algorithm} Scout工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["stage_results"][f"{current_algorithm}_scout"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": int(time.time())
            }
        
        return state
    
    def analyzer_work_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Analyzer Agent工作节点 - 执行代码分析任务"""
        current_algorithm = state["current_algorithm"]
        report_folder = state["report_folder"]
        
        print(f"📊 [Analyzer工作] 分析 {current_algorithm} 代码实现...")
        
        try:
            analyzer_input = f"""
            Master Agent调度任务: 分析 {current_algorithm} 算子的代码实现
            
            请执行以下工作:
            1. 读取 ../discovery_results/{current_algorithm}_discovered_{report_folder}.json 中的文件列表
            2. 对每个文件进行三层优化技术分析:
               - 算法层: 循环展开、分块、数据重用
               - 代码层: 缓存友好、内存对齐、预取
               - 指令层: SIMD向量化、FMA、指令并行
            3. 生成JSON格式的分析结果
            4. 先创建目录: ../analysis_results/{current_algorithm}/
            5. 然后保存到: ../analysis_results/{current_algorithm}/analysis_{current_algorithm}_{report_folder}.json
            
            请使用工具完成工作，并汇报分析的文件数量和发现的优化技术层数。
            """
            
            result = self.analyzer_agent.invoke({"input": analyzer_input})
            
            # 记录结果
            state["stage_results"][f"{current_algorithm}_analyze"] = {
                "status": "completed",
                "result": result["output"],
                "timestamp": int(time.time())
            }
            
            # 更新算子进度
            state["algorithm_progress"][current_algorithm]["analyze"] = True
            
            print(f"✅ {current_algorithm} Analyzer工作完成")
            
        except Exception as e:
            error_msg = f"{current_algorithm} Analyzer工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["stage_results"][f"{current_algorithm}_analyze"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": int(time.time())
            }
        
        return state
    
    def strategist_work_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Strategist Agent工作节点 - 执行策略提炼任务"""
        current_algorithm = state["current_algorithm"]
        report_folder = state["report_folder"]
        
        print(f"🎯 [Strategist工作] 提炼 {current_algorithm} 优化策略...")
        
        try:
            # 确保时间戳文件夹存在
            timestamp_folder = f"../strategy_reports/report_{report_folder}"
            
            strategist_input = f"""
            Master Agent调度任务: 为 {current_algorithm} 算子生成优化策略报告
            
            请执行以下工作:
            1. 读取 ../analysis_results/{current_algorithm}/analysis_{current_algorithm}_{report_folder}.json
            2. 按照三层分析框架提炼优化策略:
               - 算法设计层次: 计算逻辑优化、空间时间权衡
               - 代码优化层次: 性能加速、循环优化、代码顺序
               - 特有指令层次: 专有指令使用和优化设计
            3. 生成完整的Markdown格式策略报告
            4. **重要**: 先使用list_directory检查 {timestamp_folder} 是否存在，如不存在则创建
            5. 然后保存到: {timestamp_folder}/{current_algorithm}_optimization_analysis.md
            
            **文件夹组织说明:**
            - 每次运行都会创建新的时间戳文件夹: report_{report_folder}
            - 所有算子的策略报告都保存在同一个时间戳文件夹中
            - 最终的总结报告也会保存在这个文件夹中
            
            请使用工具完成工作，并确认策略报告已保存到指定路径。
            """
            
            result = self.strategist_agent.invoke({"input": strategist_input})
            
            # 记录结果
            state["stage_results"][f"{current_algorithm}_strategize"] = {
                "status": "completed",
                "result": result["output"],
                "timestamp": int(time.time())
            }
            
            # 更新算子进度
            state["algorithm_progress"][current_algorithm]["strategize"] = True
            
            # 将完成的算子加入已完成列表
            if current_algorithm not in state["completed_algorithms"]:
                state["completed_algorithms"].append(current_algorithm)
            
            print(f"✅ {current_algorithm} Strategist工作完成")
            
        except Exception as e:
            error_msg = f"{current_algorithm} Strategist工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["stage_results"][f"{current_algorithm}_strategize"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": int(time.time())
            }
        
        return state
    
    def summarizer_work_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Summarizer Agent工作节点 - 执行跨算子总结任务"""
        completed_algorithms = state["completed_algorithms"]
        report_folder = state["report_folder"]
        
        print(f"📝 [Summarizer工作] 生成跨算子总结报告...")
        
        try:
            timestamp_folder = f"../strategy_reports/report_{report_folder}"
            
            summarizer_input = f"""
            Master Agent调度任务: 生成多算子优化策略总结报告
            
            请执行以下工作:
            1. 列出 {timestamp_folder}/ 目录中的所有算子报告
            2. 逐个读取已完成算子的策略报告: {completed_algorithms}
            3. 进行跨算子分析:
               - 跨算子共性分析: 相同优化技术、通用设计模式
               - 架构特化对比: 不同架构的优化差异
               - 性能提升模式: 优化技术收益和适用场景
            4. 生成结构化总结报告
            5. **重要**: 保存到 {timestamp_folder}/optimization_summary_report.md
            
            **总结报告要求:**
            - 分析本次运行的所有算子: {completed_algorithms}
            - 总结报告保存在与算子报告相同的时间戳文件夹中
            - 确保报告包含跨算子的深度对比和通用优化模式提炼
            
            请使用工具完成工作，并汇报分析的算子数量和提炼的通用模式数量。
            """
            
            result = self.summarizer_agent.invoke({"input": summarizer_input})
            
            # 记录结果
            state["stage_results"]["summarize"] = {
                "status": "completed",
                "result": result["output"],
                "algorithms_count": len(completed_algorithms),
                "timestamp": int(time.time())
            }
            
            print(f"✅ 跨算子总结完成，分析了 {len(completed_algorithms)} 个算子")
            
        except Exception as e:
            error_msg = f"Summarizer工作失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["stage_results"]["summarize"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": int(time.time())
            }
        
        return state
    
    def master_quality_check_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Master Agent质量检查节点 - 使用LLM Agent检查Worker Agent工作质量"""
        current_algorithm = state["current_algorithm"]
        current_stage = state["current_stage"]
        report_folder = state["report_folder"]
        
        print(f"🔍 [Master质检] {current_algorithm} {current_stage} 阶段质量检查...")
        
        try:
            # 构建质量检查的输入
            quality_check_input = f"""
            请检查 {current_algorithm} 算子 {current_stage} 阶段的工作质量：
            
            **检查内容:**
            - 算子: {current_algorithm}
            - 阶段: {current_stage}
            - 报告文件夹: {report_folder}
            
            **需要检查的路径:**
            """
            
            if current_stage == "scout":
                quality_check_input += f"- ../discovery_results/{current_algorithm}_discovered_{report_folder}.json"
            elif current_stage == "analyze":
                quality_check_input += f"- ../analysis_results/{current_algorithm}/analysis_{current_algorithm}_{report_folder}.json"
            elif current_stage == "strategize":
                quality_check_input += f"- ../strategy_reports/report_{report_folder}/{current_algorithm}_optimization_analysis.md"
            
            quality_check_input += f"""
            
            请使用工具检查文件是否存在和内容是否符合标准，然后给出结构化的质量检查结果。
            """
            
            # 调用质量检查Agent
            result = self.quality_check_agent.invoke({"input": quality_check_input})
            
            # 解析结构化输出
            try:
                quality_result = self.factory.quality_parser.parse(result["output"])
                quality_passed = quality_result.get("quality_passed", "false").lower() == "true"
                
                # 更新状态
                stage_key = f"{current_algorithm}_{current_stage}"
                state["quality_checks"][stage_key] = quality_passed
                
                if quality_passed:
                    print(f"✅ {current_algorithm} {current_stage} 阶段质量检查通过")
                else:
                    print(f"❌ {current_algorithm} {current_stage} 阶段质量检查失败")
                    issues = quality_result.get("issues", [])
                    if issues:
                        print(f"📋 发现问题: {issues}")
                        
            except Exception as parse_error:
                print(f"⚠️ 质量检查结果解析失败: {parse_error}")
                # 回退到简单检查
                stage_result = state["stage_results"].get(f"{current_algorithm}_{current_stage}", {})
                quality_passed = stage_result.get("status") == "completed"
                state["quality_checks"][f"{current_algorithm}_{current_stage}"] = quality_passed
                
        except Exception as e:
            error_msg = f"Master质量检查失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["quality_checks"][f"{current_algorithm}_{current_stage}"] = False
        
        return state
    
    def master_next_decision_node(self, state: MasterWorkflowState) -> MasterWorkflowState:
        """Master Agent决策节点 - 使用LLM Agent智能决策下一步行动"""
        current_algorithm = state["current_algorithm"]
        current_stage = state["current_stage"]
        target_algorithms = state["target_algorithms"]
        completed_algorithms = state["completed_algorithms"]
        
        print(f"🤔 [Master决策] 智能规划下一步行动...")
        
        try:
            # 构建决策输入
            stage_key = f"{current_algorithm}_{current_stage}"
            quality_passed = state["quality_checks"].get(stage_key, False)
            retry_count = state["retry_count"]
            max_retries = state["max_retries"]
            
            decision_input = f"""
            请基于当前状态智能决策下一步行动：
            
            **当前状态:**
            - 当前算子: {current_algorithm}
            - 当前阶段: {current_stage}
            - 质量检查: {'通过' if quality_passed else '失败'}
            - 重试次数: {retry_count}/{max_retries}
            - 目标算子列表: {target_algorithms}
            - 已完成算子: {completed_algorithms}
            
            **决策规则:**
            1. 如果质量检查失败且重试次数<{max_retries}，应该重试
            2. 如果当前阶段是scout，下一阶段应该是analyze
            3. 如果当前阶段是analyze，下一阶段应该是strategize  
            4. 如果当前阶段是strategize且还有未处理算子，应该处理下一个算子
            5. 如果所有算子都完成strategize阶段，应该开始summarize
            
            请给出智能决策结果。
            """
            
            # 调用决策Agent
            result = self.decision_agent.invoke({"input": decision_input})
            
            # 解析结构化输出
            try:
                decision_result = self.factory.decision_parser.parse(result["output"])
                decision = decision_result.get("decision", "complete")
                next_stage = decision_result.get("next_stage", "")
                next_algorithm = decision_result.get("next_algorithm", "")
                reason = decision_result.get("reason", "")
                
                # 执行决策
                state["master_decision"] = decision
                
                if decision == "retry":
                    state["retry_count"] += 1
                    print(f"🔄 Master决策: 重试 - {reason}")
                    
                elif decision == "continue":
                    state["retry_count"] = 0  # 重置重试计数
                    if next_stage:
                        state["current_stage"] = next_stage
                    if next_algorithm:
                        state["current_algorithm"] = next_algorithm
                    print(f"➡️ Master决策: 继续 - {reason}")
                    
                elif decision == "summarize":
                    state["current_stage"] = "summarize"
                    print(f"📝 Master决策: 开始总结 - {reason}")
                    
                else:  # complete
                    print(f"✅ Master决策: 完成工作流 - {reason}")
                    
            except Exception as parse_error:
                print(f"⚠️ 决策结果解析失败: {parse_error}")
                # 回退到简单决策逻辑
                if not quality_passed and retry_count < max_retries:
                    state["retry_count"] += 1
                    state["master_decision"] = "retry"
                    print(f"🔄 回退决策: 第 {retry_count + 1} 次重试")
                else:
                    state["master_decision"] = "complete"
                    print(f"⚠️ 回退决策: 完成工作流")
                
        except Exception as e:
            error_msg = f"Master决策失败: {str(e)}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["master_decision"] = "complete"
        
        return state
    
    def run_master_workflow(self, user_request: str) -> MasterWorkflowState:
        """运行Master Agent调度的完整工作流"""
        # 初始化状态
        initial_state = MasterWorkflowState(
            user_request=user_request,
            analysis_type="",
            target_algorithms=[],
            current_algorithm="",
            current_stage="planning",
            completed_algorithms=[],
            algorithm_progress={},
            report_folder="",
            stage_results={},
            quality_checks={},
            master_decision="continue",
            retry_count=0,
            max_retries=3,
            errors=[],
            messages=[HumanMessage(content=user_request)]
        )
        
        # 运行工作流（官方推荐的配置方式）
        config = {
            "recursion_limit": 100,  # 增加递归限制到100次
            "max_iterations": 50,    # 最大迭代次数
        }
        
        print(f"🚀 启动Master Agent调度系统 (基于LangGraph官方规范)")
        print(f"📝 用户请求: {user_request}")
        print(f"⚙️ 配置: 递归限制={config['recursion_limit']}, 最大迭代={config['max_iterations']}")
        print()
        final_state = self.workflow.invoke(initial_state, config=config)
        
        return final_state

def main():
    """主函数 - Master Agent调度系统入口"""
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请在.env文件中设置DASHSCOPE_API_KEY")
        return
    
    # 检查OpenBLAS目录
    if not os.path.exists("./OpenBLAS-develop"):
        print("❌ 错误: 未找到OpenBLAS-develop目录")
        return
    
    # 创建Master工作流
    master_workflow = OpenBLASMasterWorkflow()
    
    # 用户交互
    print("🎯 OpenBLAS优化策略分析 - Master Agent调度系统")
    print("=" * 60)
    print("💡 智能化多Agent协作分析系统")
    print("🤖 Master Agent将自动调度和质量控制整个分析流程")
    print("=" * 60)
    print()
    print("分析选项:")
    print("1. 快速分析 - 自动分析核心BLAS算子 (gemm, axpy, dot)")
    print("2. 全面分析 - 自动分析完整BLAS算子集合")
    print("3. 自定义分析 - 指定要分析的算子")
    print("4. 直接输入分析请求")
    print()
    
    choice = input("请选择 (1-4) 或直接输入分析请求: ").strip()
    
    # 解析用户输入
    if choice == "1":
        user_request = "请进行快速分析"
    elif choice == "2":
        user_request = "请进行全面分析"
    elif choice == "3":
        algorithms = input("请输入要分析的算子 (逗号分隔): ").strip()
        user_request = f"请分析以下算子: {algorithms}"
    elif choice == "4":
        user_request = input("请输入您的分析请求: ").strip()
    else:
        # 直接作为用户请求
        user_request = choice
    
    if not user_request:
        print("❌ 未提供有效的分析请求")
        return
    
    print(f"\n🎯 Master Agent接收请求: {user_request}")
    print("🤖 正在启动智能调度系统...")
    print()
    
    try:
        # 运行Master工作流
        final_state = master_workflow.run_master_workflow(user_request)
        
        # 输出结果总结
        print("\n" + "=" * 60)
        print("📊 Master Agent调度系统执行完成")
        print("=" * 60)
        
        # 显示分析结果
        target_algorithms = final_state["target_algorithms"]
        completed_algorithms = final_state["completed_algorithms"] 
        report_folder = final_state["report_folder"]
        errors = final_state["errors"]
        
        print(f"\n🎯 分析类型: {final_state['analysis_type']}")
        print(f"📋 目标算子: {target_algorithms}")
        print(f"✅ 成功完成: {completed_algorithms} ({len(completed_algorithms)}/{len(target_algorithms)})")
        
        if len(completed_algorithms) < len(target_algorithms):
            failed = set(target_algorithms) - set(completed_algorithms)
            print(f"❌ 未完成: {list(failed)}")
        
        # 显示错误信息
        if errors:
            print(f"\n⚠️ 遇到 {len(errors)} 个错误:")
            for error in errors[:3]:
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  ... 还有 {len(errors) - 3} 个错误")
        
        # 显示生成的文件
        if report_folder:
            report_dir = f"strategy_reports/report_{report_folder}"
            print(f"\n📁 生成的报告 (新的时间戳文件夹结构):")
            print(f"  📂 时间戳文件夹: {report_dir}/")
            
            if os.path.exists(report_dir):
                files = os.listdir(report_dir)
                algo_reports = [f for f in files if f.endswith("_optimization_analysis.md")]
                summary_reports = [f for f in files if f.startswith("optimization_summary")]
                
                if algo_reports:
                    print(f"  📄 算子策略报告: {len(algo_reports)} 个")
                    for report in sorted(algo_reports):
                        print(f"    - {report}")
                
                if summary_reports:
                    print(f"  📋 跨算子总结报告: {len(summary_reports)} 个")
                    for report in sorted(summary_reports):
                        print(f"    - {report}")
            else:
                print(f"  ⚠️ 报告文件夹不存在: {report_dir}")
        
        # 显示其他生成的文件
        print(f"\n📁 其他生成的文件:")
        discovery_dir = "discovery_results"
        analysis_dir = "analysis_results"
        
        if os.path.exists(discovery_dir):
            discovery_files = [f for f in os.listdir(discovery_dir) if f.endswith('.json')]
            print(f"  🔍 发现结果: {len(discovery_files)} 个文件")
            
        if os.path.exists(analysis_dir):
            analysis_folders = [d for d in os.listdir(analysis_dir) if os.path.isdir(os.path.join(analysis_dir, d))]
            print(f"  📊 分析结果: {len(analysis_folders)} 个算子文件夹")
            for folder in sorted(analysis_folders):
                folder_path = os.path.join(analysis_dir, folder)
                json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
                print(f"    - {folder}/: {len(json_files)} 个分析文件")
        
        # 最终状态
        if completed_algorithms and final_state["stage_results"].get("summarize", {}).get("status") == "completed":
            print(f"\n🎉 Master Agent调度系统成功完成!")
            print(f"🤖 智能分析了 {len(completed_algorithms)} 个算子")
            print(f"📊 生成了完整的优化策略分析和跨算子总结")
            print(f"💡 请查看报告了解OpenBLAS的优化策略!")
        else:
            print(f"\n⚠️ Master Agent调度系统部分完成")
            print(f"🔍 请检查错误信息并重试")
            
    except Exception as e:
        print(f"\n❌ Master Agent调度系统执行失败: {str(e)}")
        print("🔧 请检查配置和环境设置")

if __name__ == "__main__":
    main() 