#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析 - 简化版LangGraph工作流
"""

import os
import time
from typing import List, Literal, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

# LangGraph imports
from langgraph.graph import StateGraph, END

# 本地imports
from analyze import OpenBLASAgentFactory

# 加载环境变量
load_dotenv()

# ===== 工作流状态定义 =====
class WorkflowState(TypedDict):
    """简化的工作流状态"""
    stage: Literal["scout", "analyze", "strategize", "summarize", "complete"]
    algorithms: List[str]
    current_algorithm: str  # 当前处理的算法
    messages: List[BaseMessage]
    scout_completed: bool
    analysis_completed: bool
    strategies_completed: bool

    summarize_completed: bool  # 新增：总结完成状态
    report_folder: str  # 新增：存储本次分析的时间戳文件夹路径
    completed_algorithms: List[str]  # 新增：已完成的算法列表
    
    errors: List[str]

# ===== LangGraph工作流 =====
class OpenBLASWorkflow:
    """OpenBLAS分析工作流 - 支持批量分析和总结"""
    
    def __init__(self):
        self.factory = OpenBLASAgentFactory()
        self.scout_agent = self.factory.create_scout_agent()
        self.analyzer_agent = self.factory.create_analyzer_agent()
        self.strategist_agent = self.factory.create_strategist_agent()
        self.summarizer_agent = self.factory.create_summarizer_agent()  # 新增总结agent
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("scout", self.scout_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("strategize", self.strategize_node)
        workflow.add_node("summarize", self.summarize_node)  # 新增总结节点
        
        # 设置入口点
        workflow.set_entry_point("scout")
        
        # 添加边
        workflow.add_edge("scout", "analyze")
        workflow.add_edge("analyze", "strategize")
        workflow.add_edge("strategize", "summarize")  # 策略后进行总结
        workflow.add_edge("summarize", END)
        
        return workflow.compile()
    
    def scout_node(self, state: WorkflowState) -> WorkflowState:
        """侦察节点 - 发现OpenBLAS文件"""
        algorithm = state["algorithms"][0]  # 现在每次只处理一个算子
        print(f"🔍 [侦察阶段] 发现 {algorithm} 算子文件...")
        
        try:
            scout_input = f"""
            请帮我发现OpenBLAS中 {algorithm} 算法的实现文件。
            
            请按步骤执行：
            1. 在OpenBLAS-develop/kernel/目录中搜索 {algorithm} 相关文件
            2. 选择3-5个不同类型的实现（generic、x86_64、arm64等）
            3. 为每个文件提供简要的实现类型说明
            4. 生成JSON格式的发现结果（使用discoveries数组）
            5. 读取已有 discovery_results/discovered_files.json（如存在），将本次发现追加到discoveries数组中，保存回同一路径，并用read_file验证，失败请重试直到成功
            6. 你可以先读取discovery_results/discovered_files.json，然后根据内容，追加到大括号中，然后保存。
            重要：必须追加到现有discoveries数组中，不能覆盖！
            仅通过工具完成保存与验证，不要在聊天中直接输出结果。
            """
            
            result = self.scout_agent.invoke({"input": scout_input})
            
            # 保存验证与重试（守护兜底）
            try:
                target = "discovery_results/discovered_files.json"
                if (not os.path.exists(target)) or os.path.getsize(target) == 0:
                    retry_input = (
                        f"请重新执行保存：读取已有discovery_results/discovered_files.json（如有），"
                        f"确保包含 {algorithm} 算法的发现结果，"
                        f"然后使用write_file写回，并用read_file验证，失败请重试直到成功。"
                        "只调用工具，不要在对话中输出其它内容。"
                    )
                    self.scout_agent.invoke({"input": retry_input})
            except Exception as _:
                pass
            
            # 最终检查
            if (not os.path.exists("discovery_results/discovered_files.json")) or os.path.getsize("discovery_results/discovered_files.json") == 0:
                raise RuntimeError("保存discovered_files.json失败")
            
            state["scout_completed"] = True
            state["stage"] = "analyze"
            print(f"✅ {algorithm} 侦察完成")
            
        except Exception as e:
            error_msg = f"侦察失败: {str(e)}"
            print(f"✗ {error_msg}")
            state["errors"].append(error_msg)
            state["scout_completed"] = False
        
        return state
    
    def analyze_node(self, state: WorkflowState) -> WorkflowState:
        """分析节点 - 深度分析代码"""
        algorithm = state["algorithms"][0]  # 现在每次只处理一个算子
        print(f"📊 [分析阶段] 深度分析 {algorithm} 代码实现...")
        
        try:
            analyze_input = f"""
            现在需要深度分析OpenBLAS中 {algorithm} 算法的实现代码。
            
            请按步骤执行：
            1. 从discovery_results/discovered_files.json中找到 {algorithm} 算法的所有文件
            2. 对每个文件进行深度分析，关注：
               - 算法层优化（循环展开、分块等）
               - 架构层优化（缓存友好、内存对齐等）
               - 指令层优化（SIMD、FMA等）
               - 微架构优化（寄存器分配、指令调度等）
            3. 确保分析结果包含具体的优化技术和代码示例
            4. 生成JSON格式的分析报告
            5. **重要保存步骤**:
               - 首先创建算子文件夹：analysis_results/{algorithm}/
               - 然后保存每个文件的分析结果到：analysis_results/{algorithm}/analysis_{algorithm}_{{实现类型}}_{{时间戳}}.json
               - 保存后用read_file验证每个文件，失败请重试直到成功
            
            仅通过工具完成保存与验证，不要在聊天中直接输出结果。
            """
            
            result = self.analyzer_agent.invoke({"input": analyze_input})
            
            # 保存验证与重试（守护兜底）
            try:
                algo_dir = f"analysis_results/{algorithm}"
                has_current_algo_json = False
                if os.path.exists(algo_dir):
                    for fname in os.listdir(algo_dir):
                        if fname.endswith(".json") and f"analysis_{algorithm}_" in fname:
                            has_current_algo_json = True
                            break
                if not has_current_algo_json:
                    retry_input = (
                        f"请读取discovery_results/discovered_files.json并对 {algorithm} 算法执行分析，"
                        f"先创建文件夹analysis_results/{algorithm}/，"
                        f"然后按每个文件单独保存至analysis_results/{algorithm}/analysis_{algorithm}_*.json，保存后用read_file验证，失败重试直到成功。"
                        "只调用工具，不要在对话中输出其它内容。"
                    )
                    self.analyzer_agent.invoke({"input": retry_input})
            except Exception as _:
                pass
            
            # 最终检查
            algo_dir = f"analysis_results/{algorithm}"
            has_current_algo_json_final = False
            if os.path.exists(algo_dir):
                for fname in os.listdir(algo_dir):
                    if fname.endswith(".json") and f"analysis_{algorithm}_" in fname:
                        has_current_algo_json_final = True
                        break
            if not has_current_algo_json_final:
                raise RuntimeError(f"未在analysis_results/{algorithm}/目录中生成 {algorithm} 的分析JSON文件")
            
            state["analysis_completed"] = True
            state["stage"] = "strategize"
            print(f"✅ {algorithm} 分析完成")
            
        except Exception as e:
            error_msg = f"分析失败: {str(e)}"
            print(f"✗ {error_msg}")
            state["errors"].append(error_msg)
            state["analysis_completed"] = False
        
        return state

    def strategize_node(self, state: WorkflowState) -> WorkflowState:
        """策略节点 - 提取优化策略"""
        algorithm = state["algorithms"][0]  # 现在每次只处理一个算子
        report_folder = state["report_folder"]  # 获取时间戳文件夹路径
        print(f"🎯 [策略阶段] 总结 {algorithm} 优化策略...")
        
        try:
            strategize_input = f"""
            现在需要主动分析OpenBLAS中 {algorithm} 算法的优化设计模式。
            
            请执行以下分析任务：
            1. 读取analysis_results/{algorithm}/目录中所有 analysis_{algorithm}_*.json 文件
            2. 专注于 {algorithm} 算法，按照三层分析框架，主动发现和分析优化设计：
            
            **算法设计层次分析：**
            - 深入分析是否有更适合计算机计算逻辑的算法设计
            - 发现以空间换时间的具体优化设计实例
            - 发现以时间换空间的具体优化设计实例
            
            **代码优化层次分析：**
            - 主动分析性能加速的代码优化技术
            - 深入分析循环优化的具体设计
            - 分析代码顺序调整对性能的影响
            
            **特有指令层次分析：**
            - 识别使用的专有指令类型
            - 分析围绕专有指令的优化设计策略
            
            3. 对每个发现的优化设计，提供：
               - 具体实现代码片段
               - 优化原理解释
               - 性能提升分析
            4. 生成完整的Markdown格式分析报告
            5. **重要保存步骤**:
               - 首先确保目录存在：strategy_reports/report_{report_folder}/
               - 然后保存到：strategy_reports/report_{report_folder}/{algorithm}_optimization_analysis.md
               - 保存后用read_file验证，失败请重试直到成功
               - 完成后输出: "✅ 已将 {algorithm} 报告保存到文件夹: strategy_reports/report_{report_folder}/"
            
            仅通过工具完成保存与验证，不要在聊天中直接输出结果。
            """
            
            result = self.strategist_agent.invoke({"input": strategize_input})
            
            # 保存验证与重试（守护兜底）
            try:
                target_dir = f"strategy_reports/report_{report_folder}"
                target_file = f"{target_dir}/{algorithm}_optimization_analysis.md"
                if not os.path.exists(target_file) or os.path.getsize(target_file) == 0:
                    retry_input = (
                        f"请根据analysis_results/{algorithm}/目录中 {algorithm} 的JSON报告生成Markdown策略报告，"
                        f"确保创建目录strategy_reports/report_{report_folder}/，"
                        f"然后保存为strategy_reports/report_{report_folder}/{algorithm}_optimization_analysis.md，保存后用read_file验证，失败重试直到成功。"
                        "只调用工具，不要在对话中输出其它内容。"
                    )
                    self.strategist_agent.invoke({"input": retry_input})
            except Exception as _:
                pass
            
            # 最终检查
            target_dir = f"strategy_reports/report_{report_folder}"
            target_file = f"{target_dir}/{algorithm}_optimization_analysis.md"
            if not os.path.exists(target_file) or os.path.getsize(target_file) == 0:
                raise RuntimeError(f"未在 {target_dir} 目录中生成 {algorithm} 的策略报告")
            
            # 记录已完成的算法到状态中
            if algorithm not in state["completed_algorithms"]:
                state["completed_algorithms"].append(algorithm)
            
            state["strategies_completed"] = True
            state["stage"] = "summarize"
            print(f"✅ {algorithm} 策略提取完成，保存到: {target_dir}/")
            
        except Exception as e:
            error_msg = f"策略提取失败: {str(e)}"
            print(f"✗ {error_msg}")
            state["errors"].append(error_msg)
            state["strategies_completed"] = False
        
        return state

    def summarize_node(self, state: WorkflowState) -> WorkflowState:
        """总结节点 - 总结多个算法的优化策略"""
        report_folder = state["report_folder"]  # 获取时间戳文件夹路径
        completed_algorithms = state["completed_algorithms"]  # 获取已完成的算法列表
        print(f"📝 [总结阶段] 生成多算法优化策略总结报告...")
        
        try:
            summarize_input = f"""
            现在需要总结和归纳多个OpenBLAS算法的优化策略，从以下已完成的算法报告中提炼共性规律。
            
            请执行以下总结任务：
            1. **读取报告文件夹** - 列出strategy_reports/report_{report_folder}/目录中的所有 *_optimization_analysis.md 文件
            2. **逐个读取报告** - 读取每个算法的优化分析报告：
               已完成的算法: {', '.join(completed_algorithms)}
            3. **横向对比分析** - 按照以下框架进行跨算法分析：
            
            **跨算法共性分析：**
            - 识别不同算法使用的相同优化技术
            - 总结通用的算法设计模式
            - 归纳共同的性能瓶颈解决方案
            
            **架构特化对比：**
            - 对比不同架构（x86_64, ARM64, RISC-V）的优化差异
            - 总结指令集特定的优化策略
            - 分析硬件特性利用的通用方法
            
            **性能提升模式：**
            - 量化各种优化技术的性能收益范围
            - 总结优化技术的适用场景
            - 提炼最佳实践组合建议
            
            4. **生成结构化总结** - 包含：
               - 对比表格和量化分析
               - 实用的优化指导原则
               - 可复用的设计模式
            5. **保存总结报告** - 保存到：strategy_reports/report_{report_folder}/optimization_summary_report.md
            6. **保存后验证** - 用read_file读取保存的文件确认内容完整
            7. **完成后输出** - "✅ 多算法优化策略总结报告已保存到: strategy_reports/report_{report_folder}/"
            
            仅通过工具完成所有操作，不要在聊天中直接输出结果。
            """
            
            result = self.summarizer_agent.invoke({"input": summarize_input})
            
            # 保存验证与重试（守护兜底）
            try:
                target_dir = f"strategy_reports/report_{report_folder}"
                summary_file = f"{target_dir}/optimization_summary_report.md"
                if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
                    retry_input = (
                        f"请读取strategy_reports/report_{report_folder}/目录中的所有算法报告，"
                        f"生成多算法优化策略总结，保存为strategy_reports/report_{report_folder}/optimization_summary_report.md，"
                        f"保存后用read_file验证，失败重试直到成功。只调用工具，不要在对话中输出其它内容。"
                    )
                    self.summarizer_agent.invoke({"input": retry_input})
            except Exception as _:
                pass
            
            # 最终检查
            target_dir = f"strategy_reports/report_{report_folder}"
            summary_file = f"{target_dir}/optimization_summary_report.md"
            if not os.path.exists(summary_file) or os.path.getsize(summary_file) == 0:
                raise RuntimeError(f"未在 {target_dir} 目录中生成总结报告")
            
            state["summarize_completed"] = True
            state["stage"] = "complete"
            print(f"✅ 多算法优化策略总结完成，保存到: {target_dir}/")
            
        except Exception as e:
            error_msg = f"总结失败: {str(e)}"
            print(f"✗ {error_msg}")
            state["errors"].append(error_msg)
            state["summarize_completed"] = False
        
        return state

    def run_single_algorithm(self, algorithm: str) -> WorkflowState:
        """运行单个算法的分析工作流"""
        # 为单算法模式生成时间戳
        single_report_timestamp = f"{algorithm}_{int(time.time())}"
        
        # 初始化状态
        initial_state = WorkflowState(
            stage="scout",
            algorithms=[algorithm],  # 只包含一个算法
            current_algorithm=algorithm, # 设置当前算法
            messages=[HumanMessage(content=f"分析OpenBLAS算法: {algorithm}")],
            scout_completed=False,
            analysis_completed=False,
            strategies_completed=False,
            summarize_completed=False, # 初始化总结完成状态
            report_folder=single_report_timestamp, # 使用算法特定的时间戳
            completed_algorithms=[], # 初始化已完成的算法列表
            errors=[]
        )
        
        print(f"🔄 处理算法: {algorithm}")
        
        # 运行工作流
        final_state = self.workflow.invoke(initial_state)
        
        return final_state

    def run_batch_algorithms(self, algorithms: List[str], report_timestamp: str) -> dict:
        """运行批量算法分析工作流，最后生成总结报告"""
        print(f"🚀 开始批量处理 {len(algorithms)} 个算法")
        print(f"📋 算法列表: {', '.join(algorithms)}")
        print(f"📁 报告文件夹: strategy_reports/report_{report_timestamp}/")
        print()
        
        completed_algorithms = []
        all_errors = []
        
        # 第一阶段：处理每个算法的侦察、分析、策略阶段
        for i, algorithm in enumerate(algorithms):
            print(f"\n{'='*60}")
            print(f"🔄 第 {i+1}/{len(algorithms)} 个算法: {algorithm}")
            print(f"{'='*60}")
            
            try:
                # 为每个算法运行侦察、分析、策略阶段（不包括总结）
                single_result = self.run_single_algorithm_phases(algorithm, report_timestamp)
                
                if single_result["strategies_completed"]:
                    completed_algorithms.append(algorithm)
                    print(f"✅ {algorithm} 策略分析完成")
                else:
                    print(f"⚠️ {algorithm} 策略分析未完全完成")
                    all_errors.extend(single_result["errors"])
                    
            except Exception as e:
                error_msg = f"{algorithm} 分析失败: {str(e)}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)
        
        # 第二阶段：如果有算法成功完成，进行总结分析
        summary_completed = False
        if completed_algorithms:
            print(f"\n{'='*60}")
            print(f"📝 总结阶段: 分析 {len(completed_algorithms)} 个算法的优化策略")
            print(f"{'='*60}")
            
            try:
                # 创建总结状态
                summary_state = WorkflowState(
                    stage="summarize",
                    algorithms=completed_algorithms,
                    current_algorithm="",
                    messages=[HumanMessage(content=f"总结优化策略: {', '.join(completed_algorithms)}")],
                    scout_completed=True,
                    analysis_completed=True,
                    strategies_completed=True,
                    summarize_completed=False,
                    report_folder=report_timestamp,
                    completed_algorithms=completed_algorithms,
                    errors=all_errors
                )
                
                # 运行总结节点
                final_summary_state = self.summarize_node(summary_state)
                summary_completed = final_summary_state["summarize_completed"]
                
                if summary_completed:
                    print(f"✅ 多算法优化策略总结完成")
                else:
                    print(f"⚠️ 总结阶段未完全完成")
                    all_errors.extend(final_summary_state["errors"])
                    
            except Exception as e:
                error_msg = f"总结阶段失败: {str(e)}"
                print(f"❌ {error_msg}")
                all_errors.append(error_msg)
        
        return {
            "completed_algorithms": completed_algorithms,
            "summary_completed": summary_completed,
            "report_folder": report_timestamp,
            "errors": all_errors
        }
    
    def run_single_algorithm_phases(self, algorithm: str, report_timestamp: str) -> WorkflowState:
        """运行单个算法的侦察、分析、策略阶段（不包括总结）"""
        # 初始化状态
        initial_state = WorkflowState(
            stage="scout",
            algorithms=[algorithm],
            current_algorithm=algorithm,
            messages=[HumanMessage(content=f"分析OpenBLAS算法: {algorithm}")],
            scout_completed=False,
            analysis_completed=False,
            strategies_completed=False,
            summarize_completed=False,
            report_folder=report_timestamp,
            completed_algorithms=[],
            errors=[]
        )
        
        print(f"🔄 处理算法: {algorithm}")
        
        # 顺序执行侦察、分析、策略三个阶段
        state = initial_state
        
        # 侦察阶段
        state = self.scout_node(state)
        if not state["scout_completed"]:
            return state
            
        # 分析阶段
        state = self.analyze_node(state)
        if not state["analysis_completed"]:
            return state
            
        # 策略阶段
        state = self.strategize_node(state)
        
        return state

def main():
    """主函数"""
    # 检查API密钥
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 请在.env文件中设置DASHSCOPE_API_KEY")
        return
    
    # 检查OpenBLAS目录
    if not os.path.exists("./OpenBLAS-develop"):
        print("❌ 错误: 未找到OpenBLAS-develop目录")
        return
    
    # 创建工作流
    workflow = OpenBLASWorkflow()
    
    # 运行选项
    print("🚀 OpenBLAS优化策略分析 - 多算法批量分析模式")
    print("分析配置:")
    print("1. 快速分析 (gemm, axpy, dot) - 生成总结报告")
    print("2. 全面分析 (gemm, axpy, dot, gemv, nrm2, ger) - 生成总结报告")
    print("3. 自定义分析 - 生成总结报告")
    
    choice = input("\n选择分析模式 (1-3): ").strip()
    
    algorithms_to_process = []
    
    if choice == "1":
        # 快速分析核心算法
        algorithms_to_process = ['gemm', 'axpy', 'dot']
        
    elif choice == "2":
        # 全面分析
        all_algorithms = ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger']
        confirm = input(f"将分析 {len(all_algorithms)} 个算法，最后生成总结报告，可能需要较长时间，继续？(y/N): ")
        if confirm.lower() == 'y':
            algorithms_to_process = all_algorithms
        else:
            print("已取消")
            return
            
    elif choice == "3":
        # 自定义分析
        algorithms_input = input("输入要分析的算法（逗号分隔，如: dot,gemm）: ").strip()
        algorithms = [a.strip() for a in algorithms_input.split(',') if a.strip()]
        
        if algorithms:
            algorithms_to_process = algorithms
        else:
            print("未输入有效的算法名称")
            return
    else:
        print("无效选择")
        return
    
    # 生成时间戳作为报告文件夹名
    report_timestamp = f"{int(time.time())}"
    
    # 执行批量分析
    print(f"\n🚀 开始批量分析模式")
    print(f"📋 算法列表: {', '.join(algorithms_to_process)}")
    print(f"📁 报告将保存到: strategy_reports/report_{report_timestamp}/")
    print()
    
    try:
        batch_result = workflow.run_batch_algorithms(algorithms_to_process, report_timestamp)
        
        # 输出总体结果
        print("\n" + "="*60)
        print("📊 批量分析完成")
        print("="*60)
        
        completed_algorithms = batch_result["completed_algorithms"]
        summary_completed = batch_result["summary_completed"]
        report_folder = batch_result["report_folder"]
        all_errors = batch_result["errors"]
        
        print(f"\n✅ 成功完成的算法 ({len(completed_algorithms)}/{len(algorithms_to_process)}):")
        for algo in completed_algorithms:
            print(f"  - {algo}")
        
        if len(completed_algorithms) < len(algorithms_to_process):
            failed_algorithms = set(algorithms_to_process) - set(completed_algorithms)
            print(f"\n❌ 未完成的算法 ({len(failed_algorithms)}):")
            for algo in failed_algorithms:
                print(f"  - {algo}")
        
        # 显示错误信息
        if all_errors:
            print(f"\n⚠️  总共遇到 {len(all_errors)} 个错误:")
            for error in all_errors[:5]:  # 只显示前5个错误
                print(f"  - {error}")
            if len(all_errors) > 5:
                print(f"  ... 还有 {len(all_errors) - 5} 个错误")
        
        # 显示生成的文件
        print(f"\n📁 生成的文件:")
        print(f"  🔍 发现结果: discovery_results/discovered_files.json")
        
        if os.path.exists("analysis_results"):
            # 统计按算子分组的分析结果
            total_analysis_files = 0
            algo_dirs = [d for d in os.listdir("analysis_results") if os.path.isdir(os.path.join("analysis_results", d))]
            print(f"  📊 分析结果: {len(algo_dirs)} 个算子文件夹 (analysis_results/)")
            
            for algo_dir in sorted(algo_dirs):
                algo_path = os.path.join("analysis_results", algo_dir)
                analysis_files = [f for f in os.listdir(algo_path) if f.endswith(".json")]
                total_analysis_files += len(analysis_files)
                print(f"    - {algo_dir}/: {len(analysis_files)} 个分析文件")
            
            print(f"  📊 总计分析文件: {total_analysis_files} 个")
        
        # 显示策略报告文件夹
        report_dir = f"strategy_reports/report_{report_folder}"
        if os.path.exists(report_dir):
            strategy_files = [f for f in os.listdir(report_dir) if f.endswith(".md")]
            print(f"  🎯 策略报告文件夹: {report_dir}/")
            print(f"    📄 算法报告: {len([f for f in strategy_files if not f.startswith('optimization_summary')])} 个文件")
            for sf in sorted([f for f in strategy_files if not f.startswith('optimization_summary')]):
                print(f"      - {sf}")
            
            # 显示总结报告
            summary_files = [f for f in strategy_files if f.startswith('optimization_summary')]
            if summary_files:
                print(f"    📋 总结报告: {len(summary_files)} 个文件")
                for sf in sorted(summary_files):
                    print(f"      - {sf}")
        
        # 最终状态总结
        if completed_algorithms and summary_completed:
            print(f"\n🎉 批量分析成功完成！")
            print(f"📂 文件结构:")
            print(f"   strategy_reports/report_{report_folder}/")
            for algo in completed_algorithms:
                print(f"   ├── {algo}_optimization_analysis.md")
            if summary_completed:
                print(f"   └── optimization_summary_report.md")
            print(f"\n💡 查看总结报告了解跨算法的通用优化策略！")
        elif completed_algorithms:
            print(f"\n⚠️ 部分算法分析完成，但总结阶段失败")
        else:
            print("\n❌ 没有算法成功完成分析，请检查错误信息")
            
    except Exception as e:
        print(f"\n❌ 批量分析过程中发生错误: {str(e)}")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main() 