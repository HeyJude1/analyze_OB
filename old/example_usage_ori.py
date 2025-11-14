#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析 - LangGraph工作流
"""

import os
import json
from typing import TypedDict, Literal, List
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# 导入我们的Agent工厂
from analyze import OpenBLASAgentFactory

load_dotenv()

# ===== 定义工作流状态 =====
class WorkflowState(TypedDict):
    """工作流状态定义"""
    # 当前阶段
    stage: Literal["scout", "analyze", "strategize", "complete"]
    
    # 要分析的算法列表
    algorithms: List[str]
    
    # 发现的文件
    discovered_files: dict
    
    # 分析结果
    analysis_results: List[dict]
    
    # 最终策略
    optimization_strategies: str
    
    # 消息历史
    messages: List[BaseMessage]
    
    # 错误信息
    errors: List[str]

# ===== 创建OpenBLAS分析工作流 =====
class OpenBLASWorkflow:
    """OpenBLAS分析工作流"""
    
    def __init__(self):
        # 创建Agent工厂
        self.factory = OpenBLASAgentFactory()
        
        # 创建各个Agent
        self.scout_agent = self.factory.create_scout_agent()
        self.analyzer_agent = self.factory.create_analyzer_agent()
        self.strategist_agent = self.factory.create_strategist_agent()
        
        # 构建工作流图
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """构建工作流图"""
        workflow = StateGraph(WorkflowState)
        
        # 添加节点
        workflow.add_node("scout", self.scout_node)
        workflow.add_node("analyze", self.analyze_node)
        workflow.add_node("strategize", self.strategize_node)
        
        # 设置入口
        workflow.set_entry_point("scout")
        
        # 添加边
        workflow.add_conditional_edges(
            "scout",
            self.should_continue_scouting,
            {
                "continue": "scout",
                "analyze": "analyze"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze",
            self.should_continue_analyzing,
            {
                "continue": "analyze",
                "strategize": "strategize"
            }
        )
        
        workflow.add_edge("strategize", END)
        
        return workflow.compile()
    
    def scout_node(self, state: WorkflowState) -> WorkflowState:
        """侦察节点 - 发现算子文件"""
        print("\n🔍 [侦察阶段] 发现算子文件...")
        
        if not state.get('discovered_files'):
            state['discovered_files'] = {}
        
        # 遍历每个算法
        for algorithm in state['algorithms']:
            if algorithm not in state['discovered_files']:
                scout_input = f"请发现并读取 {algorithm} 算法的不同类型实现文件（最多5个），确保包含generic、architecture-specific和SIMD优化版本"
                
                try:
                    result = self.scout_agent.invoke({"input": scout_input})
                    state['discovered_files'][algorithm] = result['output']
                    print(f"✓ {algorithm}")
                except Exception as e:
                    error_msg = f"侦察 {algorithm} 失败: {str(e)}"
                    print(f"✗ {algorithm}: {error_msg}")
                    state['errors'].append(error_msg)
        
        return state
    
    def analyze_node(self, state: WorkflowState) -> WorkflowState:
        """分析节点 - 深度分析代码"""
        print("\n📊 [分析阶段] 深度分析代码...")
        
        if not state.get('analysis_results'):
            state['analysis_results'] = []
        
        # 分析每个算法的文件
        for algorithm, discovery_output in state['discovered_files'].items():
            analyze_input = f"""
基于以下侦察结果，请深度分析 {algorithm} 算法的各个实现：

{discovery_output}

请识别每个实现中的关键优化技术，包括：
- 算法层优化（循环展开、分块等）
- 架构层优化（缓存优化、内存访问模式等）
- 指令层优化（SIMD向量化、FMA指令等）
- 微架构优化（寄存器使用、指令调度等）

分析完成后保存结果。
"""
            
            try:
                result = self.analyzer_agent.invoke({"input": analyze_input})
                state['analysis_results'].append({
                    'algorithm': algorithm,
                    'analysis': result['output']
                })
                print(f"✓ {algorithm}")
            except Exception as e:
                error_msg = f"分析 {algorithm} 失败: {str(e)}"
                print(f"✗ {algorithm}: {error_msg}")
                state['errors'].append(error_msg)
        
        return state
    
    def strategize_node(self, state: WorkflowState) -> WorkflowState:
        """策略总结节点 - 提取优化策略"""
        print("\n🎯 [策略总结] 提取优化策略...")
        
        strategize_input = """
请收集所有的分析结果，并从中提取通用的优化策略。

要求：
1. 按照优化层次组织（算法级、架构级、指令级、微架构级）
2. 每个策略都要有具体的代码示例
3. 说明每种优化的适用场景和性能影响
4. 总结OpenBLAS的优化哲学和最佳实践

生成一份专业的优化策略报告。
"""
        
        try:
            result = self.strategist_agent.invoke({"input": strategize_input})
            state['optimization_strategies'] = result['output']
            state['stage'] = 'complete'
            print("✓ 策略提取完成")
        except Exception as e:
            error_msg = f"策略提取失败: {str(e)}"
            print(f"✗ 策略提取失败: {error_msg}")
            state['errors'].append(error_msg)
            state['stage'] = 'complete'
        
        return state
    
    def should_continue_scouting(self, state: WorkflowState) -> str:
        """判断是否继续侦察"""
        # 检查是否所有算法都已侦察完成
        expected_algorithms = set(state['algorithms'])
        discovered_algorithms = set(state['discovered_files'].keys())
        
        if expected_algorithms.issubset(discovered_algorithms):
            return 'analyze'
        return 'continue'
    
    def should_continue_analyzing(self, state: WorkflowState) -> str:
        """判断是否继续分析"""
        # 检查是否所有算法都已分析完成
        analyzed_algorithms = set(result['algorithm'] for result in state['analysis_results'])
        expected_algorithms = set(state['algorithms'])
        
        if expected_algorithms.issubset(analyzed_algorithms):
            return 'strategize'
        return 'continue'
    
    def run(self, algorithms: List[str] = None, custom_prompt: str = None) -> WorkflowState:
        """运行完整的分析工作流"""
        if algorithms is None:
            algorithms = ['dot', 'gemm', 'copy']  # 默认分析这三个核心算法（移除axpy）
        
        print(f"🚀 OpenBLAS优化策略分析")
        print(f"📋 算法: {', '.join(algorithms)}")
        
        # 初始化状态
        initial_state = WorkflowState(
            stage="scout",
            algorithms=algorithms,
            discovered_files={},
            analysis_results=[],
            optimization_strategies="",
            messages=[],
            errors=[]
        )
        
        # 如果有自定义prompt，添加到消息中
        if custom_prompt:
            initial_state['messages'].append(HumanMessage(content=custom_prompt))
        
        # 运行工作流
        final_state = self.workflow.invoke(initial_state)
        
        # 输出结果摘要
        print("\n" + "="*50)
        print("📊 分析完成")
        print("="*50)
        
        if final_state['errors']:
            print(f"\n⚠️  遇到 {len(final_state['errors'])} 个错误:")
            for error in final_state['errors']:
                print(f"  - {error}")
        
        if final_state['optimization_strategies']:
            print("\n✅ 优化策略已生成并保存")
            print("\n策略预览:")
            print("-"*60)
            print(final_state['optimization_strategies'][:500] + "...")
            print("-"*60)
        
        return final_state

# ===== 主函数 =====
def main():
    """主函数 - 自动化运行分析流程"""
    print("🧠 OpenBLAS算子优化策略智能分析系统\n")
    
    # 创建工作流
    workflow = OpenBLASWorkflow()
    
    # 运行选项
    print("分析配置:")
    print("1. 快速分析 (dot, gemm, copy)")
    print("2. 全面分析 (所有常见算法)")
    print("3. 自定义分析")
    
    choice = input("\n选择分析模式 (1-3): ").strip()
    
    if choice == "1":
        # 快速分析核心算法
        result = workflow.run(['dot', 'gemm', 'copy'])
        
    elif choice == "2":
        # 全面分析
        all_algorithms = ['dot', 'gemm', 'copy', 'gemv', 'scal', 'asum']
        confirm = input(f"将分析 {len(all_algorithms)} 个算法，可能需要较长时间，继续？(y/N): ")
        if confirm.lower() == 'y':
            result = workflow.run(all_algorithms)
        else:
            print("已取消")
            return
            
    elif choice == "3":
        # 自定义分析
        algorithms_input = input("输入要分析的算法（逗号分隔，如: dot,gemm）: ").strip()
        algorithms = [a.strip() for a in algorithms_input.split(',') if a.strip()]
        
        if algorithms:
            custom_prompt = input("输入额外的分析要求（可选，直接回车跳过）: ").strip()
            result = workflow.run(algorithms, custom_prompt)
        else:
            print("未输入有效算法")
            return
    else:
        print("无效选择")
        return
    
    # 询问是否查看完整报告
    if 'optimization_strategies' in result and result['optimization_strategies']:
        view_full = input("\n是否查看完整的优化策略报告？(y/N): ").strip().lower()
        if view_full == 'y':
            print("\n" + "="*60)
            print("完整优化策略报告")
            print("="*60)
            print(result['optimization_strategies'])

if __name__ == "__main__":
    main() 