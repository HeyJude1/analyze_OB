#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真正的LangGraph Supervisor模式示例
与当前硬编码流程的对比
"""

from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END

# ===== 真正的Supervisor Agent =====
def create_supervisor_agent(llm, members: list[str]):
    """创建智能决策的Supervisor Agent"""
    
    system_prompt = f"""你是一个智能的任务调度supervisor。
    
你管理以下专家团队: {', '.join(members)}

每个专家的能力：
- scout: 扫描和发现算子文件
- analyzer: 分析代码优化策略  
- individual_summarizer: 总结单个算子
- final_summarizer: 跨算子总结

**你的职责**:
1. 根据当前任务状态，智能决定下一步调用哪个专家
2. 处理执行失败的情况（重试、跳过、或调整策略）
3. 优化整体执行效率
4. 确保任务完整性

**决策规则**:
- 如果某个算子失败多次，考虑跳过
- 如果资源不足，优先处理重要算子
- 根据已完成的工作量，动态调整后续计划

请根据当前状态，选择下一步行动，并简要说明原因。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"可选专家: {', '.join(members) + ', FINISH'}. 选择一个专家或FINISH:")
    ])
    
    return prompt | llm

# ===== 智能决策函数 =====
def supervisor_router(state) -> Literal["scout", "analyzer", "individual_summarizer", "final_summarizer", "FINISH"]:
    """Supervisor的智能路由决策 - 基于LLM推理"""
    
    # 构建状态描述
    context = f"""
当前状态分析：
- 已完成算子: {state.get('completed_algorithms', [])}
- 当前处理: {state.get('current_algorithm', 'None')}  
- 失败次数: {state.get('retry_count', 0)}
- 错误信息: {state.get('last_error', 'None')}
- 资源状态: {state.get('resource_status', '正常')}
- 总体进度: {len(state.get('completed_algorithms', []))}/{state.get('total_algorithms', 0)}

请智能决策下一步行动。
"""
    
    # 调用Supervisor Agent进行决策
    supervisor = create_supervisor_agent(llm, ["scout", "analyzer", "individual_summarizer", "final_summarizer"])
    response = supervisor.invoke({"messages": [("human", context)]})
    
    # 解析决策结果
    decision = response.content.strip().upper()
    
    # 验证决策有效性
    valid_choices = ["SCOUT", "ANALYZER", "INDIVIDUAL_SUMMARIZER", "FINAL_SUMMARIZER", "FINISH"]
    if decision not in valid_choices:
        return "FINISH"  # 默认结束
    
    return decision.lower()

# ===== 对比：当前硬编码 vs 真正Supervisor =====

class CurrentHardcodedFlow:
    """当前的硬编码流程"""
    def run(self, algorithms):
        # ❌ 固定顺序，无智能决策
        for algo in algorithms:
            self.scout_work(algo)      # 固定步骤1
            self.analyzer_work(algo)   # 固定步骤2  
            self.summary_work(algo)    # 固定步骤3
        self.final_summary()          # 固定最后步骤


class TrueSupervisorFlow:
    """真正的Supervisor流程"""
    def build_workflow(self):
        workflow = StateGraph(AgentState)
        
        # ✅ 动态路由 - 每次都经过Supervisor智能决策
        workflow.add_conditional_edges(
            "supervisor",
            supervisor_router,  # 🎯 LLM-based智能决策
            {
                "scout": "scout_agent",
                "analyzer": "analyzer_agent", 
                "individual_summarizer": "individual_agent",
                "final_summarizer": "final_agent",
                "FINISH": END
            }
        )
        
        # 所有Agent完成后都回到Supervisor重新评估
        for agent in ["scout_agent", "analyzer_agent", "individual_agent", "final_agent"]:
            workflow.add_edge(agent, "supervisor")
        
        return workflow

# ===== 真正Supervisor的优势 =====
"""
🎯 真正Supervisor模式的优势：

1. **自适应能力**: 
   - 根据执行结果动态调整策略
   - 处理异常情况时能智能选择替代方案

2. **容错机制**:
   - 自动重试失败的任务
   - 跳过问题节点，继续其他工作
   
3. **资源优化**:
   - 根据可用资源动态调整并发度
   - 优先处理重要或简单的任务

4. **上下文理解**:  
   - 理解任务间的依赖关系
   - 基于历史执行情况做决策

5. **灵活扩展**:
   - 容易添加新的专家Agent
   - 决策逻辑可以随需求演进
"""