#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - 真正的LangGraph Supervisor模式 Agent工厂
基于官方Supervisor模式实现智能决策的多Agent协作
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field

load_dotenv()


# ===== 智能Supervisor Agent =====
def create_supervisor_agent(llm, members: List[str]) -> ChatPromptTemplate:
    """创建智能决策的Supervisor Agent"""
    
    system_prompt = f"""你是一个智能的OpenBLAS分析任务调度supervisor。

你管理以下专家团队: {', '.join(members)}

每个专家的能力：
- scout: 扫描kernel目录，发现和分类算子文件
- analyzer: 深度分析单个算子文件的优化策略
- individual_summarizer: 总结单个算子的所有优化策略
- final_summarizer: 跨算子总结，生成最终优化策略库

**你的智能决策职责**:
1. 根据当前任务状态和执行历史，智能决定下一步调用哪个专家
2. 处理执行失败的情况（重试、跳过、或调整策略）
3. 优化整体执行效率，避免不必要的重复工作
4. 确保任务完整性和数据一致性
5. 根据资源状况动态调整执行策略

**智能决策规则**:
- 如果某个算子连续失败3次，考虑跳过该算子
- 如果API调用频繁失败，自动增加延迟时间
- 根据已完成的工作量，动态调整后续计划优先级
- 检测到重复工作时，智能跳过或合并
- 根据文件大小和复杂度，调整分析深度

**状态感知能力**:
- 理解任务间的依赖关系（scout -> analyzer -> summarizer）
- 监控执行效率和资源使用情况
- 基于历史执行情况预测和优化后续决策

请根据当前状态，选择下一步行动，并简要说明决策原因。
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("system", f"可选专家: {', '.join(members) + ', FINISH'}. 请选择一个专家继续工作，或选择FINISH结束任务。\n\n请回复格式: EXPERT_NAME|原因说明")
    ])
    
    return prompt | llm


# ===== 智能路由函数 =====
def supervisor_router(state) -> Literal["scout", "analyzer", "individual_summarizer", "final_summarizer", "FINISH"]:
    """Supervisor的智能路由决策 - 基于LLM推理"""
    
    # 构建详细的状态描述
    context = f"""
当前任务执行状态分析：

📊 **整体进度**:
- 已完成算子: {state.get('completed_algorithms', [])} ({len(state.get('completed_algorithms', []))}/{state.get('total_algorithms', 0)})
- 当前处理算子: {state.get('current_algorithm', 'None')}
- 当前阶段: {state.get('current_phase', 'None')}

⚠️ **错误和重试情况**:
- 当前算子重试次数: {state.get('retry_count', 0)}/3
- 最近错误: {state.get('last_error', 'None')}
- 总错误数: {len(state.get('all_errors', []))}

🔄 **执行历史**:
- 已完成任务: {state.get('completed_tasks', [])}
- 跳过的算子: {state.get('skipped_algorithms', [])}
- 执行时长: {state.get('execution_time', 0)} 秒

💾 **资源状态**:
- 可用算子列表: {state.get('available_algorithms', [])}
- 文件系统状态: {state.get('file_system_status', '正常')}
- API调用状态: {state.get('api_status', '正常')}

🎯 **任务依赖分析**:
- Scout完成状态: {'✅' if state.get('scout_completed') else '❌'}
- 需要分析的文件数: {state.get('pending_files_count', 0)}
- 需要总结的算子数: {state.get('pending_summary_count', 0)}

请基于以上状态信息，智能决策下一步最优行动。
"""
    
    # 获取LLM配置
    with open("/home/dgc/mjs/project/analyze_OB/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        model_config = config["model"]
    
    llm = ChatOpenAI(
        model=model_config["name"],
        temperature=0.1,  # 降低温度以获得更一致的决策
        max_tokens=model_config["max_tokens"],
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # 调用Supervisor Agent进行智能决策
    supervisor = create_supervisor_agent(llm, ["scout", "analyzer", "individual_summarizer", "final_summarizer"])
    
    try:
        response = supervisor.invoke({"messages": [("human", context)]})
        decision_text = response.content.strip()
        
        # 解析决策结果 (格式: EXPERT_NAME|原因)
        if "|" in decision_text:
            decision = decision_text.split("|")[0].strip().upper()
            reason = decision_text.split("|", 1)[1].strip()
            print(f"🧠 [Supervisor决策] {decision} - {reason}")
        else:
            decision = decision_text.upper()
            print(f"🧠 [Supervisor决策] {decision}")
        
        # 验证决策有效性
        valid_choices = ["SCOUT", "ANALYZER", "INDIVIDUAL_SUMMARIZER", "FINAL_SUMMARIZER", "FINISH"]
        if decision not in valid_choices:
            print(f"⚠️ [Supervisor] 无效决策 '{decision}', 默认结束任务")
            return "FINISH"
        
        return decision.lower()
        
    except Exception as e:
        print(f"❌ [Supervisor] 决策失败: {str(e)}, 默认结束任务")
        return "FINISH"


# ===== Agent工厂 =====
class AgentFactory:
    """Agent工厂"""
    
    def __init__(self):
        with open("/home/dgc/mjs/project/analyze_OB/config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
            model_config = config["model"]
        
        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def create_scout_agent(self) -> AgentExecutor:
        """Scout Agent - 智能文件发现和分类"""
        
        tools = [self._create_scan_tool(), self._create_file_read_tool()]
        
        scout_schemas = [
            ResponseSchema(name="algorithms", description="发现的算子种类列表，每个包含algorithm和files字段"),
            ResponseSchema(name="total_algorithms", description="算子种类总数"),
            ResponseSchema(name="total_files", description="文件总数"),
            ResponseSchema(name="scan_strategy", description="使用的扫描策略"),
            ResponseSchema(name="confidence_score", description="分类准确度评分(0-1)"),
            ResponseSchema(name="timestamp", description="扫描时间戳")
        ]
        scout_parser = StructuredOutputParser.from_response_schemas(scout_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能的OpenBLAS算子发现专家。你具备自适应扫描和智能分类能力。

🧠 **智能能力**:
1. 根据目录大小自动调整扫描策略
2. 智能识别算子模式，包括变体和特殊情况
3. 自动评估分类准确度并提供置信度分数
4. 处理异常文件和边界情况

🎯 **核心任务**:
- 扫描kernel目录，智能发现所有算子种类
- 使用模式匹配和启发式规则进行分类
- 生成高质量的算子分类报告

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=scout_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)
    
    def create_analyzer_agent(self) -> AgentExecutor:
        """Analyzer Agent - 智能代码分析"""
        
        tools = [self._create_file_read_tool(), self._create_analysis_read_tool()]
        
        analyzer_schemas = [
            ResponseSchema(name="algorithm", description="算子名称"),
            ResponseSchema(name="file_path", description="分析的文件路径"),
            ResponseSchema(name="analysis_depth", description="分析深度级别(basic/detailed/comprehensive)"),
            ResponseSchema(name="algorithm_level_optimizations", description="算法层优化策略，包含name、description、code_snippet、confidence"),
            ResponseSchema(name="code_level_optimizations", description="代码层优化策略，包含name、description、code_snippet、confidence"),
            ResponseSchema(name="instruction_level_optimizations", description="指令层优化策略，包含name、description、code_snippet、confidence"),
            ResponseSchema(name="complexity_score", description="代码复杂度评分(1-10)"),
            ResponseSchema(name="optimization_potential", description="优化潜力评估"),
            ResponseSchema(name="timestamp", description="分析时间戳")
        ]
        analyzer_parser = StructuredOutputParser.from_response_schemas(analyzer_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能的高性能计算代码分析专家。你具备深度代码理解和优化识别能力。

🧠 **智能分析能力**:
1. 根据代码复杂度自动调整分析深度
2. 智能识别优化模式，包括隐式和显式优化
3. 评估每个优化策略的置信度和重要性
4. 提供代码复杂度和优化潜力评估

🎯 **分析框架**:
- 算法层：计算逻辑、数据结构、算法设计优化
- 代码层：循环、分支、内存访问、编译器优化
- 指令层：SIMD、向量化、特殊指令、汇编优化

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=analyzer_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)
    
    def create_individual_summarizer_agent(self) -> AgentExecutor:
        """Individual Summarizer Agent - 智能策略整合"""
        
        tools = [self._create_analysis_read_tool()]
        
        individual_schemas = [
            ResponseSchema(name="algorithm", description="算子名称"),
            ResponseSchema(name="integration_strategy", description="使用的整合策略"),
            ResponseSchema(name="algorithm_level_optimizations", description="整合后的算法层优化策略"),
            ResponseSchema(name="code_level_optimizations", description="整合后的代码层优化策略"),
            ResponseSchema(name="instruction_level_optimizations", description="整合后的指令层优化策略"),
            ResponseSchema(name="redundancy_eliminated", description="消除的冗余策略数量"),
            ResponseSchema(name="quality_score", description="整合质量评分(0-1)"),
            ResponseSchema(name="timestamp", description="整合时间戳")
        ]
        individual_parser = StructuredOutputParser.from_response_schemas(individual_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能的策略整合专家。你具备高级的模式识别和策略合并能力。

🧠 **智能整合能力**:
1. 自动识别相似和重复的优化策略
2. 智能合并策略，保持最佳描述和命名
3. 评估整合质量并提供改进建议
4. 消除冗余，提升策略库的简洁性

🎯 **整合原则**:
- 保持策略的完整性和准确性
- 统一命名规范，提升可读性
- 合并相似策略，消除重复
- 保留关键差异，避免过度简化

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=individual_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)
    
    def create_final_summarizer_agent(self) -> AgentExecutor:
        """Final Summarizer Agent - 智能跨算子总结"""
        
        tools = [self._create_analysis_read_tool()]
        
        final_schemas = [
            ResponseSchema(name="analyzed_algorithms", description="分析的算子列表"),
            ResponseSchema(name="cross_algorithm_patterns", description="跨算子优化模式"),
            ResponseSchema(name="algorithm_level_optimizations", description="通用算法层优化策略库"),
            ResponseSchema(name="code_level_optimizations", description="通用代码层优化策略库"),
            ResponseSchema(name="instruction_level_optimizations", description="通用指令层优化策略库"),
            ResponseSchema(name="optimization_taxonomy", description="优化策略分类体系"),
            ResponseSchema(name="best_practices", description="最佳实践建议"),
            ResponseSchema(name="coverage_analysis", description="策略覆盖度分析"),
            ResponseSchema(name="timestamp", description="总结时间戳")
        ]
        final_parser = StructuredOutputParser.from_response_schemas(final_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能的跨算子优化专家。你具备宏观分析和模式提取能力。

🧠 **智能总结能力**:
1. 识别跨算子的通用优化模式和规律
2. 构建完整的优化策略分类体系
3. 提供策略覆盖度分析和质量评估
4. 生成实用的最佳实践建议

🎯 **总结目标**:
- 构建OpenBLAS优化策略知识库
- 发现通用优化规律和最佳实践
- 提供策略应用指导和建议
- 评估优化策略的完整性和实用性

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=final_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20)
    
    # ===== 工具方法 =====
    def _create_scan_tool(self):
        @tool
        def intelligent_scan_kernel_directory() -> str:
            """智能扫描kernel目录，自适应处理大量文件"""
            try:
                kernel_path = "/home/dgc/mjs/project/analyze_OB/openblas-output/GENERIC/kernel"
                if not os.path.exists(kernel_path):
                    return f"目录不存在: {kernel_path}"
                
                files = [f for f in os.listdir(kernel_path) if f.endswith('.c') and 'clean' in f]
                files.sort()
                
                return f"发现 {len(files)} 个.clean.c文件，准备智能分类:\n" + "\n".join(files[:50]) + \
                       (f"\n... 还有 {len(files)-50} 个文件" if len(files) > 50 else "")
            except Exception as e:
                return f"扫描失败: {str(e)}"
        
        return intelligent_scan_kernel_directory
    
    def _create_file_read_tool(self):
        @tool
        def smart_read_source_file(file_path: str) -> str:
            """智能读取源文件，自动处理大文件"""
            try:
                full_path = os.path.join("/home/dgc/mjs/project/analyze_OB/openblas-output/GENERIC/kernel", file_path)
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(20000)  # 增加读取长度
                return f"文件: {file_path}\n内容:\n{content}"
            except Exception as e:
                return f"读取失败: {str(e)}"
        
        return smart_read_source_file
    
    def _create_analysis_read_tool(self):
        @tool
        def read_analysis_results(file_path: str) -> str:
            """读取分析结果文件"""
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                return f"读取失败: {str(e)}"
        
        return read_analysis_results


# ===== 文件管理器 =====
class FileManager:
    """文件管理器"""
    
    @staticmethod
    def ensure_directories(report_folder: str):
        """创建所有必要的目录"""
        Path(report_folder).mkdir(parents=True, exist_ok=True)
        Path(f"{report_folder}/discovery_results").mkdir(exist_ok=True)
        Path(f"{report_folder}/analysis_results").mkdir(exist_ok=True)
        Path(f"{report_folder}/strategy_reports").mkdir(exist_ok=True)
        Path(f"{report_folder}/supervisor_logs").mkdir(exist_ok=True)  # 新增Supervisor日志目录
    
    @staticmethod
    def get_discovery_output_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/discovery_results/{algorithm}_discovery.json"
    
    @staticmethod
    def get_analysis_output_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/analysis_results/{algorithm}_analysis.json"
    
    @staticmethod
    def get_individual_summary_path(report_folder: str, algorithm: str) -> str:
        return f"{report_folder}/strategy_reports/{algorithm}_summary.json"
    
    @staticmethod
    def get_final_summary_path(report_folder: str) -> str:
        return f"{report_folder}/strategy_reports/final_optimization_summary.json"
    
    @staticmethod
    def get_supervisor_log_path(report_folder: str) -> str:
        return f"{report_folder}/supervisor_logs/supervisor_decisions.json"
    
    @staticmethod
    def save_content(file_path: str, content: str) -> bool:
        """保存内容到文件，支持错误恢复"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 备份现有文件
            if os.path.exists(file_path):
                backup_path = f"{file_path}.backup"
                os.rename(file_path, backup_path)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 删除备份文件
            backup_path = f"{file_path}.backup"
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            return True
        except Exception as e:
            print(f"保存文件失败 {file_path}: {e}")
            
            # 恢复备份文件
            backup_path = f"{file_path}.backup"
            if os.path.exists(backup_path):
                os.rename(backup_path, file_path)
            
            return False
    
    @staticmethod
    def log_supervisor_decision(report_folder: str, decision_data: dict):
        """记录Supervisor决策日志"""
        log_path = FileManager.get_supervisor_log_path(report_folder)
        
        # 读取现有日志
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # 添加新决策
        logs.append({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            **decision_data
        })
        
        # 保存日志
        FileManager.save_content(log_path, json.dumps(logs, ensure_ascii=False, indent=2))


# ===== 导出 =====
__all__ = [
    'AgentFactory',
    'FileManager',
    'supervisor_router',
    'create_supervisor_agent'
]
