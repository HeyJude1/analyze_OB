#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - 标准LangChain Agent + Tools架构
符合LangChain官方Agent定义：Agent使用LLM选择和执行Tools序列
"""

import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel

# 加载环境变量
load_dotenv()

# ===== 标准LangChain Tools (使用@tool装饰器) =====

@tool
def get_current_timestamp() -> str:
    """获取当前时间戳
    
    Returns:
        当前Unix时间戳和格式化时间字符串
    """
    current_time = int(time.time())
    formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
    
    return json.dumps({
        "timestamp": current_time,
        "formatted_time": formatted_time,
        "readable_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    })

@tool
def analyze_workflow_state(state_data: str) -> str:
    """分析当前工作流状态并推荐下一步行动
    
    Args:
        state_data: JSON格式的状态数据，包含current_stage, completed_tasks, algorithms等
        
    Returns:
        推荐的下一步行动和理由
    """
    try:
        state = json.loads(state_data) if isinstance(state_data, str) else state_data
        
        current_stage = state.get("current_stage", "planning")
        completed_tasks = state.get("completed_tasks", [])
        algorithms = state.get("algorithms", [])
        current_algorithm_index = state.get("current_algorithm_index", 0)
        
        # 简单的状态分析逻辑
        if current_algorithm_index < len(algorithms):
            current_algorithm = algorithms[current_algorithm_index]
            
            # 检查当前算子的完成状态
            scout_done = any(f"scout_{current_algorithm}" in task for task in completed_tasks)
            analyze_done = any(f"analyze_{current_algorithm}" in task for task in completed_tasks)
            strategize_done = any(f"strategize_{current_algorithm}" in task for task in completed_tasks)
            
            if not scout_done:
                return f"建议执行scout任务：发现{current_algorithm}算子的实现文件"
            elif not analyze_done:
                return f"建议执行analyze任务：分析{current_algorithm}算子的优化技术"
            elif not strategize_done:
                return f"建议执行strategize任务：提炼{current_algorithm}算子的优化策略"
            else:
                return f"建议移动到下一个算子：{current_algorithm}已完成，进入下一个算子"
        else:
            summarize_done = any("summarize" in task for task in completed_tasks)
            if not summarize_done:
                return "建议执行summarize任务：生成跨算子优化策略总结"
            else:
                return "建议完成工作流：所有任务已完成"
                
    except Exception as e:
        return f"状态分析失败: {str(e)}"

@tool  
def check_output_quality(file_path: str, expected_type: str) -> str:
    """检查输出文件的质量和完整性
    
    Args:
        file_path: 要检查的文件路径
        expected_type: 期望的文件类型 (json/markdown)
        
    Returns:
        质量检查结果和建议
    """
    try:
        if not os.path.exists(file_path):
            return f"质量检查：文件 {file_path} 不存在"
        
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return f"质量检查：文件 {file_path} 为空"
        elif file_size < 100:
            return f"质量检查：文件 {file_path} 内容较少 ({file_size} bytes)"
        
        # 检查文件格式（简化输出）
        if expected_type == "json":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                json.loads(content)
                return f"质量检查通过：JSON文件格式正确，大小 {file_size} bytes"
            except:
                return f"质量检查失败：JSON文件格式错误"
        elif expected_type == "markdown":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(200)
            if content.startswith("#") or "##" in content:
                return f"质量检查通过：Markdown文件格式正确，大小 {file_size} bytes"
            else:
                return f"质量检查：Markdown文件格式可能不标准"
        else:
            return f"质量检查完成：文件存在，大小 {file_size} bytes"
            
    except Exception as e:
        return f"质量检查错误: {str(e)}"

@tool
def update_workflow_progress(state_data: str, task_completed: str) -> str:
    """更新工作流进度状态
    
    Args:
        state_data: 当前状态数据 (JSON格式)
        task_completed: 完成的任务名称
        
    Returns:
        更新后的状态摘要
    """
    try:
        return f"进度更新：任务 '{task_completed}' 已完成"
    except Exception as e:
        return f"进度更新失败: {str(e)}"

@tool
def schedule_next_tasks(algorithms: str, completed_tasks: str) -> str:
    """智能调度下一批任务
    
    Args:
        algorithms: 算子列表 (JSON数组格式)
        completed_tasks: 已完成任务列表 (JSON数组格式)
        
    Returns:
        推荐的任务调度计划
    """
    try:
        alg_list = json.loads(algorithms) if isinstance(algorithms, str) else algorithms
        completed = json.loads(completed_tasks) if isinstance(completed_tasks, str) else completed_tasks
        
        # 简单的调度逻辑
        pending_tasks = []
        for alg in alg_list:
            if not any(f"scout_{alg}" in task for task in completed):
                pending_tasks.append(f"scout_{alg}")
            elif not any(f"analyze_{alg}" in task for task in completed):
                pending_tasks.append(f"analyze_{alg}")
            elif not any(f"strategize_{alg}" in task for task in completed):
                pending_tasks.append(f"strategize_{alg}")
        
        if not pending_tasks and not any("summarize" in task for task in completed):
            pending_tasks.append("summarize_all")
        
        if pending_tasks:
            return f"调度建议：下一步执行 {pending_tasks[0]}"
        else:
            return "调度完成：所有任务已完成"
            
    except Exception as e:
        return f"任务调度失败: {str(e)}"

@tool
def create_output_directory(directory_path: str) -> str:
    """创建输出目录
    
    Args:
        directory_path: 要创建的目录路径
        
    Returns:
        创建结果
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return f"目录创建成功: {directory_path}"
    except Exception as e:
        return f"目录创建失败: {str(e)}"

@tool
def get_algorithm_list(analysis_type: str) -> str:
    """获取指定分析类型的算子列表
    
    Args:
        analysis_type: 分析类型 (quick/comprehensive/custom)
        
    Returns:
        算子列表 (JSON格式)
    """
    if analysis_type == "quick":
        algorithms = ['gemm', 'axpy', 'dot']
    elif analysis_type == "comprehensive":
        algorithms = ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger']
    else:
        algorithms = ['gemm', 'axpy', 'dot']  # 默认
    
    return json.dumps({
        "algorithms": algorithms,
        "count": len(algorithms),
        "type": analysis_type
    })

# ===== 标准LangChain Agent工厂 =====
class StandardAgentFactory:
    """标准LangChain Agent工厂 - 符合官方Agent定义"""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """初始化标准Agent工厂"""
        # 加载配置
        if model_config is None:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                model_config = config["model"]
        
        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=0.1,
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 标准文件管理工具
        self.file_toolkit = FileManagementToolkit(
            root_dir="OpenBLAS-develop",
            selected_tools=["read_file", "write_file", "list_directory", "file_search"]
        )
        
        # 组合所有工具：文件工具 + 自定义业务工具
        self.all_tools = self._create_all_tools()
        
        # 确保目录存在
        Path("discovery_results").mkdir(exist_ok=True)
        Path("analysis_results").mkdir(exist_ok=True)
        Path("strategy_reports").mkdir(exist_ok=True)
    
    def _create_all_tools(self) -> List:
        """创建所有工具的组合"""
        # 文件管理工具
        file_tools = self.file_toolkit.get_tools()
        
        # 业务逻辑工具（使用@tool装饰器定义的）
        business_tools = [
            get_current_timestamp,
            analyze_workflow_state,
            check_output_quality,
            update_workflow_progress,
            schedule_next_tasks,
            create_output_directory,
            get_algorithm_list
        ]
        
        return file_tools + business_tools
    
    def create_master_coordinator_agent(self) -> AgentExecutor:
        """创建Master协调器Agent - 标准LangChain Agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS优化分析的Master协调器Agent，使用LLM推理和工具调用管理整个工作流。

🎯 **Agent职责:**
你是一个标准的LangChain Agent，通过LLM推理决定调用哪些工具来完成复杂的工作流管理。

🛠️ **可用工具:**

**文件管理工具:**
- read_file: 读取文件内容
- write_file: 写入文件
- list_directory: 列出目录内容
- file_search: 搜索文件

**工作流管理工具:**
- get_current_timestamp: 获取当前时间戳
- analyze_workflow_state: 分析当前状态并推荐下一步
- check_output_quality: 检查输出文件质量
- update_workflow_progress: 更新工作流进度
- schedule_next_tasks: 智能调度下一批任务
- create_output_directory: 创建必要的输出目录
- get_algorithm_list: 获取指定类型的算子列表

📋 **算子知识:**
- 快速分析: gemm, axpy, dot (核心BLAS算子)
- 全面分析: gemm, axpy, dot, gemv, nrm2, ger (完整BLAS算子集)

🔄 **工作流程:**
1. **规划阶段**: 使用get_algorithm_list获取算子列表，get_current_timestamp生成报告文件夹
2. **执行阶段**: 使用analyze_workflow_state分析当前状态，决定下一步行动
3. **质量控制**: 使用check_output_quality检查输出质量
4. **进度管理**: 使用update_workflow_progress更新状态

⚠️ **重要原则:**
- 你是标准的LangChain Agent，通过LLM推理决定工具调用序列
- 每次决策都基于当前观察结果和用户需求
- 工具调用后要分析结果，决定下一步行动
- 保持工作流的连续性和高效性"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 使用标准的create_openai_tools_agent
        agent = create_openai_tools_agent(self.llm, self.all_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=False,  # 减少输出
            max_iterations=30,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def create_scout_specialist_agent(self) -> AgentExecutor:
        """创建Scout专家Agent - 专注文件发现的标准Agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS文件发现专家Agent，专门负责发现和分析算子实现文件。

🎯 **专业领域:** BLAS算子实现文件的智能发现和分类

🛠️ **专用工具策略:**
- list_directory: 系统性探索OpenBLAS kernel目录结构
- file_search: 搜索特定算子的实现文件
- read_file: 快速识别文件类型和优化特征（避免输出文件内容）
- write_file: 保存结构化的发现结果JSON
- create_output_directory: 确保输出目录存在

📊 **发现标准:**
- 发现至少3种不同架构实现 (generic, x86_64, arm64等)
- 识别实现类型 (simd_optimized, microkernel, baseline等)
- 生成标准JSON格式输出

💼 **工作流程:**
1. 使用list_directory探索kernel/目录结构
2. 使用file_search搜索特定算子实现
3. 使用read_file分析关键文件特征（不输出具体内容）
4. 使用create_output_directory确保 ../discovery_results 目录存在
5. 使用write_file保存发现结果JSON

⚠️ **重要:** 
- 文件保存路径必须是 ../discovery_results/算子名_discovered_时间戳.json
- 使用read_file时只分析文件特征，不要输出具体内容
- 确保生成完整的JSON格式发现结果"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.all_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=False,  # 减少输出
            max_iterations=20,
            handle_parsing_errors=True
        )
    
    def create_analyzer_specialist_agent(self) -> AgentExecutor:
        """创建Analyzer专家Agent - 专注代码分析的标准Agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高性能计算分析专家Agent，专门负责深度分析BLAS算子代码实现。

🎯 **专业领域:** 代码优化技术的识别、分类和性能分析

🔍 **三层分析框架:**
- 算法层: 循环展开、分块、数据重用、计算重排
- 代码层: 缓存优化、内存对齐、预取、编译器优化  
- 指令层: SIMD向量化、FMA指令、指令并行、流水线

🛠️ **专用工具策略:**
- read_file: 读取发现结果和源代码（避免输出具体内容）
- create_output_directory: 创建算子专用的分析目录
- write_file: 保存详细的分析结果JSON
- check_output_quality: 验证分析结果的质量

📊 **分析标准:**
- 每个实现识别至少5种优化技术
- 按三层框架准确分类
- 评估性能影响和适用场景
- 生成结构化JSON格式

💼 **工作流程:**
1. 使用read_file读取发现结果（不输出内容）
2. 深度分析优化技术实现
3. 按三层框架分类技术特征
4. 使用create_output_directory创建 ../analysis_results/算子名/ 目录
5. 使用write_file保存分析报告JSON
6. 使用check_output_quality验证结果

⚠️ **重要:** 
- 文件保存路径必须是 ../analysis_results/算子名/analysis_算子名_时间戳.json
- 使用read_file时只进行分析，不要输出文件具体内容
- 确保生成完整的分析结果JSON"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.all_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=False,  # 减少输出
            max_iterations=25,
            handle_parsing_errors=True
        )
    
    def create_strategist_specialist_agent(self) -> AgentExecutor:
        """创建Strategist专家Agent - 专注策略提炼的标准Agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是优化策略大师Agent，专门负责从技术分析中提炼实用的优化策略。

🎯 **专业领域:** 优化策略的提炼、实践指导和价值评估

🔍 **三层策略框架:**
- 算法设计层: 计算逻辑优化、时空权衡策略
- 代码优化层: 性能加速技术、结构调整方法
- 特有指令层: 专用指令利用、硬件特性发挥

🛠️ **专用工具策略:**
- read_file: 读取分析结果（避免输出具体内容）
- get_current_timestamp: 获取时间戳用于文件夹命名
- create_output_directory: 创建带时间戳的策略报告目录
- write_file: 生成高质量Markdown策略报告
- check_output_quality: 验证策略报告质量

📝 **策略输出要求:**
- Markdown格式的结构化报告
- 包含具体代码示例和性能数据
- 提供可操作的实施步骤
- 评估适用场景和预期收益

💼 **工作流程:**
1. 使用read_file读取分析结果
2. 使用get_current_timestamp获取时间戳
3. 使用create_output_directory创建 ../strategy_reports/report_时间戳/ 目录
4. 深度理解优化技术并提炼策略
5. 使用write_file生成策略报告Markdown
6. 使用check_output_quality验证报告质量

⚠️ **重要:** 
- 必须先使用get_current_timestamp获取时间戳
- 创建带时间戳的报告目录: ../strategy_reports/report_时间戳/
- 文件保存路径: ../strategy_reports/report_时间戳/算子名_optimization_analysis.md
- 使用read_file时只进行分析，不要输出文件具体内容"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.all_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=False,  # 减少输出
            max_iterations=20,
            handle_parsing_errors=True
        )
    
    def create_summarizer_specialist_agent(self) -> AgentExecutor:
        """创建Summarizer专家Agent - 专注跨算子总结的标准Agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是跨算子总结专家Agent，专门负责发现多算子间的通用优化规律。

🎯 **专业领域:** 跨算子分析、通用模式识别和价值提炼

🔍 **跨算子分析框架:**
- 共性模式: 识别通用优化技术和设计模式
- 差异特征: 分析算子特化和架构适配策略
- 性能效果: 评估优化技术收益和适用场景

🛠️ **专用工具策略:**
- list_directory: 系统收集策略报告目录
- read_file: 读取多个算子的策略报告（避免输出具体内容）
- write_file: 生成综合性总结报告
- check_output_quality: 验证总结报告质量

📊 **总结标准:**
- 分析所有可用的算子策略报告
- 识别5个以上通用优化模式
- 提供量化性能效果评估
- 生成结构化Markdown总结报告

💼 **工作流程:**
1. 使用list_directory收集策略报告目录
2. 使用read_file逐个分析算子策略（不输出内容）
3. 横向对比发现共性和差异
4. 提炼通用优化原则和最佳实践
5. 使用write_file生成总结报告
6. 使用check_output_quality验证质量

⚠️ **重要:** 
- 文件保存在现有的时间戳目录中: ../strategy_reports/report_时间戳/optimization_summary_report.md
- 使用read_file时只进行分析，不要输出文件具体内容
- 确保生成完整的跨算子总结报告"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.all_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.all_tools,
            verbose=False,  # 减少输出
            max_iterations=25,
            handle_parsing_errors=True
        )

# ===== 导出 =====
__all__ = ['StandardAgentFactory'] 