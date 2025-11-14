#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - 标准LangChain Agent + Tools架构
使用config.json作为状态存储，Agent可随时读写状态
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

# ===== 核心Agent工具集 (去重后) =====

@tool
def get_current_timestamp() -> str:
    """【时间戳工具】获取当前时间戳，用于创建带时间标识的输出目录和文件
    
    ⚡ 使用场景：
    - 需要为分析结果创建唯一的时间戳目录
    - 生成报告时需要时间标识
    - 创建带时间戳的文件名避免覆盖
    
    Returns:
        JSON格式的时间信息，包含：
        - timestamp: Unix时间戳 (用于程序处理)
        - formatted_time: 格式化时间 YYYYMMDD_HHMMSS (用于目录名)
        - readable_time: 可读时间格式 (用于显示)
        
    🌟 示例用法：
        timestamp_info = get_current_timestamp()
        # 返回: {"timestamp": 1640995200, "formatted_time": "20220101_120000", "readable_time": "2022-01-01 12:00:00"}
    """
    current_time = int(time.time())
    formatted_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
    
    return json.dumps({
        "timestamp": current_time,
        "formatted_time": formatted_time,
        "readable_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
    })

@tool
def read_workflow_state() -> str:
    """【状态读取工具】Agent读取工作流的完整状态信息
    
    ⚡ 使用场景：
    - Agent需要了解当前工作流进度
    - 检查已完成的任务列表
    - 获取当前分析的算子和索引
    - 判断下一步应该执行什么操作
    
    Returns:
        JSON格式的完整工作流状态，包含：
        - user_request: 用户原始需求
        - analysis_type: 分析类型 (quick/comprehensive/custom)
        - current_algorithm: 当前正在分析的算子名称
        - current_algorithm_index: 当前算子在列表中的索引位置
        - completed_tasks: 已完成任务的列表
        - algorithms: 需要分析的算子列表
        - workflow_complete: 工作流是否完成
        - report_folder: 报告输出文件夹
        - iteration_count: 迭代计数
        - errors: 错误记录列表
        - next_action: 下一步应该执行的动作
        
    🌟 示例用法：
        state = read_workflow_state()
        # 返回完整的workflow状态JSON字符串
    """
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 如果没有workflow状态，初始化一个
        if "workflow" not in config:
            config["workflow"] = {
                "current_algorithm": "",
                "current_algorithm_index": 0,
                "completed_tasks": [],
                "algorithms": [],
                "workflow_complete": False,
                "analysis_type": "",
                "report_folder": "",
                "iteration_count": 0,
                "errors": [],
                "next_action": "planning"
            }
            # 保存初始化的状态
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        return json.dumps(config["workflow"], ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "error": f"读取状态失败: {str(e)}",
            "current_algorithm": "",
            "completed_tasks": [],
            "workflow_complete": False
        })

@tool
def update_workflow_state(state_updates: str) -> str:
    """【状态更新工具】Agent更新工作流状态到config.json，实现状态持久化
    
    ⚡ 使用场景：
    - Agent完成某个任务后，标记任务完成状态
    - 更新当前处理的算子信息
    - 修改下一步要执行的动作
    - 记录错误信息或工作流完成状态
    
    Args:
        state_updates: JSON格式的状态更新数据，支持的字段：
            - completed_tasks_add: 添加已完成任务（如 "scout_gemm"）
            - current_algorithm: 更新当前算子
            - current_algorithm_index: 更新算子索引
            - next_action: 设置下一步动作
            - workflow_complete: 设置工作流完成状态
            - report_folder: 设置报告文件夹
            - errors: 添加错误记录
            
    Returns:
        JSON格式的操作结果，包含success状态和错误信息
        
    🌟 示例用法：
        # 标记Scout任务完成
        update_workflow_state('{{"completed_tasks_add": "scout_gemm"}}')
        
        # 更新下一步动作
        update_workflow_state('{"next_action": "analyze"}')
        
        # 标记工作流完成
        update_workflow_state('{"workflow_complete": true}')
    """
    try:
        # 读取当前配置
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # 确保workflow状态存在
        if "workflow" not in config:
            config["workflow"] = {}
        
        # 解析更新数据
        updates = json.loads(state_updates) if isinstance(state_updates, str) else state_updates
        
        updated_fields = []
        
        # 应用所有更新
        for field, value in updates.items():
            if field == "completed_tasks_add":
                # 添加完成任务
                if "completed_tasks" not in config["workflow"]:
                    config["workflow"]["completed_tasks"] = []
                if value not in config["workflow"]["completed_tasks"]:
                    config["workflow"]["completed_tasks"].append(value)
                    updated_fields.append(f"添加任务: {value}")
            elif field == "algorithms_set":
                # 设置算子列表
                config["workflow"]["algorithms"] = value
                updated_fields.append(f"设置算子列表: {value}")
            else:
                # 直接更新字段
                config["workflow"][field] = value
                updated_fields.append(f"{field}: {value}")
        
        # 保存更新后的配置
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return json.dumps({
            "success": True,
            "updated_fields": updated_fields,
            "message": f"成功更新状态: {', '.join(updated_fields)}"
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": "状态更新失败"
        })

@tool
def analyze_and_decide_next_step() -> str:
    """【智能决策工具】Agent分析当前工作流状态并智能决定下一步行动
    
    ⚡ 使用场景：
    - Master Agent需要协调整个工作流程
    - 自动判断当前算子的完成状态
    - 决定是否进入下一个算子或工作流阶段
    - 检查工作流是否已完全完成
    
    🧠 智能逻辑：
    - 分析已完成任务列表判断当前算子状态
    - 按 scout → analyze → strategize 顺序执行
    - 自动切换到下一个算子或总结阶段
    - 处理工作流完成和异常情况
    
    Returns:
        JSON格式的决策结果，包含：
        - next_action: 下一步要执行的动作
        - reasoning: 决策理由和逻辑
        - current_algorithm: 当前处理的算子
        - recommendation: 执行建议
        
    🌟 示例用法：
        decision = analyze_and_decide_next_step()
        # 自动分析状态并决定：scout/analyze/strategize/summarize/complete
    """
    try:
        # 读取当前状态
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        workflow = config.get("workflow", {})
        algorithms = workflow.get("algorithms", [])
        current_index = workflow.get("current_algorithm_index", 0)
        completed_tasks = workflow.get("completed_tasks", [])
        
        if "error" in workflow:
            return json.dumps({
                "next_action": "complete",
                "reasoning": "状态读取失败",
                "recommendation": "检查配置文件"
            })
        
        # 简化决策逻辑：按固定顺序执行
        if current_index < len(algorithms):
            current_alg = algorithms[current_index]
            
            # 按固定顺序检查：scout → analyze → strategize → individual_summarize
            scout_done = f"scout_{current_alg}" in completed_tasks
            analyze_done = f"analyze_{current_alg}" in completed_tasks
            strategize_done = f"strategize_{current_alg}" in completed_tasks
            individual_summarize_done = f"individual_summarize_{current_alg}" in completed_tasks
            
            if not scout_done:
                next_action = "scout"
                reasoning = f"执行{current_alg}算子发现阶段"
            elif not analyze_done:
                next_action = "analyze"
                reasoning = f"执行{current_alg}算子分析阶段"
            elif not strategize_done:
                next_action = "strategize"
                reasoning = f"执行{current_alg}算子策略阶段"
            elif not individual_summarize_done:
                next_action = "individual_summarize"
                reasoning = f"执行{current_alg}算子个人总结阶段"
            else:
                # 当前算子完成，切换到下一个算子
                next_index = current_index + 1
                if next_index < len(algorithms):
                    next_alg = algorithms[next_index]
                    # 更新到下一个算子
                    update_workflow_state(json.dumps({
                        "current_algorithm_index": next_index,
                        "current_algorithm": next_alg
                    }))
                    next_action = "scout"  # 下一个算子从scout开始
                    reasoning = f"{current_alg}完成，开始处理{next_alg}算子"
                else:
                    next_action = "final_summarize"
                    reasoning = "所有算子完成，执行最终总结"
        else:
            # 检查最终总结是否完成
            final_summarize_done = "final_summarize" in completed_tasks
            if not final_summarize_done:
                next_action = "final_summarize"
                reasoning = "执行跨算子最终总结"
            else:
                next_action = "complete"
                reasoning = "所有工作已完成"
        
        # 更新下一步行动
        update_workflow_state(json.dumps({"next_action": next_action}))
        
        return json.dumps({
            "next_action": next_action,
            "reasoning": reasoning,
            "current_algorithm": workflow.get("current_algorithm", ""),
            "progress": f"{len(completed_tasks)}/{len(algorithms) * 4 + 1}" if algorithms else "完成"
        })
        
    except Exception as e:
        return json.dumps({
            "next_action": "complete",
            "reasoning": f"决策失败: {str(e)}",
            "error": str(e)
        })



@tool
def get_algorithm_list(analysis_type: str) -> str:
    """【算子配置工具】根据分析类型获取需要处理的BLAS算子列表
    
    ⚡ 使用场景：
    - 工作流初始化时确定要分析的算子范围
    - 根据用户需求选择不同复杂度的分析任务
    - 为后续的Scout、Analyzer、Strategist提供工作清单
    
    📋 支持的分析类型：
    - quick/快速: 核心算子 [gemm, axpy, dot] - 3个算子
    - comprehensive/全面: 扩展算子集 [gemm, axpy, dot, gemv, nrm2, ger] - 6个算子  
    - custom/自定义: 默认使用快速分析的算子集
    
    Args:
        analysis_type: 分析类型字符串，支持中英文
            - "quick" 或包含 "快速" 
            - "comprehensive" 或包含 "全面"
            - 其他值使用默认快速分析
        
    Returns:
        JSON格式的算子配置，包含：
        - algorithms: 算子名称列表
        - count: 算子数量
        - type: 分析类型标识
        
    🌟 示例用法：
        config = get_algorithm_list("quick")
        # 返回: {"algorithms": ["gemm", "axpy", "dot"], "count": 3, "type": "quick"}
    """
    if analysis_type == "quick" or "快速" in analysis_type:
        algorithms = ['gemm', 'axpy', 'dot']
    elif analysis_type == "comprehensive" or "全面" in analysis_type:
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
    """标准LangChain Agent工厂 - 创建真正的Agent+Tools架构"""
    
    def __init__(self, model_config: dict = None):
        if model_config is None:
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                model_config = config["model"]
        
        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 文件管理工具 - 参考analyze.py模式增强描述
        self.file_toolkit = FileManagementToolkit(
            root_dir=".",  # 项目根目录
            selected_tools=["read_file", "write_file", "list_directory", "file_search"]
        )
        self.file_tools = self._enhance_tool_descriptions(self.file_toolkit.get_tools())
        
        # 所有工具集合
        self.all_tools = [
            get_current_timestamp,
            read_workflow_state,
            update_workflow_state,
            analyze_and_decide_next_step
        ] + self.file_tools
        
        # 创建必要目录
        Path("results").mkdir(exist_ok=True)
    
    def _enhance_tool_descriptions(self, tools):
        """为通用文件工具添加OpenBLAS特定的使用描述 - 参考analyze.py模式"""
        enhanced_tools = []
        
        for tool in tools:
            if tool.name == "read_file":
                tool.description += (
                    "\n\n**OpenBLAS工作流专用格式:**\n"
                    "- 读取OpenBLAS源码: OpenBLAS-develop/kernel/目录下的.c/.S文件\n"
                    "- 读取发现结果: results/{{timestamp}}/discovery_results/{{algorithm}}_discovery.json\n"
                    "- 读取分析结果: results/{{timestamp}}/analysis_results/{{algorithm}}_analysis.json\n"
                    "- 读取策略结果: results/{{timestamp}}/strategy_reports/{{algorithm}}_strategy.md\n"
                    "- 验证保存结果: 保存后必须用此工具验证文件内容\n"
                    "- **重要**: 不要显示文件内容到控制台，只做静默读取验证"
                )
            elif tool.name == "write_file":
                tool.description += (
                    "\n\n**OpenBLAS工作流专用格式:**\n"
                    "- **文件夹结构**: results/{{timestamp}}/\n"
                    "  ├── discovery_results/\n"
                    "  │   ├── {{algorithm}}_discovery.json\n"
                    "  ├── analysis_results/\n"
                    "  │   ├── {{algorithm}}_analysis.json\n"
                    "  └── strategy_reports/\n"
                    "      ├── {{algorithm}}_strategy.md\n"
                    "      ├── {{algorithm}}_summary.md\n"
                    "      └── final_optimization_summary.md\n"
                    "- **Scout保存**: results/{{timestamp}}/discovery_results/{{algorithm}}_discovery.json\n"
                    "- **Analyzer保存**: results/{{timestamp}}/analysis_results/{{algorithm}}_analysis.json\n"
                    "- **Strategist保存**: results/{{timestamp}}/strategy_reports/{{algorithm}}_strategy.md\n"
                    "- **Individual Summarizer保存**: results/{{timestamp}}/strategy_reports/{{algorithm}}_summary.md\n"
                    "- **Final Summarizer保存**: results/{{timestamp}}/strategy_reports/final_optimization_summary.md\n"
                    "- **重要**: 每个算子独立JSON文件，不要合并多个算子到一个文件\n"
                    "- **重要**: 保存后必须用read_file验证，失败重试直到成功"
                )
            elif tool.name == "list_directory":
                tool.description += (
                    "\n\n**OpenBLAS工作流专用:**\n"
                    "- 探索OpenBLAS-develop/kernel/目录结构寻找算法实现\n"
                    "- 检查输出目录: results/{{timestamp}}/discovery_results/, results/{{timestamp}}/analysis_results/, results/{{timestamp}}/strategy_reports/\n"
                    "- 列出算子分析文件: results/{{timestamp}}/analysis_results/\n"
                    "- 检查策略报告: results/{{timestamp}}/strategy_reports/"
                )
            elif tool.name == "file_search":
                tool.description += (
                    "\n\n**OpenBLAS工作流专用:**\n"
                    "- 在OpenBLAS-develop/kernel/搜索算法实现文件\n"
                    "- 查找不同架构实现: generic, x86_64, arm64, riscv64等\n"
                    "- 支持模糊搜索: 搜索'gemm'找到所有gemm相关文件\n"
                    "- 搜索已保存结果: results/{{timestamp}}/discovery_results/, results/{{timestamp}}/analysis_results/\n"
                    "- **限制**: Scout阶段只需找3-5个代表性文件，避免搜索过多"
                )
            
            enhanced_tools.append(tool)
        
        return enhanced_tools
    
    def create_master_coordinator_agent(self) -> AgentExecutor:
        """创建Master协调器Agent"""
        
        # Master专用工具集：协调管理 + 算子配置
        master_tools = [
            # 状态管理工具
            read_workflow_state,
            update_workflow_state,
            analyze_and_decide_next_step,
            # 配置工具
            get_algorithm_list,
            get_current_timestamp
        ]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS优化分析的Master协调器Agent。

🎯 **核心目标：**
统筹管理整个OpenBLAS算子优化分析工作流，协调各专家Agent完成复杂的代码分析任务。

📋 **标准工作流程：**
**第一阶段：工作流初始化**
1. 解析用户需求，确定分析类型（快速/全面/自定义）
2. 获取对应的算子列表，设置工作流状态
3. 为当前分析任务创建时间戳报告目录

**第二阶段：逐个算子分析** (对每个算子执行以下步骤)
1. **Scout阶段：** 发现算子实现文件（3-5个代表性文件）
2. **Analyzer阶段：** 分析优化技术（三层框架分析）
3. **Strategist阶段：** 提炼优化策略（生成Markdown报告）

**第三阶段：跨算子总结**
1. **Summarizer阶段：** 分析所有算子报告，生成总结
2. **工作流完成：** 标记完成状态，结束流程

🧠 **智能调度逻辑：**
- 按算子索引顺序处理：当前算子完成后自动切换到下一个
- 阶段顺序：scout → analyze → strategize (循环) → summarize
- 异常处理：记录错误但继续处理其他算子
- 完成判断：所有算子的三个阶段完成后进行总结

📊 **管理范围：**
- **算子类型：** 快速分析(gemm,axpy,dot) / 全面分析(+gemv,nrm2,ger)
- **文件发现：** 每个算子限制3-5个代表性实现文件
- **状态持久化：** 基于config.json的集中状态管理
- **输出结构：** results/{{timestamp}}/discovery_results/ → results/{{timestamp}}/analysis_results/ → results/{{timestamp}}/strategy_reports/"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, master_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=master_tools,
            verbose=False,
            max_iterations=15,
            handle_parsing_errors=True
        )
    
    def create_scout_specialist_agent(self) -> AgentExecutor:
        """创建Scout专家Agent - 代码发现和文件整理"""
        
        # Scout专用工具集：增强版文件工具 + 状态管理
        scout_tools = [
            get_current_timestamp,
            read_workflow_state,
            update_workflow_state
        ] + self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS代码侦察专家，专门负责发现和保存算子实现文件。

🎯 **执行任务：**
1. **读取任务：** 调用read_workflow_state获取当前算子名称和报告文件夹
2. **搜索文件：** 在OpenBLAS-develop/kernel/目录搜索算子相关文件
3. **筛选文件：** 选择3-5个不同架构的代表性实现（generic、x86_64、arm64等）
4. **保存结果：** 保存到报告文件夹下的discovery_results/{{算子名}}_discovery.json

📋 **新的保存格式 (重要)：**
保存到: results/{{timestamp}}/discovery_results/{{算子名}}_discovery.json
```json
{{
  "algorithm": "算法名",
  "files": [
    {{"path": "文件路径", "type": "实现类型", "description": "架构特征"}}
  ],
  "timestamp": "发现时间",
  "session_folder": "results/{{时间戳}}"
}}
```

**实现类型：** generic, x86_optimized, simd_optimized, microkernel

🔧 **执行流程：**
1. 获取当前算子名称和报告文件夹（report_folder字段，格式：results/timestamp）
2. 搜索OpenBLAS-develop/kernel/下的相关文件
3. 选择3-5个代表性文件
4. 创建文件夹：{{report_folder}}/discovery_results/
5. 保存为：{{report_folder}}/discovery_results/{{算子名}}_discovery.json
6. 用read_file验证保存成功
7. 调用update_workflow_state标记完成：{{"completed_tasks_add": "scout_算子名"}}

⚠️ **注意：**
- 每个算子独立保存一个JSON文件
- 保存后必须验证文件内容
- 必须调用状态更新工具完成任务标记"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, scout_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=scout_tools,
            verbose=False,
            max_iterations=15,
            handle_parsing_errors=True
        )
    
    def create_analyzer_specialist_agent(self) -> AgentExecutor:
        """创建Analyzer专家Agent"""
        
        # Analyzer专用工具集：增强版文件工具 + 状态管理
        analyzer_tools = [
            get_current_timestamp,
            read_workflow_state,
            update_workflow_state
        ] + [tool for tool in self.file_tools if tool.name in ['read_file', 'write_file', 'list_directory']]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高性能计算分析专家Agent。

🎯 **核心目标：**
深度分析指定算子的性能优化技术，识别和分类所有优化策略。

🔍 **分析框架：**
- **算法层：** 循环展开、分块技术、数据重用策略
- **代码层：** 缓存优化、内存对齐、数据预取
- **指令层：** SIMD向量化、FMA指令、指令级并行

📋 **任务要求：**
1. **数据获取：** 从 {{report_folder}}/discovery_results/{{算子名}}_discovery.json 读取发现阶段结果
2. **技术识别：** 按三层框架系统性识别所有优化技术
3. **分类整理：** 将优化技术分类并详细描述实现机制
4. **结果输出：** 生成结构化分析报告保存到 {{report_folder}}/analysis_results/ 目录
5. **状态更新：** 完成分析后必须标记为已完成，格式：{{"completed_tasks_add": "analyze_算子名"}}

📊 **输出规范：**
- 读取报告文件夹路径（report_folder字段，格式：results/timestamp）
- 创建文件夹：{{report_folder}}/analysis_results/
- 保存文件：{{report_folder}}/analysis_results/{{算子名}}_analysis.json
- 内容包含：优化技术分类、实现机制、性能影响分析
- 保存后必须用read_file验证成功

🧠 **分析深度：**
- 理解代码的性能关键路径
- 识别架构相关的优化特征
- 分析优化技术的适用场景和效果"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, analyzer_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=analyzer_tools,
            verbose=False,
            max_iterations=20,
            handle_parsing_errors=True
        )
    
    def create_strategist_specialist_agent(self) -> AgentExecutor:
        """创建Strategist专家Agent"""
        
        # Strategist专用工具集：增强版文件工具 + 状态管理
        strategist_tools = [
            get_current_timestamp,
            read_workflow_state,
            update_workflow_state
        ] + [tool for tool in self.file_tools if tool.name in ['read_file', 'write_file', 'list_directory']]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是优化策略大师Agent。

🎯 **核心目标：**
将技术分析结果转化为可实施的具体优化策略和实践指南。

🔍 **策略框架：**
- **算法设计层：** 计算逻辑优化、时空复杂度权衡
- **代码优化层：** 性能加速技术、代码结构调整
- **特化指令层：** 专用指令利用、硬件特性充分发挥

📋 **任务要求：**
1. **策略提炼：** 从 {{report_folder}}/analysis_results/{{算子名}}_analysis.json 读取分析结果
2. **方案设计：** 为每个优化点设计具体的实现方案
3. **指南生成：** 创建包含代码示例的实施指南
4. **文档输出：** 生成结构化Markdown策略报告保存到 {{report_folder}}/strategy_reports/
5. **状态更新：** 完成策略制定后必须标记为已完成，格式：{{"completed_tasks_add": "strategize_算子名"}}

📊 **输出规范：**
- 读取报告文件夹路径（report_folder字段，格式：results/timestamp）
- 创建文件夹：{{report_folder}}/strategy_reports/
- 保存文件：{{report_folder}}/strategy_reports/{{算子名}}_strategy.md
- 内容包含：策略分类、实施步骤、代码示例、效果预期
- 保存后必须用read_file验证成功

🧠 **策略深度：**
- 提供具体可行的优化实施路径
- 包含性能提升的量化预期
- 考虑不同硬件平台的适用性
- 评估优化的复杂度和收益比"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, strategist_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=strategist_tools,
            verbose=False,
            max_iterations=20,
            handle_parsing_errors=True
        )
    
    def create_individual_summarizer_agent(self) -> AgentExecutor:
        """创建Individual Summarizer专家Agent - 负责单独算子的优化总结"""
        
        # Individual Summarizer专用工具集
        individual_summarizer_tools = [
            read_workflow_state,
            update_workflow_state
        ] + [tool for tool in self.file_tools if tool.name in ['read_file', 'write_file', 'list_directory']]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是单算子优化总结专家Agent。

🎯 **核心目标：**
为单个算子生成全面的优化总结报告，整合发现、分析、策略三个阶段的成果。

📋 **任务要求：**
1. **数据整合：** 从 {{report_folder}} 读取算子的 discovery_results/{{算子名}}_discovery.json、analysis_results/{{算子名}}_analysis.json、strategy_reports/{{算子名}}_strategy.md 三个文件
2. **总结生成：** 整合所有信息生成该算子的完整优化总结
3. **格式规范：** 按照final_optimization_summary.md的格式生成单算子总结
4. **保存输出：** 生成算子专属的总结报告到 {{report_folder}}/strategy_reports/
5. **状态更新：** 完成后标记为已完成，格式：{{"completed_tasks_add": "individual_summarize_算子名"}}

📊 **输出规范：**
- 读取报告文件夹路径（report_folder字段，格式：results/timestamp）
- 保存文件：{{report_folder}}/strategy_reports/{{算子名}}_summary.md
- 内容包含：算子特性、优化技术、实施策略、性能预期
- 保存后必须用read_file验证成功

🧠 **总结重点：**
- 该算子的核心特性和优化挑战
- 发现的关键优化技术和实现方案
- 具体的策略建议和实施路径
- 预期的性能提升效果"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, individual_summarizer_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=individual_summarizer_tools,
            verbose=False,
            max_iterations=20,
            handle_parsing_errors=True
        )

    def create_final_summarizer_agent(self) -> AgentExecutor:
        """创建Final Summarizer专家Agent - 负责跨算子的最终总结"""
        
        # Final Summarizer专用工具集
        final_summarizer_tools = [
            read_workflow_state,
            update_workflow_state
        ] + [tool for tool in self.file_tools if tool.name in ['read_file', 'write_file', 'list_directory']]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是跨算子最终总结专家Agent。

🎯 **核心目标：**
汇总多个算子的优化策略，发现通用规律，生成跨算子的综合性最终总结报告。

📋 **任务要求：**
1. **数据收集：** 从 {{report_folder}}/strategy_reports/ 收集所有算子的个人总结报告（{{算子名}}_summary.md文件）
2. **模式识别：** 分析跨算子的共性优化模式和差异特征
3. **规律提炼：** 总结通用的优化原则和最佳实践
4. **最终报告：** 创建综合性的跨算子优化总结文档保存到 {{report_folder}}/strategy_reports/final_optimization_summary.md
5. **工作流结束：** 标记整个工作流完成，格式：{{"completed_tasks_add": "final_summarize", "workflow_complete": true}}

📊 **输出规范：**
- 读取报告文件夹路径（report_folder字段，格式：results/timestamp）
- 保存文件：{{report_folder}}/strategy_reports/final_optimization_summary.md
- 内容包含：通用优化模式、架构差异分析、最佳实践建议、跨算子规律
- 保存后必须用read_file验证成功

🧠 **分析重点：**
- 识别跨算子的共同优化技术
- 发现不同算子的特化优化策略
- 总结硬件架构相关的优化规律
- 提供面向未来的优化指导原则"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, final_summarizer_tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=final_summarizer_tools,
            verbose=False,
            max_iterations=25,
            handle_parsing_errors=True
        )

# ===== 导出 =====
__all__ = ['StandardAgentFactory'] 