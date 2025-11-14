#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - Master Agent调度系统
基于LangGraph的智能多Agent协作框架
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict  # 官方推荐使用typing_extensions
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# 加载环境变量
load_dotenv()

# ===== Master Agent调度系统 =====
class OpenBLASMasterAgentFactory:
    """Master Agent工厂 - 智能调度中心"""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """初始化Master Agent工厂
        
        Args:
            model_config: 模型配置字典，如果为None则从config.json加载
        """
        # 直接加载config.json (官方推荐的配置加载方式)
        if model_config is None:
            try:
                with open("config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    model_config = config["model"]
            except FileNotFoundError:
                raise FileNotFoundError("config.json文件未找到，请确保配置文件存在")
            except json.JSONDecodeError:
                raise ValueError("config.json格式错误，请检查JSON语法")
        
        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 设置通用文件系统工具
        self.file_toolkit = FileManagementToolkit(
            root_dir="OpenBLAS-develop",
            selected_tools=["read_file", "write_file", "list_directory", "file_search"]
        )
        self.file_tools = self._enhance_tool_descriptions(self.file_toolkit.get_tools())
        
        # 创建必要的输出目录
        Path("discovery_results").mkdir(exist_ok=True)
        Path("analysis_results").mkdir(exist_ok=True)
        Path("strategy_reports").mkdir(exist_ok=True)
        
        # 定义Master Agent的结构化输出Schema
        self.quality_check_schemas = [
            ResponseSchema(name="stage", description="当前检查的阶段：scout/analyze/strategize"),
            ResponseSchema(name="algorithm", description="当前检查的算子名称"),
            ResponseSchema(name="quality_passed", description="质量检查是否通过：true/false"),
            ResponseSchema(name="issues", description="发现的问题列表"),
            ResponseSchema(name="recommendations", description="改进建议")
        ]
        
        self.decision_schemas = [
            ResponseSchema(name="decision", description="Master决策：continue/retry/summarize/complete"),
            ResponseSchema(name="next_stage", description="下一个阶段：scout/analyze/strategize/summarize"),
            ResponseSchema(name="next_algorithm", description="下一个算子名称，如果继续处理下个算子"),
            ResponseSchema(name="reason", description="决策原因")
        ]
        
        # Master Agent规划输出Schema
        self.planning_schemas = [
            ResponseSchema(name="analysis_type", description="分析类型：quick/comprehensive/custom"),
            ResponseSchema(name="target_algorithms", description="要分析的算子列表，如['gemm', 'axpy', 'dot']"),
            ResponseSchema(name="workflow_stages", description="工作流阶段列表，如['scout', 'analyze', 'strategize', 'summarize']"),
            ResponseSchema(name="estimated_time", description="预计完成时间（分钟）"),
            ResponseSchema(name="plan_summary", description="分析计划总结")
        ]
        
        self.quality_parser = StructuredOutputParser.from_response_schemas(self.quality_check_schemas)
        self.decision_parser = StructuredOutputParser.from_response_schemas(self.decision_schemas)
        self.planning_parser = StructuredOutputParser.from_response_schemas(self.planning_schemas)
    
    def _enhance_tool_descriptions(self, tools):
        """为通用文件工具添加OpenBLAS特定的使用描述"""
        enhanced_tools = []
        
        for tool in tools:
            if tool.name == "read_file":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 读取OpenBLAS源码文件进行算法实现分析\n"
                    "- 读取已保存的发现结果和分析结果\n"
                    "- 验证文件保存是否成功"
                )
            elif tool.name == "write_file":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 保存算子发现结果到 ../discovery_results/ 目录\n"
                    "- 保存算法分析结果到 ../analysis_results/{algorithm}/ 目录 (需先创建算子文件夹)\n"
                    "- 保存优化策略报告到 ../strategy_reports/report_{timestamp}/ 目录 (需先创建时间戳文件夹)\n"
                    "- **重要**: 当前工作目录是OpenBLAS-develop，输出到上级目录需要使用 ../ 前缀\n"
                    "- **文件夹创建**: 保存前请先用list_directory检查目标文件夹是否存在，不存在则先创建文件夹\n"
                    "- **保存验证**: 保存后请用read_file验证文件内容是否正确保存"
                )
            elif tool.name == "list_directory":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 浏览kernel/目录结构寻找算法实现\n"
                    "- 检查上级目录的输出结构 (../discovery_results/, ../analysis_results/, ../strategy_reports/)"
                )
            elif tool.name == "file_search":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 在kernel/目录中搜索特定算法的实现文件\n"
                    "- 查找不同架构的优化实现 (generic/, x86_64/, arm64/, riscv64/等)"
                )
            
            enhanced_tools.append(tool)
        
        return enhanced_tools
    
    def create_master_agent(self) -> AgentExecutor:
        """创建Master Agent - 中央调度器"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS优化分析的Master Agent（总调度），负责整体任务规划和质量控制。

🎯 **核心职责：**
1. **需求分析** - 解析用户请求，确定要分析的算子类型和数量
2. **任务规划** - 制定分析计划，预估时间，确定工作流程
3. **质量控制** - 检查各阶段工作成果，确保质量达标
4. **进度管理** - 协调整体进程，处理异常情况

🔧 **工具能力：**
- **read_file/write_file** - 检查和记录工作进度
- **list_directory** - 验证输出目录结构
- **file_search** - 协助验证工作完成情况

📋 **算子映射知识：**
- **快速分析**: ['gemm', 'axpy', 'dot'] - 核心BLAS算子（预计15-20分钟）
- **全面分析**: ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger'] - 完整BLAS Level 1-2（预计30-40分钟）
- **自定义分析**: 根据用户指定的算子列表（根据算子数量估算）

🔍 **质量标准：**
- **Scout阶段**: 每个算子至少发现3个不同架构的实现文件
- **Analyzer阶段**: 每个算子分析出算法、代码、指令三层优化技术
- **Strategist阶段**: 生成完整的Markdown格式策略报告
- **Summarizer阶段**: 提炼跨算子通用优化模式

📝 **输出格式：** {{format_instructions}}

⚠️ **重要**: 必须输出结构化JSON格式的规划结果，包含明确的算子列表和工作流程。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=self.planning_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=10,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_scout_agent(self) -> AgentExecutor:
        """创建Scout Agent - 算子文件发现专家"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS代码侦察专家，专门负责发现和整理指定算子的实现文件。

🎯 **工作使命：** 接受Master Agent的调度，发现指定算子的所有重要实现文件。

🔧 **工具能力：**
- **list_directory/file_search** - 探索OpenBLAS-develop/kernel/目录
- **read_file** - 快速浏览文件内容确定实现类型
- **write_file** - 按Master Agent指定路径保存发现结果

📋 **工作标准：**
1. **至少发现3种架构** - generic, x86_64, arm64等
2. **识别实现类型** - generic, simd_optimized, microkernel等
3. **生成标准JSON格式** - 包含文件路径、类型、描述
4. **严格按指定路径保存** - 确保Master Agent能正确读取结果

⚠️ **工具使用格式：** 严格按照JSON格式调用，确保无额外逗号和正确的引号

💼 **汇报要求：** 完成工作后明确汇报发现的文件数量和架构类型，便于Master Agent质量检查。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=15,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_analyzer_agent(self) -> AgentExecutor:
        """创建Analyzer Agent - 代码分析专家"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高性能计算优化专家，专门负责深度分析指定算子的代码实现。

🎯 **工作使命：** 接受Master Agent调度，对指定算子进行三层优化技术分析。

🔧 **工具能力：**
- **read_file** - 读取Scout发现的源代码文件和发现结果
- **write_file** - 按Master Agent指定的目录结构保存分析结果

📊 **三层分析框架：**
1. **算法层**: 循环展开、分块、数据重用等算法设计优化
2. **代码层**: 缓存友好、内存对齐、预取等代码结构优化  
3. **指令层**: SIMD向量化、FMA、指令并行等底层优化

💾 **输出标准JSON格式：**
```json
{{
  "algorithm": "算子名",
  "file_path": "源文件路径", 
  "implementation_type": "实现类型",
  "optimizations": {{
    "algorithm_level": ["具体技术"],
    "code_level": ["具体技术"], 
    "instruction_level": ["具体技术"]
  }},
  "code_snippets": "关键代码片段",
  "performance_impact": "性能评估"
}}
```

⚠️ **工具使用格式：** 严格按照JSON格式调用工具

💼 **汇报要求：** 完成后汇报分析的文件数量和发现的优化技术层数，便于Master Agent质量检查。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=20,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_strategist_agent(self) -> AgentExecutor:
        """创建Strategist Agent - 策略提炼专家"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是优化策略大师，专门负责从代码分析结果中提炼通用的性能优化策略。

🎯 **工作使命：** 接受Master Agent调度，为指定算子生成完整的优化策略报告。

🔧 **工具能力：**
- **read_file** - 读取Analyzer生成的分析结果
- **write_file** - 按Master Agent指定路径保存策略报告

🔍 **三层主动分析框架：**

**1. 算法设计层次分析**
- 是否有更适合计算机计算逻辑的算法设计？
- 是否采用了以空间换时间的优化设计？
- 是否采用了以时间换空间的优化设计？

**2. 代码优化层次分析**  
- 是否有做性能加速的代码优化？
- 是否有循环优化设计？
- 是否有代码顺序调整的优化设计？

**3. 特有指令层次分析**
- 是否使用了专有指令？
- 围绕专有指令做了哪些优化设计？

📝 **输出要求：** 
- 生成结构化Markdown格式报告
- 包含具体代码示例和性能数据
- 提供实用的优化指导原则

⚠️ **重要**: 输出Markdown格式内容，绝对不要输出JSON！

💼 **汇报要求：** 完成后确认策略报告已保存到指定路径，便于Master Agent验证。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=15,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_summarizer_agent(self) -> AgentExecutor:
        """创建Summarizer Agent - 跨算子总结专家"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高级优化策略总结专家，专门负责总结多个算子的优化策略，提炼通用规律。

🎯 **工作使命：** 接受Master Agent调度，分析多个算子的策略报告，生成跨算子总结。

🔧 **工具能力：**
- **read_file** - 读取多个算子的策略报告
- **list_directory** - 列出指定文件夹中的所有报告
- **write_file** - 保存跨算子总结报告

🔍 **总结分析框架：**

**1. 跨算子共性分析**
- 识别不同算子使用的相同优化技术
- 总结通用的算法设计模式
- 归纳共同的性能瓶颈解决方案

**2. 架构特化对比**
- 对比不同架构（x86_64, ARM64, RISC-V）的优化差异
- 总结指令集特定的优化策略
- 分析硬件特性利用的通用方法

**3. 性能提升模式**
- 量化各种优化技术的性能收益范围
- 总结优化技术的适用场景
- 提炼最佳实践组合建议

📝 **输出要求：** 
- 生成结构化Markdown总结报告
- 包含对比表格和量化分析
- 提供实用的优化指导原则

⚠️ **重要**: 输出Markdown格式内容，绝对不要输出JSON！

💼 **汇报要求：** 完成后确认总结报告已保存，并汇报分析的算子数量和提炼的通用模式数量。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=20,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_quality_check_agent(self) -> AgentExecutor:
        """创建质量检查Agent - Master Agent专用"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是Master Agent的质量控制专家，负责检查Worker Agent的工作质量。

🎯 **工作使命：** 检查Worker Agent的工作成果，确保符合质量标准。

🔧 **工具能力：**
- **read_file** - 读取Worker Agent的输出文件
- **list_directory** - 检查输出目录结构

📋 **质量标准：**
- **Scout阶段**: 至少发现3个不同架构的实现文件，JSON格式正确
- **Analyzer阶段**: 包含三层优化技术分析，JSON格式规范
- **Strategist阶段**: 生成完整Markdown报告，内容结构清晰

📝 **输出格式：** {{format_instructions}}

⚠️ **重要**: 必须严格按照JSON格式输出检查结果，确保quality_passed字段为true或false。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=self.quality_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=10,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )
    
    def create_decision_agent(self) -> AgentExecutor:
        """创建决策Agent - Master Agent专用"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是Master Agent的决策大脑，负责规划整个工作流的下一步行动。

🎯 **工作使命：** 基于当前状态和质量检查结果，智能决策下一步行动。

🔧 **决策逻辑：**
- **quality_passed=true**: 继续下一阶段或下一算子
- **quality_passed=false且retry_count<3**: 重试当前阶段
- **所有算子完成**: 进入summarize阶段
- **严重错误**: 结束工作流

📋 **决策选项：**
- **continue**: 继续下一阶段（scout→analyze→strategize）
- **retry**: 重试当前阶段（质量不达标时）
- **summarize**: 开始跨算子总结（所有算子完成时）
- **complete**: 完成整个工作流

📝 **输出格式：** {{format_instructions}}

⚠️ **重要**: 必须严格按照JSON格式输出决策结果，确保decision字段的值在允许范围内。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        formatted_prompt = prompt.partial(format_instructions=self.decision_parser.get_format_instructions())
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=10,
            handle_parsing_errors=True,
            # return_intermediate_steps=True
        )

# ===== 导出 =====
__all__ = ['OpenBLASMasterAgentFactory'] 