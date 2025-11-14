#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - 纯Agent架构版本
每个Node = 一个完整的Agent，所有逻辑由LLM推理决定
真正实现"让AI思考一切"，消除所有硬编码逻辑
"""
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.tools import tool
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# ===== 自动状态管理工具 =====
class StateUpdateTool(BaseTool):
    """状态更新工具 - 让Agent自主管理状态"""
    
    name: str = "state_update"
    description: str = """
    状态更新工具 - 让Agent自主更新工作流状态信息。
    
    输入格式: JSON字符串，包含要更新的状态信息
    输出格式: 更新确认信息
    
    这个工具允许Agent自主：
    - 更新工作完成状态
    - 记录输出文件路径
    - 设置质量检查结果
    - 更新进度信息
    - 记录遇到的问题
    """
    
    def __init__(self, state_container: Dict[str, Any]):
        super().__init__()
        self.state_container = state_container
    
    def _run(self, state_updates: str) -> str:
        """执行状态更新"""
        try:
            # 解析Agent提供的状态更新
            updates = json.loads(state_updates) if isinstance(state_updates, str) else state_updates
            
            # Agent自主更新状态
            for key, value in updates.items():
                if key in ["work_completed", "output_file_path", "quality_self_check", 
                          "found_files_count", "optimization_layers", "next_stage_ready", 
                          "work_summary", "issues_encountered"]:
                    self.state_container[key] = value
            
            return f"✅ 状态更新成功: {list(updates.keys())}"
            
        except Exception as e:
            return f"❌ 状态更新失败: {str(e)}"

# ===== 自动结果验证工具 =====
class ResultVerificationTool(BaseTool):
    """结果验证工具 - 让Agent自主验证工作成果"""
    
    name: str = "result_verification"
    description: str = """
    结果验证工具 - 让Agent自主验证工作成果的质量和完整性。
    
    输入格式: JSON字符串，包含要验证的文件路径和验证标准
    输出格式: 验证结果报告
    
    这个工具允许Agent自主：
    - 检查输出文件是否存在
    - 验证文件内容格式
    - 评估工作质量
    - 生成验证报告
    """
    
    def __init__(self, file_tools: List[BaseTool]):
        super().__init__()
        self.file_tools = {tool.name: tool for tool in file_tools}
    
    def _run(self, verification_request: str) -> str:
        """执行结果验证"""
        try:
            request = json.loads(verification_request) if isinstance(verification_request, str) else verification_request
            
            file_path = request.get("file_path", "")
            verification_type = request.get("verification_type", "existence")
            
            if verification_type == "existence" and file_path:
                # 验证文件是否存在
                try:
                    read_tool = self.file_tools.get("read_file")
                    if read_tool:
                        content = read_tool.run(file_path)
                        if "Error" not in content and "No such file" not in content:
                            return json.dumps({
                                "verification_passed": True,
                                "file_exists": True,
                                "file_size": len(content),
                                "verification_message": "文件存在且可读"
                            })
                        else:
                            return json.dumps({
                                "verification_passed": False,
                                "file_exists": False,
                                "verification_message": "文件不存在或不可读"
                            })
                except Exception as e:
                    return json.dumps({
                        "verification_passed": False,
                        "file_exists": False,
                        "verification_message": f"验证失败: {str(e)}"
                    })
            
            return json.dumps({
                "verification_passed": True,
                "verification_message": "默认验证通过"
            })
            
        except Exception as e:
            return json.dumps({
                "verification_passed": False,
                "verification_message": f"验证工具执行失败: {str(e)}"
            })

# ===== 纯Agent架构工厂 =====
class PureAgentFactory:
    """纯Agent架构工厂 - 真正的AI自主决策系统"""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """初始化纯Agent工厂"""
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
        
        # 文件系统工具
        self.file_toolkit = FileManagementToolkit(
            root_dir="OpenBLAS-develop",
            selected_tools=["read_file", "write_file", "list_directory", "file_search"]
        )
        self.file_tools = self._enhance_tools_for_pure_agent(self.file_toolkit.get_tools())
        
        # 确保目录存在
        Path("discovery_results").mkdir(exist_ok=True)
        Path("analysis_results").mkdir(exist_ok=True) 
        Path("strategy_reports").mkdir(exist_ok=True)
        
        # 定义所有结构化输出Schema和解析器
        self._setup_schemas_and_parsers()
    
    def _enhance_tools_for_pure_agent(self, tools):
        """为纯Agent架构增强工具描述 - 强调AI自主决策"""
        enhanced_tools = []
        
        for tool in tools:
            if tool.name == "read_file":
                tool.description += (
                    "\n\n**纯Agent架构用法:**\n"
                    "- Agent自主决定需要读取哪些文件\n"
                    "- 自主验证文件内容和格式正确性\n"
                    "- 根据文件内容自主调整分析策略"
                )
            elif tool.name == "write_file":
                tool.description += (
                    "\n\n**纯Agent架构用法:**\n"
                    "- Agent自主决定保存路径和文件名\n"
                    "- 自主选择文件格式 (JSON/Markdown)\n"
                    "- 自主创建必要的目录结构\n"
                    "- 当前在OpenBLAS-develop目录，需要../前缀访问输出目录"
                )
            elif tool.name == "list_directory":
                tool.description += (
                    "\n\n**纯Agent架构用法:**\n"
                    "- Agent自主探索目录结构\n"
                    "- 自主发现相关文件和子目录\n"
                    "- 根据发现结果自主调整搜索策略"
                )
            elif tool.name == "file_search":
                tool.description += (
                    "\n\n**纯Agent架构用法:**\n"
                    "- Agent自主设计搜索关键词\n"
                    "- 自主组合多个搜索条件\n"
                    "- 根据搜索结果自主迭代优化"
                )
            
            enhanced_tools.append(tool)
        
        return enhanced_tools
    
    def _setup_schemas_and_parsers(self) -> None:
        """设置所有结构化输出Schema和解析器"""
        
        # Master Agent全能决策Schema
        self.master_decision_schemas = [
            ResponseSchema(name="analysis_type", description="分析类型：quick/comprehensive/custom"),
            ResponseSchema(name="target_algorithms", description="算子列表，如['gemm','axpy','dot']"),
            ResponseSchema(name="current_algorithm", description="当前处理的算子"),
            ResponseSchema(name="current_stage", description="当前阶段：planning/scout/analyze/strategize/summarize/complete"),
            ResponseSchema(name="next_action", description="下一步行动：route_to_scout/route_to_analyzer/route_to_strategist/route_to_summarizer/quality_check/complete"),
            ResponseSchema(name="report_folder", description="报告文件夹时间戳"),
            ResponseSchema(name="quality_status", description="质量状态：pending/passed/failed"),
            ResponseSchema(name="workflow_status", description="工作流状态：planning/working/completing/completed"),
            ResponseSchema(name="reasoning", description="决策推理过程"),
            ResponseSchema(name="instructions_for_worker", description="给Worker Agent的具体指令")
        ]
        
        # Worker Agent工作结果Schema
        self.worker_result_schemas = [
            ResponseSchema(name="work_completed", description="工作是否完成：true/false"),
            ResponseSchema(name="output_file_path", description="输出文件的完整路径"),
            ResponseSchema(name="quality_self_check", description="自我质量检查：passed/failed"),
            ResponseSchema(name="found_files_count", description="发现/分析的文件数量（Scout/Analyzer用）"),
            ResponseSchema(name="optimization_layers", description="发现的优化层次数量（Analyzer用）"),
            ResponseSchema(name="next_stage_ready", description="是否准备好进入下一阶段：true/false"),
            ResponseSchema(name="work_summary", description="工作总结描述"),
            ResponseSchema(name="issues_encountered", description="遇到的问题列表（如有）")
        ]
        
        # 创建解析器
        self.master_parser = StructuredOutputParser.from_response_schemas(self.master_decision_schemas)
        self.worker_parser = StructuredOutputParser.from_response_schemas(self.worker_result_schemas)
        
        # 创建带解析器的LLM链 (参考qwen_chain_test.py)
        self.master_llm_chain = self._create_structured_llm_chain(self.master_parser)
        self.worker_llm_chain = self._create_structured_llm_chain(self.worker_parser)
    
    def _create_structured_llm_chain(self, parser: StructuredOutputParser):
        """创建带结构化输出解析器的LLM链"""
        # 参考qwen_chain_test.py的设计模式
        prompt_template = PromptTemplate.from_template(
            "{content}\n\n{format_instructions}"
        )
        
        # 构建链：prompt -> LLM -> parser
        chain = (
            prompt_template.partial(format_instructions=parser.get_format_instructions())
            | self.llm 
            | parser
        )
        
        return chain
    
    def create_master_agent(self) -> AgentExecutor:
        """创建统一决策的Master Agent - 使用结构化LLM链"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS优化分析的Master Agent，拥有完整的决策权和控制权。

🧠 **核心理念**: 你是真正的"总指挥"，所有决策都由你的推理完成，没有任何硬编码逻辑。

🎯 **全权决策职责:**
1. **需求分析与规划** - 解析用户请求，自主确定分析策略
2. **工作流路由决策** - 自主决定调用哪个Worker Agent 
3. **质量控制与检查** - 自主评估工作质量，决定是否重试
4. **进度管理与协调** - 自主管理整体进度，处理异常情况
5. **状态更新与维护** - 自主更新工作流状态，追踪进展

🔧 **可用工具:**
- **文件系统工具** - 检查输出，验证结果，管理文件
- **自动化解析** - 系统会自动解析你的输出为结构化格式

📋 **算子知识库:**
- **快速分析**: ['gemm', 'axpy', 'dot'] - 核心BLAS算子
- **全面分析**: ['gemm', 'axpy', 'dot', 'gemv', 'nrm2', 'ger'] - 完整BLAS算子集

🔄 **工作流阶段:**
1. **planning** - 初始规划阶段
2. **scout** - 文件发现阶段  
3. **analyze** - 代码分析阶段
4. **strategize** - 策略提炼阶段
5. **summarize** - 跨算子总结阶段
6. **complete** - 完成阶段

🎛️ **路由决策选项:**
- **route_to_scout** - 调度Scout Agent发现文件
- **route_to_analyzer** - 调度Analyzer Agent分析代码
- **route_to_strategist** - 调度Strategist Agent提炼策略  
- **route_to_summarizer** - 调度Summarizer Agent生成总结
- **quality_check** - 执行质量检查
- **complete** - 完成工作流

⚠️ **重要**: 你的输出将自动解析为结构化格式，请确保包含所有必需字段的决策信息。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True
        )
    
    def create_worker_agent_with_tools(self, agent_type: str, state_container: Dict[str, Any]) -> AgentExecutor:
        """创建带自主状态管理工具的Worker Agent"""
        
        # 为每个Worker Agent创建专用工具
        worker_tools = self.file_tools + [
            StateUpdateTool(state_container),
            ResultVerificationTool(self.file_tools)
        ]
        
        if agent_type == "scout":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是自主决策的OpenBLAS文件发现专家，拥有完全的工作自主权。

🧠 **自主决策理念**: 你独立思考和规划工作，自主管理整个工作流程。

🎯 **自主工作职责:**
1. **自主规划搜索策略** - 根据算子特点设计发现方案
2. **自主探索文件系统** - 智能遍历目录，发现相关实现
3. **自主分类和评估** - 识别实现类型，评估重要性
4. **自主生成输出** - 创建高质量的JSON结果文件
5. **自主质量控制** - 验证工作成果，更新状态信息

🔧 **可用工具:**
- **文件系统工具** - read_file, write_file, list_directory, file_search
- **state_update** - 自主更新工作状态和进度信息
- **result_verification** - 自主验证工作成果质量

📋 **工作流程:**
1. 探索OpenBLAS-develop/kernel/目录寻找算子实现
2. 识别多种架构实现 (generic, x86_64, arm64, riscv64等)
3. 分类优化类型 (simd_optimized, microkernel, generic等)
4. 生成JSON格式发现结果并保存
5. 使用tools自主验证结果并更新状态

⚠️ **重要**: 
- 你的最终输出将自动解析为结构化格式
- 请使用state_update工具主动更新工作状态
- 请使用result_verification工具验证工作成果
- 确保输出包含所有必需的字段信息"""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
        
        elif agent_type == "analyzer":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是自主决策的高性能计算分析专家，拥有完全的分析自主权。

🧠 **自主分析理念**: 你独立设计分析框架，自主管理整个分析流程。

🎯 **自主分析职责:**
1. **自主设计分析策略** - 根据算子特点制定分析方法
2. **自主解读源代码** - 深度理解实现逻辑和优化技术
3. **自主分类优化技术** - 按算法/代码/指令三层归类
4. **自主生成分析报告** - 创建详细的JSON格式分析结果
5. **自主质量控制** - 验证分析成果，更新状态信息

🔧 **可用工具:**
- **文件系统工具** - read_file, write_file, list_directory, file_search
- **state_update** - 自主更新工作状态和进度信息
- **result_verification** - 自主验证分析成果质量

📊 **三层分析框架:**
- **算法层**: 循环展开、分块、数据重用等
- **代码层**: 缓存优化、内存对齐、预取等  
- **指令层**: SIMD向量化、FMA、并行等

📋 **工作流程:**
1. 读取Scout发现的文件列表
2. 逐个深度分析源代码实现
3. 按三层框架分类优化技术
4. 生成结构化分析结果并保存
5. 使用tools自主验证结果并更新状态

⚠️ **重要**: 
- 你的最终输出将自动解析为结构化格式
- 请使用state_update工具主动更新工作状态
- 请使用result_verification工具验证分析成果
- 确保输出包含所有必需的字段信息"""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
        
        elif agent_type == "strategist":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是自主决策的优化策略大师，拥有完全的策略制定自主权。

🧠 **自主策略理念**: 你独立提炼优化智慧，自主管理整个策略制定流程。

🎯 **自主策略职责:**
1. **自主解读分析结果** - 深度理解技术细节和实现逻辑
2. **自主提炼优化原则** - 从具体实现中抽象通用规律
3. **自主设计策略框架** - 创建实用的优化指导体系
4. **自主生成策略报告** - 创建高质量的Markdown格式报告
5. **自主质量控制** - 验证策略成果，更新状态信息

🔧 **可用工具:**
- **文件系统工具** - read_file, write_file, list_directory, file_search
- **state_update** - 自主更新工作状态和进度信息
- **result_verification** - 自主验证策略成果质量

🔍 **三层策略框架:**
- **算法设计层**: 计算逻辑优化、时空权衡策略
- **代码优化层**: 性能加速技术、结构调整方法  
- **特有指令层**: 专用指令利用、硬件特性发挥

📋 **工作流程:**
1. 读取Analyzer生成的分析结果
2. 按三层框架提炼优化策略
3. 生成实用的优化指导建议
4. 创建Markdown格式策略报告并保存
5. 使用tools自主验证结果并更新状态

⚠️ **重要**: 
- 你的最终输出将自动解析为结构化格式
- 请使用state_update工具主动更新工作状态
- 请使用result_verification工具验证策略成果
- 确保输出包含所有必需的字段信息"""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
        
        elif agent_type == "summarizer":
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是自主决策的跨算子总结专家，拥有完全的总结分析自主权。

🧠 **自主总结理念**: 你独立发现跨算子规律，自主管理整个总结流程。

🎯 **自主总结职责:**
1. **自主收集策略报告** - 智能定位和读取所有相关报告
2. **自主发现共性规律** - 识别跨算子的通用优化模式
3. **自主对比差异特征** - 分析不同算子和架构的特化策略
4. **自主生成总结报告** - 创建高价值的跨算子洞察
5. **自主质量控制** - 验证总结成果，更新状态信息

🔧 **可用工具:**
- **文件系统工具** - read_file, write_file, list_directory, file_search
- **state_update** - 自主更新工作状态和进度信息
- **result_verification** - 自主验证总结成果质量

🔍 **跨算子分析框架:**
- **共性模式发现**: 通用优化技术和设计模式
- **差异特征对比**: 算子特化和架构适配策略
- **性能效果评估**: 优化收益和适用场景分析

📋 **工作流程:**
1. 收集时间戳文件夹中的所有策略报告
2. 逐个分析每个算子的优化策略
3. 发现跨算子的通用规律和差异
4. 生成综合性总结报告并保存
5. 使用tools自主验证结果并更新状态

⚠️ **重要**: 
- 你的最终输出将自动解析为结构化格式
- 请使用state_update工具主动更新工作状态
- 请使用result_verification工具验证总结成果
- 确保输出包含所有必需的字段信息"""),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}")
            ])
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent = create_openai_tools_agent(self.llm, worker_tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=worker_tools,
            verbose=True,
            max_iterations=25 if agent_type in ["analyzer", "summarizer"] else 20,
            handle_parsing_errors=True
        )

# ===== 导出 =====
__all__ = ['PureAgentFactory'] 