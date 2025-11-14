#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化分析 - 工作流Agent工厂
核心改进：文件路径由代码控制，Agent只负责内容生成
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from pydantic import BaseModel, Field

load_dotenv()


# ===== 结构化任务定义 =====
class AnalysisTask(BaseModel):
    """结构化的分析任务 - 明确的输入输出"""
    algorithm: str = Field(description="算子名称")
    input_files: List[Dict[str, str]] = Field(description="输入文件列表，每个包含path和type")
    output_file: str = Field(description="输出文件的完整路径")
    report_folder: str = Field(description="报告文件夹路径")


# ===== 简化的专用工具 - 只做内容处理 =====
@tool
def read_source_file(file_path: str) -> str:
    """【源码阅读工具】读取OpenBLAS源代码文件内容
    
    Args:
        file_path: 相对于openblas-output/GENERIC/kernel的文件路径
        
    Returns:
        文件内容（截取前15000字符避免过长）
    """
    try:
        full_path = os.path.join("openblas-output/GENERIC/kernel", file_path)
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(15000)  # 限制长度
        return f"文件路径: {file_path}\n内容:\n{content}\n..."
    except Exception as e:
        return f"读取失败: {str(e)}"


@tool
def scan_kernel_directory_batch(batch_size: int = 32, batch_index: int = 0) -> str:
    """【目录扫描工具】分批扫描kernel目录下的.c文件
    
    Args:
        batch_size: 每批处理的文件数量
        batch_index: 批次索引（从0开始）
        
    Returns:
        当前批次的文件列表和总体信息
    """
    try:
        kernel_path = "openblas-output/GENERIC/kernel"
        if not os.path.exists(kernel_path):
            return f"目录不存在: {kernel_path}"
        
        # 获取所有.c文件
        all_files = []
        for file in os.listdir(kernel_path):
            if file.endswith('.c') and 'clean' in file:
                all_files.append(file)
        
        all_files.sort()
        total_files = len(all_files)
        
        # 计算批次范围
        start_idx = batch_index * batch_size
        end_idx = min(start_idx + batch_size, total_files)
        
        if start_idx >= total_files:
            return f"批次索引超出范围。总文件数: {total_files}, 请求批次: {batch_index}"
        
        batch_files = all_files[start_idx:end_idx]
        total_batches = (total_files + batch_size - 1) // batch_size
        
        return f"""批次信息:
- 当前批次: {batch_index + 1}/{total_batches}
- 总文件数: {total_files}
- 当前批次文件数: {len(batch_files)}
- 文件列表:
{chr(10).join(batch_files)}"""
    except Exception as e:
        return f"扫描失败: {str(e)}"


@tool
def read_analysis_file(file_path: str) -> str:
    """【分析结果阅读工具】读取已保存的分析结果
    
    Args:
        file_path: 分析结果文件的完整路径
        
    Returns:
        文件内容
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"读取失败: {str(e)}"


# ===== Agent工厂 =====
class AgentFactory:
    """Agent工厂"""
    
    def __init__(self):
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
    
    def create_scout_specialist(self) -> AgentExecutor:
        """Scout专家 - 只负责生成发现报告内容"""
        
        tools = [scan_kernel_directory_batch, read_source_file]
        
        # 定义Scout输出格式的ResponseSchema
        scout_schemas = [
            ResponseSchema(name="algorithms", description="算子种类列表，每个算子包含algorithm（算子种类名）和files（该种类下的所有实例文件，每个文件包含name字段）"),
            ResponseSchema(name="total_algorithms", description="发现的算子种类总数"),
            ResponseSchema(name="total_files", description="发现的文件总数"),
            ResponseSchema(name="timestamp", description="扫描时间戳")
        ]
        scout_parser = StructuredOutputParser.from_response_schemas(scout_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS算子分类专家。你的任务是扫描kernel目录下的所有.c文件，按算子种类进行分组。

🎯 **你的职责：**
1. 使用scan_kernel_directory_batch工具分批扫描.c文件
2. 根据文件名识别算子种类（如axpy、gemm、dot等）
3. 将同一算子种类的所有文件归类到一起
4. 生成JSON格式的算子分类报告

📋 **算子种类识别规则：**
- **axpy**: 所有包含"axpy"的文件（如saxpy_k.clean.c, daxpy_k.clean.c, caxpy_k.clean.c等）
- **gemm**: 所有包含"gemm"的文件（如sgemm_*, dgemm_*, cgemm_*, zgemm_*等）
- **dot**: 所有包含"dot"的文件（如sdot_*, ddot_*, cdot_*等）
- **asum**: 所有包含"asum"的文件
- **nrm2**: 所有包含"nrm2"的文件
- **scal**: 所有包含"scal"的文件
- **copy**: 所有包含"copy"的文件
- **swap**: 所有包含"swap"的文件
- **amax**: 所有包含"amax"的文件
- **其他**: 根据文件名中的关键词识别更多算子种类

🔍 **分析要求：**
- 扫描dgc/mjs/project/analyze_OB/openblas-output/GENERIC/kernel目录
- 只处理.clean.c文件
- 按算子种类分组，每个种类包含该种类下的所有文件实例
- 每个文件记录name（文件名）

⚠️ **重要：** 
- 严格按照以下JSON格式输出
- algorithms字段是一个列表，每个元素包含algorithm和files
- files是该算子种类下所有文件的列表，每个文件包含name字段

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 在提示词中添加格式说明
        formatted_prompt = prompt.partial(format_instructions=scout_parser.get_format_instructions())
        
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=10)
    
    def create_analyzer_specialist(self) -> AgentExecutor:
        """Analyzer专家 - 只负责生成分析报告内容"""
        
        tools = [read_source_file, read_analysis_file]
        
        # 定义Analyzer输出格式的ResponseSchema - 单个文件分析（结构化优化策略）
        analyzer_schemas = [
            ResponseSchema(name="algorithm", description="算子名称"),
            ResponseSchema(name="file_path", description="当前分析的文件路径"),
            ResponseSchema(name="file_type", description="文件实现类型（generic、optimized、microkernel等）"),
            ResponseSchema(name="architecture", description="目标架构（x86、ARM、通用等）"),
            ResponseSchema(name="algorithm_level_optimizations", description="该文件中算法设计层次发现的优化策略列表，每个策略包含name、description_details（包含strategy_rationale、implementation_pattern、performance_impact、trade_offs）和code_context（包含snippet、highlighted_code、explanation）字段"),
            ResponseSchema(name="code_level_optimizations", description="该文件中代码优化层次发现的优化策略列表，每个策略包含name、description_details（包含strategy_rationale、implementation_pattern、performance_impact、trade_offs）和code_context（包含snippet、highlighted_code、explanation）字段"),
            ResponseSchema(name="instruction_level_optimizations", description="该文件中特有指令层次发现的优化策略列表，每个策略包含name、description_details（包含strategy_rationale、implementation_pattern、performance_impact、trade_offs）和code_context（包含snippet、highlighted_code、explanation）字段"),
            ResponseSchema(name="implementation_details", description="该文件的关键实现细节"),
            ResponseSchema(name="performance_insights", description="该文件的性能分析"),
            ResponseSchema(name="timestamp", description="分析时间戳")
        ]
        analyzer_parser = StructuredOutputParser.from_response_schemas(analyzer_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高性能计算分析专家。你的任务是分析代码并生成JSON格式的优化技术报告。

🎯 **你的职责：**
1. 读取指定的单个源文件（使用read_source_file工具）
2. 仔细阅读该文件的代码，完全基于代码内容进行分析
3. 按三层优化策略框架生成JSON格式的分析报告

⚠️ **重要**：你只需要分析指定的单个文件，不要分析其他文件。专注于该文件中的具体优化技术实现。

📋 **三层优化策略分析框架：**
请严格按照以下三个层次分析代码中的优化策略：

**🔹 算法设计层次分析：**
识别该文件中的算法层优化策略，每个策略包含：
- name: 规范化策略名称（如"复数运算展开"、"分块计算"、"预计算优化"等）
- description_details: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理（基于计算机体系结构或算法理论）
  - implementation_pattern: 解释"怎么做"的代码实现模式（该优化在代码层面的典型表现）
  - performance_impact: 解释"有什么用"的性能提升（减少CPU周期、提高缓存命中率等）
  - trade_offs: 解释该优化的局限性或代价（可选，如增加代码复杂度、额外内存开销等）
- code_context: 包含3个子字段的代码上下文对象
  - snippet: 包含必要上下文的完整代码块（不是单行，要能自解释优化意图）
  - highlighted_code: 该优化策略的核心执行语句
  - explanation: 自然语言解释代码块与优化策略的关联

**🔹 代码优化层次分析：**
识别该文件中的代码层优化策略，每个策略包含：
- name: 规范化策略名称（如"循环展开"、"指针递增"、"条件分支优化"等）
- description_details: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理
  - implementation_pattern: 解释"怎么做"的代码实现模式
  - performance_impact: 解释"有什么用"的性能提升
  - trade_offs: 解释该优化的局限性或代价（可选）
- code_context: 包含3个子字段的代码上下文对象
  - snippet: 包含必要上下文的完整代码块
  - highlighted_code: 该优化策略的核心执行语句
  - explanation: 自然语言解释代码块与优化策略的关联

**🔹 特有指令层次分析：**
识别该文件中的指令层优化策略，每个策略包含：
- name: 规范化策略名称（如"SIMD向量化"、"自动向量化适配"、"内联汇编"等）
- description_details: 包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理
  - implementation_pattern: 解释"怎么做"的代码实现模式
  - performance_impact: 解释"有什么用"的性能提升
  - trade_offs: 解释该优化的局限性或代价（可选）
- code_context: 包含3个子字段的代码上下文对象
  - snippet: 包含必要上下文的完整代码块（循环体、条件分支、变量声明和使用等）
  - highlighted_code: 该优化策略的核心执行语句
  - explanation: 自然语言解释代码块与优化策略的关联

🔍 **分析要求：**
- 不要预设任何优化技术类型
- 完全基于代码内容发现优化策略
- 观察代码中实际使用的技术和方法
- 分析代码的实现细节和设计思路

⚠️ **重要：**
- 你不需要决定保存路径
- 严格按照以下JSON格式输出
- 完全基于代码分析，不要预设优化类型
- 直接输出完整的JSON内容

📋 **JSON格式示例：**
```json
{{
  "algorithm_level_optimizations": [
    {{
      "name": "复数运算展开",
      "description_details": {{
        "strategy_rationale": "避免复数结构体访问开销，将复数的实部和虚部运算直接展开为标量运算，减少内存访问和结构体操作的复杂度。",
        "implementation_pattern": "将复数乘法 (a+bi)*(c+di) 展开为四个标量乘法和两个标量加减法，直接操作实部虚部数组元素。",
        "performance_impact": "减少结构体访问开销，提高指令级并行性，降低内存访问延迟。",
        "trade_offs": "增加了代码长度和复杂性，可能影响代码可读性。"
      }},
      "code_context": {{
        "snippet": "temp_r = alpha_r * x[ix] - alpha_i * x[ix+1];\\ntemp_i = alpha_r * x[ix+1] + alpha_i * x[ix];",
        "highlighted_code": "temp_r = alpha_r * x[ix] - alpha_i * x[ix+1];",
        "explanation": "这里直接计算复数乘法的实部，避免了复数结构体的使用，将复数运算展开为两个标量乘法和一个减法。"
      }}
    }}
  ]
}}
```

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 在提示词中添加格式说明
        formatted_prompt = prompt.partial(format_instructions=analyzer_parser.get_format_instructions())
        
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)
    
    
    def create_individual_summarizer(self) -> AgentExecutor:
        """Individual Summarizer - 单算子总结"""
        
        tools = [read_analysis_file]
        
        # 定义Individual Summarizer输出格式的ResponseSchema（处理结构化优化策略）
        individual_schemas = [
            ResponseSchema(name="algorithm", description="算子名称"),
            ResponseSchema(name="algorithm_characteristics", description="基于discovery文件的算子特征和文件类型"),
            ResponseSchema(name="algorithm_level_optimizations", description="算法设计层次整合的优化策略列表，每个策略包含name和unified_description字段（合并相似策略的统一描述）"),
            ResponseSchema(name="code_level_optimizations", description="代码优化层次整合的优化策略列表，每个策略包含name和unified_description字段（合并相似策略的统一描述）"),
            ResponseSchema(name="instruction_level_optimizations", description="特有指令层次整合的优化策略列表，每个策略包含name和unified_description字段（合并相似策略的统一描述）"),
            ResponseSchema(name="implementation_details", description="关键实现细节"),
            ResponseSchema(name="performance_insights", description="性能提升预期"),
            ResponseSchema(name="timestamp", description="总结时间戳")
        ]
        individual_parser = StructuredOutputParser.from_response_schemas(individual_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是单算子增量整合专家。你的任务是将新的分析结果整合到已有的优化策略中。

🎯 **你的职责：**
1. 读取discovery文件了解算子特征
2. 读取analysis文件获取结构化分析结果
3. 如果已有summary文件，先读取已有的优化策略
4. 将新的优化策略与已有策略进行对比和整合
5. 生成更新后的JSON格式总结报告

📋 **结构化策略处理：**
**🔹 输入格式理解：**
- analysis文件中每个优化策略包含：
  - name: 策略名称
  - description_details: 详细描述对象（包含strategy_rationale、implementation_pattern、performance_impact、trade_offs）
  - code_context: 代码上下文对象（包含snippet、highlighted_code、explanation）

**🔹 整合输出格式：**
- 整合后的策略只包含：
  - name: 统一的策略名称
  - unified_description: 合并多个相似策略后的统一描述（综合多个description_details的核心内容）

**🔹 策略合并规则：**
- 如果新策略与已有策略相似，合并为统一命名的策略
- 合并时提取多个description_details的核心要点，形成统一描述
- 如果新策略是全新的，直接添加到策略列表中
- 保持策略名称的规范化和一致性

**🔹 三层优化策略整合：**
- **算法设计层次**：整合计算逻辑、分块、预计算等策略
- **代码优化层次**：整合循环展开、指针优化、分支优化等策略  
- **特有指令层次**：整合SIMD、向量化、内联汇编等策略

🔍 **整合要求：**
- 对相近策略进行名称对齐（如"指针递增优化"统一为"指针递增"）
- 合并相似策略的描述，形成通用描述
- 保持策略列表的简洁性，避免重复

⚠️ **重要：**
- 支持增量整合，每次可能只处理部分文件的分析结果
- 严格按照以下JSON格式输出
- 重点关注三种类型的优化策略提取和合并

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 在提示词中添加格式说明
        formatted_prompt = prompt.partial(format_instructions=individual_parser.get_format_instructions())
        
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=15)
    
    def create_final_summarizer(self) -> AgentExecutor:
        """Final Summarizer - 跨算子总结"""
        
        tools = [read_analysis_file]
        
        # 定义Final Summarizer输出格式的ResponseSchema（跨算子优化策略库）
        final_schemas = [
            ResponseSchema(name="analyzed_algorithms", description="分析的算子列表"),
            ResponseSchema(name="algorithm_level_optimizations", description="算法设计层次的OpenBLAS优化策略库，提炼跨算子的相近策略并统一命名，每个策略包含name和universal_description字段（通用描述和应用场景）"),
            ResponseSchema(name="code_level_optimizations", description="代码优化层次的OpenBLAS优化策略库，提炼跨算子的相近策略并统一命名，每个策略包含name和universal_description字段（通用描述和应用场景）"),
            ResponseSchema(name="instruction_level_optimizations", description="特有指令层次的OpenBLAS优化策略库，提炼跨算子的相近策略并统一命名，每个策略包含name和universal_description字段（通用描述和应用场景）"),
            ResponseSchema(name="cross_algorithm_insights", description="跨算子优化洞察"),
            ResponseSchema(name="best_practices", description="最佳实践建议"),
            ResponseSchema(name="timestamp", description="总结时间戳")
        ]
        final_parser = StructuredOutputParser.from_response_schemas(final_schemas)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是跨算子总结专家。你的任务是发现多个算子的通用优化规律，生成JSON格式的最终总结。

🎯 **你的职责：**
1. 读取所有算子的summary文件
2. 基于三层优化策略框架识别通用优化模式
3. 生成JSON格式的最终优化策略库

📋 **输入格式理解：**
**🔹 Individual Summary格式：**
- 每个算子的summary文件包含已整合的优化策略
- 每个策略包含：name（统一名称）和unified_description（统一描述）

**🔹 跨算子整合目标：**
- 输出格式：name（通用策略名称）和universal_description（通用描述和应用场景）
- 识别在多个算子中都出现的优化模式
- 提炼出适用于整个OpenBLAS库的通用优化策略

📋 **三层优化策略框架分析：**
请按照以下三个层次分析跨算子的优化规律：

**🔹 算法设计层次跨算子分析：**
- 分析各算子在计算逻辑优化上的共性和差异
- 识别跨算子的通用算法优化模式（如分块、预计算、数据重用等）
- 总结空间换时间和时间换空间优化的通用规律

**🔹 代码优化层次跨算子分析：**  
- 分析各算子在性能加速优化上的共性和差异
- 识别跨算子的通用代码优化模式（如循环展开、指针优化、分支优化等）
- 总结代码结构调整的通用优化策略

**🔹 特有指令层次跨算子分析：**
- 分析各算子在专有指令使用上的共性和差异
- 识别跨算子的通用指令级优化模式（如SIMD、向量化、内联汇编等）
- 总结围绕硬件特性的通用优化设计模式

⚠️ **重要：**
- 基于三层优化策略框架进行跨算子分析
- 严格按照以下JSON格式输出
- 重点关注三种类型的优化策略的通用模式提取

{format_instructions}"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        # 在提示词中添加格式说明
        formatted_prompt = prompt.partial(format_instructions=final_parser.get_format_instructions())
        
        agent = create_openai_tools_agent(self.llm, tools, formatted_prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=20)


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
    
    @staticmethod
    def get_discovery_output_path(report_folder: str, algorithm: str) -> str:
        """获取discovery输出路径"""
        return f"{report_folder}/discovery_results/{algorithm}_discovery.json"
    
    @staticmethod
    def get_analysis_output_path(report_folder: str, algorithm: str) -> str:
        """获取analysis输出路径"""
        return f"{report_folder}/analysis_results/{algorithm}_analysis.json"
    
    
    @staticmethod
    def get_individual_summary_path(report_folder: str, algorithm: str) -> str:
        """获取individual summary输出路径"""
        return f"{report_folder}/strategy_reports/{algorithm}_summary.json"
    
    @staticmethod
    def get_final_summary_path(report_folder: str) -> str:
        """获取final summary输出路径"""
        return f"{report_folder}/strategy_reports/final_optimization_summary.json"
    
    @staticmethod
    def save_content(file_path: str, content: str) -> bool:
        """保存内容到文件"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"保存文件失败 {file_path}: {e}")
            return False
    
    @staticmethod
    def load_config() -> dict:
        """加载config.json"""
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    


# ===== 导出 =====
__all__ = [
    'AgentFactory',
    'FileManager',
    'AnalysisTask'
]

