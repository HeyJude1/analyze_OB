#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - 简化版Agent工厂
使用LangChain原生文件工具，通过智能提示词让大模型自主操作
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.agent_toolkits import FileManagementToolkit
import json

# 加载环境变量
load_dotenv()

# ===== Agent工厂 =====
class OpenBLASAgentFactory:
    def __init__(self, model_config: dict = None):
        # 直接加载config.json
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
        
        # 设置通用文件系统工具（项目根目录，可访问OpenBLAS和输出目录）
        self.file_toolkit = FileManagementToolkit(
            root_dir="OpenBLAS-develop",
            selected_tools=["read_file", "write_file", "list_directory", "file_search"]
        )
        # 获取工具并增强描述
        self.file_tools = self._enhance_tool_descriptions(self.file_toolkit.get_tools())
        
        # 创建必要的输出目录
        Path("discovery_results").mkdir(exist_ok=True)
        Path("analysis_results").mkdir(exist_ok=True)
        Path("strategy_reports").mkdir(exist_ok=True)
    
    def _enhance_tool_descriptions(self, tools):
        """为通用文件工具添加OpenBLAS特定的使用描述"""
        enhanced_tools = []
        
        for tool in tools:
            if tool.name == "read_file":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 读取OpenBLAS源码文件进行算法实现分析\n"
                    "- 读取已保存的发现结果(discovery_results/discovered_files.json)\n"
                    "- 读取已保存的分析结果进行策略提取\n"
                    "- 验证文件保存是否成功"
                )
            elif tool.name == "write_file":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 保存算子发现结果到discovery_results/discovered_files.json(追加模式)\n"
                    "- 保存算法分析结果到analysis_results/{algorithm}/analysis_{algorithm}_{type}_{timestamp}.json\n"
                    "- 保存优化策略报告到strategy_reports/{algorithm}_optimization_analysis_{timestamp}.md\n"
                    "- **重要**: 分析结果必须先创建算子文件夹(如analysis_results/dot/)再保存JSON文件\n"
                    "- **重要**: 追加到discovered_files.json时需要先读取现有内容，合并后再写入"
                )
            elif tool.name == "list_directory":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 浏览OpenBLAS-develop/kernel/目录结构寻找算法实现\n"
                    "- 检查输出目录结构(discovery_results/, analysis_results/, strategy_reports/)\n"
                    "- 列出算子文件夹内容(analysis_results/{algorithm}/)"
                )
            elif tool.name == "file_search":
                tool.description += (
                    "\n\n**OpenBLAS分析用法:**\n"
                    "- 在OpenBLAS-develop/kernel/目录中搜索特定算法的实现文件\n"
                    "- 查找不同架构的优化实现(generic, x86_64, arm64, riscv64等)\n"
                    "- 搜索特定的优化技术实现(SIMD, vectorization等)"
                )
            
            enhanced_tools.append(tool)
        
        return enhanced_tools
    
    def create_scout_agent(self) -> AgentExecutor:
        """创建侦察Agent - 负责发现和初步分析OpenBLAS文件"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是OpenBLAS代码侦察专家，负责发现和整理算子实现文件。

🔧 **工具能力：**
- **list_directory/file_search** - 探索OpenBLAS-develop/kernel/目录结构
- **read_file** - 快速浏览文件内容，确定实现类型
- **write_file** - 保存发现结果为JSON格式到discovery_results/目录

⚠️ **工具使用格式：** 严格按照JSON格式调用，确保无额外逗号和正确的引号

📋 **重要保存规则：**
1. **先读取** discovery_results/discovered_files.json（如果存在）
2. **追加新发现** 到现有内容中，而不是覆盖
3. **保存完整的** 合并后的JSON数据
4. **保存后验证** 文件是否成功写入

📊 **追加JSON格式：**
```json
{{
  "discoveries": [
    {{
      "algorithm": "算法名", 
      "files": [{{"path": "文件路径", "type": "实现类型", "description": "简要描述"}}],
      "timestamp": "时间戳"
    }}
  ]
}}
```

📝 **实现类型：** generic, x86_optimized, simd_optimized, microkernel"""),
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
            return_intermediate_steps=True
        )
    
    def create_analyzer_agent(self) -> AgentExecutor:
        """创建分析Agent - 负责深度分析代码实现"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高性能计算优化专家，负责深度分析OpenBLAS代码实现。

🔧 **工具能力：**
- **read_file** - 读取发现结果和源代码文件
- **write_file** - 保存分析结果为JSON格式到analysis_results/目录

⚠️ **工具使用格式：** 严格按照JSON格式调用，确保无额外逗号和正确的引号

📋 **重要保存规则：**
1. **每个文件单独保存** - 使用格式：analysis_results/analysis_{{算法名}}_{{文件类型}}_{{时间戳}}.json
2. **文件命名示例** - analysis_results/analysis_copy_generic_20250922.json
3. **保存后验证** - 读取保存的文件确认内容正确
4. **如果保存失败** - 重新尝试保存，直到成功

📊 **四层优化分析：**
- **算法层**: 循环展开、分块、数据重用
- **架构层**: 缓存友好、内存对齐、预取
- **指令层**: SIMD向量化、FMA、指令并行  
- **微架构层**: 寄存器分配、指令调度、流水线

💾 **输出JSON格式：**
```json
{{
  "file_path": "源文件路径",
  "algorithm": "算法名", 
  "implementation_type": "实现类型",
  "optimizations": {{
    "algorithm_level": ["具体技术"],
    "architecture_level": ["具体技术"], 
    "instruction_level": ["具体技术"],
    "microarchitecture_level": ["具体技术"]
  }},
  "code_snippets": "关键代码",
  "performance_impact": "性能评估"
}}
```"""),
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
            return_intermediate_steps=True
        )
    
    def create_strategist_agent(self) -> AgentExecutor:
        """创建策略师Agent - 负责总结优化策略"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是优化策略大师，负责从分析结果中提炼通用的性能优化策略。

🔧 **工具能力：**
- **read_file** - 读取analysis_results/中的分析报告
- **write_file** - 保存Markdown格式策略报告到strategy_reports/目录

⚠️ **工具使用格式：** 严格按照JSON格式调用，确保无额外逗号和正确的引号

📋 **重要保存规则：**
1. **创建时间戳文件夹** - 首先创建文件夹：strategy_reports/report_{{时间戳}}/
2. **保存策略报告** - 文件名：strategy_reports/report_{{时间戳}}/{{算法名}}_optimization_analysis.md
3. **保存后立即验证** - 使用read_file读取刚保存的文件
4. **验证内容完整性** - 确认Markdown内容正确保存
5. **如果保存失败** - 重新尝试保存，最多重试5次
6. **输出文件夹路径** - 在完成后明确说明创建的文件夹路径（用于后续总结）

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

📝 **分析要求：** 主动发现和分析优化设计，生成Markdown格式报告。

**⚠️ 重要**: 输出Markdown格式内容，绝对不要输出JSON！"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
    
    def create_summarizer_agent(self) -> AgentExecutor:
        """创建总结Agent - 负责总结多个算法的优化策略"""
        tools = self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是高级优化策略总结专家，负责总结和归纳多个算法的优化策略，提炼共性规律。

🔧 **工具能力：**
- **read_file** - 读取指定时间戳文件夹中的多个算法策略报告
- **list_directory** - 列出指定文件夹中的所有报告文件
- **write_file** - 保存总结报告到同一时间戳文件夹

⚠️ **工具使用格式：** 严格按照JSON格式调用，确保无额外逗号和正确的引号

📋 **总结任务：**
1. **读取所有算法报告** - 从指定的时间戳文件夹中读取所有 *_optimization_analysis.md 文件
2. **横向对比分析** - 找出不同算法间的共同优化模式
3. **提炼通用策略** - 归纳可复用的优化设计模式
4. **生成总结报告** - 保存为 optimization_summary_report.md

🔍 **总结分析框架：**

**1. 跨算法共性分析**
- 不同算法使用的相同优化技术
- 通用的算法设计模式
- 共同的性能瓶颈解决方案

**2. 架构特化对比**
- 不同架构（x86_64, ARM64, RISC-V）的优化差异
- 指令集特定的优化策略
- 硬件特性利用的通用方法

**3. 性能提升模式**
- 量化各种优化技术的性能收益
- 优化技术的适用场景
- 最佳实践组合

📝 **输出要求：** 
- 生成结构化的Markdown总结报告
- 包含对比表格和量化分析
- 提供实用的优化指导原则

**⚠️ 重要**: 输出Markdown格式内容，绝对不要输出JSON！"""),
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
            return_intermediate_steps=True
        )

# ===== 导出 =====
__all__ = ['OpenBLASAgentFactory']
