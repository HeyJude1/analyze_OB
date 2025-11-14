#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS算子优化策略分析工具 - Agent工厂和工具定义
"""

import os
import json
import time
import random
import re
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_community.agent_toolkits import FileManagementToolkit
from pydantic import BaseModel, Field
import pandas as pd

# 加载环境变量
load_dotenv()

# ===== 工具输入模型定义 =====
class FileDiscoveryInput(BaseModel):
    """文件发现输入参数"""
    algorithm: str = Field(description="算法类型：dot, gemm, copy等")
    count: int = Field(default=5, description="要发现的文件数量")

class FileAnalysisInput(BaseModel):
    """文件分析输入参数"""
    file_path: str = Field(description="要分析的文件路径")
    algorithm: str = Field(description="算法类型")

# 删除了StrategyExtractionInput，不再需要

# ===== OpenBLAS文件系统管理 =====
class OpenBLASFileSystem:
    """OpenBLAS文件系统管理器"""
    
    def __init__(self, openblas_path: str = "./OpenBLAS-develop"):
        self.openblas_path = Path(openblas_path)
        self.available_files = self._discover_files()
    
    def _discover_files(self) -> Dict[str, List[Dict]]:
        """发现OpenBLAS中的算子文件"""
        files_by_algorithm = {}
        kernel_path = self.openblas_path / "kernel"
        
        if not kernel_path.exists():
            return {}
        
        # 算子模式匹配
        algorithm_patterns = {
            'dot': r'.*dot.*\.(c|S)$',
            'gemm': r'.*gemm.*\.(c|S)$',
            'gemv': r'.*gemv.*\.(c|S)$',
            'scal': r'.*scal.*\.(c|S)$',
            'asum': r'.*asum.*\.(c|S)$',
            'copy': r'.*copy.*\.(c|S)$'
        }
        
        # 实现类型分类
        def classify_implementation(path_str: str) -> str:
            path_lower = path_str.lower()
            if 'generic' in path_lower:
                return 'generic'
            elif any(x in path_lower for x in ['avx512', 'microk', 'skylakex']):
                return 'avx512_optimized'
            elif any(x in path_lower for x in ['sse', 'avx']):
                return 'sse_optimized'
            elif any(x in path_lower for x in ['x86_64', 'arm64', 'power']):
                return 'architecture_specific'
            else:
                return 'unknown'
        
        for algo, pattern in algorithm_patterns.items():
            files_by_algorithm[algo] = []
            
            for file_path in kernel_path.rglob("*"):
                if file_path.is_file() and re.match(pattern, file_path.name, re.IGNORECASE):
                    relative_path = file_path.relative_to(self.openblas_path)
                    impl_type = classify_implementation(str(relative_path))
                    
                    files_by_algorithm[algo].append({
                        'path': str(relative_path),
                        'type': impl_type,
                        'name': file_path.name,
                        'algorithm': algo
                    })
        
        return files_by_algorithm
    
    def get_files_by_algorithm(self, algorithm: str, count: int = None) -> List[Dict]:
        """获取指定算法的文件"""
        files = self.available_files.get(algorithm, [])
        if count and len(files) > count:
            # 优先选择不同类型的实现
            selected = []
            types_priority = ['generic', 'architecture_specific', 'sse_optimized', 'avx512_optimized']
            
            for impl_type in types_priority:
                type_files = [f for f in files if f['type'] == impl_type]
                if type_files:
                    selected.extend(type_files[:max(1, count // len(types_priority))])
            
            # 如果还不够，随机补充
            remaining = count - len(selected)
            if remaining > 0:
                unselected = [f for f in files if f not in selected]
                selected.extend(random.sample(unselected, min(remaining, len(unselected))))
            
            return selected[:count]
        return files
    
    def read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        full_path = self.openblas_path / file_path
        if not full_path.exists():
            return f"文件不存在: {file_path}"
        
        with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # 限制内容长度
            if len(content) > 3000:
                return content[:3000] + "\n... (文件过长，已截取前3000字符)"
            return content

# 全局文件系统实例
openblas_fs = OpenBLASFileSystem()

# ===== 工具定义（符合LangChain规范）=====

@tool("discover_openblas_files", args_schema=FileDiscoveryInput)
def discover_openblas_files(algorithm: str, count: int = 5) -> str:
    """发现OpenBLAS中指定算法的实现文件，返回文件列表信息"""
    files = openblas_fs.get_files_by_algorithm(algorithm, count)
    if not files:
        return f"未找到算法 {algorithm} 的实现文件"
    
    result = f"发现 {algorithm} 算法的 {len(files)} 个实现文件:\n"
    for i, f in enumerate(files, 1):
        result += f"{i}. {f['path']} (类型: {f['type']})\n"
    
    # 保存到专门的目录
    discovery_dir = Path("discovery_results")
    discovery_dir.mkdir(exist_ok=True)
    
    discovery_data = {
        'algorithm': algorithm,
        'files': files,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    discovery_file = discovery_dir / f'discovered_{algorithm}_files.json'
    with open(discovery_file, 'w', encoding='utf-8') as f:
        json.dump(discovery_data, f, ensure_ascii=False, indent=2)
    
    return result

# read_openblas_file工具已被LangChain的read_file工具替代

@tool("save_analysis_results")
def save_analysis_results(analysis_content: str, algorithm: str) -> str:
    """保存单个文件的分析结果"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"analysis_{algorithm}_{timestamp}.json"
    
    # 构造结果数据
    result_data = {
        'algorithm': algorithm,
        'analysis': analysis_content,
        'timestamp': timestamp
    }
    
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    
    return f"分析结果已保存到: {output_path}"

@tool("collect_all_analyses")
def collect_all_analyses() -> str:
    """收集所有分析结果文件，准备进行策略提取"""
    analysis_dir = Path("analysis_results")
    if not analysis_dir.exists():
        return "没有找到分析结果目录"
    
    all_analyses = []
    for file_path in analysis_dir.glob("analysis_*.json"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_analyses.append(data)
    
    if not all_analyses:
        return "没有找到任何分析结果"
    
    # 保存汇总文件
    summary_file = "all_analyses_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, ensure_ascii=False, indent=2)
    
    return f"已收集 {len(all_analyses)} 个分析结果，保存到 {summary_file}"

@tool("save_optimization_strategies")
def save_optimization_strategies(strategies_content: str) -> str:
    """保存优化策略报告（接收已格式化的markdown内容）"""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # 创建策略报告目录
    strategies_dir = Path("strategy_reports")
    strategies_dir.mkdir(exist_ok=True)
    
    filename = strategies_dir / f"optimization_strategies_{timestamp}.md"
    
    # 保存markdown格式的策略内容
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# OpenBLAS优化策略总结\n\n")
        f.write(f"生成时间: {timestamp}\n\n")
        f.write("---\n\n")
        # 确保内容是markdown格式
        if strategies_content.strip():
            # 如果内容看起来像JSON，提示错误
            if strategies_content.strip().startswith('{') and strategies_content.strip().endswith('}'):
                f.write("⚠️ **格式错误**：接收到JSON格式数据，应为Markdown格式的策略报告。\n\n")
                f.write("```json\n")
                f.write(strategies_content)
                f.write("\n```\n")
            else:
                f.write(strategies_content)
        else:
            f.write("策略内容为空，请检查分析过程。")
    
    return f"优化策略已保存到: {filename}"

# ===== Agent工厂 =====
class OpenBLASAgentFactory:
    """OpenBLAS Agent工厂"""
    
    def __init__(self, model_config: dict = None):
        if model_config is None:
            model_config = {
                "name": "qwen-plus-2025-09-11",
                "temperature": 0.3,
                "max_tokens": 4000
            }
        
        # 配置HTTP客户端
        import httpx
        http_client = httpx.Client(
            verify=False,
            timeout=60.0,
            follow_redirects=True
        )
        
        self.llm = ChatOpenAI(
            model=model_config["name"],
            temperature=model_config["temperature"],
            max_tokens=model_config["max_tokens"],
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=http_client
        )
        
        # 设置文件系统工具（限制在OpenBLAS目录）
        self.file_toolkit = FileManagementToolkit(
            root_dir="./OpenBLAS-develop"
        )
        self.file_tools = self.file_toolkit.get_tools()
    
    def create_scout_agent(self) -> AgentExecutor:
        """创建侦察Agent - 负责发现和读取文件"""
        # 结合自定义工具和LangChain文件工具
        tools = [discover_openblas_files] + self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个OpenBLAS代码侦察专家。你的任务是：
1. 使用discover_openblas_files工具发现指定算法的实现文件
2. 使用read_file工具读取具体文件内容
3. 使用list_directory工具浏览目录结构
4. 为每个文件生成初步的代码结构分析

可用工具说明：
- discover_openblas_files: 发现OpenBLAS中指定算法的实现文件
- read_file: 读取具体文件内容  
- list_directory: 列出目录内容
- file_search: 搜索文件

注意：优先选择generic、architecture_specific、sse_optimized等不同类型的实现进行对比分析。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
    
    def create_analyzer_agent(self) -> AgentExecutor:
        """创建分析Agent - 负责深度分析代码"""
        # 结合自定义分析工具和LangChain文件工具
        tools = [save_analysis_results] + self.file_tools
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个高性能计算和编译器优化专家。你的任务是深度分析OpenBLAS的代码实现，识别关键的优化技术。

对于每个代码文件，你需要分析：
1. **算法优化**: 数学层面的优化（如分块、循环展开）
2. **架构优化**: CPU架构相关的优化（如缓存友好、内存对齐）
3. **指令优化**: SIMD向量化、指令级并行
4. **微架构优化**: 寄存器分配、指令调度、流水线优化

可用工具：
- read_file: 读取源代码文件
- list_directory: 浏览目录结构
- save_analysis_results: 保存分析结果

使用save_analysis_results保存每个文件的详细分析结果。"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)
    
    def create_strategist_agent(self) -> AgentExecutor:
        """创建策略师Agent - 负责总结优化策略"""
        tools = [collect_all_analyses, save_optimization_strategies]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个优化策略专家。你的任务是：
1. 使用collect_all_analyses收集所有的分析结果  
2. 从这些分析中提取通用的优化模式和策略
3. 按照优化层次组织策略（算法级、架构级、指令级、微架构级）
4. 为每个策略提供具体的代码示例和性能影响分析
5. 将分析结果整理成专业的Markdown格式报告
6. 使用save_optimization_strategies保存最终报告

**重要**：你必须输出标准的Markdown格式内容，包含标题、代码块、表格等。绝对不要输出JSON格式。

示例格式：
## 算法级优化
### 策略1: 矩阵分块
- **代码示例**: 
```c
for (int i = 0; i < N; i += BLOCK_SIZE) {
    // 分块代码
}
```
- **性能影响**: 提升2x性能
- **适用场景**: 大矩阵运算"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# ===== 导出 =====
__all__ = [
    'OpenBLASAgentFactory',
    'discover_openblas_files',
    'save_analysis_results',
    'collect_all_analyses',
    'save_optimization_strategies'
]
