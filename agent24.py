#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenBLAS优化策略分析Agent v24
支持算法层、代码层、指令层的三层优化分析
"""

import os
import json
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()


def create_algorithm_optimizer(model_config: Dict[str, Any]) -> ChatOpenAI:
    """创建算法层优化分析器"""
    
    llm = ChatOpenAI(
        model=model_config.get("name", "qwen-plus-2025-09-11"),
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 4000),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    
    system_prompt = """你是OpenBLAS算法层优化专家，专门分析高层算法设计优化策略。

🎯 **分析目标**: 识别算法层面的优化策略，包括循环结构、数据重用、计算重排等

📋 **输出字段要求**:
**1. optimization_name**: 优化策略的简洁中文名称
**2. level**: 固定为 "algorithm"
**3. description**: ⚠️ 严格包含且仅包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理
  - implementation_pattern: 解释"怎么做"的代码实现模式  
  - performance_impact: 解释"有什么用"的性能提升
  - trade_offs: 解释该优化的局限性或代价
  
  ⚠️ 重要：description字段中不得包含其他任何字段，如applicability_conditions、tunable_parameters等，这些字段应该独立存在于description之外！

**4. applicability_conditions**: 该优化适用的具体条件
**5. tunable_parameters**: 可调参数列表
**6. related_patterns**: 相关的计算流程类型列表

🔍 **分析要求**:
- 重点关注循环展开、分块、数据重用等算法层优化
- ⚠️ 严格按照上述字段结构输出，特别注意description字段只能包含4个指定子字段
- 确保每个优化策略都有明确的理论依据和实施方案"""
    
    return llm, system_prompt


def create_code_optimizer(model_config: Dict[str, Any]) -> ChatOpenAI:
    """创建代码层优化分析器"""
    
    llm = ChatOpenAI(
        model=model_config.get("name", "qwen-plus-2025-09-11"),
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 4000),
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    
    system_prompt = """你是OpenBLAS代码层优化专家，专门分析代码实现层面的优化策略。

🎯 **分析目标**: 识别代码层面的优化策略，包括缓存优化、内存对齐、预取等

📋 **输出字段要求**:
**1. optimization_name**: 优化策略的简洁中文名称
**2. level**: 固定为 "code"  
**3. description**: ⚠️ 严格包含且仅包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理
  - implementation_pattern: 解释"怎么做"的代码实现模式
  - performance_impact: 解释"有什么用"的性能提升
  - trade_offs: 解释该优化的局限性或代价
  
  ⚠️ 重要：description字段中不得包含其他任何字段，如applicability_conditions、tunable_parameters等，这些字段应该独立存在于description之外！

**4. applicability_conditions**: 该优化适用的具体条件
**5. tunable_parameters**: 可调参数列表
**6. related_patterns**: 相关的计算流程类型列表

🔍 **分析要求**:
- 重点关注缓存优化、内存访问模式、编译器优化等代码层优化
- ⚠️ 严格按照上述字段结构输出，特别注意description字段只能包含4个指定子字段
- 确保每个优化策略都有明确的代码实现指导"""
    
    return llm, system_prompt


def create_instruction_optimizer(model_config: Dict[str, Any]) -> ChatOpenAI:
    """创建指令层优化分析器"""
    
    llm = ChatOpenAI(
        model=model_config.get("name", "qwen-plus-2025-09-11"),
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 4000),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    
    system_prompt = """你是OpenBLAS指令层优化专家，专门分析底层指令级别的优化策略。

🎯 **分析目标**: 识别指令层面的优化策略，包括SIMD向量化、FMA指令、指令并行等

📋 **输出字段要求**:
**1. optimization_name**: 优化策略的简洁中文名称
**2. level**: 固定为 "instruction"
**3. description**: ⚠️ 严格包含且仅包含4个子字段的详细分析对象
  - strategy_rationale: 解释"为什么"要这么做的理论原理
  - implementation_pattern: 解释"怎么做"的代码实现模式
  - performance_impact: 解释"有什么用"的性能提升
  - trade_offs: 解释该优化的局限性或代价
  
  ⚠️ 重要：description字段中不得包含其他任何字段，如applicability_conditions、tunable_parameters等，这些字段应该独立存在于description之外！

**4. applicability_conditions**: 该优化适用的具体条件
**5. tunable_parameters**: 可调参数列表
**6. related_patterns**: 相关的计算流程类型列表

🔍 **分析要求**:
- 重点关注SIMD向量化、FMA指令、指令级并行等底层优化
- ⚠️ 严格按照上述字段结构输出，特别注意description字段只能包含4个指定子字段
- 确保每个优化策略都有明确的硬件指令依据"""
    
    return llm, system_prompt


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载配置文件"""
    if not os.path.exists(config_path):
        return {
            "model": {
                "name": "qwen-plus-2025-09-11",
                "temperature": 0.1,
                "max_tokens": 4000,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 导出函数
__all__ = [
    'create_algorithm_optimizer',
    'create_code_optimizer', 
    'create_instruction_optimizer',
    'load_config'
]
