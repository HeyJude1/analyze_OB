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

from ..utils.prompt_loader import get_prompt_loader

load_dotenv("config/.env")

_prompts = get_prompt_loader()


def create_algorithm_optimizer(model_config: Dict[str, Any]) -> ChatOpenAI:
    """创建算法层优化分析器"""

    llm = ChatOpenAI(
        model=model_config.get("name", "qwen-plus-2025-09-11"),
        temperature=model_config.get("temperature", 0.1),
        max_tokens=model_config.get("max_tokens", 4000),
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=model_config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )

    system_prompt = _prompts.load_system_prompt("analysis/algorithm_optimizer.yaml")

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

    system_prompt = _prompts.load_system_prompt("analysis/code_optimizer.yaml")

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

    system_prompt = _prompts.load_system_prompt("analysis/instruction_optimizer.yaml")

    return llm, system_prompt


def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
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
