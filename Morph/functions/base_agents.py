"""
定义基本的智能体
"""
from dataclasses import dataclass, field
# std imports
from argparse import ArgumentParser
import json
import os
import re
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# tpl imports
from tqdm import tqdm

# langchain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class AgentFactory:
    """智能体工厂，负责统一管理模型配置和智能体创建"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.llm = self._create_llm()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        # 尝试在当前目录或上级目录查找config.json
        possible_paths = [
            config_path,
            os.path.join(os.path.dirname(__file__), config_path),
            "config.json",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        print(f"⚠️ 警告: 未找到配置文件 {config_path}，将使用默认配置")
        return {}

    def _create_llm(self, temperature: float = None, top_p: float = None, max_tokens: int = None):
        """创建LLM实例"""
        model_config = self.config.get("model", {})
        
        # 优先使用传入的参数，否则使用配置文件的值
        temp = temperature if temperature is not None else model_config.get("temperature", 0.1)
        tp = top_p if top_p is not None else model_config.get("top_p", 0.9)
        tokens = max_tokens if max_tokens is not None else model_config.get("max_tokens", 8192)
        
        return ChatOpenAI(
            model=model_config.get("name", "qwen-plus-2025-09-11"),
            temperature=temp,
            max_tokens=tokens,
            top_p=tp,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=model_config.get("base_url")
        )

    def create_code_generate_agent(self, temperature: float = None, top_p: float = None, max_tokens: int = None):
        """创建代码生成Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个乐于助人的编程助手。
你正在帮助程序员编写一个 C++ 函数。请编写函数体并将其放在 markdown 代码块中。
不要编写任何其他代码或解释。不要包含任何注释或解释，避免多余的标点符号。
"""),
            ("human", "{input}"),
        ])

        # 如果需要特定的参数，创建一个新的临时LLM实例
        if any(x is not None for x in [temperature, top_p, max_tokens]):
            specialized_llm = self._create_llm(temperature, top_p, max_tokens)
            return prompt | specialized_llm
        return prompt | self.llm

    def create_code_review_agent(self):
        """创建代码修正Agent"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个乐于助人的代码修正助手。
你正在帮助程序员修改一个 C++ 函数。修改无法运行的函数体，使其能够运行，并将其放在 markdown 代码块中。
不要编写任何其他代码或解释。不要包含任何注释或解释，避免多余的标点符号。确保已进行修改。
"""),
            ("human", "{input}"),
        ])
        return prompt | self.llm
