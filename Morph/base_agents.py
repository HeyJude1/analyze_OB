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
from typing import Optional

# tpl imports
from tqdm import tqdm
from openai import OpenAI

# agent imports
import asyncio
from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner

# 代码生成智能体，利用prompt生成函数
def CodeGenerate(model, model_settings):
    agent = Agent(
        name = "代码生成智能体",
        #你是一个有用的编码助手。
        #您正在帮助程序员编写C++函数。编写函数体，并将其放在markdown代码块中。
        #不要编写任何其他代码或解释。不要包括任何注释或解释，并避免额外的标点符号。
        instructions="""You are a helpful coding assistant.
You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block.
Do not write any other code or explanations. Do not include any comments or explanation, and avoid extra punctuation.
""",
        model=model,
        model_settings=model_settings
    )
    return agent

# 代码二次生成智能体，负责将一次生成，但是无法运行的代码进行修正，使其可以运行。
def CodeReview(model, model_settings):
    agent = Agent(
        name="代码修正智能体",
        #你是一个有用的代码修正助手。
        #你正在帮助程序员修改C++函数，你的任务是修正无法运行的函数体，使其可以运行，并将其放在markdown代码块中
        #不要编写任何其他代码或解释。不要包括任何注释或解释，并避免额外的标点符号。确保一定进行了修改
        instructions="""You are a helpful code correction assistant
You are helping a programmer modify a C++ function. Modify the body of the function that cannot be run, make it run, and put it in a markdown code block.
Do not write any other code or explanations. Do not include any comments or explanation, and avoid extra punctuation.Make sure the changes are made
""",
        model=model,
        model_settings=model_settings
    )
    return agent

# prompt生成智能体，负责二次生成prompt
def PromptGenerate(model, model_settings):
    agent = Agent(
        name="prompt生成智能体",
        # 你是一个有用的优化策略总结助手。
        # 你的任务是根据输入的硬件环境，总结出优化策略用于帮助其他模型生成代码，其代码运行的主要的平台为cuda、serial和omp
        # 针对三个不同的平台分别编写优化策略，并给出对应的提示词，请用json格式给出，格式为：
        # {
        # "cuda": "优化策略",
        # "serial": "优化策略",
        # "omp": "优化策略"
        # }
        # 除此之外不要生成多余内容。请用英文输出
        instructions="""You are a helpful optimization strategy summarization assistant.  
Your task is to summarize optimization strategies based on the input hardware environment, to assist other models in generating code. The primary target platforms for the code are CUDA, serial, and OMP.  
Develop optimization strategies for these three platforms separately and provide corresponding prompts. Output in JSON format as follows:  
{  
"cuda": "optimization strategy",  
"serial": "optimization strategy",  
"omp": "optimization strategy"  
}  
Do not generate any additional content beyond this. Please output in English.""",
        model=model,
        model_settings=model_settings
    )
    return agent

