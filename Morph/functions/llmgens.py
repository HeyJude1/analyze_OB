"""
利用智能体调用函数，生成多段代码
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
from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig, function_tool
from base_agents import CodeGenerate
# 工具函数
import tiktoken
from utils import get_env_var, get_function_name, postprocess


# prompt 定义
SYSTEM_TEMPLATE = """You are a helpful coding assistant.
You are helping a programmer write a C++ function. Write the body of the function and put it in a markdown code block.
Do not write any other code or explanations. Do not include any comments or explanation, and avoid extra punctuation.
"""

PROMPT_TEMPLATE = """Complete the C++ function {function_name}. Only write the body of the function {function_name}.

```cpp
{prompt}
```
"""

def CodeGens(
    model: str = "qwen-plus-2025-04-28", # "qwen-turbo"
    input_path: str = "prompts.json",
    output_path: str = "results/prompts_code.json",
    api_key: str = "sk-1b0daa78e21d42509a094cec569b94f3",
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    max_requests: int = 10000,
    max_tokens_per_second: int = 1000,
    max_requests_per_second: int = 10,
    dry: bool = False,
    overwrite: bool = False,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    num_samples_per_prompt: int = 4
    )-> None:

    # 读取输入的prompts
    with open(input_path, "r") as prompts_json:
        prompts = json.load(prompts_json)

    # 如果已有输出文件并且不覆盖，加载旧输出合并
    if not overwrite and os.path.exists(output_path):
        with open(output_path, 'r') as output_json:
            outputs = json.load(output_json)

        # 将已经存在的内容复制到结果中
        copy_count = 0
        for prompt in prompts:
            for o in outputs:
                if o["prompt"] == prompt["prompt"] and \
                   o["name"] == prompt["name"] and \
                   o["parallelism_model"] == prompt["parallelism_model"] and \
                   "outputs" in o and \
                   len(o["outputs"]) == num_samples_per_prompt and \
                   o["temperature"] == temperature and \
                   o["top_p"] == top_p:
                    for col in ["temperature", "top_p", "do_sample", "max_new_tokens", "outputs"]:
                        prompt[col] = o[col]
                    copy_count += 1
                    break
        print(f"Copied {copy_count} existing outputs.")

    client = OpenAI(api_key=api_key, base_url=openai_base_url)

    # get 速率限制参数
    MAX_TOKENS_PER_SECOND = max_tokens_per_second
    MAX_REQUESTS_PER_SECOND = max_requests_per_second
    MAX_REQUESTS = max_requests

    # 初始化计数器
    request_counter = 0
    request_rate_counter = 0
    token_counter = 0
    token_rate_counter = 0
    token_timer = time.time()
    request_timer = time.time()

    # 遍历每一个prompt
    for prompt in tqdm(prompts, desc="Generating outputs"):
        # 如果已经生成好了，就skip
        if not overwrite and "outputs" in prompt:
            continue

        # get prompt
        original_prompt = prompt["prompt"]
        function_name = get_function_name(original_prompt, prompt["parallelism_model"])
        prompt_text = PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name)

        # test：在参数输入中开启dry进入测试模型，只输出prompt不会生成代码
        if dry:
            print("system", SYSTEM_TEMPLATE)
            print("prompt", prompt_text)
            continue

        # LLM 生成参数
        prompt["temperature"] = temperature
        prompt["top_p"] = top_p
        prompt["do_sample"] = True
        prompt["max_new_tokens"] = max_new_tokens

        
        # 发送请求并获取返回结果
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            n=num_samples_per_prompt
        )

        outputs = [c.message.content for c in completion.choices]
        outputs = [postprocess(original_prompt, o) for o in outputs]
        prompt["outputs"] = outputs

        # 更新计数器
        request_counter += 1
        request_rate_counter += 1
        token_counter += completion.usage.total_tokens
        token_rate_counter += completion.usage.total_tokens

        # 限流处理
        if MAX_REQUESTS is not None and request_counter >= MAX_REQUESTS:
            print(f"Stopping after {request_counter} requests.")
            break

        # 检查是否限流超标，超出了就睡眠一段时间
        tokens_per_second = token_rate_counter / (time.time() - token_timer)
        if MAX_TOKENS_PER_SECOND is not None and tokens_per_second > (MAX_TOKENS_PER_SECOND*0.9):
            sleep_time = 30
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            token_timer = time.time()
            token_rate_counter = 0

        requests_per_second = request_rate_counter / (time.time() - request_timer)
        if MAX_REQUESTS_PER_SECOND is not None and requests_per_second > (MAX_REQUESTS_PER_SECOND*0.95):
            sleep_time = 60
            print(f"Sleeping for {sleep_time} seconds.")
            time.sleep(sleep_time)
            request_timer = time.time()
            request_rate_counter = 0

        # 写入中间结果（断点续跑）
        with open(output_path, 'w') as output_json:
            json.dump(prompts, output_json, indent=2)

    # 打印统计信息
    print(f"Submitted {request_counter} requests.")
    print(f"Used {token_counter} tokens.")

    # 最终写入全部输出结果
    with open(output_path, 'w') as output_json:
        json.dump(prompts, output_json, indent=2)


