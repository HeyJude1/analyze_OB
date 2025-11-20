"""
利用智能体再次修改代码的功能在此定义
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
from agents import Agent, AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig
from base_agents import CodeReview
# 工具函数
import tiktoken
from utils import get_env_var, get_function_name, postprocess

# prompt 定义
PROMPT_TEMPLATE = """The C++ function {function_name} generated according to prompt has been run, but an error has occurred. 
Please modify the function according to the error and the original function so that it can run. Only write the body of the function {function_name}
prompt:
```cpp
{prompt}
```
code:
```cpp
{code}
```
{error_type}:
```
{error}
"""

# 定义异步主函数以运行 Agent
async def run_agent(agent: Agent, input: str):
    result = await Runner.run(agent, input=input, run_config=RunConfig(tracing_disabled=True))
    return result.final_output

def CodeRe(
        model: str = "qwen-plus-2025-04-28",
        input_path: str = "results/code_run.json",
        output_path: str = "results/code_review.json",
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
        num_samples_per_prompt: int = 1 # 智能体目前不支持生成多个输出，后续可以考虑修改prompt
    )-> None:  

    # 读取输入的代码
    with open(input_path, "r") as code_json:
        codes = json.load(code_json)

    # 创建agent用于修改代码
    external_client = AsyncOpenAI(
        api_key=api_key,
        base_url=openai_base_url
    )
    model = OpenAIChatCompletionsModel(
        model=model,
        openai_client=external_client
    )
    model_settings = ModelSettings(temperature=temperature, max_tokens=max_new_tokens, top_p=top_p)
    code_agent = CodeReview(model, model_settings)

    # get 速率限制参数
    MAX_TOKENS_PER_SECOND = max_tokens_per_second
    MAX_REQUESTS_PER_SECOND = max_requests_per_second
    MAX_REQUESTS = max_requests

    # 初始化计数器
    request_counter = 0
    request_rate_counter = 0
    encoding = tiktoken.encoding_for_model("gpt-4")
    token_counter = 0
    token_rate_counter = 0
    token_timer = time.time()
    request_timer = time.time()

    # 遍历所有的prompts，通过了就不需二次修改，没通过的才需要二次修改
    for code in tqdm(codes, desc="Reviewing Codes"):
        # 如果无错误，说明运行通过，可以不用再二次生成，否则就需要二次修改代码
        if code["outputs"][0]["build_stderr"] != "":
            error_type = "build_error"
            error = code["outputs"][0]["build_stderr"]
        elif code["outputs"][0]["runs"][0]["stderr"] != "":
            error_type = "run_error"
            error = code["outputs"][0]["runs"][0]["stderr"]
        else:
            continue

        # get prompt
        original_prompt = code["prompt"]
        original_code = code["outputs"][0]["generated_output"]
        function_name = get_function_name(original_prompt, code["parallelism_model"])
        prompt_text = PROMPT_TEMPLATE.format(prompt=original_prompt, function_name=function_name, code=original_code, error_type=error_type, error=error)

        # test：在参数输入中开启dry进入测试模型，只输出prompt不会生成代码
        if dry:
            print("prompt", prompt_text)
            continue

        outputs = asyncio.run(run_agent(code_agent, prompt_text))
        # 统计模型输出的token
        tokens = encoding.encode(outputs)
        outputs = [postprocess(original_prompt, outputs)]
        code["outputs"] = outputs
        # print(outputs)

        # 更新计数器
        request_counter += 1
        request_rate_counter += 1
        token_counter += len(tokens)
        token_rate_counter += len(tokens)

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
            json.dump(codes, output_json, indent=2)

    # 打印统计信息
    print(f"Submitted {request_counter} requests.")
    print(f"Used {token_counter} tokens.")

    # 最终写入全部输出结果
    with open(output_path, 'w') as output_json:
        json.dump(codes, output_json, indent=2)

        





        







    