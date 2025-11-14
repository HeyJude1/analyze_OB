# 设置了分页的功能来实现提取多篇论文（数量通常很大）

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 1. 导入构建自定义工具所需的核心组件
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool, tool
# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.utilities import ArxivAPIWrapper
from dotenv import load_dotenv
import os
from typing import List

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 升级工具：构建一个支持分页的 ArXiv 工具 (描述已中文化) ---

# 定义工具的输入参数模型，让 Agent 知道该提供什么参数
class ArxivSearchInput(BaseModel):
    query: str = Field(description="用于 ArXiv 搜索的查询字符串。")
    offset: int = Field(default=0, description="获取结果的起始点（偏移量）。用于分页获取数据。")

# 创建一个配置好的底层 Wrapper
arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, load_max_docs=5, doc_content_chars_max=4000)

# 使用 @tool 装饰器轻松创建自定义工具
@tool("arxiv_paginated_search", args_schema=ArxivSearchInput)
def search_arxiv_with_offset(query: str, offset: int = 0) -> str:
    """
    带分页功能，用于搜索 ArXiv 上的论文。
    通过改变 offset 参数来分批次获取论文。
    例如，要获取前5篇论文，使用 offset=0。要获取接下来的5篇，使用 offset=5。
    """
    arxiv_wrapper.arxiv_search.offset = offset
    results = arxiv_wrapper.run(query)
    return results

# Python REPL 工具 (描述已中文化)
python_tool = PythonAstREPLTool()
python_tool.description = "一个 Python 代码执行器。当你需要进行数据处理、计算、保存文件或追踪任务进度时，可以使用此工具。输入必须是有效的 Python 代码。"

# 将我们所有强大的工具放入工具箱
tools = [search_arxiv_with_offset, python_tool]

# --- 2. 升级大脑：创建一个教 Agent 如何循环的中文 Prompt ---
system_prompt = """
你是一位非常聪明的科研助理。你的任务是理解用户的需求，并规划一系列步骤来完成它。
**你的目标:** 完整地满足用户的请求，这可能需要获取大量的论文。
**你的策略:**
1.  **规划:** 将用户的请求分解成更小、可管理的步骤。
2.  **行动:** 首先，使用 `arxiv_paginated_search` 工具，并设置 `offset=0` 来搜索第一批论文。
3.  **保存与追踪:** 使用 `python_repl` 工具来处理搜索结果，将它们保存到一个列表或文件中（比如一个csv文件中），并记录你已经收集了多少篇论文。
4.  **检查与循环:** 检查已收集的论文数量是否满足了用户的要求。
5.  **如果数量不够:** 再次调用 `arxiv_paginated_search` 工具，但这一次，**增加 `offset` 的值**（例如，如果你已经获取了5篇，下一次的 offset 就应该是5）。
6.  **重复** 第 3 至 5 步，直到目标完成。
7.  **完成:** 当所有数据都收集完毕后，然后向用户报告最终结果。
开始！
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- 3. 模型和 AgentExecutor 保持不变 ---
model = ChatOpenAI(
    model="qwen-plus-2025-09-11",
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
)

agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)

# --- 4. 定义一个中文的用户任务 ---
user_task = "请帮我查找关于 '大型语言模型在代码生成领域的最新进展' 的10篇相关论文，将它们的标题、作者和摘要总结保存到一个名为 'papers.csv' 的文件中。"

print(f"用户任务: {user_task}\n")
response = agent_executor.invoke({"input": user_task})

# --- 打印最终结果 ---
print("\n--- Agent 的最终回答 ---")
print(response["output"])