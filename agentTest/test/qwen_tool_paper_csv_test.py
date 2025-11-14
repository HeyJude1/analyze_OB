from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 1. 导入所有需要的工具
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain_experimental.tools import PythonAstREPLTool
# 2. 导入正确的工具解析器
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 建议 1: 采纳更通用的系统提示 ---
# 不再提及任何具体的工具名称，让模型自己决策
system = """
你是一个强大的研究助理。
你可以使用你拥有的工具来回答用户的问题。
请根据用户的提问，思考需要使用哪个工具，然后生成调用该工具所需的参数。
"""

# --- 提示词模板保持不变的结构 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{input}")
])

# --- 模型初始化保持不变 ---
model = ChatOpenAI(
    model="qwen-plus", # 模型名称已更新
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2,
)

# --- 建议 2 & 3: 创建一个包含多个工具的工具箱 ---
# 限制 Arxiv 返回数量为 3
arxiv_wrapper = ArxivAPIWrapper(top_k_results=5, load_max_docs=2)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    description="当需要从 ArXiv.org 查找关于科学、技术等领域的学术论文时使用此工具。输入应该是一个精确的搜索查询字符串。"
)

# Python REPL 工具，用于数据分析和文件操作
python_tool = PythonAstREPLTool()

# 将所有工具放入一个列表
tools = [arxiv_tool, python_tool]

# --- 模型绑定整个工具箱 ---
model_with_tools = model.bind_tools(tools)

# --- 构建 Agent 链 ---
# 这个链现在能处理两种工具的调用

# 这是一个更通用的解析器，它会解析出模型调用的任何工具的参数
# 我们这里为了演示，仍然可以针对性地处理，但更复杂的 Agent 会有路由逻辑
# 这里的简化逻辑是：我们期望模型先调用 arxiv，再调用 python
arxiv_parser = JsonOutputKeyToolsParser(key_name=arxiv_tool.name, first_tool_only=True)
python_parser = JsonOutputKeyToolsParser(key_name=python_tool.name, first_tool_only=True)

# 简单的路由逻辑：先搜论文，然后处理结果
def process_arxiv_results(summaries: str):
    """一个自定义函数，接收 Arxiv 的摘要，生成 Python 代码来保存和分析"""
    print("\n--- ArXiv 搜索完成，正在生成 Python 代码 ---")
    
    # 将论文摘要格式化，以便写入 Python 字符串
    # 使用三引号来处理多行文本
    formatted_summaries = summaries.replace('"', '""')

    # 生成用于创建 DataFrame 并保存为 CSV 的 Python 代码
    code_to_run = f"""
import pandas as pd
import io

data = \"\"\"{formatted_summaries}\"\"\"

# 简单的解析逻辑：按 "Published" 分割论文
papers = data.strip().split('Published: ')[1:]
parsed_data = []
for paper in papers:
    lines = paper.split('\\n')
    published = lines[0].strip()
    title = lines[1].replace('Title: ', '').strip()
    authors = lines[2].replace('Authors: ', '').strip()
    summary = ' '.join([l.replace('Summary: ', '').strip() for l in lines[3:]])
    parsed_data.append({{
        'published': published,
        'title': title,
        'authors': authors,
        'summary': summary
    }})

df = pd.DataFrame(parsed_data)
df.to_csv('arxiv_results.csv', index=False)
print("论文已成功保存到 arxiv_results.csv")
print("以下是数据的前2行摘要统计：")
print(df[['published', 'title']].head(2).to_markdown())
"""
    return code_to_run

# --- 构建新的、更强大的复合链 ---

# 链的第一部分：从用户输入到 Arxiv 搜索
arxiv_search_chain = prompt | model_with_tools | arxiv_parser | arxiv_tool

# 链的第二部分：处理 Arxiv 结果，生成并执行 Python 代码
results_csv = RunnableLambda(process_arxiv_results)
analysis_chain = results_csv | python_tool

# 最终的完整链
full_chain = arxiv_search_chain | analysis_chain

# --- 调用链 ---
user_topic = "大型语言模型在代码生成领域的最新进展"
print(f"用户主题: {user_topic}\n")
response = full_chain.invoke({"input": user_topic})

# --- 打印最终结果 ---
# 最终结果是 Python REPL 工具的执行输出
print("\n--- Python REPL 工具的最终执行输出 ---")
print(response)