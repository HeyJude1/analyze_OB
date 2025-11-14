from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 1. 导入 Arxiv 工具
from langchain_community.tools import ArxivQueryRun
# 2. 导入正确的工具解析器
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 核心修改点 1: 重新设计系统提示 ---
# 告诉模型它的新角色：一个 ArXiv 搜索专家。
# 它的任务不再是写 Python 代码，而是生成一个精确的搜索查询字符串。
system = """
你是一个精通 ArXiv 论文检索的专家助理。
你拥有一个名为 'ArxivQueryRun' 的工具，可以用来搜索学术论文。
根据用户提供的研究主题，你的任务是生成一个最精准、最有效的搜索查询字符串，以便传递给这个工具。
只返回查询字符串，不要包含任何其他解释或文字。
"""

# --- 提示词模板保持不变的结构 ---
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{input}")
])

# --- 模型初始化保持不变 ---
model = ChatOpenAI(
    model="qwen-plus-2025-09-11", # 模型名称已更新
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.2, # 对于生成精确查询，较低的温度通常更好
    max_tokens=2048,
    top_p=0.9,
)

# --- 核心修改点 2: 更换工具 ---
# 将 PythonAstREPLTool 替换为 ArxivQueryRun 工具
tool = ArxivQueryRun()

# --- 核心修改点 3: 配置新的解析器 ---
# 解析器现在需要寻找 ArxivQueryRun 工具的输出。
# ArxivQueryRun 工具的默认名称是 "arxiv"。
# 它的输入参数是 'query'，所以解析器会提取出一个包含 'query' 键的字典。
parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

# --- 模型绑定工具的方式保持不变 ---
model_with_tool = model.bind_tools([tool])

# --- 调试函数，现在用于打印搜索查询 ---
def query_print(res):
    print("--- 即将运行的 ArXiv 搜索查询 ---")
    print(res['query'])
    print("---------------------------------")
    return res
query_debugger = RunnableLambda(query_print)

# --- 构建新的链 ---
# 流程：提示 -> 绑定工具的模型 -> 解析出工具参数 -> (可选)打印参数 -> 执行工具
arxiv_chain = prompt | model_with_tool | parser | query_debugger | tool

# --- 调用链 ---
# 用户的输入是一个高层次的研究主题
user_topic = "大型语言模型在代码生成领域的最新进展"
print(f"用户主题: {user_topic}\n")
response = arxiv_chain.invoke({"input": user_topic})

# --- 打印最终结果 ---
print("\n--- ArXiv 工具返回的结果 ---")
print(response)