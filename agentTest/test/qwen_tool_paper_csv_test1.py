# 基础的实现保存2篇论文的功能

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 1. 导入 Agent 需要的核心组件
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools import ArxivQueryRun
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.utilities import ArxivAPIWrapper
from dotenv import load_dotenv
import os

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 创建并正确配置我们的工具箱 ---

# Arxiv 工具
arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=2, 
    load_max_docs=2, 
    ARXIV_MAX_QUERY_LENGTH = 300,
    doc_content_chars_max=4000
)

arxiv_tool = ArxivQueryRun(
    name="arxiv_search", # 给工具一个明确的英文名
    api_wrapper=arxiv_wrapper,
    description="当需要从 ArXiv.org 查找关于科学、技术等领域的学术论文时使用此工具。输入应该是一个精确的搜索查询字符串。"
)

# Python REPL 工具
# 我们需要让它能够访问 Arxiv 工具的输出，所以 locals 是动态的
python_tool = PythonAstREPLTool()
# 我们可以给这个工具也加上描述，让 Agent 知道它的用途
python_tool.description = "Use this tool to execute Python code. You can use it for data analysis, calculations, or file operations. The input must be valid Python code."


# 将所有工具放入一个列表
tools = [arxiv_tool, python_tool]

# --- 2. 创建一个为 Agent 优化的 Prompt ---
# 这个 Prompt 模板是 LangChain 推荐的、专门用于工具调用的标准模板
# 它包含了 'input' 和一个特殊的 'agent_scratchpad' 变量
# 'agent_scratchpad' 会被 AgentExecutor 自动填充，用来记录 Agent 的思考过程和工具调用历史
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a powerful research assistant. You have access to a suite of tools. Use them to answer the user's questions."),
        ("human", "{input}"),
        # MessagesPlaceholder 是 Agent 思考过程的“记忆”
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- 3. 模型初始化保持不变 ---
model = ChatOpenAI(
    model="qwen-plus-2025-09-11",
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.3, # Agent 模式下，通常使用更低的温度以保证决策的稳定性
)

# --- 4. 创建 Agent 的“大脑” ---
# create_openai_tools_agent 会将模型、工具和提示组合成一个 Runnable
# 这个 Runnable 负责根据输入和历史，决定下一步是调用工具还是直接回答
agent = create_openai_tools_agent(llm=model, tools=tools, prompt=prompt)

# --- 5. 创建 Agent 的“运行时” ---
# AgentExecutor 是驱动 Agent 运行的循环
# 它会反复调用 agent (大脑)，执行工具，并将结果反馈给 agent，直到任务完成
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- 6. 调用 AgentExecutor ---
user_task = "请帮我查找关于 '大型语言模型在代码生成领域的最新进展' 的论文，将结果保存到 'papers.csv' 文件中，并且确保该csv文件的格式正确，然后告诉我该文件中保存了几个搜索结果，最后告诉我这个文件的第一个结果的论文名称是什么。"

print(f"用户任务: {user_task}\n")
response = agent_executor.invoke({"input": user_task})

# --- 打印最终结果 ---
print("\n--- Agent 的最终回答 ---")
print(response["output"])