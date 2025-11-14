from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
# 1. 核心修正：直接从 pydantic 导入，解决弃用警告
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.utilities import ArxivAPIWrapper
from dotenv import load_dotenv
import os
import pandas as pd
# 导入 arxiv 库的特定错误类型，以便精确捕获
import arxiv

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 创建工具箱 ---

class ArxivSearchInput(BaseModel):
    query: str = Field(description="用于 ArXiv 搜索的查询字符串。")


# 我们请求20条，这是我们的目标
scout_arxiv_wrapper = ArxivAPIWrapper(top_k_results=20, load_max_docs=20, doc_content_chars_max=4000)

# 2. 核心修正：为工具添加强大的错误处理
@tool("arxiv_scout_search", args_schema=ArxivSearchInput)
def scout_search_arxiv(query: str) -> str:
    """
    用于初步侦察，快速搜索 ArXiv 以获取论文的元数据列表（标题、作者、ID）。
    它不会返回详细的摘要。输入应该是一个精确的搜索查询字符串。
    """
    try:
        results = scout_arxiv_wrapper.run(query)
        return results
    except arxiv.UnexpectedEmptyPageError:
        # 当没有更多结果时，返回一个明确的信息给 Agent
        return "已到达搜索结果的末尾，没有更多论文了。"
    except Exception as e:
        # 捕获其他所有可能的错误
        return f"搜索 ArXiv 时发生了一个未知错误: {e}"

python_tool_scout = PythonAstREPLTool()
python_tool_scout.description = "一个 Python 代码执行器。用于解析侦察工具返回的文本，并将其保存为一个名为 'paper_list.csv' 的初始文件。"
scout_tools = [scout_search_arxiv, python_tool_scout]

# 深耕 Agent 的工具保持不变
# ... (summarize_and_update_csv 工具的代码) ...

# --- 2. 升级大脑 (Prompt) ---
# 3. 核心修正：让 Prompt 更智能，能理解不完整的结果

# scout_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个高效的“侦察兵”，任务是快速收集情报。使用你的工具找到论文的ID、标题和作者，然后用Python工具将它们保存到 'paper_list.csv' 文件中。注意：如果搜索到的论文数量少于用户的要求，请直接处理你已有的结果，不要再次搜索。"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

# --- 核心修改点：在 scout_prompt 中明确指定 CSV 的列名 ---
scout_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
你是一个高效的“侦察兵”，任务是快速收集情报。
使用你的工具找到论文，然后用Python工具将它们保存到 'paper_list.csv' 文件中。
**至关重要**: 在生成 CSV 文件时，必须包含以下三列，并且列名必须完全一样：
1. `entry_id` (这是 ArXiv 的唯一ID)
2. `title` (论文标题)
3. `authors` (作者列表)

不要去获取摘要。如果搜索到的论文数量少于用户的要求，请直接处理你已有的结果。
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# ... (Summarizer Agent 的 Prompt 和其他组件保持不变) ...
summarizer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个勤奋的“研究员”，任务是完成数据整理。你的桌上有一个叫 'paper_list.csv' 的文件。使用你的工具为文件中的每一篇论文补全摘要。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
# --- 3. 初始化模型和两个 Agent ---
model = ChatOpenAI(
    model="qwen-plus-2025-09-11", 
    api_key=qwen_api_key, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    temperature=0.3
)

# 创建侦察 Agent
scout_agent = create_openai_tools_agent(llm=model, tools=scout_tools, prompt=scout_prompt)
scout_executor = AgentExecutor(agent=scout_agent, tools=scout_tools, verbose=True, max_iterations=5)

# 创建深耕 Agent (为了完整性，这里也包含了深耕部分的代码)
deep_dive_arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, load_max_docs=1, doc_content_chars_max=10000)
@tool("summarize_and_update_csv")
def summarize_and_update_csv(csv_path: str) -> str:
    """为CSV文件中的论文添加摘要"""
    try:
        df = pd.read_csv(csv_path)
        if 'entry_id' not in df.columns: return "错误：CSV 文件中缺少 'entry_id' 列。"
        
        # 数据清洗：确保所有数据都是字符串类型
        df = df.astype(str)
        df = df.replace('nan', '')
        
        summaries = []
        for entry_id in df['entry_id']:
            # 确保entry_id是字符串并清理空白
            query = str(entry_id).strip()
            if not query or query == 'nan':
                summaries.append("无法获取摘要：无效的ArXiv ID")
                continue
                
            try:
                result_str = deep_dive_arxiv_wrapper.run(query)
                summary = result_str.split("Summary: ")[-1]
                summaries.append(summary)
                print(f"已获取 ID: {entry_id} 的摘要...")
            except Exception as e:
                summaries.append(f"获取摘要失败: {str(e)}")
                print(f"获取 ID: {entry_id} 的摘要失败: {e}")
                
        df['summary'] = summaries
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return f"任务完成！已成功为 {len(df)} 篇论文添加摘要并更新了 {csv_path} 文件。"
    except Exception as e: return f"处理文件时发生错误: {e}"

summarizer_tools = [summarize_and_update_csv]
summarizer_agent = create_openai_tools_agent(llm=model, tools=summarizer_tools, prompt=summarizer_prompt)
summarizer_executor = AgentExecutor(agent=summarizer_agent, tools=summarizer_tools, verbose=True, max_iterations=5)

# --- 4. 执行两阶段工作流 ---
scout_task = "请帮我查找关于 'Multimodal Large Language Models' 的20篇相关论文，并将它们的 ArXiv ID, 标题和作者保存到 'paper_list.csv' 文件中。"
print(f"--- 开始阶段一：侦察任务 ---\n任务: {scout_task}\n")
scout_response = scout_executor.invoke({"input": scout_task})
print(f"\n--- 侦察任务完成 ---\nAgent 回答: {scout_response['output']}\n")

# 检查侦察是否成功创建了文件
if os.path.exists('paper_list.csv'):
    summarizer_task = "现在，请为 'paper_list.csv' 文件中的所有论文添加摘要。"
    print(f"--- 开始阶段二：深耕任务 ---\n任务: {summarizer_task}\n")
    summarizer_response = summarizer_executor.invoke({"input": summarizer_task})
    print(f"\n--- 深耕任务完成 ---\nAgent 回答: {summarizer_response['output']}")
else:
    print("侦察阶段未能成功创建 'paper_list.csv'，深耕任务跳过。")