from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_experimental.tools import PythonAstREPLTool
from langchain_community.utilities import ArxivAPIWrapper
from dotenv import load_dotenv
import os
import pandas as pd
import arxiv
from datetime import datetime, timedelta
from typing import Optional

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 升级工具：构建一个支持高级搜索的 ArXiv 工具 ---

# 升级 Pydantic 模型，包含分类和日期
class AdvancedArxivSearchInput(BaseModel):
    keywords: str = Field(description="核心搜索关键词，例如 'large language models'。")
    category: Optional[str] = Field(None, description="ArXiv 的分类代码，例如 'cs.AI' 代表人工智能, 'cs.CL' 代表计算与语言。")
    start_date: Optional[str] = Field(None, description="搜索的开始日期，格式为 YYYY-MM-DD。")
    end_date: Optional[str] = Field(None, description="搜索的结束日期，格式为 YYYY-MM-DD。")

# 配置好的底层 Wrapper
scout_arxiv_wrapper = ArxivAPIWrapper(top_k_results=20, load_max_docs=20, doc_content_chars_max=1)

# 创建一个新的、更强大的自定义工具
@tool("advanced_arxiv_search", args_schema=AdvancedArxivSearchInput)
def advanced_search_arxiv(keywords: str, category: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """
    一个高级 ArXiv 搜索工具，支持按关键词、分类和提交日期范围进行搜索。
    使用此工具来执行精确的、多条件的论文查找。
    """
    # 动态构建查询字符串
    query_parts = [keywords]
    if category:
        query_parts.append(f"cat:{category}")
    if start_date and end_date:
        # 将 YYYY-MM-DD 转换为 ArXiv API 需要的 YYYYMMDD 格式
        start = start_date.replace('-', '')
        end = end_date.replace('-', '')
        query_parts.append(f"submittedDate:[{start} TO {end}]")
    
    final_query = " AND ".join(query_parts)
    print(f"--- 最终生成的 ArXiv 查询语句: {final_query} ---")
    
    try:
        return scout_arxiv_wrapper.run(final_query)
    except arxiv.UnexpectedEmptyPageError:
        return "已到达搜索结果的末尾，没有更多论文了。"
    except Exception as e:
        return f"搜索 ArXiv 时发生了一个未知错误: {e}"

python_tool_scout = PythonAstREPLTool()
python_tool_scout.description = "一个 Python 代码执行器。用于解析侦察工具返回的文本，并将其保存为一个名为 'paper_list.csv' 的初始文件。"
scout_tools = [advanced_search_arxiv, python_tool_scout]


# --- 2. 升级大脑 (Prompt)，教它使用新工具 ---
scout_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
你是一个高效的“侦察兵”，任务是根据用户的复杂需求，使用高级工具收集情报。

你拥有一个名为 `advanced_arxiv_search` 的强大工具，它有以下参数：
- `keywords`: 核心关键词。
- `category`: ArXiv 的分类代码。常见的计算机科学分类有: `cs.AI` (人工智能), `cs.CL` (计算与语言), `cs.CV` (计算机视觉), `cs.LG` (机器学习), `cs.RO` (机器人学)。
- `start_date`: 开始日期 (YYYY-MM-DD)。
- `end_date`: 结束日期 (YYYY-MM-DD)。

你的任务是：
1.  从用户的请求中解析出所有这些参数。如果用户提到“上个月”或“最近一周”，你需要自己计算出对应的日期。
2.  调用 `advanced_arxiv_search` 工具。
3.  使用 `python_repl` 工具将返回的结果保存到 'paper_list.csv' 文件中。
**至关重要**: CSV 文件必须包含 `entry_id`, `title`, `authors` 这三列。
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# ... (深耕 Agent 的部分保持不变) ...
# --- 为了完整性，这里是完整的代码结构 ---
# Summarizer Tools
deep_dive_arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, load_max_docs=1, doc_content_chars_max=10000)
@tool("summarize_and_update_csv")
def summarize_and_update_csv(csv_path: str) -> str:
    """... (函数内容同前) ..."""
    # ... (此处省略函数体)
    try:
        df = pd.read_csv(csv_path)
        if 'entry_id' not in df.columns: return "错误：CSV 文件中缺少 'entry_id' 列。"
        df = df.astype(str).replace('nan', '')
        summaries = []
        for entry_id in df['entry_id']:
            query = str(entry_id).strip()
            if not query or query == 'nan':
                summaries.append("无法获取摘要：无效的ArXiv ID")
                continue
            try:
                result_str = deep_dive_arxiv_wrapper.run(query)
                summary = result_str.split("Summary: ")[-1] if "Summary: " in result_str else "摘要未找到。"
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
summarizer_prompt = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{input}"), ("placeholder", "{agent_scratchpad}")]) # 省略

# --- 3. 初始化模型和 Agent ---
model = ChatOpenAI(model="qwen-plus", api_key=qwen_api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", temperature=0.1)
scout_agent = create_openai_tools_agent(llm=model, tools=scout_tools, prompt=scout_prompt)
scout_executor = AgentExecutor(agent=scout_agent, tools=scout_tools, verbose=True, max_iterations=5)
summarizer_agent = create_openai_tools_agent(llm=model, tools=summarizer_tools, prompt=summarizer_prompt)
summarizer_executor = AgentExecutor(agent=summarizer_agent, tools=summarizer_tools, verbose=True, max_iterations=5)

# --- 4. 执行两阶段工作流，使用一个更复杂的任务 ---
# 动态计算上个月的日期范围
today = datetime.now()
first_day_of_this_month = today.replace(day=1)
last_day_of_last_month = first_day_of_this_month - timedelta(days=1)
first_day_of_last_month = last_day_of_last_month.replace(day=1)
start_date_str = first_day_of_last_month.strftime('%Y-%m-%d')
end_date_str = last_day_of_last_month.strftime('%Y-%m-%d')

scout_task = f"请帮我查找上个月（{start_date_str} 到 {end_date_str}）在 ArXiv 的'计算机科学-人工智能 (cs.AI)'分类下，关于 '大型语言模型 (Large Language Models)' 的最多20篇论文，并将它们的 ArXiv ID, 标题和作者保存到 'paper_list.csv' 文件中。"

print(f"--- 开始阶段一：侦察任务 ---\n任务: {scout_task}\n")
scout_response = scout_executor.invoke({"input": scout_task})
print(f"\n--- 侦察任务完成 ---\nAgent 回答: {scout_response['output']}\n")

if os.path.exists('paper_list.csv'):
    summarizer_task = "现在，请为 'paper_list.csv' 文件中的所有论文添加摘要。"
    print(f"--- 开始阶段二：深耕任务 ---\n任务: {summarizer_task}\n")
    summarizer_response = summarizer_executor.invoke({"input": summarizer_task})
    print(f"\n--- 深耕任务完成 ---\nAgent 回答: {summarizer_response['output']}")
else:
    print("侦察阶段未能成功创建 'paper_list.csv'，深耕任务跳过。")