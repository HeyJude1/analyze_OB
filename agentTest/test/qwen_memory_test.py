from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
# 1. 导入新的记忆管理核心
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(override=True)
Qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 1. 创建一个简单的“会话历史存储库” ---
# 我们可以用一个简单的字典来模拟，为不同的会话ID存储历史
# 在这个单用户脚本中，我们只会有一个 "default" 会话
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取历史记录"""
    if session_id not in store:
        print(f"创建新的会话历史记录: {session_id}")
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- 2. 创建基础链（不包含记忆） ---
# 这个链的结构和你之前写的很像
prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫小智，是一名乐于助人的助手。"),
    MessagesPlaceholder("history"), # 历史占位符
    ("human", "{input}") # 用户当前输入
])

model = ChatOpenAI(
    model="qwen-plus", # 模型名称已更新
    api_key=Qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)

parser = StrOutputParser()

# 基础链
base_chain = prompt | model | parser

# --- 3. 使用 RunnableWithMessageHistory 包装基础链，为其“注入”记忆功能 ---
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",      # 告诉它，用户的输入在哪个键里
    history_messages_key="history",  # 告诉它，历史记录应该填充到哪个 MessagesPlaceholder
)

# --- 4. 修改调用循环 ---
print("🔹 输入 exit 结束对话")
while True:
    user_query = input("👤 你：")
    if user_query.lower() in {"exit", "quit"}:
        break
    
    # 调用时，需要提供一个 config，其中包含 session_id
    # 这让记忆系统知道该为哪个对话存储历史
    config = {"configurable": {"session_id": "default_session"}}
    
    # 将用户输入包装在 "input" 键中
    response = chain_with_history.invoke({"input": user_query}, config=config)
    print("🤖 小智：", response)

print()