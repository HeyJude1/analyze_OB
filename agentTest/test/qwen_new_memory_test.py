from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# 1. 导入新的、正确的摘要历史记录类
from langchain_community.chat_message_histories import ConversationSummaryBufferChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(override=True)
Qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

# --- 模型、Prompt、解析器、基础链的定义完全不变 ---
model = ChatOpenAI(
    model="qwen-plus",
    api_key=Qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫小智，是一名乐于助人的助手。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
parser = StrOutputParser()
base_chain = prompt | model | parser

# --- 2. 创建会话历史存储库 ---
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """根据 session_id 获取历史记录"""
    if session_id not in store:
        # 核心修改点：使用新的 ConversationSummaryBufferChatMessageHistory 类
        store[session_id] = ConversationSummaryBufferChatMessageHistory(
            llm=model,             # 它需要一个 LLM 来生成摘要
            max_token_limit=400,   # 当历史记录超过 400 token 时，开始进行摘要
            return_messages=True,
        )
    return store[session_id]

# --- 3. RunnableWithMessageHistory 的创建和使用完全不变 ---
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# --- 4. 调用循环完全不变 ---
print("🔹 输入 exit 结束对话")
while True:
    user_query = input("👤 你：")
    if user_query.lower() in {"exit", "quit"}:
        break
    
    config = {"configurable": {"session_id": "default_session"}}
    response = chain_with_history.invoke({"input": user_query}, config=config)
    print("🤖 小智：", response)