# openai_langchain_app.py

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- 1. 检查API密钥 ---
# 确保OpenAI的API密钥已经通过环境变量设置好了
# if "OPENAI_API_KEY" not in os.environ:
#     print("错误：请先设置环境变量 OPENAI_API_KEY")
#     # 可以在这里提供一个输入提示，但不推荐在生产代码中这样做
#     # os.environ["OPENAI_API_KEY"] = input("请输入您的OpenAI API密钥：")
#     exit()

OPENAI_API_KEY = os.getenv("LANGSMITH_API_KEY")

# --- 2. 初始化OpenAI模型 ---
# 从 langchain_openai 导入并实例化 ChatOpenAI 类。
# 这是与OpenAI聊天模型（如gpt-3.5-turbo, gpt-4）交互的核心组件。
# 我们可以指定模型名称、温度（控制创造性）等参数。
print("正在初始化OpenAI模型...")
llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0.7,
    max_tokens=2048,
    timeout=30,
    max_retries=2,
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
print("模型初始化完成。")


# --- 3. 创建提示词模板 (Prompt Template) ---
# 从 langchain_core.prompts 导入 ChatPromptTemplate。
# 提示词模板用于将用户的输入动态地包装成一个完整的、结构化的提示。
# 这里我们定义了一个简单的模板，它会接收一个名为 'question' 的变量。
prompt = ChatPromptTemplate.from_template("请回答以下问题：{question}")


# --- 4. 创建输出解析器 (Output Parser) ---
# 从 langchain_core.output_parsers 导入 StrOutputParser。
# 模型返回的通常是一个包含元数据等信息的复杂对象（AIMessage）。
# 这个解析器可以方便地将其中的内容提取为简单的字符串。
output_parser = StrOutputParser()


# --- 5. 构建链 (Chain) ---
# 使用LangChain表达式语言（LCEL）的管道符号 `|` 将各个组件连接起来。
# 这是一个标准的、现代的LangChain工作流：
# 1. 输入首先进入 `prompt` 进行格式化。
# 2. 格式化后的提示被发送给 `model`。
# 3. 模型的输出结果交给 `output_parser` 进行解析。
print("正在构建处理链...")
chain = prompt | model | output_parser
print("处理链构建完成。")


# --- 6. 运行链并获取结果 ---
# 使用 .invoke() 方法来执行链。
# 我们传入一个字典，其键是在提示模板中定义的变量名 'question'。
print("\n正在向OpenAI提问...")
user_question = "LangChain是什么？它有什么主要用途？"
response = chain.invoke({"question": user_question})


# --- 7. 打印结果 ---
print("="*50)
print(f"问题: {user_question}")
print("-"*50)
print("模型的回答:")
print(response)
print("="*50)