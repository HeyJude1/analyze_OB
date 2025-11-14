# import numpy as np
# import pandas as pd

# dataset = pd.read_csv("archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# pd.set_option('max_colwidth',200)
# dataset.head(5)
# dataset.info()

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableLambda
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv(override=True)
qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

system = f"""
你可以访问一个名为 `df` 的 pandas 数据框，你可以使用df.head().to_markdown() 查看数据集的基本信息， \
请根据用户提出的问题，编写 Python 代码来回答。只返回代码，不返回其他内容。只允许使用 pandas 和内置库。
"""

# 使用from_messages，工具调用就失败
# prompt = ChatPromptTemplate.from_messages([
#     SystemMessage(content=system),
#     HumanMessage(content="{input}")
# ])

# prompt = ChatPromptTemplate([
#     SystemMessage(content=system),
#     HumanMessage(content="{input}")
# ])

# 使用from_template，工具调用就成功？
prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{input}")
])

model = ChatOpenAI(
    model="qwen-plus-2025-09-11",
    api_key=qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
)

df = pd.read_csv("archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
tool = PythonAstREPLTool(locals={"df": df})
# result = tool.invoke("df['SeniorCitizen'].mean()")
# print(result)
# df['MonthlyCharges'].mean()

parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

model_with_tool = model.bind_tools([tool])

def code_print(res):
    print("即将运行Python代码:", res['query'])
    return res
code = RunnableLambda(code_print)

chat_chain = prompt | model_with_tool | parser | code | tool

# response = chat_chain.invoke({"input": "请问MonthlyCharges取值最高的用户ID是？"})
response = chat_chain.invoke({"input": "请帮我分析gender、SeniorCitizen和Churn三个字段之间的相关关系。"})
# response = chat_chain.invoke({"input": "请问MonthlyCharges取值最高的用户ID是？"})

# print(response.model_dump_json(indent=4))
print(response)