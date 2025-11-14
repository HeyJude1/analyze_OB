from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv(override=True)
Qwen_API_KEY = os.getenv("DASHSCOPE_API_KEY")

run_metadata = {
    "user_name": "mjs",
    "request_source": "qwen_test1",
    "prompt_template_version": 1.0
}

chatLLM = ChatTongyi(
    model="qwen-plus-2025-07-14",   # 此处以qwen-max为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    api_key=Qwen_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming=False,
    temperature=0.9,
    max_tokens=2048,
    top_p=0.9,
    timeout=30,
    verbose=True,
)
# messages = [
#     SystemMessage(content="You are a helpful assistant.", response_metadata={"logprobs": True}),
#     HumanMessage(content="你是谁？")
# ]

messages = [
    ("system", "你是一名专业的翻译家，可以将用户的中文翻译为英文。"),
    ("human", "我喜欢编程。")
]


chatLLM.invoke(messages,config={"metadata": run_metadata},stop=["pro"])
# print(chatLLM.invoke(messages))
print(chatLLM.invoke(messages).model_dump_json(indent=2))

# response = chatLLM.invoke(messages)
# print(response.model_dump_json(indent=2))
# print(chatLLM.invoke(messages))

