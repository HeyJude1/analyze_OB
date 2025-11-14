from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv(override=True)
Qwen_API_KEY = os.getenv("DASHSCOPE_API_KEY")

#使用OpenAI的API调用Qwen模型接入langchain
chatLLM = ChatOpenAI(
    api_key=Qwen_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-07-14",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    timeout=(10,30),
    logprobs=False,
    # streaming=True,
    # stream_options={"include_usage": True}    #用来显示在流式输出中token使用情况
    
)

basic_qa_chain = chatLLM | StrOutputParser()


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}]
response = chatLLM.invoke(messages)
# print(response.model_dump_json(indent=2))
print(response)

# response_stream = chatLLM.stream(messages)
# for chunk in response_stream:
#     # chunk.content 会包含一小块文本
#     print(chunk, end="")
# print() # 换行