from langchain_community.embeddings import DashScopeEmbeddings
from dotenv import load_dotenv
import os

load_dotenv(override=True)
Qwen_API_KEY = os.getenv("DASHSCOPE_API_KEY")

embeddings = DashScopeEmbeddings(
    model="text-embedding-v2",
    dashscope_api_key=Qwen_API_KEY,
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # other params...
)

text = "This is a test document."

query_result = embeddings.embed_query(text)
print(query_result)
print("文本向量长度：", len(query_result), sep='')

doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ])
print("文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep='')