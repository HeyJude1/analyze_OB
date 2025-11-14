from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import json

load_dotenv(override=True)
Qwen_api_key = os.getenv("DASHSCOPE_API_KEY")

model = ChatOpenAI(
    model="qwen-plus-2025-07-14",
    api_key=Qwen_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.7,
    max_tokens=2048,
    top_p=0.9,
    logprobs=False
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个算子生成器，请根据以下算子名称，补充算子代码以及算子描述："),
    MessagesPlaceholder("messages")
])

chat_chain = chat_prompt | model
messages_list=[]
print("请输入exit，退出程序")
while True:
    user_query = input("👤 你：")
    if user_query.lower() in {"exit", "quit"}:
        break
    
    messages_list.append(HumanMessage(content=user_query))

    ai_reply = chat_chain.invoke({"messages":messages_list})
    print("🤖 小智：", ai_reply.content)

    messages_list.append(AIMessage(content=ai_reply.content))

    messages_list=messages_list[-50:]

# gen_chain = gen_prompt | model

# schemas=[
#     ResponseSchema(name="operator_name", description="算子名称"),
#     ResponseSchema(name="operator_code", description="算子代码"),
#     ResponseSchema(name="operator_description", description="算子描述"),
# ]

# parser = StructuredOutputParser.from_response_schemas(schemas)

# summary_prompt = PromptTemplate.from_template(
#     "\n\n请根据以下算子信息，提取关键信息，以JSON格式返回：\n\n{operator_info}\n\n{format_information}"
# )

# summary_chain = (
#     summary_prompt.partial(format_information=parser.get_format_instructions())
#     | model 
#     | parser
# )

# def gen_operator_info(operator_info):
#     print("算子信息：", operator_info)
#     return operator_info

# debug_node = RunnableLambda(gen_operator_info)

# # full_chain = gen_chain | summary_chain
# full_chain = gen_chain | debug_node | summary_chain 
# response = full_chain.invoke({"operator_name" : "gemm()"})

# print(json.dumps(response, indent=4, ensure_ascii=False))

