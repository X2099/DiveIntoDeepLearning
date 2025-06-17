# -*- coding: utf-8 -*-
"""
@File    : langsmith_demo.py
@Time    : 2025/5/26 16:57
@Desc    : 
"""
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# os.environ['LANGSMITH_TRACING'] = "True"
# os.environ['LANGCHAIN_TRACING_V2'] = 'true'

prompt = ChatPromptTemplate.from_messages([
    ("system", "您是一位得力的智能助手。请仅根据给定的上下文，使用合适的语言响应用户的请求。"),
    ("user", "问题: {question}\n上下文: {context}")
])
model = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0
)
output_parser = StrOutputParser()

chain = prompt | model | output_parser

# question = "你能总结一下今天上午的会议吗？"
# context = "在今天上午的会议上，我们解决了世界上所有的冲突。"
# result = chain.invoke({"question": question, "context": context})
# print(result)


from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name='default')
chain.invoke({"question": "我是否正在使用回调？", "context": "我正在使用回调。"}, config={"callbacks": [tracer]})

from langchain_core.tracers.context import tracing_v2_enabled

with tracing_v2_enabled(project_name='default'):
    chain.invoke({"question": "我是否正在使用上下文管理器？", "context": "我正在使用上下文管理器。"})
