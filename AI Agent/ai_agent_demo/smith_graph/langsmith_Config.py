# -*- coding: utf-8 -*-
"""
@File    : langsmith_Config.py
@Time    : 2025/5/27 14:23
@Desc    : 
"""
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ['LANGCHAIN_TRACING_V2'] = 'true'

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个乐于助人的人工智能。"),
    ("user", "{input}")
])

# 标签“model-tag”和元数据 {“model-key”：“model-value”} 将仅附加到 ChatOpenAI 运行
chat_model = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base='https://api.deepseek.com/v1',
    temperature=0
).with_config({"tags": ["model-tag"], "metadata": {"model-key": "model-value"}})
output_parser = StrOutputParser()

# 可以使用 RunnableConfig 配置标签和元数据
chain = (prompt | chat_model | output_parser).with_config(
    {"tags": ["config-tag"], "metadata": {"config-key": "config-value"}})

# # 标签和元数据也可以在运行时传递
# chain.invoke({"input": "生命的意义是什么？"},
#              {"tags": ["invoke-tag"], "metadata": {"invoke-key": "invoke-value"}})

# # 在 LangChain 内跟踪时，运行名称默认为被跟踪对象的类名（例如，“ChatOpenAI”）。
# configured_chain = chain.with_config({"run_name": "MyCustomChain"})
# configured_chain.invoke({"input": "生命的意义是什么？"})
#
# # 您还可以在调用时配置运行名称，如下所示
# chain.invoke({"input": "生命的意义是什么？"}, {"run_name": "MyCustomChain"})

import uuid

my_uuid = uuid.uuid4()
# You can configure the run ID at invocation time:
chain.invoke({"input": "What is the meaning of life?"}, {"run_id": my_uuid})
