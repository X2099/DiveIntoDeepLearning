# -*- coding: utf-8 -*-
"""
@File    : 3.LangChain Agent 原理剖析与实战：理解与驾驭 ReAct 智能体.py
@Time    : 2025/5/9 11:19
@Desc    : 
"""
import os
from pprint import pprint

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType

# 初始化搜索工具
search = DuckDuckGoSearchRun()
tools = [search]

# 初始化语言模型（可替换为 OpenAI 或 DeepSeek 等）
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base='https://api.deepseek.com',
    max_tokens=1024
)

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# 发起查询
response = agent.invoke({
    "input": "能不能介绍几个可以在线试用的 AI Agent 示例网站？请用中文回答。"
})
pprint(response)
