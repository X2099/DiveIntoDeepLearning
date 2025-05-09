# -*- coding: utf-8 -*-
"""
@File    : 2.构建第一个 AI Agent：LangChain 入门实战.py
@Time    : 2025/5/7 16:43
@Desc    : 
"""
import os
# 导入 DuckDuckGo 搜索工具，用于联网搜索信息
from langchain_community.tools import DuckDuckGoSearchRun
# 导入 ChatOpenAI 类，用于与兼容 OpenAI 接口的语言模型交互
from langchain_community.chat_models import ChatOpenAI
# 导入 initialize_agent 和 AgentType，用于创建 LangChain 智能体
from langchain.agents import initialize_agent, AgentType

# 实例化 DuckDuckGo 搜索工具，作为 agent 可用的工具之一
search = DuckDuckGoSearchRun()
tools = [search]  # 工具列表，目前只包含搜索工具

API_KEY = os.getenv('DEEPSEEK_API_KEY')

# 创建一个 LLM（大型语言模型）客户端，连接到 DeepSeek 的 OpenAI 兼容接口
llm = ChatOpenAI(
    model='deepseek-chat',  # 指定使用 deepseek-chat 模型
    openai_api_key=API_KEY,  # 使用你的 API 密钥进行认证
    openai_api_base='https://api.deepseek.com',  # DeepSeek 的 API 接口地址
    max_tokens=1024  # 最大生成 token 数
)

# 初始化智能体 Agent，使用 Zero-Shot ReAct 模式（零样本推理 + 工具调用）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # agent 类型：基于 ReAct 的 Zero-Shot 智能体
    verbose=True,  # 输出 agent 的推理过程，便于调试和观察
    handle_parsing_errors=True  # 容错处理，避免格式解析出错时中断
)

response = agent.invoke({"input": "请用中文回答这个问题：最近有哪些关于AI Agent的新研究？"})
print(response)
