# -*- coding: utf-8 -*-
"""
@File    : 4.提示工程的艺术——如何“教会”你的 Agent 更聪明.py
@Time    : 2025/5/9 14:52
@Desc    : 
"""
import os
from pprint import pprint

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate

# 搜索工具实例
search = DuckDuckGoSearchRun()
tools = [search]

API_KEY = os.getenv('DEEPSEEK_API_KEY')

# 初始化语言模型
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=API_KEY,
    openai_api_base='https://api.deepseek.com/v1'
)

# 自定义 Prompt 模板
custom_prompt = PromptTemplate.from_template("""
你是一个专业的中文技术助手，擅长使用工具搜索并总结信息。
请执行以下任务：

任务：{user_task}

要求：
1. 用中文回答；
2. 每个项目简要介绍功能和特点；
3. 如果找不到信息，也请说明原因；
4. 遇到复杂问题请多次 Thought / Action 循环后再给出 Final Answer；
5. 使用普通的文本格式输出，不需要markdown格式。

请使用如下格式回答：
Question: 任务原文  
Thought: 你的思考  
Action: 使用的工具  
Action Input: 工具的输入  
Observation: 工具返回的结果  
...（可以多次循环）  
Final Answer: 总结输出
""")

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prefix": custom_prompt.format(user_task="{input}")},  # 插入到上下文中
    handle_parsing_errors=True
)

# 调用 Agent，输入任务
response = agent.invoke({
    "input": "列举 2024 年值得关注的 AI Agent 开源项目"
})

print("\n🤖 Agent 回答：\n")
pprint(response)
