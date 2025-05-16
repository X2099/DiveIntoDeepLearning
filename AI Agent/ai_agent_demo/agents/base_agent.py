# -*- coding: utf-8 -*-
"""
@File    : base_agent.py
@Time    : 2025/5/15 15:13
@Desc    : 
"""
import os

from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from memory.memory import memory
from tools.search_tool import search_web
from tools.doc_reader import load_pdf_content
from tools.vectorstore import build_vectorstore_from_pdf
from multimodal.image_captioning import caption_image

# 构建文档向量检索器
vectorstore = build_vectorstore_from_pdf("data/GB+38031-2025.pdf")
retriever = vectorstore.as_retriever()


def retrieve_doc(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


def build_agent():
    tools = [
        Tool(name="DuckDuckGo Search", func=search_web, description="用于在线搜索最新信息"),
        Tool(name="PDF Reader", func=load_pdf_content, description="适合读取本地PDF文档内容"),
        Tool(name="PDF Semantic Search", func=retrieve_doc, description="对PDF内容进行语义检索"),
        Tool(name="Image Captioning", func=caption_image, description="根据图像生成文字描述")
    ]
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
        openai_api_base='https://api.deepseek.com/v1',
        temperature=0
    )
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template="""
    你是一个多工具智能体，具备搜索、文档读取、图像识别和文档语义检索能力。
    请根据用户的问题选择合适的工具，并给出准确简洁的答案。
    回答问题时使用中文，直接输出纯文本，不要任何格式。

    你可以调用的工具有：
    - DuckDuckGo Search：用于搜索网络信息
    - PDF Reader：用于获取指定本地PDF的全部内容
    - PDF Semantic Search：用于对PDF进行基于语义的问答
    - Image Captioning：根据上传的图像生成文字描述

    请尽量以自然语言回答问题，如果需要使用工具，请先思考，再行动。

    问题：{input}
    {agent_scratchpad}
    """
    )
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent
