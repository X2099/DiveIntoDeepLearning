# -*- coding: utf-8 -*-
"""
@File    : 6.LangChain 框架下的 AI Agent 构建与部署实践.py
@Time    : 2025/5/12 16:36
@Desc    : 
"""
import os
from langchain_community.chat_models import ChatOpenAI

API_KEY = os.getenv('DEEPSEEK_API_KEY')

llm = ChatOpenAI(
    model='deepseek-chat',  # 指定使用的模型名称，如 gpt-4、deepseek-chat 等
    openai_api_key=API_KEY,  # 设置 API 密钥
    openai_api_base='https://api.deepseek.com/v1',  # 自定义模型地址（如 DeepSeek）
    temperature=0.7,  # 控制输出的随机性，越低越保守，越高越发散
    max_tokens=1024,  # 设置生成回复的最大 token 数
    model_kwargs={  # 额外的模型参数（可选）
        "top_p": 1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    },
    request_timeout=60,  # 请求超时时间（秒）
    streaming=False,  # 是否使用流式响应
    verbose=False  # 是否打印调试信息
)

response = llm.invoke("请简要说明“量子计算”和“经典计算”的主要区别。")
print(response)

from langchain_community.embeddings import HuggingFaceEmbeddings

# 初始化一个中文 BGE 嵌入模型，用于将文本转换为向量表示
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",  # 使用 BAAI 提供的 bge-small-zh 中文语义嵌入模型
    model_kwargs={"device": "cpu"},  # 指定运行设备为 CPU，如有 GPU 可改为 "cuda"
    encode_kwargs={"normalize_embeddings": True}  # 对输出的向量进行归一化，有助于相似度计算
)

from langchain_community.document_loaders import TextLoader

loader = TextLoader('./docs/智能体纪元第1篇.md', encoding='utf-8')
documents = loader.load()

from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader('./docs', glob='**/*.md', loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
documents = loader.load()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./docs/GB+38031-2025.pdf')
documents = loader.load()
documents

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
)
split_docs = text_splitter.split_documents(documents)

from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(split_docs, embed_model)

# 保存本地索引以供后续检索调用
vectorstore.save_local("faiss_index")

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

query = "电动汽车用动力蓄电池按照8.2.1进行振动试验后应达到什么要求？"
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs, 1):
    print(f"[文档 {i}]: {doc.page_content}\n")

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("最近一次 SpaceX 火箭发射是什么时候？成功了吗？")

from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "你是一个智能问答助手。你可以使用以下工具：\n{tools}\n"
    "请根据用户问题和工具结果给出答案。\n\n"
    "使用中文纯文本输出答案。\n\n"
    "用户问题: {input}\n"
    "{agent_scratchpad}"
)

from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.REACT_DOCSTORE,  # 或 AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION
    memory=memory,
    verbose=True,
    agent_kwargs={"prefix": prompt.format(user_task="{input}")}  # 插入到上下文中
)

from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/chat")
async def chat_api(request: Request):
    data = await request.json()
    query = data.get("query", "")
    # 调用 AgentExecutor 得到回答
    result = agent_executor.invoke({"input": query})
    answer = result["output"]
    return {"answer": answer}
