# -*- coding: utf-8 -*-
"""
@File    : 5.RAG 技术详解：让 Agent 学会读文档.py
@Time    : 2025/5/12 10:31
@Desc    : 
"""
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1. 加载文档
loader = PyPDFLoader("GB+38031-2025.pdf")
pages = loader.load()

# 2. 文本切分
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(pages)

# 3. 构建向量数据库

# embedding_model = OpenAIEmbeddings()

# 创建中文 Embedding 模型（bge-small-zh）
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh",
    model_kwargs={"device": "cpu"},  # 如果你有 GPU，可改为 "cuda"
    encode_kwargs={"normalize_embeddings": True}  # 官方推荐开启
)

db = FAISS.from_documents(docs, embedding_model)

# retriever = db.as_retriever()
# results = retriever.get_relevant_documents("文档中提到的某个关键词")
#
# for i, doc in enumerate(results):
#     print(f"匹配结果 {i+1}: {doc.page_content}")


# 4. 初始化语言模型
API_KEY = os.getenv('DEEPSEEK_API_KEY')
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=API_KEY,
    openai_api_base='https://api.deepseek.com/v1'
)

# 5. 构建 RAG 问答链
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True
)

# 6. 提问
query = "请用中文回答：这份文档的核心内容有哪些？"
response = rag_chain.invoke({"query": query})
print(response["result"])
