# -*- coding: utf-8 -*-
"""
@File    : vectorstore.py
@Time    : 2025/5/15 16:23
@Desc    : 
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_vectorstore_from_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore
