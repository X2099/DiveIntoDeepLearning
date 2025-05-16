# -*- coding: utf-8 -*-
"""
@File    : doc_reader.py
@Time    : 2025/5/15 15:19
@Desc    : 
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


def load_pdf_content(file_path: str) -> str:
    if not Path(file_path).exists():
        return f"文件 {file_path} 不存在"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])
