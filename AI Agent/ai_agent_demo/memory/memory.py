# -*- coding: utf-8 -*-
"""
@File    : memory.py
@Time    : 2025/5/15 15:14
@Desc    : 
"""
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")
