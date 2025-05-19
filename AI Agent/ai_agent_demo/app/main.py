# -*- coding: utf-8 -*-
"""
@File    : main.py
@Time    : 2025/5/15 16:28
@Desc    : 
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from agents.base_agent import build_agent

app = FastAPI()
agent = build_agent()


class Query(BaseModel):
    query: str


@app.post("/chat")
async def chat(q: Query):
    response = agent.invoke(q.query)
    return {"response": response}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    path = f"temp_{file.filename}"
    with open(path, "wb") as f:
        f.write(content)
    response = agent.invoke(f"请阅读 {path} 文件内容")
    return {"response": response}
