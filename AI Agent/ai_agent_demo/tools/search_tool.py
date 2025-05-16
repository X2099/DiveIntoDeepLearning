# -*- coding: utf-8 -*-
"""
@File    : search_tool.py
@Time    : 2025/5/15 15:13
@Desc    : 
"""
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()


def search_web(query: str) -> str:
    try:
        result = search(query)
    except DuckDuckGoSearchException:
        result = "搜索引擎异常，没有搜到任何信息。"
    return result


if __name__ == '__main__':
    rsp = search_web("最近逝世的乌拉圭总统")
    print(rsp)
