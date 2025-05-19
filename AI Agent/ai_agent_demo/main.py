# -*- coding: utf-8 -*-
"""
@File    : main.py
@Time    : 2025/5/15 15:12
@Desc    : 
"""
from agents.base_agent import build_agent


def run():
    agent = build_agent()

    print("🔧 智能体已启动... 输入 exit 退出")

    while True:
        user_input = input("\n🧑 用户: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 再见！")
            break
        response = agent.invoke(user_input)
        print(f"\n🤖 Agent: {response}")


if __name__ == "__main__":
    run()
