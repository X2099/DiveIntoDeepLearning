import os
from langsmith import RunTree
from langchain_openai import ChatOpenAI

# ------------------ 环境变量配置 ------------------
# 确保在环境变量或 .env 文件中配置了以下内容：
# LANGCHAIN_API_KEY: LangSmith 的 API Key
# DEEPSEEK_API_KEY: deepseek 模型的 API Key
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ------------------ 初始化 LLM ------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0,
)

# ------------------ 用户输入 ------------------
user_query = "我忘记了登录密码，怎么才能尽快重新设置？"

# ------------------ 顶层追踪节点 ------------------
run_tree = RunTree(
    name="customer_service_agent_trace",
    inputs={"user_query": user_query},
    tags=["langsmith-demo", "run_tree", "full_trace"]
)

try:
    # ------------------ 子任务 1：调用 LLM 生成回复 ------------------
    prompt = (
        "请为用户生成一段简洁友好的密码重置指引，"
        "说明操作步骤，并提醒用户注意账户安全。"
    )
    llm_child = RunTree(
        name="llm_inference",
        inputs={"prompt": prompt},
        parent_id=run_tree.id  # 👈 建立层级
    )
    response = llm.invoke(prompt)
    llm_child.add_outputs({"response": response})
    llm_child.post()  # 👈 上传子节点

    # ------------------ 子任务 2：格式化输出 ------------------
    format_child = RunTree(
        name="format_response",
        inputs={"raw_response": response},
        parent_id=run_tree.id
    )
    formatted = f"您好，{response} 如有疑问请联系人工客服协助处理。"
    format_child.add_outputs({"formatted_response": formatted})
    format_child.post()

    # ------------------ 顶层输出 ------------------
    run_tree.add_outputs({"final_response": formatted})

except Exception as e:
    run_tree.add_outputs({"error": str(e)})
    raise

finally:
    run_tree.post()  # 👈 上传主节点到 LangSmith 平台

# 控制台展示最终结果
print(formatted)
