import os
from typing import TypedDict, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tracers import LangChainTracer
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# LangSmith配置
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "default"
tracing_handler = LangChainTracer()


class AgentState(TypedDict):
    """代理状态类型定义
    Attributes:
        user_input: str 用户输入文本
        intent: Optional[Literal["普通问题", "技术问题", "投诉问题"]] 问题分类结果
        response: Optional[str] 生成的响应内容
    """
    user_input: str
    intent: Optional[Literal["普通问题", "技术问题", "投诉问题"]]
    response: Optional[str]


# 初始化大语言模型
llm = ChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),  # 从环境变量获取API密钥
    openai_api_base='https://api.deepseek.com/v1',  # DeepSeek API端点
    temperature=0,  # 控制生成结果的随机性
    callbacks=[tracing_handler]  # 集成 LangSmith
)
output_parser = StrOutputParser()  # 用于解析LLM输出的纯文本结果


def safe_set_intent(value: str) -> Optional[Literal["普通问题", "技术问题", "投诉问题"]]:
    """安全设置问题分类
    Args:
        value: 待验证的分类字符串
    Returns:
        当输入为有效分类时返回原值，否则返回None
    """
    return value if value in ("普通问题", "技术问题", "投诉问题") else None


def classify_intent(state: AgentState) -> AgentState:
    """问题分类节点
    Args:
        state: 当前对话状态
    Returns:
        更新后的状态(包含分类结果)
    """
    prompt = ChatPromptTemplate.from_template(
        "请判断用户问题类型，只返回普通问题/技术问题/投诉问题，不要包含任何格式符号或额外文字。问题内容：{input}"
    )
    chain = prompt | llm | output_parser  # 构建处理链
    intent = chain.invoke({"input": state["user_input"]}).strip()
    return {
        "user_input": state["user_input"],
        "intent": safe_set_intent(intent),  # 安全设置分类
        "response": None
    }


def handle_general(state: AgentState) -> AgentState:
    """普通问题处理节点"""
    prompt = ChatPromptTemplate.from_template(
        "你是一个客服助手，请用纯文本回答以下普通问题，不要包含任何格式符号或标记。问题：{input}"
    )
    chain = prompt | llm | output_parser
    response = chain.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "intent": state.get("intent"),
        "response": response
    }


def handle_tech(state: AgentState) -> AgentState:
    """技术问题处理节点"""
    prompt = ChatPromptTemplate.from_template(
        "你是一名技术支持工程师，请用纯文本回答以下技术问题，不要包含任何格式符号或标记。技术问题：{input}"
    )
    chain = prompt | llm | output_parser
    response = chain.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "intent": state.get("intent"),
        "response": response
    }


def handle_complaint(state: AgentState) -> AgentState:
    """投诉问题处理节点"""
    prompt = ChatPromptTemplate.from_template(
        "你是一名投诉处理专员，请用纯文本回答以下投诉，不要包含任何格式符号或标记。投诉内容：{input}"
    )
    chain = prompt | llm | output_parser
    response = chain.invoke({"input": state["user_input"]})
    return {
        "user_input": state["user_input"],
        "intent": state.get("intent"),
        "response": response
    }


def decide_next_step(state: AgentState) -> str:
    """路由决策函数
    Args:
        state: 当前对话状态
    Returns:
        下一节点的名称
    """
    return state.get("intent", "普通问题")  # 默认路由到普通问题


# 构建工作流
workflow = StateGraph(AgentState)

# 注册节点
workflow.add_node("classify", classify_intent)  # 分类节点
workflow.add_node("普通问题", handle_general)  # 注意：节点名称改为中文
workflow.add_node("技术问题", handle_tech)
workflow.add_node("投诉问题", handle_complaint)

# 设置条件路由（修正后的映射关系）
workflow.add_conditional_edges(
    "classify",
    decide_next_step,
    {
        "普通问题": "普通问题",  # 键值统一使用中文
        "技术问题": "技术问题",
        "投诉问题": "投诉问题"
    }
)

# 设置终止边
workflow.add_edge("普通问题", END)
workflow.add_edge("技术问题", END)
workflow.add_edge("投诉问题", END)

# 设置入口点
workflow.set_entry_point("classify")

# 编译应用
app = workflow.compile()

if __name__ == "__main__":
    test_cases = [
        "你们的工作时间是几点？",
        "我的账号登录不上了",
        "我对你们的服务非常不满意！"
    ]

    for query in test_cases:
        print("用户提问: " + query)
        result = app.invoke({"user_input": query})
        print("识别类型: " + str(result.get("intent")))
        print("系统回复: " + str(result.get("response")))
