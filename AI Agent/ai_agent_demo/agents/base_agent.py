# -*- coding: utf-8 -*-
"""
@File    : base_agent.py
@Time    : 2025/5/15 15:13
@Desc    : 
"""
import os

from langchain.agents import Tool, initialize_agent, AgentType

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from memory.memory import memory
from tools.search_tool import search_web
from tools.doc_reader import load_pdf_content
from tools.vectorstore import build_vectorstore_from_pdf
from multimodal.image_captioning import caption_image

# 构建文档向量检索器
vectorstore = build_vectorstore_from_pdf("data/GB+38031-2025.pdf")
retriever = vectorstore.as_retriever()


def retrieve_doc(query: str) -> str:
    docs = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in docs])


def build_agent():
    tools = [
        Tool(name="Tavily Search", func=search_web, description="用于在线搜索最新的互联网信息"),
        Tool(name="PDF Reader", func=load_pdf_content, description="适合读取本地PDF文档内容"),
        Tool(name="PDF Semantic Search", func=retrieve_doc, description="对PDF内容进行语义检索"),
        Tool(name="Image Captioning", func=caption_image, description="根据图像生成文字描述")
    ]
    llm = ChatOpenAI(
        model='deepseek-chat',
        openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
        openai_api_base='https://api.deepseek.com/v1',
        temperature=0
    )
    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad"],
        template="""
    你是一个多工具智能体，具备搜索、文档读取、图像识别和文档语义检索能力。
    请根据用户的问题选择合适的工具，并给出准确简洁的答案。

    你可以调用的工具有：
    - Tavily Search：用于搜索网络信息
    - PDF Reader：用于获取指定本地PDF的全部内容
    - PDF Semantic Search：用于对PDF进行基于语义的问答
    - Image Captioning：根据上传的图像生成文字描述

    请尽量以自然语言回答问题，如果需要使用工具，请先思考，再行动。
    
    使用以下格式进行推理：
    Question: 用户的问题
    Thought: 你对问题的思考
    Action: 要调用的工具名称（必须完全一致）
    Action Input: 工具的输入
    
    Observation: 工具返回的结果
    
    ...（可以多轮）...
    
    最终请使用：
    Final Answer: 你的最终回答
    
    现在开始：
    
    Question: {input}
    {agent_scratchpad}
    """
    )
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        prompt=prompt,
        memory=memory,
        verbose=True,
        # handle_parsing_errors=True
    )
    return agent


if __name__ == '__main__':
    ai_agent = build_agent()
    rsp = ai_agent.invoke({"input": "请同时完成：1) 将'人工智能将改变未来'翻译成英语 2) 计算235×47 3) 写一首关于春天的五行诗"})
    print(rsp)
    """
    > Entering new AgentExecutor chain...
    我需要查询北京今天的天气情况，这需要最新的互联网信息。
    
    Action: Tavily Search
    Action Input: "北京今天的天气"
    
    Observation: [{'title': '北京 - 百度', 'url': 'https://weathernew.pae.baidu.com/weathernew/pc?query=%E5%8C%97%E4%BA%AC%E5%A4%A9%E6%B0%94&srcid=4982', 'content': '北京 今天：晴 -1°~12°C 西南风1级\n02月26日 周三 农历正月廿九\n12°\n56 良\n晴 西南风 2级\n湿度11% 日出06:52 日落18:02\n24小时预报\n\n温度\n降水量\n风力风向\n空气质量\n\n13°C6°C-1°C\n现在\n17:00\n18:00\n19:00\n20:00\n21:00\n22:00\n23:00\n00:00\n01:00\n02:00\n03:00\n04:00\n05:00\n06:00\n07:00\n08:00\n09:00\n10:00\n11:00\n12:00\n13:00\n14:00\n15:00\n15天天气预报\n\n\n昨天\n02/25\n3级\n优\n\n\n今天\n02/26\n1级\n良\n\n\n明天\n02/27\n2级\n良\n\n\n周五\n02/28\n2级\n轻度\n\n\n周六\n03/01\n1级\n轻度\n\n\n周日\n03/02\n2级\n优\n\n\n周一\n03/03\n2级\n优\n\n\n周二\n03/04\n2级\n良\n\n\n周三\n03/05\n2级\n\n\n周四\n03/06\n2级\n\n\n周五\n03/07\n2级\n\n\n周六\n03/08\n2级\n\n\n周日\n03/09\n2级\n\n\n周一\n03/10\n3级\n\n\n周二\n03/11\n2级', 'score': 0.7750779}, {'title': '预报- 北京 - 中国天气网', 'url': 'https://www.weather.com.cn/weather/101010100.shtml', 'content': '天空蔚蓝 可见透彻蓝天，或有蓝天白云美景。\n天空淡蓝 天空不够清澈，以浅蓝色为主。\n天空阴沉 阴天或有雨雪，天空灰暗。\n天空灰霾 出现霾或沙尘，天空灰蒙浑浊。\n\n11时__14时__17时__20时__23时__02时__05时__08时\n15℃__18℃__19℃__14℃__11℃__9℃__7℃__10℃\n西南风__西南风__西南风__西南风__西南风__西南风__西南风__西南风\n<3级__<3级__<3级__<3级__<3级__<3级__<3级__<3级\n\n\n少发 _感冒指数_无明显降温，感冒机率较低。\n\n\n适宜 _运动指数_天气较好，尽情感受运动的快乐吧。\n\n\n易发 _过敏指数_应减少外出，外出需采取防护措施。\n\n\n 较舒适 _穿衣指数_建议穿薄外套或牛仔裤等服装。\n\n\n适宜 _洗车指数_天气较好，适合擦洗汽车。\n\n\n很强 _紫外线指数_涂擦SPF20以上，PA++护肤品，避强光。\n\n\n少发 _感冒指数_无明显降温，感冒机率较低。\n\n\n适宜 _运动指数_天气较好，尽情感受运动的快乐吧。\n\n\n易发 _过敏指数_应减少外出，外出需采取防护措施。 [...] 过去24小时AQI最高值: 59\n最高气温: 21.4℃ , 最低气温: 9.5℃\n过去24小时最大相对湿度: 39%\n过去24小时最大风力: 4级\n过去24小时总降水量：0mm\n08091011121314151617181920212223000102030405060708\n2623201714118\n(h) ℃ (μg/m³) (%) (mm) (级)空气质量指数：简称AQI，是定量描述空气质量状况的无量纲指数。（数据由生态环境部提供）\n相对湿度：空气中实际水汽压与当时气温下的饱和水汽压之比，以百分比（%）表示。\n温度：表示大气冷热程度的物理量，气象上给出的温度是指离地面1.5米高度上百叶箱中的空气温度。\n降水量：某一时段内的未经蒸发、渗透、流失的降水，在水平面上积累的深度，以毫米（mm）为单位。\n风力风向：空气的水平运动，风力是风吹到物体上所表示出力量的大小，风向指风的来向。\n优 良 轻度 中度 重度 严重\n\n30°C\n25°C\n20°C\n15°C\n10°C\n5°C\n\n0°C\n\n\n20\n7℃\n\n\n20\n8℃\n\n\n20\n7℃\n\n\n21\n9℃\n\n\n22\n10℃ [...] 热门城市\n曼谷 东京 首尔 吉隆坡 新加坡 巴黎 罗马 伦敦 雅典 柏林 纽约 温哥华 墨西哥城 哈瓦那 圣何塞 巴西利亚 布宜诺斯艾利斯 圣地亚哥 利马 基多 悉尼 墨尔本 惠灵顿 奥克兰 苏瓦 开罗 内罗毕 开普敦 维多利亚 拉巴特\n选择洲际\n亚洲 欧洲 北美洲 南美洲 非洲 大洋洲\n\n全国> 北京 > 城区\n07:30更新 | 数据来源 中央气象台\n\n今天\n7天\n8-15天\n40天\n\n雷达图\n\n\n2日（今天）\n    ======\n晴\n20/7℃\n<3级\n\n\n3日（明天）\n    ======\n多云\n20/8℃\n<3级\n\n\n4日（后天）\n    ======\n多云转晴\n20/7℃\n3-4级\n\n\n5日（周六）\n    ======\n多云\n21/9℃\n<3级\n\n\n6日（周日）\n    ======\n晴转多云\n22/10℃\n<3级\n\n\n7日（周一）\n    ======\n多云\n25/10℃\n<3级\n\n\n8日（周二）\n    ======\n阴\n21/12℃\n<3级转3-4级\n\n\n分时段预报 生活指数\n蓝天预报\n蓝天预报综合天气现象、能见度、空气质量等因子，预测未来一周的天空状况。', 'score': 0.7602419}, {'title': '中国气象局-天气预报- 北京', 'url': 'https://weather.cma.cn/web/weather/54511.html', 'content': 'æ\x97¶é\x97´ | 11:00 | 14:00 | 17:00 | 20:00 | 23:00 | 02:00 | 05:00 | 08:00\nå¤©æ°\x94 |  |  |  |  |  |  |  | \næ°\x94æ¸© | 11.4â\x84\x83 | 11.5â\x84\x83 | 11â\x84\x83 | 10.6â\x84\x83 | 10.2â\x84\x83 | 10.6â\x84\x83 | 11â\x84\x83 | 12.8â\x84\x83\né\x99\x8dæ°´ | 0.5mm | 11.3mm | 8mm | 4.3mm | 1.5mm | 0.4mm | æ\x97\xa0é\x99\x8dæ°´ | æ\x97\xa0é\x99\x8dæ°´\né£\x8eé\x80\x9f | 7.3m/s | 7.9m/s | 6.4m/s | 5.8m/s | 3.3m/s | 3m/s | 2.2m/s | 3.3m/s\né£\x8eå\x90\x91 | ä¸\x9cå\x8c\x97é£\x8e | ä¸\x9cå\x8c\x97é£\x8e | ä¸\x9cå\x8c\x97é£\x8e | è¥¿å\x8c\x97é£\x8e | è¥¿å\x8c\x97é£\x8e | è¥¿å\x8c\x97é£\x8e | è¥¿å\x8c\x97é£\x8e | è¥¿å\x8c\x97é£\x8e [...] é£\x8eå\x90\x91 | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8d\x97é£\x8e | è¥¿å\x8c\x97é£\x8e\næ°\x94å\x8e\x8b | 1006.8hPa | 1005.1hPa | 1001.7hPa | 998.2hPa | 997.4hPa | 998.4hPa | 997.9hPa | 997.7hPa\næ¹¿åº¦ | 50.4% | 20.9% | 26.3% | 39.3% | 42.1% | 50.6% | 61.3% | 72.8%\näº\x91é\x87\x8f | 3.8% | 3.3% | 10% | 10% | 10% | 78.8% | 56.4% | 27.1%\næ\x97¶é\x97´ | 08:00 | 11:00 | 14:00 | 17:00 | 20:00 | 23:00 | 02:00 | 05:00\nå¤©æ°\x94 |  |  |  |  |  |  |  | [...] æ°\x94å\x8e\x8b | 1010.1hPa | 1010hPa | 1010.6hPa | 1012.4hPa | 1012.7hPa | 1010.6hPa | 1010.9hPa | 1012.3hPa\næ¹¿åº¦ | 93.5% | 95.8% | 98% | 96.9% | 95.8% | 88% | 89.3% | 83.9%\näº\x91é\x87\x8f | 95.2% | 92.2% | 97.1% | 88.2% | 92.1% | 94.9% | 82.5% | 73.4%\næ\x97¶é\x97´ | 08:00 | 11:00 | 14:00 | 17:00 | 20:00 | 23:00 | 02:00 | 05:00\nå¤©æ°\x94 |  |  |  |  |  |  |  | \næ°\x94æ¸© | 12.8â\x84\x83 | 22.5â\x84\x83 | 22.8â\x84\x83 | 21.9â\x84\x83 | 18.3â\x84\x83 | 15.5â\x84\x83 | 13.3â\x84\x83 | 12.2â\x84\x83', 'score': 0.72768575}, {'title': '北京旅游指数/交通指数，24小时/未来天气预报查询 - 飞猪天气预报', 'url': 'https://tianqi.fliggy.com/110100/', 'content': '北京5月7日09:13天气预报，今天14℃-25℃，多云, 风向情况：南风，总降水量：29.50 ... 国内热门景点天气. 其他天气推荐. 杭州天气 · 上海天气 · 重庆天气 · 北京天气 · 广州', 'score': 0.7023877}, {'title': '22.7 - 北京市气象局', 'url': 'http://bj.cma.gov.cn/fzlm/', 'content': '北京市气象局\n现在天气一周预报旅游预报\n2025.2.28星期五 农历二月初一（08:37:32实况）\n5.0℃ 晴\n气压：1012.7hPa 相对湿度：56% 风向风力西北风1级\n精细化预报（2025-02-28 05:09:32北京市气象台发布）\n07:00\n晴\xa03℃风向：西北风力：1级\n08:00\n晴\xa04℃风向：西北风力：1级\n09:00\n晴\xa06℃风向：西北风力：1级\n36小时天气预报（2025-02-28 05:09:32北京市气象台发布）\n\n今天白天最高气温17℃晴转多云2、3级北转南风\n今天夜间最低气温3℃多云1、2级南转北风\n明天白天最高气温14℃多云转阴2、3级北转东风 [...] 中央气象台发布\n北京地区景点12小时天气预报时效：2025年02月28日发布\n景区天气气温\n昌平十三陵晴转多云18/2\n门头沟百花山晴转多云15/-1\n密云黑龙潭晴转多云16/1\n司马台长城晴转多云15/2\n门头沟妙峰山晴转多云13/-3\n天安门晴转多云17/5\n八达岭长城晴转多云13/0\n延庆龙庆峡晴转多云14/1\n平谷金海湖晴转多云18/8\n怀柔红螺寺晴转多云16/3\n怀柔慕田峪长城晴转多云16/0\n密云云蒙山晴转多云16/1\n门头沟灵山晴转多云15/-1\n房山十渡晴转多云18/5\n区局实况云图雷达图指数预报\n怀柔\n-5.40.8级\n延庆\n-1.60.7级\n密云\n0.74.0级\n昌平\n4.91.8级\n顺义\n3.52.2级\n平谷\n1.00.0级\n门头沟\n6.00.7级\n海淀\n0.80.6级\n石景山\n5.50.6级\n市中心\n5.0 1.0级\n朝阳\n5.82.0级\n房山\n2.81.1级\n丰台\n3.7 0.4级\n通州\n3.00.4级\n大兴\n3.80.0级\n2025-02-28 08:37:32\n发布时间：2025-02-26 16:44:32', 'score': 0.6852132}]
    Thought:Final Answer: 北京今天的天气是晴，气温在-1°C到12°C之间，西南风1级，空气质量为良。
    
    > Finished chain.
    {'input': '北京今天的天气怎么样？', 'chat_history': '', 'output': '北京今天的天气是晴，气温在-1°C到12°C之间，西南风1级，空气质量为良。'}
    """
