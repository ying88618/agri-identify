import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()   # 本模块 import 时即读取 TAVILY_API_KEY，须先加载 .env

_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


async def web_search(query: str) -> tuple[str, list[dict]]:
    """返回 (给模型看的文本, Tavily 的原始结果列表)。

    为什么要一并返回原始结果：api/chat.py 需要把来源(url/title)以 SSE 事件
    回传给前端做溯源。若只返回拼好的文本，URL 就在这一层被"埋"进字符串里、
    再也取不出来 —— 本模块原来就是这样，所以来源功能没法只改上层。

    注意：_client.search 是同步调用，放在 async 函数里会阻塞事件循环
    （联网搜索期间所有 SSE 流都会被卡住）。这是既有问题，本次未动。
    """
    resp = _client.search(
        query=query,
        max_results=5,
        search_depth="basic",
        topic="general",
    )
    results = resp.get("results", [])

    if not results:
        return "(联网搜索未找到相关内容)", results
    text = "\n\n".join(
        f"标题:{r.get('title','')}\n来源:{r.get('url','')}\n摘要:{r.get('content','')}"
        for r in results
    )
    return text, results
