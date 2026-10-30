from langchain.agents import create_agent

from langchain_core.tools import tool
from .config import DEFAULT_SCORE_THRESHOLD, RECALL_K, TOP_K
from .retriever import retrieve
from .sources import kb_sources, web_sources
from .web_search import web_search as tavily_web_search
from .llm import llm
from .mw.limits import (
    build_repeat_guard,
    build_step_limit,
    build_token_budget,
    build_tool_limit,
)
from .mw.trace import TraceMiddleware
import logging
import contextvars

# 阈值与检索参数的唯一定义在 core/config.py（含标定依据，换 rerank 模型必须重标定）。
# 这里 import 进来属于 re-export：`agent.DEFAULT_SCORE_THRESHOLD` 仍可用。

_current_collection: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_collection"
)
_current_threshold: contextvars.ContextVar[float] = contextvars.ContextVar(
    "current_threshold", default=DEFAULT_SCORE_THRESHOLD
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("knowledge_agent")


# response_format="content_and_artifact" 让工具返回 (给模型看的文本, artifact) 二元组：
# content 仍是给模型的那段文本（LLM 的输入完全不变），artifact 则挂到 ToolMessage 上，
# 由 api/chat.py 在消息流里取出来、以 SSE 的 sources 事件回传给前端做溯源。
# 走这条通道（而不是自己另开旁路，如 contextvar）的理由：它显式、可调试，
# 一轮里多次工具调用天然产生多个 artifact，且不涉及跨上下文的写入传播问题。
@tool(response_format="content_and_artifact")
async def knowledge_base_search(query: str, crop: str = "", section: str = ""):
    """当用户需要私人知识库(农业病虫害资料)信息时调用。

    Args:
        query: 检索用的症状描述。尽量写具体特征(颜色/形状/部位/边缘/有无霉层)，
               不要只写"叶子有病"。
        crop: 可选但**强烈建议**。作物名，如"番茄""玉米""葡萄"。
              已知作物时务必传入——不同作物的相似症状对应完全不同的病害，
              不传会大幅降低准确率。
        section: 可选。限定章节，取值：危害症状 / 防治方法 / 发生因素 / 病原 /
                 侵染循环 / 简介。问"是什么病"用 危害症状；问"怎么防治"用 防治方法。
    """
    collection_name = _current_collection.get()
    score_threshold = _current_threshold.get()
    logger.info("[TOOL CALLED] knowledge_base_search query=%r crop=%r section=%r",
                query, crop, section)
    results = await retrieve(
        query,
        collection_name,
        k=TOP_K,
        hybrid=True,          # 向量 + BM25 混合, 单一向量通道在农业术语上偏弱
        recall_k=RECALL_K,    # 宽召回: 候选池含正确答案的比例 66.6% -> 95.5%
        score_threshold=score_threshold,
        crop=crop or None,
        section=section or None,
    )
    logger.info(
        "[TOOL RETURNED] %d docs (threshold=%.2f)", len(results), score_threshold
    )
    if not results:
        # 即使无结果也必须返回 artifact：response_format="content_and_artifact"
        # 要求工具返回 (content, artifact) 二元组。空列表到前端就等于"没有来源可展示"。
        return "(知识库中未找到相关资料)", {"sources": []}
    text = "\n\n".join(f"来源:{r['file_name']}\n{r['content']}" for r in results)
    return text, {"sources": kb_sources(results)}


@tool(response_format="content_and_artifact")
async def web_search(query: str):
    """当需要实时/互联网信息，或知识库无相关内容时调用，返回联网检索结果"""
    logger.info("[TOOL CALLED] web_search query=%r", query)
    text, results = await tavily_web_search(query)
    return text, {"sources": web_sources(results)}


def set_request_context(collection_name: str,
                        score_threshold: float = DEFAULT_SCORE_THRESHOLD):
    _current_collection.set(collection_name)
    _current_threshold.set(score_threshold)


# middleware 实例是全局共享的（本模块只执行一次），所以 TraceMiddleware 里
# 不允许有任何可变实例状态 —— 所有 per-request 信息走 core/run_state 的 ContextVar。
AGENT = create_agent(
    llm,
    tools=[web_search, knowledge_base_search],
    # 顺序 = 洋葱层级：trace 在最外层，能观测到后面两个中间件的效果。
    # limits 内部只做计数与跳转，不改请求内容。
    middleware=[
        TraceMiddleware(),
        build_step_limit(),
        build_tool_limit(),
        build_token_budget(),
        # 放在最后：它在 wrap_tool_call 里短路，trace（最外层）仍能记录到这次调用
        build_repeat_guard(),
    ],
)


def get_agent():
    """返回全局唯一agent实例"""
    return AGENT
