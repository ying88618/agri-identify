"""MCP server:把私域知识库检索暴露为标准MCP工具。
启动：python mcp_server.py"""

import logging

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core.config import KB_COLLECTION  # noqa: E402
from core.retriever import retrieve  # noqa: E402

DEFAULT_COLLECTION = KB_COLLECTION

mcp = FastMCP("private-kb-server")
logger = logging.getLogger(__name__)


@mcp.tool()
async def knowledge_search(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = 4,
) -> str:
    """检索用户私域知识库。
    当用户的问题需要内部文档、公司资料、私有文件中的信息时调用本工具；
    对于实时新闻、公开互联网信息请不要使用本工具（应使用联网搜索）。
    Args:
        query: 用户自然语言问题
        collection_name: 知识库集合名(不同业务/用户隔离)
        k: 返回最相关的k条chunk
    """
    try:
        results = await retrieve(
            query,
            collection_name,
            k=k,
            score_threshold=None,
        )
    except Exception as e:
        logger.error(f"检索失败：{e}", exc_info=True)
        return f"知识库检索失败:{e}"

    if not results:
        return "知识库中未找到相关资料"

    formatted = []
    for r in results:
        parts = [f"来源：{r.get('file_name','未知文件')}"]
        if r.get("score") is not None:
            score = r.get("score")
            parts.append(f"相关度：{score:.3f}")
        parts.append(r.get("content", ""))
        formatted.append("\n".join(parts))
    return "\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run()