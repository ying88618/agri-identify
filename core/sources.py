"""检索来源的整理：构造 / 去重 / 合并。

【为什么单独成模块】
api/chat.py 要把检索与联网的元数据以 SSE 事件回传给前端（"依据：《番茄早疫病》
的「危害症状」章节"）。构造、去重、合并全是纯函数，放这里就能零依赖单测 ——
与 core/catalog.py、core/images.py 是同一思路。

【excerpt 的取舍】
只回传元数据（病害/章节/文件）说服力不足：用户点不开原文就等于没有依据；
带上完整正文又会让 SSE 帧膨胀（一轮最多可能十几条来源）。折中为截断摘录。

注意 excerpt 是给**人**看的，绝不能进 memory / 对话历史 ——
历史里只该有模型看到过的内容（见 api/chat.py 的 append_turn）。

【relevance 不是概率】
它是 rerank 的相关性分数，只用于**排序比较**，不同 query 之间不可比。
字段名刻意避开 confidence：农业涉及用药安全，一个被前端误读成"83% 确定"
的数字，比不给数字更危险。
"""
import logging

logger = logging.getLogger("sources")

# 摘录长度（按字符）。一轮最多可能十几条来源，太长会让 SSE 帧明显膨胀。
EXCERPT_CHARS = 160


def excerpt(text: str, limit: int = EXCERPT_CHARS) -> str:
    """把正文压成单行摘录：折叠所有连续空白，超长截断并加省略号。"""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def kb_sources(results: list[dict] | None) -> list[dict]:
    """core/retriever.retrieve() 的结果 -> 给前端看的来源条目。"""
    out: list[dict] = []
    for r in results or []:
        out.append(
            {
                "kind": "kb",
                "crop": r.get("crop") or "",
                "disease": r.get("disease") or "",
                "section": r.get("section") or "",
                # 旧库用 file_name、农业库用 source_file，retriever 已统一过，
                # 这里再兜一次是为了容忍直接手写 dict 的调用方
                "source_file": r.get("source_file") or r.get("file_name") or "",
                "relevance": round(float(r.get("score") or 0.0), 4),
                "excerpt": excerpt(r.get("content", "")),
            }
        )
    return out


def web_sources(results: list[dict] | None) -> list[dict]:
    """Tavily 的原始结果 -> 给前端看的来源条目。

    没有 url 的条目直接丢弃：来源的核心价值是可点开溯源，
    没 url 的条目既不能核对也不能点开。
    """
    out: list[dict] = []
    for r in results or []:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        out.append(
            {
                "kind": "web",
                "title": (r.get("title") or "").strip(),
                "url": url,
                "excerpt": excerpt(r.get("content", "")),
            }
        )
    return out


def _dedup_key(item: dict) -> tuple:
    """去重键用元数据，不用正文。

    core/retriever._rrf_fusion 里是用 content[:200] 做键的，那是为了把不同召回
    通道的同一篇文档融合起来，语义不同，别混用。
    """
    if item.get("kind") == "web":
        return ("web", item.get("url", ""))
    return (
        "kb",
        item.get("crop", ""),
        item.get("disease", ""),
        item.get("section", ""),
        item.get("source_file", ""),
    )


def merge_sources(*groups: list[dict] | None) -> list[dict]:
    """合并多次工具调用的来源并去重，保持传入顺序。

    这是**必然路径**而不是防御性代码：模型一轮里会发多个并行工具调用
    （实测出现过"防治方法"与"危害症状"两次），两次检索很可能召回同一篇文档的
    不同章节；不去重前端就会看到重复条目。

    顺序按传入顺序保留（即调用顺序）：kb 与 web 的分数不可比，
    没有统一的分数可以排序。
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for group in groups:
        for item in group or []:
            key = _dedup_key(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
    return out
