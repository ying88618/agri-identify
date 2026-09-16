import asyncio
import logging
import os
import httpx
from dotenv import load_dotenv

from .config import BM25_FILTER_POOL
from .vectorstore import build_vs
from .bm25_index import get_bm25

load_dotenv()   # 本模块 import 时即读取 RERANK_MODEL/OPENAI_BASE_URL，须先加载 .env

logger = logging.getLogger(__name__)

RERANK_API = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")

# 复用连接池; 按事件循环缓存, 避免跨 loop 复用报 "Event loop is closed"
_client: "httpx.AsyncClient | None" = None
_client_loop = None


def _get_client() -> httpx.AsyncClient:
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(timeout=30)
        _client_loop = loop
    return _client


async def _rerank(query: str, docs: list[str], top_n: int, retries: int = 5):
    """调用 rerank API, 返回 [{"index": i, "relevance_score": s}, ...]; 失败返回 None

    【必须用异步客户端】同步 httpx.post 会阻塞事件循环, 使调用方并发退化为串行:
    实测吞吐 0.71/s(同步) vs 9.58/s(异步), 差 10 倍; 且高并发下超时失败率上升,
    而调用方(_rerank 失败即降级)拿不到 score, 会引起下游 KeyError。

    【必须指数退避重试】该接口在持续负载下会返回 429 Too Many Requests, 且限流是
    "账户级/持续一段时间"而非单次抖动 —— 短退避(0.4s)完全无效, 实测连续 3 次全失败。
    一旦最终失败, 调用方会静默降级为 RRF 排序: 分数被补成 0.0、丢失精排效果,
    评测指标随之失真(实测降级率可达 15%~63%)。故退避需足够长以跨过限流窗口。
    """
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = await _get_client().post(
                f"{RERANK_API}/rerank",
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                json={
                    "model": RERANK_MODEL,
                    "query": query,
                    "documents": docs,
                    "top_n": top_n,
                    "return_documents": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["results"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                # 指数退避: 1,2,4,8,16s, 用于跨过 429 的限流窗口
                await asyncio.sleep(min(2 ** attempt, 16))
    logger.warning("rerank 连续 %d 次失败, 降级为 RRF 排序: %s: %s",
                   retries + 1, type(last_err).__name__, last_err)
    return None  # 失败时降级为双塔排序


def _rrf_fusion(lists: list[list[dict]], rank_constant: int = 60) -> list[dict]:
    """按排名融合检索结果"""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for lst in lists:
        for rank, item in enumerate(lst):
            key = item["content"][:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank + 1)
            items[key] = item
    ordered = sorted(scores, key=lambda k: -scores[k])
    return [items[k] for k in ordered]


def _safe_filter_value(value: str | None) -> str | None:
    """净化用于 Milvus 过滤表达式的值；不合法时返回 None（等价于"不过滤该字段"）。

    【为什么必须有】表达式是 f-string 直拼的:
        f'crop == "{crop}"'
    而 crop/section 来自模型生成的工具参数(core/agent.py 的
    knowledge_base_search), 用户可用提示词引导模型填入畸形值 ——
    值里带一个双引号就足以破坏表达式结构(Milvus 表达式注入)。

    这里只做「语法层」校验: 纯字符串判断、不依赖任何外部数据, 因此无论
    Milvus / 知识库是否可用, 这道防线都成立。白名单、别名归一化那类
    「语义层」校验不属于本函数职责。
    """
    if not isinstance(value, str) or not value.strip():
        return None
    if any(ch in value for ch in ('"', "\\", "\n", "\r")):
        logger.warning("过滤值含非法字符, 已丢弃该过滤条件: %r", value)
        return None
    return value


async def retrieve(
    question: str,
    collection_name: str,
    k: int = 4,
    score_threshold: float | None = None,
    recall_k: int = 20,
    hybrid: bool = False,
    crop: str | None = None,
    section: str | None = None,
) -> list[dict]:
    """两阶段检索: 向量召回 Top-recall_k → rerank 精排 → 取 Top-k

    crop:    可选, 限定作物(农业库 kb_agri 的 crop 字段); 传 None 表示不过滤
    section: 可选, 限定字段章节(如 "危害症状"/"防治方法"); 传 None 表示不过滤
    """
    vs = build_vs(collection_name)

    # 过滤值先净化再拼表达式。放在这里(而不是各调用方)是因为这里是表达式
    # 唯一的产生地, 于是 agent / MCP server / 评测脚本都受同一道保护。
    # 净化后的值同时用于下面的 BM25 后过滤, 保持两条通道条件一致。
    crop = _safe_filter_value(crop)
    section = _safe_filter_value(section)

    conds = []
    if crop:
        conds.append(f'crop == "{crop}"')
    if section:
        conds.append(f'section == "{section}"')
    expr = " and ".join(conds) if conds else None
    docs_and_scores = await vs.asimilarity_search_with_score(
        question, k=recall_k, expr=expr
    )
    vec_cands = []
    for d, s in docs_and_scores:
        meta = dict(d.metadata or {})
        # 命名兼容: 旧库(kb_default)用 file_name, 农业库(kb_agri)用 source_file
        src = meta.get("file_name") or meta.get("source_file") or "未知"
        vec_cands.append(
            {
                **meta,                 # 透传 disease/crop/section 等业务字段
                "content": d.page_content,
                "file_name": src,       # 统一命中字段
                "source_file": src,
                "score": float(s),
            }
        )

    if hybrid:
        # 有过滤条件时, 必须先取更大的 BM25 候选池再过滤:
        # 否则全局 Top-k 里可能几乎没有目标 crop/section 的片段, 过滤后所剩无几, BM25 通道形同虚设
        bm_pool = recall_k * BM25_FILTER_POOL if (crop or section) else recall_k
        bm_cands = get_bm25(collection_name).search(question, k=bm_pool)
        if crop:
            bm_cands = [c for c in bm_cands if c.get("crop") == crop]
        if section:
            bm_cands = [c for c in bm_cands if c.get("section") == section]
        bm_cands = bm_cands[:recall_k]
        candidates = _rrf_fusion([vec_cands, bm_cands])[:recall_k]
    else:
        candidates = vec_cands

    if not candidates:
        return []

    ranked = await _rerank(question, [c["content"] for c in candidates], k)
    if ranked is not None:
        results = []
        for r in ranked:
            c = dict(candidates[r["index"]])
            c["score"] = float(r["relevance_score"])
            results.append(c)
    else:
        results = candidates[:k]  # 降级: 双塔
        # BM25 通道进来的候选没有 score 字段, 补默认值, 否则下游 res[0]["score"] 会 KeyError
        for r in results:
            r.setdefault("score", 0.0)
        if score_threshold is not None:
            results = [r for r in results if r["score"] >= score_threshold]
    return results
