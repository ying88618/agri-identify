# self_correct.py — 自纠正闭环: 检测拒答 → query改写 → 合并重检索 → 再生成
# 注意: 评测用独立非流式客户端(避免 llm.py 的 streaming=True 在高并发下 ReadError)
import asyncio
import json
import logging
import os
import re

from openai import AsyncOpenAI

from .retriever import retrieve

logger = logging.getLogger("self_correct")

# 独立非流式客户端(评测/离线场景专用, 不依赖 llm.py 的流式实例)
_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
MODEL = os.getenv("MODEL_NAME")

# 拒答信号检测(文本级匹配, 不调 LLM, 快且稳定)
REFUSAL_PATTERNS = [
    r"暂无相关资料", r"未找到相关资料", r"没有找到相关资料",
    r"未检索到", r"知识库中无", r"无法回答", r"资料不足",
]


def detect_refusal(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return any(re.search(p, answer) for p in REFUSAL_PATTERNS)


# 生成 prompt 与 eval_generation.py 完全一致, 保证评测可比
GEN_PROMPT = """基于下面的检索片段回答用户问题。
规则:
1. 只能使用片段中的信息, 禁止使用你自己的知识;
2. 片段信息不足时, 直接回复"知识库中暂无相关资料", 不要编造;
3. 回答要简洁。

用户问题: {question}
检索片段:
{contexts}"""

REWRITE_PROMPT = """你是检索查询改写器。把用户的问题改写成 1~2 个更具体、更适合在知识库中检索的查询。
要求:
1. 保留原问题的核心实体与关键词;
2. 把代词/省略补全为具体名称(如"它""该函数"补全为实际名字);
3. 不要加解释, 每个查询独立成句。
只输出 JSON 数组, 例如: ["具体查询1", "具体查询2"]"""


async def _llm_chat(messages: list[dict], temperature: float, retries: int = 2) -> str:
    """非流式 LLM 调用 + 指数退避重试 + 硬超时; 全部失败返回空串"""
    for attempt in range(retries + 1):
        try:
            resp = await asyncio.wait_for(
                _client.chat.completions.create(
                    model=MODEL, temperature=temperature, messages=messages,
                ),
                timeout=60.0,          # 硬超时 60s
            )
            return (resp.choices[0].message.content or "").strip()
        except asyncio.TimeoutError:
            logger.warning("LLM 调用超时(第%s次)", attempt + 1)
        except Exception as e:
            logger.warning("LLM 调用失败(第%s次): %s: %s",
                           attempt + 1, type(e).__name__, e)
        if attempt == retries:
            return ""
        await asyncio.sleep(2 ** attempt)
    return ""


async def _generate(question: str, chunks: list[dict]) -> str:
    contexts = (
        "\n\n".join(f"[片段{j+1}]\n{r['content']}" for j, r in enumerate(chunks))
        if chunks
        else "(检索无结果)"
    )
    text = await _llm_chat(
        [{"role": "user", "content": GEN_PROMPT.format(question=question, contexts=contexts)}],
        temperature=0.3,
    )
    # 调用失败也视为"拒答", 交给上层走纠正; 保证流程永不崩溃
    return text or "知识库中暂无相关资料"


async def rewrite_query(question: str) -> list[str]:
    """LLM 改写问题 → 1~2 个更适合检索的 query; 失败时回退为原问题"""
    text = await _llm_chat(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    if not text:
        return [question]
    try:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"```(?:json)?|```", "", text)
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            return [question]
        queries = json.loads(text[start:end + 1])
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return cleaned[:2] if cleaned else [question]
    except Exception as e:
        logger.warning("rewrite_query 解析失败, 回退原问题: %s", e)
        return [question]


async def merged_retrieve(question: str, rewritten: list[str],
                          collection_name: str, k: int = 8) -> list[dict]:
    """原始 query + 改写 query 的检索结果合并去重, 扩大召回"""
    all_results: list[dict] = []
    seen: set[str] = set()
    for q in [question] + rewritten:
        results = await retrieve(q, collection_name, k=k)
        for r in results:
            key = r["content"][:200]   # 按内容前缀近似去重
            if key not in seen:
                seen.add(key)
                all_results.append(r)
    return all_results[:k]


async def self_correct_answer(question: str, collection_name: str,
                              max_attempts: int = 2, k: int = 4) -> dict:
    """自纠正主流程:
    1) 初始检索+生成; 2) 若拒答 → 改写query → 合并重检索 → 再生成;
    最多重试 max_attempts 轮; 仍拒答则返回最终拒答。"""
    chunks = await retrieve(question, collection_name, k=k)
    answer = await _generate(question, chunks)
    corrected = False
    attempts = 0
    while detect_refusal(answer) and attempts < max_attempts:
        attempts += 1
        rewritten = await rewrite_query(question)
        chunks = await merged_retrieve(question, rewritten, collection_name, k=8)
        answer = await _generate(question, chunks)
        corrected = True
    return {
        "question": question,
        "answer": answer,
        "chunks": chunks,
        "attempts": attempts,
        "corrected": corrected,
    }

