"""跨轮次上下文压缩：把早期对话压成一段摘要，写回 Redis。

【为什么只压跨轮，不压单次运行内】
一次"检索 → 回答"最多 8 次模型调用，消息量不大；而工具返回的文档原文既是
回答依据也是溯源依据，压掉等于丢证据。真正会膨胀的是跨轮历史（每轮 2 条）。

【策略】
    首条原文（含图片 VL 描述） + 早期对话摘要 + 最近 N 条原文

- 首条永不压：丢了它模型就"失明"（见 core/memory.py 的【首条常驻】）
- 摘要只覆盖 [1, total-keep) 这一段，游标 upto 保证幂等（不会把同一段反复摘要成糊状）
- **摘要生成失败不抛异常**，只是这次不压 —— 压缩绝不能把对话打挂

【为什么失败时保留旧摘要】
旧摘要已经覆盖了更早的内容，丢掉它等于让模型忘掉前文，比不压更糟。
"""
from __future__ import annotations

import asyncio
import logging

from openai import AsyncOpenAI

from . import memory
from .config import (
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    SUMMARY_ENABLED,
    SUMMARY_KEEP_RECENT,
    SUMMARY_MAX_CHARS,
    SUMMARY_TRIGGER_TURNS,
)

logger = logging.getLogger("knowledge_agent.summary")

# 与 core/diagnose.py 一样，用独立的非流式客户端：
# 摘要要的是完整文本与确定性输出，不要复用主对话那个 streaming、temperature=0.3 的实例。
_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

SUMMARY_TIMEOUT = 30

PROMPT = """下面是一段农业病虫害咨询对话（可能附带更早的摘要）。请把它压缩成一段背景摘要。

要求：
1. 保留：作物是什么、用户描述过的关键症状、已经排除的病害、已经给出的结论、尚未确认的问题。
2. 不要补充对话里没有的信息，不要下新的结论。
3. 用第三人称陈述，300 字以内，不要分点编号。
4. 直接输出摘要正文，不要任何前缀或解释。

{old_part}对话：
{segment}
"""


def compress_end(total: int, keep: int, upto: int, trigger: int) -> int | None:
    """算出这次该压到哪个位置（不含）。不需要压时返回 None。

    可压区间是 [1, total - keep)：
      - 从 1 开始：index 0 是首条，永久保留原文
      - 到 total - keep 结束：最近 keep 条保留原文

    返回 None 的两种情况：
      - 总条数还没到阈值
      - 该区间已被上次摘要覆盖（游标 >= 起点），说明没有新内容可压
    """
    if total <= trigger:
        return None
    end = total - keep
    if end <= max(upto, 1):
        return None
    return end


def segment_to_text(segment: list[dict]) -> str:
    """把消息列表拼成给模型看的对话文本。"""
    lines = []
    for m in segment:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        role = "用户" if m.get("role") == "user" else "助手"
        if m.get("partial"):
            # 被熔断截断的回答要标出来，否则摘要会把它当成完整结论
            role += "（该回答被截断）"
        lines.append(f"{role}：{content}")
    return "\n".join(lines)


async def _call_llm(prompt: str) -> str:
    resp = await asyncio.wait_for(
        _client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,  # 摘要要稳定，不要发挥
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=SUMMARY_TIMEOUT,
    )
    return (resp.choices[0].message.content or "").strip()


async def maybe_summarize(user_id: int, session_id: str) -> None:
    """在读历史之前调用。不满足条件时立即返回，满足时把新摘要写回 Redis。

    任何失败都只记 warning，不影响本次对话。
    """
    if not SUMMARY_ENABLED:
        return

    try:
        total = memory.history_length(user_id, session_id)
        upto = memory.read_upto(user_id, session_id)
    except Exception as e:
        logger.warning("读取历史长度失败，跳过压缩: %s: %s", type(e).__name__, e)
        return

    end = compress_end(total, SUMMARY_KEEP_RECENT, upto, SUMMARY_TRIGGER_TURNS)
    if end is None:
        return

    segment = memory.read_range(user_id, session_id, upto, end)
    if not segment:
        return

    old = memory.read_summary(user_id, session_id)
    old_part = f"更早的摘要：\n{old}\n\n" if old else ""
    prompt = PROMPT.format(old_part=old_part, segment=segment_to_text(segment))

    try:
        text = await _call_llm(prompt)
    except Exception as e:
        # 必须带 type(e).__name__：TimeoutError 的 str() 是空串
        logger.warning(
            "摘要生成失败（本次不压缩，旧摘要保留）: %s: %s", type(e).__name__, e
        )
        return

    # 防御性 strip：虽然 _call_llm 已经 strip 过，但空值判断不能依赖上游自觉。
    # 不 strip 的话，一段纯空白会被写进 Redis 并推进游标 —— 那段历史就永久丢失了。
    text = (text or "").strip()
    if not text:
        logger.warning("摘要结果为空，本次不压缩")
        return

    text = text[:SUMMARY_MAX_CHARS]
    try:
        memory.write_summary(user_id, session_id, text, end)
    except Exception as e:
        logger.warning("摘要写入失败: %s: %s", type(e).__name__, e)
        return

    logger.info(
        "已压缩 %d 条历史（%d -> %d，保留最近 %d 条）",
        max(end - upto, 0),
        total,
        total - max(end - upto, 0),
        SUMMARY_KEEP_RECENT,
    )
