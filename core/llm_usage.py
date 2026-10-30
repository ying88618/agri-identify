"""流式用量采集：绕开 LangChain 的 chunk 合并。

【问题】
本项目的 provider（SiliconFlow）**在每个流式 chunk 里都带一份"累计" usage**，
不符合 OpenAI 规范（规范只在最后一个 chunk 给一次）。

而 LangChain 合并 AIMessageChunk 时，`langchain_core/messages/ai.py` 用的是
`add_usage(usage_metadata, other.usage_metadata)` —— **相加**。

于是 N 个 chunk 的累计值被加了 N 次。实测：一个 12 字问题的
prompt_tokens 报出 60256，约为真实值（≈772）的 78 倍。

【修法】
不自己数 token（推理模型的 reasoning_tokens 不经过输出流，数不到），
而是**在 _astream 里截住最后一个 chunk 的 usage**——它本身就是完整累计值。
拿到的是 provider 的原始真值，只是不经过 LangChain 的聚合路径。

【代价】
依赖 `_astream` 这个私有方法。LangChain 升级时要回归
tests/test_llm_usage.py 里那条"取最后一块、不是相加"的用例。
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI

from . import run_state


def extract_usage(chunk) -> dict | None:
    """从 ChatGenerationChunk 里取归一化后的 usage。取不到返回 None。"""
    message = getattr(chunk, "message", None)
    if message is None:
        return None
    return getattr(message, "usage_metadata", None)


class _UsageCollector:
    """收集流式块里的 usage。

    【关键语义】后一块覆盖前一块，**绝不相加**。provider 给的是累计值。
    """

    def __init__(self) -> None:
        self._last: dict | None = None

    def feed(self, chunk) -> None:
        usage = extract_usage(chunk)
        if usage:
            self._last = usage

    @property
    def value(self) -> dict | None:
        return self._last


class UsageTrackingChatOpenAI(ChatOpenAI):
    """在流式过程中收集真实用量，一次模型调用结束后累加进 RunContext。"""

    async def _astream(self, *args, **kwargs) -> AsyncIterator:
        collector = _UsageCollector()
        async for chunk in super()._astream(*args, **kwargs):
            collector.feed(chunk)
            yield chunk
        # 循环结束才累加：此时拿到的是最后一个 chunk 的完整累计值
        if collector.value:
            run_state.add_usage(collector.value)
