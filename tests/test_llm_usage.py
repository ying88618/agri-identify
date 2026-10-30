"""用量采集的回归测试。

核心保护的是一条：**provider 每个 chunk 都带累计 usage，采集必须取最后一块，
不能相加。** 一旦有人"顺手"改成累加，线上就会重现 78 倍膨胀。
"""
import asyncio
from types import SimpleNamespace

from langchain_openai import ChatOpenAI

from core import run_state
from core.llm_usage import UsageTrackingChatOpenAI, _UsageCollector, extract_usage


def _chunk(input_tokens=5, output_tokens=10, reasoning=0):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "output_token_details": {"reasoning": reasoning},
    }
    return SimpleNamespace(message=SimpleNamespace(usage_metadata=usage))


def test_collector_keeps_last_not_sum():
    col = _UsageCollector()
    # 模拟 provider 行为：每块都是"累计值"，不是增量
    for n in (10, 200, 1638):
        col.feed(_chunk(output_tokens=n))

    assert col.value["output_tokens"] == 1638
    # 若实现成相加，这里会变成 1848 —— 正是线上 78 倍膨胀的来源
    assert col.value["output_tokens"] != 10 + 200 + 1638


def test_collector_ignores_chunks_without_usage():
    col = _UsageCollector()
    col.feed(_chunk(output_tokens=100))
    col.feed(SimpleNamespace(message=SimpleNamespace(usage_metadata=None)))
    assert col.value["output_tokens"] == 100


def test_extract_usage_handles_missing_message():
    assert extract_usage(SimpleNamespace()) is None


def test_add_usage_accumulates_and_delta_is_per_call():
    """trace 里记的是"后减前"的增量，所以累加必须单调递增。"""

    async def main():
        run_state.start(1, "s-usage")
        run_state.add_usage(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "output_token_details": {"reasoning": 20},
            }
        )
        before = run_state.usage()
        run_state.add_usage(
            {
                "input_tokens": 110,
                "output_tokens": 90,
                "total_tokens": 200,
                "output_token_details": {"reasoning": 60},
            }
        )
        return before, run_state.usage()

    before, after = asyncio.run(main())

    assert before["total_tokens"] == 150
    assert after["total_tokens"] == 350
    # 单次调用的增量
    assert after["prompt_tokens"] - before["prompt_tokens"] == 110
    assert after["reasoning_tokens"] == 80  # 20 + 60


def test_add_usage_tolerates_none():
    async def main():
        run_state.start(1, "s-usage-2")
        run_state.add_usage(None)
        return run_state.usage()

    assert asyncio.run(main())["total_tokens"] == 0


def test_tracking_model_is_chatopenai():
    """我们的子类必须仍然是 ChatOpenAI，否则 create_agent 的模型能力探测会失效。"""
    assert issubclass(UsageTrackingChatOpenAI, ChatOpenAI)
