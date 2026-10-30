"""跨轮压缩的行为契约：阈值边界、幂等、失败降级。

不调真实 LLM：`_call_llm` 被 monkeypatch 掉，只验证"什么时候压、压完写什么、
失败时不动旧数据"这三件事。
"""
import asyncio
import json

import core.memory as memory
import core.summary as summary


class FakeStore:
    """够用的假 Redis：list + kv。"""

    def __init__(self, turns):
        self.items = [json.dumps(t, ensure_ascii=False) for t in turns]
        self.kv: dict[str, str] = {}

    def llen(self, key):
        return len(self.items)

    def lrange(self, key, start, end):
        if end == -1:
            end = len(self.items) - 1
        return self.items[start:end + 1]

    def get(self, key):
        return self.kv.get(key)

    def set(self, key, value):
        self.kv[key] = value

    def expire(self, key, ttl):
        return True


def _turns(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(n)]


# ---------------- 纯逻辑 ----------------


def test_compress_end_respects_trigger_and_keep():
    assert summary.compress_end(10, 6, 1, 12) is None, "没到阈值不该压"
    assert summary.compress_end(14, 6, 1, 12) == 8, "可压区间是 [1, total-keep)"
    assert summary.compress_end(14, 6, 8, 12) is None, "游标已到，没有新内容"
    assert summary.compress_end(20, 6, 8, 12) == 14, "有新增才继续压"


def test_compress_end_never_touches_first_turn():
    """无论多少条，可压起点至少是 1 —— 首条（图片 VL 描述）永远保留原文。"""
    assert summary.compress_end(100, 6, 0, 12) == 94
    assert summary.compress_end(15, 6, 0, 12) > 1


def test_segment_to_text_marks_partial_and_skips_empty():
    text = summary.segment_to_text(
        [
            {"role": "user", "content": "番茄有黑点"},
            {"role": "assistant", "content": "疑似晚疫病", "partial": True},
            {"role": "user", "content": "   "},
        ]
    )
    assert "用户：番茄有黑点" in text
    # 被熔断截断的回答必须标注，否则摘要会把它当完整结论
    assert "（该回答被截断）" in text
    assert len(text.splitlines()) == 2  # 空内容被跳过


# ---------------- 主流程 ----------------


def test_writes_summary_and_cursor(monkeypatch):
    store = FakeStore(_turns(14))
    monkeypatch.setattr(memory, "_redis", store)

    async def fake_llm(prompt):
        return "用户咨询番茄病害，已排除早疫病。"

    monkeypatch.setattr(summary, "_call_llm", fake_llm)

    asyncio.run(summary.maybe_summarize(1, "s"))

    assert store.kv["user_memory:1:s:summary"].startswith("用户咨询番茄")
    assert store.kv["user_memory:1:s:upto"] == "8"  # 14 - 6


def test_second_run_is_idempotent(monkeypatch):
    store = FakeStore(_turns(14))
    monkeypatch.setattr(memory, "_redis", store)
    calls = []

    async def fake_llm(prompt):
        calls.append(prompt)
        return "摘要"

    monkeypatch.setattr(summary, "_call_llm", fake_llm)

    asyncio.run(summary.maybe_summarize(1, "s"))
    asyncio.run(summary.maybe_summarize(1, "s"))

    assert len(calls) == 1, "同一区间不能被重复摘要"


def test_failure_keeps_old_summary_and_cursor(monkeypatch):
    store = FakeStore(_turns(14))
    store.kv["user_memory:1:s:summary"] = "旧摘要"
    monkeypatch.setattr(memory, "_redis", store)

    async def boom(prompt):
        raise RuntimeError("模型超时")

    monkeypatch.setattr(summary, "_call_llm", boom)

    asyncio.run(summary.maybe_summarize(1, "s"))  # 不应抛异常

    assert store.kv["user_memory:1:s:summary"] == "旧摘要"
    assert "user_memory:1:s:upto" not in store.kv, "失败时不能推进游标"


def test_empty_result_not_written(monkeypatch):
    store = FakeStore(_turns(14))
    monkeypatch.setattr(memory, "_redis", store)

    async def empty(prompt):
        return "   "

    monkeypatch.setattr(summary, "_call_llm", empty)

    asyncio.run(summary.maybe_summarize(1, "s"))
    assert store.kv == {}


def test_disabled_does_nothing(monkeypatch):
    store = FakeStore(_turns(30))
    monkeypatch.setattr(memory, "_redis", store)
    monkeypatch.setattr(summary, "SUMMARY_ENABLED", False)

    asyncio.run(summary.maybe_summarize(1, "s"))
    assert store.kv == {}


def test_redis_error_does_not_raise(monkeypatch):
    class Broken(FakeStore):
        def llen(self, key):
            raise ConnectionError("redis down")

    monkeypatch.setattr(memory, "_redis", Broken(_turns(20)))

    asyncio.run(summary.maybe_summarize(1, "s"))  # 不应抛异常
