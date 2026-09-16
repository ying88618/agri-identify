"""多轮记忆测试：守护「首条常驻」这一关键策略。

背景：首条 user 消息里带着图片的 VL 描述，而后续轮次不再传图。
若只取「最近 n 条」，第 5 轮起描述会被挤出窗口，模型从此「失明」。
故策略是：首条 + 最近 (n-1) 条。这里用假 Redis 隔离，不需要真实服务。
"""
import json

import core.memory as memory


class FakeRedis:
    """只实现 load_history 用到的 llen / lrange。

    raw=False：把 dict 序列化成 JSON，模拟 append_turn 的正常写入。
    raw=True ：直接返回列表原值，用于模拟 Redis 里已存在脏数据的情况。
    """

    def __init__(self, turns, raw=False):
        self.items = list(turns)
        self.raw = raw

    def llen(self, key):
        return len(self.items)

    def lrange(self, key, start, end):
        if end == -1:
            end = len(self.items) - 1
        page = self.items[start:end + 1]
        if self.raw:
            return list(page)
        return [json.dumps(t, ensure_ascii=False) for t in page]


def _turns(n, prefix="m"):
    return [{"role": "user", "content": f"{prefix}{i}"} for i in range(n)]


def test_keeps_first_turn_when_history_exceeds_window(monkeypatch):
    monkeypatch.setattr(memory, "_redis", FakeRedis(_turns(12)))
    out = memory.load_history(user_id=1, session_id="s", n=6)

    assert len(out) == 6, "窗口大小应等于 n"
    assert out[0]["content"] == "m0", \
        "首条必须常驻 —— 图片的 VL 症状描述存在这里，丢了模型就『失明』"
    assert out[-1]["content"] == "m11", "尾部应取最近一条"


def test_returns_all_when_history_shorter_than_window(monkeypatch):
    monkeypatch.setattr(memory, "_redis", FakeRedis(_turns(3)))
    out = memory.load_history(user_id=1, session_id="s", n=6)
    assert [o["content"] for o in out] == ["m0", "m1", "m2"]


def test_drops_leading_assistant_message(monkeypatch):
    """流式中断会留下不配对的 assistant 尾巴，序列不能以 assistant 开头。"""
    turns = [
        {"role": "assistant", "content": "孤儿消息"},
        {"role": "user", "content": "问题"},
        {"role": "assistant", "content": "回答"},
    ]
    monkeypatch.setattr(memory, "_redis", FakeRedis(turns))
    out = memory.load_history(user_id=1, session_id="s", n=6)

    assert out[0]["role"] == "user"
    assert [o["content"] for o in out] == ["问题", "回答"]


def test_skips_corrupted_json_entries(monkeypatch):
    fake = FakeRedis(
        ['{"role": "user", "content": "ok"}', "{不是合法JSON}"],
        raw=True,
    )
    monkeypatch.setattr(memory, "_redis", fake)

    out = memory.load_history(user_id=1, session_id="s", n=6)
    assert [o["content"] for o in out] == ["ok"], "脏数据应跳过而不是抛异常"


def test_skips_non_dict_json_entries(monkeypatch):
    """合法 JSON 但非对象的值同样不能打挂请求（曾会 AttributeError）。"""
    fake = FakeRedis(
        ['"abc"', "123", '{"role": "user", "content": "ok"}'],
        raw=True,
    )
    monkeypatch.setattr(memory, "_redis", fake)

    out = memory.load_history(user_id=1, session_id="s", n=6)
    assert [o["content"] for o in out] == ["ok"]


def test_key_is_scoped_by_user_and_session():
    assert memory._key(7, "abc") == "user_memory:7:abc"
