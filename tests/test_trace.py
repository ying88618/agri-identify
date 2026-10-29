"""core/trace.py 的行为契约。

不测中间件（那需要真模型），只测"事件怎么落下、落不下时怎么退化"。
"""
import asyncio
import hashlib
import json

import core.trace as trace


def _read(tmp_path) -> list[dict]:
    files = sorted(tmp_path.glob("*.jsonl"))
    assert len(files) == 1, f"期望 1 个文件，实际 {[f.name for f in files]}"
    lines = files[0].read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines]


def test_events_written_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path)

    async def run():
        await trace.start_run(42, "s-1")
        trace.trace_event("run_start", question_len=3)
        trace.trace_event("tool_end", tool="knowledge_base_search")
        await trace.shutdown()

    asyncio.run(run())

    rows = _read(tmp_path)
    assert [r["event"] for r in rows] == ["run_start", "tool_end"]
    # step 从 1 起递增，用于复现"第几步开始发散"
    assert [r["step"] for r in rows] == [1, 2]
    # 同一请求的事件必须能串起来
    assert rows[0]["request_id"]
    assert rows[0]["request_id"] == rows[1]["request_id"]
    # user 只落 hash 前缀，不落明文
    assert rows[0]["user"] == hashlib.sha1(b"42").hexdigest()[:8]
    assert rows[1]["tool"] == "knowledge_base_search"
    assert rows[0]["session"] == "s-1"


def test_queue_full_drops_instead_of_raising(monkeypatch, tmp_path):
    """队列满时必须静默丢弃 —— trace 不允许拖垮业务。"""
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(trace, "TRACE_QUEUE_MAX", 1)
    monkeypatch.setattr(trace, "_dropped", 0)

    async def run():
        await trace.start_run(1, "s-2")
        for i in range(5):
            trace.trace_event("tool_end", seq=i)
        await trace.shutdown()

    asyncio.run(run())  # 不抛异常即通过

    assert trace._dropped >= 1
    # 落盘条数少于写入条数（其余被丢弃）
    assert len(_read(tmp_path)) < 5


def test_disabled_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(trace, "TRACE_DIR", tmp_path)
    monkeypatch.setattr(trace, "TRACE_ENABLED", False)

    async def run():
        await trace.start_run(1, "s-3")
        trace.trace_event("run_start")  # 不应抛异常
        await trace.shutdown()

    asyncio.run(run())

    assert list(tmp_path.glob("*.jsonl")) == []
