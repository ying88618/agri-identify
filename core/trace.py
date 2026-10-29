"""结构化 trace：每步一条 JSONL，异步落盘。

【三个设计约束，每条都对应一个真实故障模式】

1. 绝不阻塞事件循环。
   钩子跑在 asyncio 里，直接 open().write() 会让整个 SSE 流卡住。
   所以：入队用 put_nowait，落盘交给后台 task + asyncio.to_thread。

2. trace 不能拖垮主流程。
   队列有上限；满了就丢弃并计数，只每 100 条打一次 warning。
   观测系统的正确失败方式是"静默降级"，而不是让业务挂掉。

3. 进程退出必须 flush。
   run_end 常常是最后一条记录，也是排查时最需要的一条。
   靠 main.py 的 lifespan 在 shutdown 阶段调用 shutdown()。

【并发说明】
现在是单 worker 部署（见 Dockerfile 的 --workers 1），多个协程共用一个 writer task。
若将来开多 worker，多个进程会写同一个按天文件 —— 单行写入是原子的（<PIPE_BUF），
但不同进程的 flush 顺序不保证。真要多 worker 就把文件名带上 PID。
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from . import run_state
from .config import TRACE_DIR, TRACE_ENABLED, TRACE_QUEUE_MAX

logger = logging.getLogger("knowledge_agent.trace")

_queue: asyncio.Queue[str] | None = None
_task: asyncio.Task[None] | None = None
_loop: asyncio.AbstractEventLoop | None = None
_dropped = 0
_SENTINEL = object()
BATCH_SIZE = 50
FLUSH_INTERVAL = 1.0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _append_lines(lines: list[str]) -> None:
    """线程内执行：一次 open + 批量追加，按天切文件。

    路径在写入时计算，所以跨零点的批次会落到新一天的文件里（符合直觉）。
    """
    path = TRACE_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


async def _flush(batch: list[str]) -> None:
    try:
        await asyncio.to_thread(_append_lines, list(batch))
    except Exception as e:
        # 磁盘满 / 权限问题：只告警，不影响业务
        logger.warning("trace 落盘失败: %s: %s", type(e).__name__, e)


async def _writer_loop(q: asyncio.Queue[str]) -> None:
    batch: list[str] = []
    while True:
        try:
            item = await asyncio.wait_for(q.get(), timeout=FLUSH_INTERVAL)
        except asyncio.TimeoutError:
            if batch:
                await _flush(batch)
                batch.clear()
            continue

        if item is _SENTINEL:
            while True:
                try:
                    batch.append(q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if batch:
                await _flush(batch)
            return

        batch.append(item)
        if len(batch) >= BATCH_SIZE:
            await _flush(batch)
            batch.clear()


async def _ensure_started() -> None:
    """惰性创建队列与后台 task。

    必须在事件循环里调用（start_run 是 async），否则 Task 无法创建。

    【为什么不需要锁】asyncio 是单线程协作式调度，下面几行之间没有 await，
    检查与赋值不会被其它协程插入，天然原子。

    【为什么要判断 loop 变了】asyncio.Queue 一旦被使用就绑定到当时的 loop，
    跨 loop 复用会抛 "bound to a different event loop"。生产只有一个 loop，
    但测试里每次 asyncio.run() 都会新建一个，所以这里按 loop 重置句柄。
    """
    global _queue, _task, _loop
    if not TRACE_ENABLED:
        return
    loop = asyncio.get_running_loop()
    if _loop is not loop:
        _loop = loop
        _queue = asyncio.Queue(maxsize=TRACE_QUEUE_MAX)
        _task = None
    if _task is None or _task.done():
        _task = loop.create_task(_writer_loop(_queue))


async def start_run(user_id: int, session_id: str) -> None:
    """每个请求开头调一次：初始化运行上下文并启动 writer。

    user_id 在 run_state.start 里被转成 sha1 前 8 位 —— 排查时能区分
    "同一用户反复触发"，又不把用户 ID 明文写进日志文件。
    """
    if not TRACE_ENABLED:
        return
    await _ensure_started()
    run_state.start(user_id, session_id)


def trace_event(event: str, **fields) -> None:
    """记录一条事件。同步、非阻塞、永不抛异常。

    在入队前就序列化：这样调用方之后修改传入的对象也不会影响已记录的内容。
    """
    global _dropped
    if not TRACE_ENABLED or _queue is None:
        return

    # 注意：必须读同一个 RunContext 对象的字段。
    # 若改回多个独立 ContextVar，中间件（跑在子 task 里）的写入不会回流到 chat.py，
    # step 会永远停在 1、stop_reason 也传不出去（P1 的真实缺陷，P2 修正）。
    ctx = run_state.ensure()
    record: dict = {
        "ts": _now_iso(),
        "request_id": ctx.request_id,
        "user": ctx.user_key,
        "session": ctx.session_id,
        "step": run_state.next_step(),
        "event": event,
    }
    record.update(fields)

    try:
        _queue.put_nowait(json.dumps(record, ensure_ascii=False, default=str))
    except asyncio.QueueFull:
        _dropped += 1
        if _dropped % 100 == 1:
            logger.warning("trace 队列已满，累计丢弃 %d 条", _dropped)


async def shutdown() -> None:
    """停止 writer 并 flush 剩余记录。在 main.py 的 lifespan 退出阶段调用。"""
    global _task
    if _queue is None or _task is None:
        return
    try:
        await asyncio.wait_for(_queue.put(_SENTINEL), timeout=2)
        await asyncio.wait_for(_task, timeout=5)
    except Exception as e:
        logger.warning("trace flush 失败（可能丢最后几条）: %s: %s", type(e).__name__, e)
    finally:
        _task = None
