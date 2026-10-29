"""run_state 的跨 task 语义 —— P1 缺陷的回归测试。

背景：早期版本用 5 个独立 ContextVar，而 asyncio.create_task 复制的是
context 的映射，子 task 里 set() 不会回流。LangGraph 把每个节点跑在独立
task 里，结果就是 step 永远是 1、stop_reason 传不回 chat.py。

改成共享 RunContext 对象后，子 task 改字段父 task 必须可见。
"""
import asyncio
import hashlib

from core import run_state


def test_child_task_mutation_visible_in_parent():
    async def main():
        ctx = run_state.start(7, "s-1")

        async def child():
            run_state.set_stop_reason("max_model_calls")
            return run_state.next_step()

        step = await asyncio.create_task(child())
        # 在父任务的上下文里读，必须看到子任务的写入
        return ctx, step, run_state.get_stop_reason()

    ctx, step, reason = asyncio.run(main())

    assert step == 1
    assert ctx.step == 1                      # 共享对象：累加对父侧可见
    assert ctx.stop_reason == "max_model_calls"
    assert reason == "max_model_calls"        # 父任务读得到


def test_start_initialises_fields():
    async def main():
        ctx = run_state.start(42, "s-2")
        return ctx

    ctx = asyncio.run(main())

    assert len(ctx.request_id) == 12
    # 只落 sha1 前 8 位，不落明文用户 ID
    assert ctx.user_key == hashlib.sha1(b"42").hexdigest()[:8]
    assert ctx.session_id == "s-2"
    assert ctx.step == 0
    assert ctx.stop_reason is None


def test_step_increments_monotonically():
    async def main():
        run_state.start(1, "s-3")
        return [run_state.next_step() for _ in range(5)]

    assert asyncio.run(main()) == [1, 2, 3, 4, 5]
