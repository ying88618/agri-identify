"""预算类中间件：模型调用上限、工具调用上限。

【墙钟超时不在这里】
它由 api/chat.py 的 asyncio.timeout 执行。原因：超时之后必须还能向后端推
SSE 事件（告诉用户"被截断了"），而中间件没有发事件的能力。
把"停"和"说清楚为什么停"分开，是这一层的核心约束。

【工具上限为什么用 continue 而不是 end】
continue 会给模型返回一条"别再调这个工具"的 ToolMessage，让模型拿已有信息作答。
对诊断类应用，硬中断的产物是"正在为您检索…"，而 continue 的产物是一个
带不确定性的结论 —— 后者对用户有用得多。
"""
from __future__ import annotations

from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)

from .. import run_state
from ..config import MAX_MODEL_CALLS, MAX_TOOL_CALLS


class StepLimitMiddleware(ModelCallLimitMiddleware):
    """在官方件外面套一层，把"触发了上限"写进 RunContext。

    官方件超限时只会返回 {"jump_to": "end", ...}，不会通知任何人。
    没有这个桥接，chat.py 就不知道这次结束是熔断还是正常结束 ——
    用户看到的就是静默截断。
    """

    # 【装饰器必须重加】create_agent 用 _get_can_jump_to() 读的是
    # `类.before_model.__can_jump_to__` 这个类属性。子类覆写后若不重加，
    # 属性就丢了 → jump 静默失效，且只在真跑到上限时才暴露。
    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        result = super().before_model(state, runtime)
        if result is not None:
            run_state.set_stop_reason("max_model_calls")
        return result


def build_step_limit() -> StepLimitMiddleware:
    # run_limit 而非 thread_limit：thread 级计数需要 checkpointer 才有意义，
    # 本项目没有 checkpointer，thread 计数永远是 0。
    return StepLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end")


def build_tool_limit() -> ToolCallLimitMiddleware:
    return ToolCallLimitMiddleware(run_limit=MAX_TOOL_CALLS, exit_behavior="continue")
