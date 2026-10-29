"""预算中间件的行为契约。

这里不跑真模型（那需要网络），只验证两件容易静默失效的事：
1. 子类覆写后 @hook_config 还在 —— 丢了就 jump 不了，熔断变成空操作
2. 触发上限时会写 stop_reason —— 不写的话上层看到的是"正常结束"
"""
from core import run_state
from core.config import MAX_TOOL_CALLS
from core.mw.limits import StepLimitMiddleware, build_tool_limit


def _state(n: int) -> dict:
    """模拟 ModelCallLimitState：两个计数都置为 n。"""
    return {"thread_model_call_count": n, "run_model_call_count": n}


def test_hook_config_present():
    """漏掉 @hook_config 时 create_agent 会认为该中间件不能 jump_to end，
    熔断静默失效 —— 只在真跑到上限时才暴露，所以用断言钉住。"""
    assert getattr(StepLimitMiddleware.before_model, "__can_jump_to__", None) == ["end"]


def test_below_limit_returns_none_and_keeps_reason_empty():
    run_state.start(1, "s-limit")

    mw = StepLimitMiddleware(run_limit=2, exit_behavior="end")
    assert mw.before_model(_state(0), None) is None
    assert run_state.get_stop_reason() is None


def test_reaching_limit_jumps_and_sets_reason():
    run_state.start(1, "s-limit")

    mw = StepLimitMiddleware(run_limit=2, exit_behavior="end")
    out = mw.before_model(_state(2), None)

    assert isinstance(out, dict)
    assert out.get("jump_to") == "end"
    assert run_state.get_stop_reason() == "max_model_calls"


def test_tool_limit_uses_continue_semantics():
    """工具超限不终止运行：给模型一条提示，让它用已有信息作答。"""
    mw = build_tool_limit()
    assert mw.exit_behavior == "continue"
    assert mw.run_limit == MAX_TOOL_CALLS
