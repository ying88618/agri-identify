"""预算中间件的行为契约。

这里不跑真模型（那需要网络），只验证两件容易静默失效的事：
1. 子类覆写后 @hook_config 还在 —— 丢了就 jump 不了，熔断变成空操作
2. 触发上限时会写 stop_reason —— 不写的话上层看到的是"正常结束"
"""
import asyncio

from langchain_core.messages import ToolMessage

from core import run_state
from core.config import MAX_REPEAT_ACTIONS, MAX_TOOL_CALLS, MAX_TOTAL_TOKENS
from core.mw.limits import (
    RepeatActionMiddleware,
    StepLimitMiddleware,
    TokenBudgetMiddleware,
    action_signature,
    build_repeat_guard,
    build_token_budget,
    build_tool_limit,
)


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


def test_token_budget_below_limit_does_nothing():
    run_state.start(1, "s-token")
    mw = TokenBudgetMiddleware(max_total_tokens=1000)
    run_state.add_usage({"input_tokens": 100, "output_tokens": 100, "total_tokens": 200})

    assert mw.after_model({}, None) is None
    assert run_state.get_stop_reason() is None


def test_token_budget_exceeded_jumps_and_sets_reason():
    run_state.start(1, "s-token")
    mw = TokenBudgetMiddleware(max_total_tokens=1000)
    run_state.add_usage({"input_tokens": 600, "output_tokens": 600, "total_tokens": 1200})

    assert mw.after_model({}, None) == {"jump_to": "end"}
    assert run_state.get_stop_reason() == "max_tokens"


def test_token_budget_counts_reasoning_tokens():
    """推理 token 计入 total：provider 的 total_tokens 已包含它。"""
    run_state.start(1, "s-token")
    mw = TokenBudgetMiddleware(max_total_tokens=500)
    run_state.add_usage(
        {
            "input_tokens": 100,
            "output_tokens": 450,
            "total_tokens": 550,
            "output_token_details": {"reasoning": 400},
        }
    )

    assert mw.after_model({}, None) == {"jump_to": "end"}
    assert run_state.usage()["reasoning_tokens"] == 400


def test_token_budget_hook_config_on_both_paths():
    """同步与异步路径都要带能力标记：astream 走 aafter_model，
    漏了装饰器就会在异步路径上静默失去 jump 能力。"""
    assert getattr(TokenBudgetMiddleware.after_model, "__can_jump_to__", None) == ["end"]
    assert getattr(TokenBudgetMiddleware.aafter_model, "__can_jump_to__", None) == ["end"]


def test_token_budget_async_path_delegates():
    async def main():
        run_state.start(1, "s-token-async")
        mw = TokenBudgetMiddleware(max_total_tokens=10)
        run_state.add_usage({"total_tokens": 20})
        return await mw.aafter_model({}, None)

    assert asyncio.run(main()) == {"jump_to": "end"}


def test_token_budget_reads_config():
    assert build_token_budget().max_total_tokens == MAX_TOTAL_TOKENS


# ---------------- 重复动作检测 ----------------


class _Req:
    """最小化的 ToolCallRequest 替身：只提供 wrap_tool_call 会用到的字段。"""

    def __init__(self, name, args, call_id="c1"):
        self.tool_call = {"name": name, "args": args, "id": call_id}


def test_action_signature_normalises():
    # None 与空串等价；多余空白被压缩；key 顺序无关
    assert action_signature("kb", {"crop": "", "query": "番茄  黑点"}) == action_signature(
        "kb", {"query": "番茄 黑点", "crop": None}
    )
    # 不同工具、不同参数必须区分
    assert action_signature("kb", {"q": "a"}) != action_signature("web", {"q": "a"})
    assert action_signature("kb", {"q": "a"}) != action_signature("kb", {"q": "b"})


def test_repeat_blocks_third_identical_call():
    async def main():
        run_state.start(1, "s-repeat")
        mw = RepeatActionMiddleware(max_repeats=3)
        executed = []

        async def handler(request):
            executed.append(request.tool_call["args"])
            return ToolMessage(content="ok", tool_call_id="c1", name="kb")

        results = [
            await mw.awrap_tool_call(_Req("kb", {"query": "番茄 黑点"}), handler)
            for _ in range(4)
        ]
        return executed, results

    executed, results = asyncio.run(main())

    assert len(executed) == 2  # 第 3 次起被短路，工具不再执行
    assert results[2].content.startswith("同一检索已连续执行 3 次")
    assert results[3].content.startswith("同一检索已连续执行 4 次")


def test_repeat_resets_on_different_args():
    async def main():
        run_state.start(1, "s-repeat-2")
        mw = RepeatActionMiddleware(max_repeats=2)
        executed = []

        async def handler(request):
            executed.append(1)
            return ToolMessage(content="ok", tool_call_id="c1")

        for q in ("a", "b", "a", "b"):
            await mw.awrap_tool_call(_Req("kb", {"query": q}), handler)
        return executed

    # 交替参数永远不构成"连续相同"，一次都不该被拦
    assert len(asyncio.run(main())) == 4


def test_repeat_treats_empty_and_none_as_same():
    async def main():
        run_state.start(1, "s-repeat-3")
        mw = RepeatActionMiddleware(max_repeats=2)
        executed = []

        async def handler(request):
            executed.append(1)
            return ToolMessage(content="ok", tool_call_id="c1")

        await mw.awrap_tool_call(_Req("kb", {"query": "番茄", "crop": ""}), handler)
        await mw.awrap_tool_call(_Req("kb", {"query": "番茄", "crop": None}), handler)
        return executed

    # 模型在两次调用里一次传 ""、一次传 None，必须视为同一个动作
    assert len(asyncio.run(main())) == 1


def test_repeat_guard_reads_config():
    assert build_repeat_guard().max_repeats == MAX_REPEAT_ACTIONS
