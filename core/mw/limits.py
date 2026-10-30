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

import hashlib
import json

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    hook_config,
)
from langchain_core.messages import ToolMessage

from .. import run_state
from ..config import (
    MAX_MODEL_CALLS,
    MAX_REPEAT_ACTIONS,
    MAX_TOOL_CALLS,
    MAX_TOTAL_TOKENS,
)
from ..trace import trace_event


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


class TokenBudgetMiddleware(AgentMiddleware):
    """累计 token 超限即熔断。

    【为什么挂 after_model】
    用量只有一次调用结束后才知道。想提前在 before_model 拦，就得预估"下一次要花多少"，
    而推理模型的 reasoning_tokens 根本估不出来（同一句话实测 1 到 631 都出现过）。
    代价是会多花一次调用的 token 才熔断 —— 这是精度与严格性的取舍，不是缺陷。

    【为什么不用 response.usage_metadata】
    本 provider 在每个流式 chunk 都带累计 usage，LangChain 合并时相加，数值膨胀上百倍。
    这里读 core/run_state 采集的真值（见 core/llm_usage.py）。
    """

    def __init__(self, max_total_tokens: int) -> None:
        super().__init__()
        # 不可变配置放实例上是安全的：中间件实例全局共享，但上限本来就不随请求变化
        self.max_total_tokens = max_total_tokens

    @hook_config(can_jump_to=["end"])
    def after_model(self, state, runtime):
        used = run_state.usage()["total_tokens"]
        if used >= self.max_total_tokens:
            run_state.set_stop_reason("max_tokens")
            # 不注入 messages：jump_to=end 已跳过后续模型调用，
            # 塞一条文案只会污染 state，且英文文案对用户没有意义。
            # 用户侧的提示由 api/chat.py 的 truncated 事件负责。
            return {"jump_to": "end"}
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state, runtime):
        # astream 走异步路径；必须同步声明装饰器，否则 _get_can_jump_to 取不到能力标记
        return self.after_model(state, runtime)


def build_token_budget() -> TokenBudgetMiddleware:
    return TokenBudgetMiddleware(max_total_tokens=MAX_TOTAL_TOKENS)


def action_signature(tool_name: str | None, args) -> str:
    """把 (工具名, 参数) 归一化成可比较的签名。

    归一化做三件事，缺一条都会漏检，而漏检的表现是"死循环没被拦住"，很难复现：
      - None 与空串都视为"没传"（模型有时传 crop=""，有时传 None）
      - 字符串压掉多余空白（"番茄 黑点" 与 "番茄   黑点" 是同一个动作）
      - key 排序（同一个 dict 的插入顺序可能不同）
    """
    if isinstance(args, dict):
        cleaned: dict = {}
        for k, v in args.items():
            if v is None:
                continue
            if isinstance(v, str):
                v = " ".join(v.split())
                if not v:
                    continue
            cleaned[k] = v
    else:
        cleaned = args

    try:
        body = json.dumps(cleaned, ensure_ascii=False, sort_keys=True)
    except Exception:
        body = repr(cleaned)
    return f"{tool_name}:{body}"


class RepeatActionMiddleware(AgentMiddleware):
    """同一工具 + 同一参数连续 N 次 → 短路，不再执行。

    【为什么在 wrap_tool_call 里拦，而不是等下一次 before_model】
    多等一步就多烧一次检索（Milvus 召回 + rerank 都是钱），而且结果必然一样。
    这里直接返回一条 ToolMessage，模型拿到"别再调了"的提示，自己去组织回答 ——
    比强行中断温和，用户仍能拿到结论。

    【为什么计数放 RunContext 而不是 self】
    中间件实例全局共享。放 self 上，两个用户问同一个作物时，
    第二次会被误判成"重复动作"并直接短路 —— 而"同作物同症状"是高频正常情况。
    """

    def __init__(self, max_repeats: int) -> None:
        super().__init__()
        self.max_repeats = max_repeats

    async def awrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name")
        sig = action_signature(name, call.get("args"))

        ctx = run_state.ensure()
        if sig == ctx.last_action:
            ctx.repeat_count += 1
        else:
            ctx.last_action = sig
            ctx.repeat_count = 1

        if ctx.repeat_count >= self.max_repeats:
            trace_event(
                "tool_blocked",
                tool=name,
                repeat=ctx.repeat_count,
                args_sha1=hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12],
            )
            return ToolMessage(
                content=(
                    f"同一检索已连续执行 {ctx.repeat_count} 次，结果不会变化。"
                    "请不要再用相同参数调用该工具：要么换一个明显不同的检索词，"
                    "要么直接基于已有信息回答。"
                ),
                tool_call_id=call.get("id") or "",
                name=name,
            )

        return await handler(request)


def build_repeat_guard() -> RepeatActionMiddleware:
    return RepeatActionMiddleware(max_repeats=MAX_REPEAT_ACTIONS)
