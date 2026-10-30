"""每请求运行状态。

【为什么不用中间件实例属性】
core/agent.py 的 AGENT 是模块级构建一次，中间件实例被所有请求共享。
把计数/标识放在 self 上会跨请求串号：
    - 用户 A 的第 3 步会让用户 B 提前熔断
    - A 的 request_id 被 B 覆盖，trace 记到别人名下
    - 重复检测的"上一次签名"跨用户残留，两个用户问同一作物时互相误杀

【为什么是一个可变对象，而不是多个 ContextVar】
asyncio.create_task 会**复制**当前 context 的映射，子任务里 `var.set(x)`
只改子任务自己那份。而 LangGraph 把每个节点（含中间件钩子）跑在独立 task 里，于是：
    - request_id 由父任务写、子任务读 → 继承正常 ✅
    - step 由子任务写、子任务读 → 每个钩子都从父任务那份 0 开始 ❌
    - stop_reason 由子任务写、父任务读 → 写了也读不到 ❌

把状态收进一个 dataclass 对象、只 set 一次引用后，子任务拿到的是**同一个对象**，
改字段父任务立刻可见。这是跨 task 回传状态少数几个可靠解法之一。

【并发前提】
单线程 asyncio + 读改写之间没有 await，所以 `ctx.step += 1` 不存在竞态。
若将来在中间件里引入 await，必须重新审视这一点。
"""
from __future__ import annotations

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass
from uuid import uuid4


@dataclass
class RunContext:
    """"""

    request_id: str = ""
    # 只存 sha1 前 8 位，不落明文用户 ID
    user_key: str = ""
    session_id: str = ""
    # 事件序号，1 起
    step: int = 0
    # None = 正常结束；其余取值见 api/chat.py 的 truncated 事件
    stop_reason: str | None = None

    # 累计用量。由 core/llm_usage.py 在每次模型调用结束时写入。
    # 【为什么不用 LangChain 的 usage_metadata】provider 在每个流式 chunk 里都带
    # "累计" usage，而 LangChain 合并 chunk 时用 add_usage 相加
    # （langchain_core/messages/ai.py），结果是 N 个 chunk 被加了 N 次。
    # 实测一个 12 字问题报出 60256 prompt tokens —— 膨胀约 78 倍。
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # 推理模型把思维链也算进输出，这部分不经过流式文本，只能从 usage 里取
    reasoning_tokens: int = 0
    total_tokens: int = 0

    # 重复动作检测：上一步的工具签名，以及它连续出现的次数。
    # 【必须放这里而不是中间件实例上】中间件实例是全局共享的，
    # 若把"上一次签名"存在 self 上，两个用户问同一作物时会互相误判成重复动作 ——
    # 而"同作物同症状"恰恰是高频正常情况。
    last_action: str | None = None
    repeat_count: int = 0


_CTX: ContextVar[RunContext | None] = ContextVar("run_ctx", default=None)


def current() -> RunContext | None:
    return _CTX.get()


def ensure() -> RunContext:
    """取当前上下文；不存在时创建一个（仅在异常路径下会走到）。"""
    ctx = _CTX.get()
    if ctx is None:
        ctx = RunContext()
        _CTX.set(ctx)
    return ctx


def start(user_id: int, session_id: str) -> RunContext:
    """每个请求开头调用一次。"""
    ctx = RunContext(
        request_id=uuid4().hex[:12],
        user_key=hashlib.sha1(str(user_id).encode()).hexdigest()[:8],
        session_id=session_id,
    )
    _CTX.set(ctx)
    return ctx


def next_step() -> int:
    ctx = ensure()
    ctx.step += 1
    return ctx.step


def set_stop_reason(reason: str) -> None:
    ensure().stop_reason = reason


def get_stop_reason() -> str | None:
    ctx = _CTX.get()
    return ctx.stop_reason if ctx else None


_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")


def usage() -> dict[str, int]:
    """取用量快照。调用方用「后减前」得到单次调用的增量。"""
    ctx = ensure()
    return {f: getattr(ctx, f) for f in _USAGE_FIELDS}


def add_usage(usage_metadata: dict | None) -> None:
    """把一次模型调用的真实用量累加进上下文。

    入参是 LangChain 归一化后的 UsageMetadata：
        {"input_tokens": 9, "output_tokens": 1638, "total_tokens": 1647,
         "output_token_details": {"reasoning": 631}}
    provider 原始字段名不同，这里统一按归一化后的键读。
    """
    if not usage_metadata:
        return
    ctx = ensure()
    prompt = int(usage_metadata.get("input_tokens") or 0)
    completion = int(usage_metadata.get("output_tokens") or 0)
    total = int(usage_metadata.get("total_tokens") or (prompt + completion))
    details = usage_metadata.get("output_token_details") or {}
    reasoning = int(details.get("reasoning") or 0)

    ctx.prompt_tokens += prompt
    ctx.completion_tokens += completion
    ctx.reasoning_tokens += reasoning
    ctx.total_tokens += total
