"""trace 中间件：在模型调用与工具调用的边界上记录结构化事件。

【为什么只挂 wrap_* 两个钩子】
- wrap_model_call / wrap_tool_call 能拿到入参、耗时、usage、异常类型，
  这正是 P0 探针验证过的三条能力。
- 请求级的 run_start / run_end 放在 api/chat.py 里记：
  那里天然有请求生命周期，也能拿到 stop_reason 与最终文本长度。
  这样中间件里不需要 before_agent/after_agent（少两个未验证的钩子签名）。

【本类不允许有可变实例状态】
见 core/mw/__init__.py。所有 per-request 信息都从 ContextVar 取。
"""
from __future__ import annotations

import hashlib
import json
import time

from langchain.agents.middleware import AgentMiddleware

from ..config import TRACE_ARGS_PREVIEW
from ..trace import trace_event


def _args_fingerprint(args) -> tuple[int, str, str]:
    """返回 (长度, sha1 前 12 位, 预览)。

    工具入参里是用户描述的症状文本 —— 属敏感内容，不整段落盘。
    保留 hash 是为了统计"同参数重复率"（重复检测上线后有用），
    保留预览是为了人工排查时能看懂发生了什么。
    """
    try:
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        raw = repr(args)
    return (
        len(raw),
        hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12],
        raw[:TRACE_ARGS_PREVIEW],
    )


def _usage_fields(response) -> dict:
    """从 ModelResponse 里抽 usage。不同 provider 可能只在其中一处给。"""
    usage = None
    for m in getattr(response, "result", None) or []:
        usage = getattr(m, "usage_metadata", None) or usage
    if usage is None:
        usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _tool_calls_requested(response) -> int:
    """这一轮模型发起了几个工具调用（含并行调用）。"""
    n = 0
    for m in getattr(response, "result", None) or []:
        n += len(getattr(m, "tool_calls", None) or [])
    return n


class TraceMiddleware(AgentMiddleware):
    """记录 model_end / tool_end 两类事件，附带耗时与用量。"""

    async def awrap_model_call(self, request, handler):
        t0 = time.perf_counter()
        try:
            response = await handler(request)
        except Exception as e:
            # 只记类型不记 message：异常文案可能带 URL / 请求 ID / 密钥片段
            trace_event(
                "model_error",
                duration_ms=round((time.perf_counter() - t0) * 1000),
                error_type=type(e).__name__,
            )
            raise

        trace_event(
            "model_end",
            duration_ms=round((time.perf_counter() - t0) * 1000),
            tool_calls=_tool_calls_requested(response),
            **_usage_fields(response),
        )
        return response

    async def awrap_tool_call(self, request, handler):
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name")
        args_len, args_sha1, args_preview = _args_fingerprint(call.get("args"))

        t0 = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as e:
            trace_event(
                "tool_error",
                tool=name,
                duration_ms=round((time.perf_counter() - t0) * 1000),
                error_type=type(e).__name__,
                args_len=args_len,
                args_sha1=args_sha1,
                args_preview=args_preview,
            )
            raise

        trace_event(
            "tool_end",
            tool=name,
            duration_ms=round((time.perf_counter() - t0) * 1000),
            args_len=args_len,
            args_sha1=args_sha1,
            args_preview=args_preview,
        )
        return result
