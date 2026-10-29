"""P0 探针：验证 LangChain 中间件在本项目里是否可用。

只做一件事：确认三个前提成立，再决定后面四个功能怎么落地。
不碰 Milvus / MySQL / Redis —— 用一个假工具，只有一次 LLM 调用。

三个必须通过的前提：
    A. awrap_tool_call 在 astream(stream_mode="messages") 下真的被调用
    B. 能拿到 request.tool_call["args"]（重复检测和 trace 都依赖它）
    C. awrap_model_call 的响应里有 usage_metadata（token 预算依赖它）

附加实验（回答一个悬而未决的问题）：
    D. ModelCallLimitMiddleware 注入的英文 AIMessage，会不会作为 token 流出来？
       如果不流出来，说明熔断对用户是"静默截断" —— 必须自己补 SSE 事件。

运行：
    python scripts/probe_middleware.py

依赖 .env 里的 OPENAI_API_KEY / OPENAI_BASE_URL / MODEL_NAME。
不 import core/*，因为 core.config 在缺 DATABASE_URL 时会直接抛错。
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import (  # noqa: E402
    AgentMiddleware,
    ModelCallLimitMiddleware,
)
from langchain_core.messages import AIMessageChunk  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

SYSTEM = "你必须先调用 echo_probe 工具一次，然后再回答问题，不要省略工具调用。"


@tool
async def echo_probe(text: str) -> str:
    """探针专用工具：把输入原样返回，不访问任何外部服务。"""
    return f"echo: {text}"


class ProbeMiddleware(AgentMiddleware):
    """只观测、不干预。

    注意：这里把结果存在 self 上，是**仅限单次运行的探针写法**。
    生产代码里这样做会跨请求串号（见 core/mw 的规划：状态必须走 ContextVar 或 state）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.tool_events: list[dict] = []
        self.model_events: list[dict] = []

    async def awrap_tool_call(self, request, handler):
        t0 = time.perf_counter()
        call = getattr(request, "tool_call", None) or {}
        name = call.get("name")
        args = call.get("args")
        print(f"  [tool:start] {name}  args={args}")

        try:
            result = await handler(request)
        except Exception as e:  # 记录错误类型，trace 要用
            print(f"  [tool:error] {name}  {type(e).__name__}  "
                  f"{(time.perf_counter() - t0) * 1000:.0f}ms")
            raise

        ms = (time.perf_counter() - t0) * 1000
        self.tool_events.append({"name": name, "args": args, "ms": ms})
        print(f"  [tool:end]   {name}  {ms:.0f}ms  -> {str(result)[:60]}")
        return result

    async def awrap_model_call(self, request, handler):
        t0 = time.perf_counter()
        response = await handler(request)
        ms = (time.perf_counter() - t0) * 1000

        usage = None
        for m in getattr(response, "result", None) or []:
            usage = getattr(m, "usage_metadata", None) or usage
        if usage is None:
            usage = getattr(response, "usage_metadata", None)

        self.model_events.append({"ms": ms, "usage": usage})
        print(f"  [model:end]  {ms:.0f}ms  usage={usage}")
        return response


async def _drain(agent, question: str) -> str:
    """消费一遍 astream，返回拼起来的文本（只看 AI 文本块）。"""
    parts: list[str] = []
    async for chunk, _meta in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and isinstance(chunk.content, str):
            parts.append(chunk.content)
    return "".join(parts)


async def experiment_ab(llm: ChatOpenAI) -> dict[str, bool]:
    print("=" * 68)
    print("实验 1：钩子是否触发 / args 能否拿到 / usage 是否可用")
    print("=" * 68)

    probe = ProbeMiddleware()
    agent = create_agent(
        llm,
        tools=[echo_probe],
        system_prompt=SYSTEM,
        middleware=[probe],
    )

    try:
        text = await _drain(agent, "请调用工具，参数 text 用 'hello-probe'，然后告诉我结果。")
    except Exception as e:
        print(f"\n运行失败：{type(e).__name__}: {e}")
        return {"A": False, "B": False, "C": False}

    print(f"\n  模型最终文本（前 120 字）：{text[:120]!r}")

    has_tool_hook = len(probe.tool_events) > 0
    has_args = any(ev.get("args") for ev in probe.tool_events)
    has_usage = any(ev.get("usage") for ev in probe.model_events)

    print(f"\n  A 工具钩子被调用: {'是' if has_tool_hook else '否'}"
          f"（{len(probe.tool_events)} 次）")
    print(f"  B 拿到工具入参  : {'是' if has_args else '否'}")
    print(f"  C 拿到 token 用量: {'是' if has_usage else '否'}"
          f"（{len(probe.model_events)} 次模型调用）")

    return {"A": has_tool_hook, "B": has_args, "C": has_usage}


async def experiment_d(llm: ChatOpenAI) -> bool:
    """熔断注入的 AIMessage 会不会以 token 形式流出？"""
    print()
    print("=" * 68)
    print("实验 2：run_limit=1 熔断时，注入的英文 AIMessage 会不会流给用户")
    print("=" * 68)

    agent = create_agent(
        llm,
        tools=[echo_probe],
        system_prompt=SYSTEM,
        middleware=[ModelCallLimitMiddleware(run_limit=1, exit_behavior="end")],
    )

    try:
        text = await _drain(agent, "请调用工具，参数 text 用 'x'，然后回答。")
    except Exception as e:
        print(f"  运行失败：{type(e).__name__}: {e}")
        return False

    leaked = "limit" in text.lower()
    print(f"  流给用户的文本：{text[:160]!r}")
    print(f"  是否包含限制文案：{'是（会污染 UI）' if leaked else '否（用户只看到静默截断）'}")
    return leaked


async def main() -> int:
    model = os.getenv("MODEL_NAME")
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    missing = [k for k, v in {
        "MODEL_NAME": model,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": api_key,
    }.items() if not v]
    if missing:
        print(f"缺少环境变量：{', '.join(missing)}（检查 {ROOT / '.env'}）")
        return 2

    print(f"模型: {model}    网关: {base_url}")
    print("stream_usage=True —— 这是拿到 usage_metadata 的前提\n")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=True,
        stream_usage=True,   # ← 本探针唯一的关键开关
    )

    r = await experiment_ab(llm)
    leaked = await experiment_d(llm)

    print()
    print("=" * 68)
    print("结论")
    print("=" * 68)
    print(f"  A 工具钩子   : {'通过' if r['A'] else '不通过'}")
    print(f"  B 工具入参   : {'通过' if r['B'] else '不通过'}")
    print(f"  C token 用量 : {'通过' if r['C'] else '不通过'}")
    print(f"  D 熔断文案   : {'会漏给用户' if leaked else '不会漏（需自己发 SSE 事件）'}")

    if r["A"] and r["B"] and r["C"]:
        print("\n  三个前提全部成立 → 可以按 P1~P4 推进。")
        return 0
    if r["A"] and r["B"] and not r["C"]:
        print("\n  A/B 成立，C 不成立 → 该 provider 不支持流式 usage。")
        print("  对策：token 预算改用近似计数（字符数或 tiktoken），精度损失可接受。")
        return 1
    print("\n  前提未成立 → 先别写业务中间件，把这里查清楚。")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
