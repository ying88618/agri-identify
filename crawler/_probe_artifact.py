# -*- coding: utf-8 -*-
"""一次性探针：确认 @tool(response_format="content_and_artifact") 的 artifact
能否在 agent.astream(stream_mode="messages") 里拿到。

为什么要分开看两件事：
  Q1  工具返回的 artifact 会不会挂到 ToolMessage 上（LangChain 行为）
  Q2  这个 ToolMessage 会不会随 messages 流出来（langgraph 行为）
只看"流里没见到 artifact"无法区分是 Q1 失败还是 Q2 失败，
所以这里同时收集流和最终状态：最终状态里一定有 ToolMessage（如果模型调了工具）。

用法：
    python crawler/_probe_artifact.py
"""
import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from core.llm import llm


@tool(response_format="content_and_artifact")
def probe_tool(query: str):
    """Look up internal test info. Always call this tool when asked to look something up."""
    text_for_model = "TEXT_SEEN_BY_MODEL"
    artifact = {"sources": [{"kind": "kb", "disease": "PROBE_DISEASE"}]}
    return text_for_model, artifact


agent = create_agent(llm, tools=[probe_tool])


async def main() -> None:
    hist = Counter()
    streamed_tool_msgs = []
    last_state = None

    async for mode, payload in agent.astream(
        {"messages": [{"role": "user",
                       "content": "Please call probe_tool to look something up."}]},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, _meta = payload
            hist[type(chunk).__name__] += 1
            if isinstance(chunk, ToolMessage):
                streamed_tool_msgs.append(chunk)
        else:
            last_state = payload

    print("stream chunk histogram:", dict(hist))
    print("ToolMessage in stream:", len(streamed_tool_msgs))

    final_msgs = (last_state or {}).get("messages", [])
    print("final message types:", [type(m).__name__ for m in final_msgs])

    state_tool_msgs = [m for m in final_msgs if isinstance(m, ToolMessage)]
    print("ToolMessage in final state:", len(state_tool_msgs))
    for m in state_tool_msgs:
        print("  content :", repr(m.content)[:70])
        print("  artifact:", getattr(m, "artifact", "<no attr>"))

    print("---")
    print("Q1 artifact on ToolMessage:",
          "OK" if any(getattr(m, "artifact", None) for m in state_tool_msgs) else "FAIL")
    print("Q2 ToolMessage streamed  :",
          "OK" if streamed_tool_msgs else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
