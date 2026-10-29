"""对话路由：SSE 流式问答。"""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessageChunk, ToolMessage
from sse_starlette.event import JSONServerSentEvent
from sse_starlette.sse import EventSourceResponse

from api.deps import get_current_user
from core import run_state
from core.agent import get_agent, set_request_context
from core.config import DEFAULT_SCORE_THRESHOLD, KB_COLLECTION, WALL_CLOCK_S
from core.images import (
    ImageError,
    ensure_exists,
    read_cached_description,
    resolve_image_url,
    write_cached_description,
)
from core.llm import SYSTEM_PROMPT
from core.memory import append_turn, load_history
from core.sources import merge_sources
from core.trace import start_run, trace_event
from core.vision import describe_image

logger = logging.getLogger("knowledge_agent")

router = APIRouter()

HISTORY_TURNS = 6

# 注入到 user 消息里的作物标记前缀。
# 提成常量是为了让"是否已注入过"的判断与生成逻辑共用同一个字符串 ——
# 两处各写一遍字面量的话，改一处就会让去重静默失效，退化成每轮都注入。
CROP_NOTE_PREFIX = "已知作物："


class ChatRequest(BaseModel):
    # 这里没有 user_id：身份一律来自 Authorization 头里的令牌，
    session_id: str
    question: str
    # 前端作物输入框的值（可选）。只用于"让模型知道作物"，不构成服务端过滤约束 ——
    # 过滤仍由模型填的 knowledge_base_search(crop=...) 决定。
    # max_length 是防超长输入把 prompt 撑爆；实测作物名都很短。
    crop: str | None = Field(default=None, max_length=32)
    image_id: str | None = None
    image_url: str | None = None


def _build_user_text(question: str, image_desc: str, crop: str = "") -> str:
    """把已知作物与图片的客观描述作为上下文注入。
    """
    # 压掉换行，避免用户输入把上下文结构撑开。
    # 这不是安全措施 —— question 本来就是用户可控的自由文本，同样的问题本来就存在。
    crop = (crop or "").replace("\n", " ").replace("\r", " ").strip()
    if crop:
        question = f"{CROP_NOTE_PREFIX}{crop}。{question}"
    if image_desc:
        return f"用户上传了一张图片，图片内容描述如下：\n{image_desc}\n\n用户问题：{question}"
    return question


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, user_id: int = Depends(get_current_user)):
    # user_id 来自令牌（api/deps.py），而不是请求体。

    if req.image_id:
        try:
            # 先确认图片存在且属于当前用户（顺带校验 image_id 格式）。
            ensure_exists(user_id, req.image_id)
        except ImageError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from None

        # 先查 VL 描述缓存。前端每一轮都会带同一个 image_id，不缓存的话每轮都要
        image_desc = read_cached_description(user_id, req.image_id)
        if image_desc is None:
            image_desc = await describe_image(
                resolve_image_url(user_id, req.image_id), req.question
            )
            if image_desc:
                write_cached_description(user_id, req.image_id, image_desc)
    elif req.image_url:
        image_desc = await describe_image(req.image_url, req.question)
    else:
        image_desc = ""
    # 历史要在拼 user_text 之前取：作物标记是否需要注入，取决于历史里有没有。
    history = load_history(user_id, req.session_id, n=HISTORY_TURNS)

    # 作物标记只在历史里还没有的时候注入：
    #   · 首轮注入后，core/memory.py 的【首条常驻】会让它一直留在上下文窗口里，
    #     所以不必每轮重复（重复还会被 append_turn 写进历史，变成噪声）
    #   · 但用户可能中途才在 UI 上填作物，所以要允许后续轮次补注
    crop_arg = ""
    if req.crop and not any(
        CROP_NOTE_PREFIX in (m.get("content") or "") for m in history
    ):
        crop_arg = req.crop

    user_text = _build_user_text(req.question, image_desc, crop_arg)
    append_turn(user_id, req.session_id, "user", user_text)

    agent = get_agent()
    # 注意: 检索阈值实际通过 set_request_context 的 contextvar 传给工具,
    # 这里的 config 是历史遗留参数, 不参与阈值传递。
    config = {"configurable": {"score_threshold": DEFAULT_SCORE_THRESHOLD}}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": user_text})

    async def event_generator():
        set_request_context(KB_COLLECTION, score_threshold=DEFAULT_SCORE_THRESHOLD)

        # trace 起点：初始化 per-request 上下文（request_id / user hash / step=0）。
        # 必须在第一次 trace_event 之前 —— 否则事件里 request_id 为空，多请求无法区分。
        await start_run(user_id, req.session_id)
        t0 = time.monotonic()
        trace_event(
            "run_start",
            question_len=len(req.question),
            has_image=bool(req.image_id or req.image_url),
            history_len=len(history),
            crop=bool(crop_arg),
        )

        full = []
        # 各次工具调用产出的来源，按调用顺序累积（合并去重见 core/sources.py）
        source_groups: list[list[dict]] = []

        def _record_end() -> None:
            """收尾记录。四条路径（正常/墙钟/断开/异常）都要落一条。

            不用 finally：finally 会在收尾代码之前执行，导致 trace 里 run_end
            排在 truncated 之前，读起来像是"先结束再报错"。
            """
            trace_event(
                "run_end",
                duration_ms=round((time.monotonic() - t0) * 1000),
                answer_len=len("".join(full)),
                source_groups=len(source_groups),
                stop_reason=run_state.get_stop_reason(),
                partial=bool(run_state.get_stop_reason()),
            )

        try:
            # 墙钟上限只能在这一层做：中间件没有发 SSE 事件的能力，
            # 超时后用户就只会看到"说到一半没了"。
            # 取消会传给 httpx 的在途请求，不会留下悬挂的模型调用。
            async with asyncio.timeout(WALL_CLOCK_S):
                async for msg_chunk, _meta in agent.astream(
                    {"messages": messages},
                    config=config,
                    stream_mode="messages",
                ):
                    # ToolMessage 携带工具的结构化返回（artifact），来源在这里收集。
                    # 关键：只读 artifact，绝不碰 content —— content 是检索到的文档原文，
                    # 推给前端会污染回答、写进历史会污染后续多轮的记忆（见下方说明）。
                    if isinstance(msg_chunk, ToolMessage):
                        artifact = getattr(msg_chunk, "artifact", None)
                        if isinstance(artifact, dict) and artifact.get("sources"):
                            source_groups.append(artifact["sources"])
                        continue

                    # stream_mode="messages" 会把 ToolMessage 一并流出(检索到的文档原文、
                    # 联网搜索结果)。若不过滤, 这些原始内容会被当成"回答"推给前端,
                    # 并被 append_turn 写进对话历史, 污染后续多轮的记忆。
                    # 只放行模型自身产生的文本块。
                    if not isinstance(msg_chunk, AIMessageChunk):
                        continue
                    text = getattr(msg_chunk, "content", "")
                    if text and isinstance(text, str):
                        full.append(text)
                        yield JSONServerSentEvent(data={"type": "token", "content": text})

        except TimeoutError:
            # asyncio.timeout 超时抛的是内置 TimeoutError（3.11+ 与 asyncio.TimeoutError 同一类）
            run_state.set_stop_reason("wall_clock")
            logger.warning("超过墙钟上限 %.0fs，中断并返回部分结果", WALL_CLOCK_S)
            trace_event("wall_clock_exceeded", limit_s=WALL_CLOCK_S)
        except asyncio.CancelledError:
            logger.info("SSE client disconnected, abort streaming.")
            # 断开也要留痕：否则日志里只剩半截回答，事后无法区分
            # "模型正常答完" 与 "用户中途关页面"。
            run_state.set_stop_reason("client_disconnect")
            _record_end()
            raise
        except Exception as e:
            # 非预期异常也必须留痕，否则 trace 里那条 run_end 看起来像正常结束
            run_state.set_stop_reason(f"error:{type(e).__name__}")
            _record_end()
            raise

        # 走到这里只可能是"正常结束"或"墙钟超时"——另两条路径已经 raise。
        reason = run_state.get_stop_reason()
        answer = "".join(full)

        # 半截回答也要入库（否则下一轮问"那怎么治"会上下文断裂），
        # 但必须带 partial 标记：core/diagnose.py 会把历史里的助手结论抽出来当依据，
        # 不标记的话，一句被截断的"建议使用甲霜灵"会被当成有效结论。
        append_turn(user_id, req.session_id, "assistant", answer, partial=bool(reason))

        # 来源统一在末尾推一次，而不是每次工具返回就推：
        # 一轮里模型可能发多个并行工具调用（实测出现过"防治方法"+"危害症状"两次），
        # 逐次推的话前端还得自己做合并去重，这里一次给全。
        sources = merge_sources(*source_groups)
        if sources:
            yield JSONServerSentEvent(data={"type": "sources", "items": sources})

        # 顺序固定：truncated → done。前端收到 done 时状态已完整。
        if reason:
            yield JSONServerSentEvent(data={"type": "truncated", "reason": reason})

        yield JSONServerSentEvent(
            data={"type": "done", "content": answer, "partial": bool(reason)}
        )
        _record_end()

    return EventSourceResponse(
        event_generator(),
        media_type="text/event-stream",
        ping=15,
    )

r"""
检查数据
cd E:\agri-agent
.\.venv\Scripts\python.exe -c "from pymilvus import MilvusClient; c=MilvusClient(uri='http://localhost:19530'); cols=list(dict.fromkeys(c.list_collections())); print('collections:', cols); [ (c.load_collection(x), print(' ', x, c.query(x, filter='', output_fields=['count(*)'])[0]['count(*)'])) for x in cols ]"
"""