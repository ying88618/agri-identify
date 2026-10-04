"""结构化诊断路由。

业务逻辑在 core/diagnose.py —— 本文件只做 HTTP 层：
取令牌里的 user_id、把领域异常转成状态码。
（见 api/__init__.py 的分工说明）
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from api.deps import get_current_user
from core.diagnose import DiagnoseError, diagnose

router = APIRouter()


class DiagnoseRequest(BaseModel):
    session_id: str
    # 前端作物输入框的值（可选）。优先级高于从对话里抽取的作物：
    # 用户自己知道种的是什么，比让模型从对话里认更可靠。
    # 认错作物会让检索被过滤到错误的作物上，表现为"知识库无相关资料"——
    # 这个失败模式会误导人，而显式传入是最便宜的规避手段。
    crop: str | None = Field(default=None, max_length=32)


@router.post("/diagnose")
async def diagnose_route(
    req: DiagnoseRequest,
    user_id: int = Depends(get_current_user),
) -> dict:
    # 这里用 async def，与 api/auth.py 的 def 不同 ——
    # 区别在于耗时的性质：auth 那边是 bcrypt 的 CPU 密集阻塞调用，必须丢线程池；
    # 这里的耗时全在 IO 等待（抽取 LLM + Milvus + rerank），
    # 丢进线程池反而白占一个线程，留在事件循环里 await 才是对的。
    try:
        return await diagnose(user_id, req.session_id, req.crop)
    except DiagnoseError as e:
        # 503 而不是 200 + abstain：抽取失败与"确实没查出结论"必须能区分，
        # 否则前端会把一次服务故障渲染成"没找到病害"。
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from None
