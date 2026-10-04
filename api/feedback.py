"""用户反馈：收集诊断对错，供 badcase 闭环使用。

只做 HTTP 层。日志逻辑（导出）不在这里：那要读所有用户的对话快照，
而项目没有 admin 概念 —— 见 crawler/feedback_to_samples.py 开头说明。
"""

from typing import Literal

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.deps import get_current_user
from api.models import Feedback, get_db
from core.memory import load_history

router = APIRouter()

# 取快照时的轮数上限。取"大到等于全量"是有意的：
# 这是快照，不是喂给模型的历史窗口，不该被 core/memory.py 的【首条常驻】策略截断。
SNAPSHOT_TURNS = 200


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    # max_length 必须与 DDL 的 VARCHAR 长度一致：MySQL 上是硬限制，
    # 超长会报 1406/截断变成 500，而不是像 SQLite 那样默默存下去。
    disease: str = Field(min_length=1, max_length=64)
    verdict: Literal["correct", "wrong"]
    correct_disease: str | None = Field(default=None, max_length=64)
    comment: str | None = Field(default=None, max_length=500)


@router.post("/feedback", status_code=status.HTTP_201_CREATED)
def submit_feedback(
    req: FeedbackRequest,
    user_id: int = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    # 用同步 def：get_db 给的是同步 SQLAlchemy Session，是阻塞的，
    # 交给 FastAPI 的线程池执行（与 api/auth.py 同理）。
    snapshot = load_history(user_id, req.session_id, n=SNAPSHOT_TURNS) or None

    # verdict=correct 时用户不会填正确病害，此时把 disease 自己存进去 ——
    # 导出样本时 truth 字段直接可用，前端不必多传一个字段。
    truth = req.correct_disease or (req.disease if req.verdict == "correct" else None)

    row = Feedback(
        user_id=user_id,
        session_id=req.session_id,
        disease=req.disease,
        verdict=req.verdict,
        correct_disease=truth,
        comment=req.comment,
        snapshot=snapshot,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # 返回 snapshot_turns 是为了让前端/调试能一眼看出有没有抓到对话 ——
    # 抓到 0 条说明 Redis 里已过期，这条 badcase 进不了评测集。
    return {"id": row.id, "snapshot_turns": len(snapshot) if snapshot else 0}