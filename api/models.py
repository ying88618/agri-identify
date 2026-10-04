"""SQLAlchemy 模型与会话。"""

from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from core.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=5,
)


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """返回 naive UTC 时间。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Feedback(Base):
    """诊断结果的人工反馈。

    【为什么必须存快照】core/memory.py 的 CHAT_HISTORY_TTL=1800，
    用户点差评时 Redis 里那段对话很可能已经过期；而 /diagnose 是无状态的，
    服务端不留任何东西的话，这条 badcase 就只剩"某病害被判错了"这一句。

    【为什么用 JSON 而不是 Text】落库/读取由驱动负责序列化，读出来直接是 dict；
    分析时能直接用 snapshot->>'$.xxx' 查询，不必先取出来再解析。
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 来自令牌，绝不接受请求体传入 —— 与 api/chat.py 的 user_id 同一条规则。
    # index 是给"按用户查自己的反馈"和将来做统计用的。
    user_id: Mapped[int] = mapped_column(index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    # 被评价的那个候选病害（前端从 /diagnose 的 candidates[].disease 原样回传）
    disease: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(16))  # correct / wrong
    # verdict=correct 时写入 disease 自身，导出样本时 truth 字段直接可用
    correct_disease: Mapped[str | None] = mapped_column(String(64), default=None)
    comment: Mapped[str | None] = mapped_column(String(500), default=None)
    # 可为 NULL：历史已过期时反馈仍然收下（"某病害被判错"本身就是信号），
    # 只是这条进不了评测集 —— 见 crawler/feedback_to_samples.py 的 skipped 统计。
    snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """建表checkfirst=True（默认）使其幂等：表已存在则跳过，不发任何 DDL。"""
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    """FastAPI 依赖：每个请求一个 Session，请求结束必关。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
    