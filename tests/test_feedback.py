"""反馈功能测试：写入路由 + 导出脚本。

分两层：
  · /feedback 路由 —— 用内存 SQLite 覆盖 get_db，**不依赖 MySQL**。
    Feedback 用的 JSON 是 SQLAlchemy 通用类型，SQLite 上同样成立，
    所以这张表可以做到"每个用例一个全新空库"，不像 users 表那样必须连真库。
  · crawler/feedback_to_samples.py 的纯逻辑 —— 只测解析与跳过判定，不碰数据库。
"""
import pytest
from fastapi.testclient import TestClient


# 一段"有图 + 有作物标记"的对话，格式与 api/chat.py 的 _build_user_text 一致。
HISTORY = [
    {
        "role": "user",
        "content": (
            "用户上传了一张图片，图片内容描述如下：\n"
            "叶背有白色霉层，病斑水渍状\n\n"
            "用户问题：已知作物：番茄。这是什么病"
        ),
    },
    {"role": "assistant", "content": "最可能是番茄晚疫病。"},
]


def _default_history(*_a, **_k):
    return list(HISTORY)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def db_session_factory():
    """把 app 的 get_db 换成绑定内存 SQLite 的 sessionmaker，并返回它供断言使用。"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from api.models import Base, get_db
    from main import app

    # StaticPool 是必须的：SQLite 的 ":memory:" 下每个新连接都是一个**独立的空库**，
    # 不加它的话建表用的连接和请求用的连接不是同一个库，会报 "no such table: feedback"。
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _override_get_db():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield factory
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client(db_session_factory, monkeypatch):
    from main import app

    # load_history 是路由内部直接调用的普通函数，不是 FastAPI 依赖，
    # 只能 monkeypatch（用例里再 patch 一次即可覆盖这里的默认值）。
    monkeypatch.setattr("api.feedback.load_history", _default_history)
    return TestClient(app)


def _auth(user_id: int = 1) -> dict:
    from api.security import create_token

    return {"Authorization": f"Bearer {create_token(user_id)}"}


def _body(**over) -> dict:
    body = {"session_id": "s1", "disease": "番茄晚疫病", "verdict": "wrong"}
    body.update(over)
    return body


# ---------------------------------------------------------------- 写入路由

def test_feedback_route_is_registered():
    from main import app

    assert "/feedback" in app.openapi()["paths"]


def test_feedback_requires_token(client):
    r = client.post("/feedback", json=_body())
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_feedback_rejects_unknown_verdict(client):
    """verdict 用 Literal 约束：'maybe' 这类自由文本必须被挡在 422。"""
    r = client.post("/feedback", json=_body(verdict="maybe"), headers=_auth())
    assert r.status_code == 422


def test_feedback_rejects_overlong_disease(client):
    """★ 必须挡在 pydantic 层：DDL 里 disease 是 VARCHAR(64)，MySQL 上是硬限制。
    不挡的话一条 100 字病名会变成 MySQL 报错 → 500，而不是干净的 422。"""
    r = client.post("/feedback", json=_body(disease="病" * 100), headers=_auth())
    assert r.status_code == 422


def test_feedback_rejects_overlong_comment(client):
    r = client.post("/feedback", json=_body(comment="x" * 501), headers=_auth())
    assert r.status_code == 422


def test_feedback_rejects_empty_disease(client):
    r = client.post("/feedback", json=_body(disease=""), headers=_auth())
    assert r.status_code == 422


def test_feedback_stores_snapshot(client, db_session_factory):
    """核心断言：对话快照必须在提交时落库。

    Redis 里那段对话 30 分钟后就会过期（CHAT_HISTORY_TTL=1800），
    而 /diagnose 是无状态的 —— 不在这里抓下来，badcase 就只剩一句"某病被判错了"。
    """
    r = client.post("/feedback", json=_body(correct_disease="番茄早疫病"), headers=_auth())
    assert r.status_code == 201
    assert r.json()["snapshot_turns"] == len(HISTORY)

    from api.models import Feedback

    db = db_session_factory()
    try:
        row = db.query(Feedback).filter_by(id=r.json()["id"]).one()
        assert row.snapshot == HISTORY, "快照必须原样保存（含图片描述那一轮）"
        assert row.correct_disease == "番茄早疫病"
        assert row.verdict == "wrong"
    finally:
        db.close()


def test_feedback_correct_verdict_backfills_truth(client, db_session_factory):
    """verdict=correct 时用户不会填正确病害，correct_disease 应回填 disease 自身 ——
    导出样本时 truth 字段直接可用，前端不必多传一个参数。"""
    r = client.post("/feedback", json=_body(verdict="correct"), headers=_auth())
    assert r.status_code == 201

    from api.models import Feedback

    db = db_session_factory()
    try:
        row = db.query(Feedback).filter_by(id=r.json()["id"]).one()
        assert row.correct_disease == "番茄晚疫病"
    finally:
        db.close()


def test_feedback_user_id_comes_from_token_not_body(client, db_session_factory):
    """★ 身份必须来自令牌。请求体里的 user_id 会被 Pydantic 默认的 extra='ignore'
    静默丢弃 —— 否则任何人都能以别人的名义留下反馈（甚至是恶意刷 badcase）。"""
    r = client.post("/feedback", json=_body(user_id=999), headers=_auth(1))
    assert r.status_code == 201

    from api.models import Feedback

    db = db_session_factory()
    try:
        row = db.query(Feedback).filter_by(id=r.json()["id"]).one()
        assert row.user_id == 1
    finally:
        db.close()


def test_feedback_accepts_expired_session(client, monkeypatch, db_session_factory):
    """历史已过期（Redis 里没了）时仍要收下反馈，只是 snapshot 为 NULL。

    这类记录进不了评测集（导出脚本会跳过），但"某病害被判错"本身就是信号，
    不能因为快照丢了就整个拒收。
    """
    monkeypatch.setattr("api.feedback.load_history", lambda *a, **k: [])
    r = client.post("/feedback", json=_body(), headers=_auth())
    assert r.status_code == 201
    assert r.json()["snapshot_turns"] == 0

    from api.models import Feedback

    db = db_session_factory()
    try:
        assert db.query(Feedback).filter_by(id=r.json()["id"]).one().snapshot is None
    finally:
        db.close()


# ---------------------------------------------------------------- 导出脚本

def _row(**over):
    """构造一个不入库的 Feedback 实例，用于测导出逻辑。"""
    from api.models import Feedback

    row = Feedback(
        session_id="s1", disease="番茄晚疫病", verdict="wrong", user_id=1,
        snapshot=list(HISTORY), correct_disease="番茄早疫病",
    )
    row.id = 7
    for k, v in over.items():
        setattr(row, k, v)
    return row


def test_parse_first_turn_extracts_desc_and_crop():
    from crawler.feedback_to_samples import parse_first_turn

    desc, crop = parse_first_turn(HISTORY[0]["content"])
    assert desc == "叶背有白色霉层，病斑水渍状"
    assert crop == "番茄"


def test_parse_first_turn_without_image_returns_no_desc():
    """没有图片的对话解析不出 desc —— 导出时必须计入 skipped，不能静默丢弃。"""
    from crawler.feedback_to_samples import parse_first_turn

    desc, crop = parse_first_turn("用户问题：已知作物：番茄。叶子有斑点")
    assert desc is None
    assert crop == "番茄"


def test_parse_first_turn_without_crop_note():
    from crawler.feedback_to_samples import parse_first_turn

    desc, crop = parse_first_turn(
        "用户上传了一张图片，图片内容描述如下：\n叶面有褐色斑点\n\n用户问题：这是什么病"
    )
    assert desc == "叶面有褐色斑点"
    assert crop is None


def test_to_sample_produces_eval_sample():
    """产出的形状必须对齐 eval_multiturn.py 的样本源（vl_desc_*.jsonl）。"""
    from crawler.feedback_to_samples import to_sample

    sample, why = to_sample(_row())
    assert why is None
    assert sample["crop"] == "番茄"
    assert sample["truth"] == "番茄早疫病"   # truth 来自 correct_disease
    assert sample["desc"] == "叶背有白色霉层，病斑水渍状"
    assert sample["cls"] == "" and sample["img"] == ""  # 真实用户照片无 PlantVillage 标签
    assert sample["source"] == "feedback:7"  # 溯源，eval_multiturn 会忽略这个额外字段


def test_to_sample_skip_reasons():
    """三种入不了样本集的情况必须各有明确的跳过原因，便于在统计里区分。"""
    from crawler.feedback_to_samples import to_sample

    assert to_sample(_row(snapshot=None))[1] == "no_snapshot"
    assert to_sample(_row(snapshot=[]))[1] == "no_snapshot"
    assert to_sample(_row(correct_disease=None))[1] == "no_truth"
    assert to_sample(
        _row(snapshot=[{"role": "user", "content": "没有图片，纯文字提问"}])
    )[1] == "no_desc"


def test_to_sample_skips_when_first_user_turn_has_no_image():
    """首轮无图、次轮才传图 —— 图片描述不在首条里，同样解析不到，归入 no_desc。"""
    from crawler.feedback_to_samples import to_sample

    snap = [
        {"role": "user", "content": "叶子有斑点"},
        {"role": "user", "content": "用户上传了一张图片，图片内容描述如下：\n褐斑\n\n用户问题：这是什么病"},
    ]
    assert to_sample(_row(snapshot=snap))[1] == "no_desc"
