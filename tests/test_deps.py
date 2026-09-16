"""get_current_user 行为测试：只依赖 JWT_SECRET，不依赖数据库 / Redis。"""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from api.deps import get_current_user
from api.security import JWT_SECRET, create_token


@pytest.fixture
def client() -> TestClient:
    """最小应用：只挂一个受保护路由，用来观察 get_current_user 的行为。"""
    app = FastAPI()

    @app.get("/probe")
    def probe(user_id: int = Depends(get_current_user)):
        return {"user_id": user_id}

    return TestClient(app)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_valid_token_yields_user_id(client):
    r = client.get("/probe", headers=_auth(create_token(42)))
    assert r.status_code == 200
    assert r.json() == {"user_id": 42}, "返回的必须是 int，不能是字符串 '42'"


def test_missing_header_is_401_not_403(client):
    """守住 auto_error=False 这个选择：HTTPBearer 的默认行为是 403。"""
    r = client.get("/probe")
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_wrong_scheme_is_401(client):
    r = client.get("/probe", headers={"Authorization": "Token abc"})
    assert r.status_code == 401


def test_tampered_payload_is_401(client):
    header, payload, sig = create_token(1).split(".")
    # 改 payload 末位但保留原签名 -> 签名必然不匹配
    new_payload = payload[:-1] + ("A" if payload[-1] != "A" else "B")
    r = client.get("/probe", headers=_auth(f"{header}.{new_payload}.{sig}"))
    assert r.status_code == 401


def test_expired_token_is_401(client):
    expired = jwt.encode(
        {"sub": "1", "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
        JWT_SECRET, algorithm="HS256",
    )
    r = client.get("/probe", headers=_auth(expired))
    assert r.status_code == 401


def test_non_numeric_sub_is_401(client):
    t = jwt.encode(
        {"sub": "abc", "exp": datetime.now(timezone.utc) + timedelta(days=1)},
        JWT_SECRET, algorithm="HS256",
    )
    r = client.get("/probe", headers=_auth(t))
    assert r.status_code == 401


def test_token_for_nonexistent_user_passes(client):
    """★ 这是一条「记录取舍」的测试，不是在测 bug。

    get_current_user 不查库，所以库里不存在的用户 id 也能通过校验，
    且「已删除用户」的令牌在过期前依然有效。
    要即时失效需要 Redis 黑名单（项目里已有 Redis，见 docker-compose.yml）。
    把这个行为写成测试，是为了让将来改代码的人明确知道这是有意为之，
    而不是一个需要"修复"的遗漏。
    """
    r = client.get("/probe", headers=_auth(create_token(999999)))
    assert r.status_code == 200
    assert r.json() == {"user_id": 999999}
