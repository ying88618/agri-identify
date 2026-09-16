from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from core.config import JWT_ALGORITHM, JWT_EXPIRE_DAYS, JWT_SECRET

_MIN_SECRET_BYTES = 32

if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET 未配置。请在 .env 中设置，生成方式：\n"
        '  python -c "import secrets;print(secrets.token_hex(32))"'
    )

if len(JWT_SECRET.encode("utf-8")) < _MIN_SECRET_BYTES:
    raise RuntimeError(
        f"JWT_SECRET 过短（{len(JWT_SECRET.encode('utf-8'))} 字节），"
        f"至少需要 {_MIN_SECRET_BYTES} 字节。重新生成：\n"
        '  python -c "import secrets;print(secrets.token_hex(32))"'
    )

BCRYPT_MAX_BYTES = 72


def _to_bytes(password: str) -> bytes:
    """把密码编码为 bcrypt 可接受的字节串。"""
    return password.encode("utf-8")[:BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """返回 bcrypt 哈希"""
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_token(user_id: int) -> str:
    """签发访问令牌。"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": now + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> int:
    """校验签名与过期时间，返回 user_id。"""
    payload = jwt.decode(
        token,
        JWT_SECRET,
        algorithms=[JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
    try:
        return int(payload["sub"])
    except (TypeError, ValueError):
        raise jwt.InvalidTokenError("sub 不是合法的 user_id") from None
