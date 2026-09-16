"""security.py 纯函数单测：不依赖数据库 / Redis / 网络。"""
import jwt
import pytest

from api.security import (
    BCRYPT_MAX_BYTES, create_token, decode_token, hash_password, verify_password,
)


def test_hash_is_bcrypt_and_salted():
    h1, h2 = hash_password("secret123"), hash_password("secret123")
    assert h1.startswith("$2b$")
    assert len(h1) == 60
    assert h1 != h2, "相同密码必须产生不同哈希（salt 未生效）"


def test_verify_password():
    h = hash_password("secret123")
    assert verify_password("secret123", h) is True
    assert verify_password("secret124", h) is False


def test_verify_password_tolerates_broken_hash():
    # 库中脏数据不应导致 500，只应返回 False
    assert verify_password("secret123", "") is False
    assert verify_password("secret123", "not-a-bcrypt-hash") is False


def test_long_password_does_not_raise():
    # bcrypt 4.1+ 对 >72 字节直接抛 ValueError，故 hash_password 必须自己截断
    hash_password("a" * 200)


def test_token_roundtrip():
    assert decode_token(create_token(42)) == 42


def test_token_rejects_tampering():
    t = create_token(1)
    tampered = t[:-1] + ("A" if t[-1] != "A" else "B")
    with pytest.raises(jwt.PyJWTError):
        decode_token(tampered)


def test_token_rejects_wrong_secret():
    t = create_token(1)
    with pytest.raises(jwt.PyJWTError):
        # 故意用 36 字节：短于 32 字节的密钥会让 PyJWT 发 InsecureKeyLengthWarning，
        # 干扰测试输出。这里测的是"密钥不对"，不是"密钥太短"，两件事别混。
        jwt.decode(t, "wrong-secret" * 3, algorithms=["HS256"])



def test_token_requires_exp():
    # 无 exp 的令牌应被拒绝（options={"require": ["exp"]}）
    from api.security import JWT_SECRET
    never_expires = jwt.encode({"sub": "1"}, JWT_SECRET, algorithm="HS256")
    with pytest.raises(jwt.PyJWTError):
        decode_token(never_expires)


def test_token_rejects_non_numeric_sub():
    from api.security import JWT_SECRET
    t = jwt.encode({"sub": "abc"}, JWT_SECRET, algorithm="HS256")
    with pytest.raises(jwt.PyJWTError):
        decode_token(t)
