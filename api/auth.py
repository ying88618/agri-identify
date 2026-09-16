"""认证路由：注册 / 登录。"""

from fastapi import APIRouter, Depends, HTTPException, status
from httpx import get
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.models import User, get_db
from api.security import BCRYPT_MAX_BYTES, create_token, hash_password, verify_password

router = APIRouter()


class AuthRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_]+$")
    password: str = Field(min_length=6, max_length=BCRYPT_MAX_BYTES)

    @field_validator("password")
    @classmethod
    def _check_bcrypt_byte_limit(cls, v: str) -> str:
        """按字节数校验密码长度 —— 这是权威判断。"""
        nbytes = len(v.encode("utf-8"))
        if nbytes > BCRYPT_MAX_BYTES:
            raise ValueError(f"密码过长：{nbytes} 字节，上限 {BCRYPT_MAX_BYTES} 字节")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(req: AuthRequest, db: Session = Depends(get_db)) -> TokenResponse:
    if db.scalar(select(User).where(User.username == req.username)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已被占用")
    user = User(username=req.username, password_hash=hash_password(req.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "用户名已被占用") from None
    return TokenResponse(access_token=create_token(user.id))


@router.post("/auth/login", response_model=TokenResponse)
def login(req: AuthRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.username == req.username))
    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(access_token=create_token(user.id))
