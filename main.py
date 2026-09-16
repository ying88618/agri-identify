"""Agri-Agent 服务入口：挂载 api/* 各 router。

本地启动:
    uvicorn main:app --reload
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.files import router as files_router
from api.kb import router as kb_router
from api.models import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建表：checkfirst 保证幂等，表已存在时不发任何 DDL。；
    init_db()
    yield


app = FastAPI(title="Agri Pest Diagnostic Agent", version="0.1.0", lifespan=lifespan)

# 各 router 自带完整路径（如 /chat/stream、/auth/login），不额外加前缀
app.include_router(chat_router, tags=["chat"])
app.include_router(auth_router, tags=["auth"])
app.include_router(files_router, tags=["files"])
app.include_router(kb_router, tags=["kb"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

#uvicorn main:app --reload
