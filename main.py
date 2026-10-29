"""Agri-Agent 服务入口：挂载 api/* 各 router。

本地启动:
    uvicorn main:app --reload
    uvicorn main:app --host 0.0.0.0 --port 8001
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.auth import router as auth_router
from api.chat import router as chat_router
from api.diagnose import router as diagnose_router
from api.feedback import router as feedback_router
from api.files import router as files_router
from api.kb import router as kb_router
from api.models import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 建表：checkfirst 保证幂等，表已存在时不发任何 DDL。；
    init_db()
    yield
    # 关闭时必须 flush：trace writer 是后台 task，进程直接退出会丢掉队列里剩下的记录，
    # 而 run_end 往往是最后一条 —— 也是排查时最需要的那条。
    from core.trace import shutdown as trace_shutdown

    await trace_shutdown()


app = FastAPI(title="Agri Pest Diagnostic Agent", version="0.1.0", lifespan=lifespan)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, bool]:
    """容器存活探针。

    刻意只返回常量，不检查 Milvus / MySQL / Redis：
    依赖抖一下就重启应用，会把可恢复的抖动放大成连环重启（Milvus 冷启动本身就要几十秒）。
    若将来需要"就绪"判断，应另开 /readyz 交给负载均衡摘节点，而不是写进 restart 策略。
    """
    return {"ok": True}


# 各 router 自带完整路径（如 /chat/stream、/auth/login），不额外加前缀
app.include_router(chat_router, tags=["chat"])
app.include_router(auth_router, tags=["auth"])
app.include_router(files_router, tags=["files"])
app.include_router(kb_router, tags=["kb"])
app.include_router(diagnose_router, tags=["diagnose"])
app.include_router(feedback_router, tags=["feedback"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

#uvicorn main:app --reload
