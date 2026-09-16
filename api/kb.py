"""知识库目录路由：向前端提供作物等元数据。

对应用户在对话页用的作物输入框（作物有几十个，用自动补全而不是下拉）。
业务逻辑在 core/catalog.py —— 本文件只做 HTTP 层，见 api/__init__.py 的分工说明。
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.deps import get_current_user
from core import catalog

router = APIRouter()


class CropItem(BaseModel):
    crop: str
    disease_count: int


@router.get("/kb/crops", response_model=list[CropItem])
def list_crops(
    refresh: bool = False,
    user_id: int = Depends(get_current_user),
) -> list[CropItem]:
    """作物列表，含每个作物的病害数（kb_agri）。

    【契约】crop 的值与 Milvus 里**逐字节一致**，前端拿到后必须原样回传给
    /chat/stream 的 crop 字段。任何 trim / 大小写 / 本地化处理都会让
    core/retriever.py 里的 crop == "..." 静默匹配不到，表现为"检索结果变差
    但不报错"。

    refresh=true 用于重灌知识库后强制刷新进程内缓存（见 core/catalog.py）。
    用同步 def 而不是 async def：Milvus 客户端是阻塞的，交给 FastAPI 的
    线程池执行，不要卡住事件循环（对话流是 SSE 长连接，卡住影响面很大）。
    """
    try:
        items = catalog.crop_list(refresh=refresh)
    except Exception as e:
        # 依赖不可用给 503 而不是 500：这是 Milvus 的问题，不是代码缺陷。
        # 刻意不在启动时预热缓存，所以本端点故障不会连带影响鉴权与对话。
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"知识库元数据暂不可用: {type(e).__name__}",
        ) from e
    return [CropItem(**it) for it in items]
