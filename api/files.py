"""图片上传路由。
"""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel

from api.deps import get_current_user
from core.config import MAX_UPLOAD_BYTES
from core.images import ImageError, save_upload

router = APIRouter()


class UploadedImage(BaseModel):
    image_id: str
    width: int
    height: int
    bytes: int


@router.post("/files", response_model=UploadedImage,
             status_code=status.HTTP_201_CREATED)
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user),
) -> UploadedImage:
    # 【第一道】先看声明的大小，能在接收请求体之前就拒掉绝大多数超大上传。
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限",
        )

    # 【第二道】只多读 1 字节用于判断超限。
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限",
        )

    try:
        # 刻意不用 file.filename（客户端能传 ../../etc/passwd），
        info = save_upload(user_id, data)
    except ImageError as e:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)
        ) from None

    return UploadedImage(**info)
