"""图片上传与读取：本地文件 <-> VL 可用的 data URL。"""

import base64
import io
import json
import logging
import re
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from core.config import (
    IMAGE_JPEG_QUALITY,
    MAX_IMAGE_DIM,
    MAX_UPLOAD_BYTES,
    UPLOAD_DIR,
    VL_MODEL_NAME,
)

logger = logging.getLogger("images")

Image.MAX_IMAGE_PIXELS = 50_000_000
_IMAGE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}

class ImageError(ValueError):
    """图片不可用（格式不支持 / 损坏 / 超限）。调用方转成 4xx。"""

def user_dir(user_id: int) -> Path:
    """用户专属目录"""
    d = UPLOAD_DIR / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_upload(user_id: int, data: bytes) -> dict:
    """校验并落盘一张图片，返回 {image_id, width, height, bytes}。"""
    if not data:
        raise ImageError("空文件")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageError(f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限")

    try:
        img = Image.open(io.BytesIO(data))
        img_format = (img.format or "").upper()
        if img_format not in _ALLOWED_FORMATS:
            raise ImageError(f"不支持的图片格式: {img_format or '未知'}")
        img.load()
    except ImageError:
        raise
    except UnidentifiedImageError as e:
        raise ImageError("不是有效的图片文件") from e
    except Exception as e:
        raise ImageError(f"图片无法解码: {type(e).__name__}") from e    

    img = ImageOps.exif_transpose(img)

    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))

    if img.mode != "RGB":
        img = img.convert("RGB")

    image_id = uuid.uuid4().hex
    path = user_dir(user_id) / f"{image_id}.jpg"
    img.save(path, format="JPEG", quality=IMAGE_JPEG_QUALITY, optimize=True)

    size = path.stat().st_size
    logger.info("saved image user=%s id=%s %dx%d %dB",
                user_id, image_id, img.width, img.height, size)
    return {
        "image_id": image_id,
        "width": img.width,
        "height": img.height,
        "bytes": size,
    }

def image_path(user_id: int, image_id: str) -> Path:
    if not _IMAGE_ID_RE.fullmatch(image_id or ""):
        raise ImageError("image_id 格式非法")
    return user_dir(user_id) / f"{image_id}.jpg"

def to_data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")

def ensure_exists(user_id: int, image_id: str) -> None:
    """校验 image_id 合法且图片确实存在。只 stat，不读内容。

    单独拆出这个函数，是因为 api/chat.py 需要"先确认图片存在"再决定要不要读缓存。
    若直接调 resolve_image_url，它会把整个文件读出来并做 base64 ——
    而一旦缓存命中，这次读取就完全白费了。
    """
    if not image_path(user_id, image_id).is_file():
        raise ImageError("图片不存在或不属于当前用户")


def resolve_image_url(user_id: int, image_id: str) -> str:
    """image_id -> VL 可用的 data URL。找不到时抛 ImageError。
    """
    path = image_path(user_id, image_id)
    if not path.is_file():
        raise ImageError("图片不存在或不属于当前用户")
    return to_data_url(path.read_bytes())


# ---------------------------------------------------------------------------
# VL 描述缓存
# ---------------------------------------------------------------------------
# 缓存文件与图片同目录，好处有三：
#   · 生命周期一致（图片没了，描述也没有意义）
#   · 复用 image_path() 的 image_id 校验，同样不可能路径穿越
#   · 不引入额外存储依赖（不用 Redis，也就不会被 TTL 提前淘汰）
#
# 有意的副作用：缓存命中后，后续轮次不再把"当轮问题"作为提示传给 VL
# （即 describe_image 的 question 参数）。这是刻意的 —— 症状描述应当客观、
# 与问题无关，而"多轮看到同一份描述"比"贴合当轮问题"更重要：
# core/memory.py 的【首条常驻】机制就是把首轮那份描述一直沿用下去。
#
# 已知局限：若 DESCRIBE_PROMPT 有实质改动，旧描述同样会过时。目前只按 model
# 字段区分；真要严格，应把 prompt 的版本/哈希一并写进 payload。
_DESC_SUFFIX = ".desc.json"


def description_path(user_id: int, image_id: str) -> Path:
    return image_path(user_id, image_id).with_suffix(_DESC_SUFFIX)


def read_cached_description(user_id: int, image_id: str) -> str | None:
    """读缓存的 VL 描述。未命中返回 None（注意不是空串）。

    三种情况都算未命中：文件不存在、文件损坏、VL 模型换过。
    """
    path = description_path(user_id, image_id)
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # 写入中途被打断会留下半个文件。这种脏数据不该让整个对话 500，
        # 当作未命中重新描述即可 —— 与 core/memory.py 处理 Redis 坏 JSON 的做法一致。
        logger.warning("VL 描述缓存损坏, 按未命中处理: %s", path)
        return None

    if not isinstance(obj, dict) or obj.get("model") != VL_MODEL_NAME:
        # 换过 VL 模型：旧描述是另一个模型产出的，不能复用，否则多轮对话里
        # 会混入不同模型的描述。本项目对这类隐性漂移很敏感 —— 参见 core/config.py
        # 里"换 rerank 模型必须重新标定阈值"的同款教训。
        return None

    text = obj.get("text")
    return text if isinstance(text, str) and text else None


def write_cached_description(user_id: int, image_id: str, text: str) -> None:
    """写入 VL 描述缓存。调用方须保证 text 非空（空串代表调用失败，不该被缓存）。"""
    path = description_path(user_id, image_id)
    try:
        path.write_text(
            json.dumps({"model": VL_MODEL_NAME, "text": text}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        # 缓存写失败不影响本次回答（描述已经在手里了），记日志即可。
        logger.warning("VL 描述缓存写入失败 %s: %s", path, e)