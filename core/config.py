"""集中配置：环境变量 + 全局业务常量。
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


# 项目根目录。
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR") or (BASE_DIR / "uploads"))

# 单文件上限。
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))

# 缩放上限。
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "1280"))
IMAGE_JPEG_QUALITY = int(os.getenv("IMAGE_JPEG_QUALITY", "85"))

# 全项目唯一的集合名出口。客户端不允许指定 collection_name
KB_COLLECTION = "kb_agri"
MCP_DEFAULT_COLLECTION = KB_COLLECTION


DEFAULT_SCORE_THRESHOLD = 0.5

RECALL_K = 100          # 宽召回候选数（候选池含正确答案的比例 66.6% -> 95.5%）
TOP_K = 6               # rerank 精排后返回条数
BM25_FILTER_POOL = 20   # 有 crop/section 过滤时 BM25 候选池放大倍数


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
VL_MODEL_NAME = os.getenv("VL_MODEL_NAME", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

MILVUS_URI = os.getenv("MILVUS_URI", "localhost:19530")
REDIS_URL = os.getenv("REDIS_URL")
CHAT_HISTORY_TTL = int(os.getenv("CHAT_HISTORY_TTL", "1800"))


JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7"))

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL 未配置。格式见 .env.example，例如：\n"
        "  mysql+pymysql://agri:<password>@127.0.0.1:3306/agri_agent?charset=utf8mb4"
    )
