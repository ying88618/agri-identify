"""集中配置：环境变量 + 全局业务常量。"""

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

RECALL_K = 100  # 宽召回候选数（候选池含正确答案的比例 66.6% -> 95.5%）
TOP_K = 6  # rerank 精排后返回条数
BM25_FILTER_POOL = 20  # 有 crop/section 过滤时 BM25 候选池放大倍数


# VL 图片描述 prompt 的当前版本号，用于 VL 描述缓存的失效判断（core/images.py）。
# 实质改动 core/vision.py 的 DESCRIBE_PROMPT 时必须 +1，否则已缓存的老图会一直沿用
# 旧版描述，线上效果停留在旧 prompt —— 与"换 rerank 模型必须重标定阈值"同属一类
# 隐性漂移。版本沿革与实测依据见 crawler/data/PROJECT2_RESULTS.md §5。
DESCRIBE_PROMPT_VERSION = 5


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

SECTION_SYMPTOM = "危害症状"
SECTION_CONTROL = "防治方法"

DIAGNOSE_HISTORY_TURNS = 40
DIAGNOSE_MAX_CANDIDATES = 3
DIAGNOSE_EVIDENCE_K = 30
DIAGNOSE_PLAN_K = 20
DIAGNOSE_MAX_EVIDENCE = 3
DIAGNOSE_MAX_PLAN = 5
DIAGNOSE_PROMPT_VERSION = 1


# ---------------- 结构化 trace ----------------
# 事件落在 traces/YYYY-MM-DD.jsonl，每行一条。见 core/trace.py 与 core/mw/trace.py。
# 关掉只需 TRACE_ENABLED=0：此时队列与 writer 都不会创建，钩子里的 trace_event 直接返回。
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "1") not in ("0", "false", "False")
TRACE_DIR = Path(os.getenv("TRACE_DIR") or (BASE_DIR / "traces"))
# 队列上限：满了丢记录而不是阻塞业务。观测系统必须能静默降级。
TRACE_QUEUE_MAX = int(os.getenv("TRACE_QUEUE_MAX", "2000"))
# 工具入参的预览长度。入参是用户描述的症状文本，属敏感内容，只留开头供人工排查。
TRACE_ARGS_PREVIEW = int(os.getenv("TRACE_ARGS_PREVIEW", "80"))


# ---------------- 硬停止（P2）----------------
# 模型调用次数上限。一次"检索 → 回答"通常 2~3 次模型调用，给 6 次足够；
# 超过还能继续，基本可以判定模型在反复试错。
MAX_MODEL_CALLS = int(os.getenv("MAX_MODEL_CALLS", "6"))
# 工具调用次数上限。continue 语义：超限只是提示模型别再调，不终止本次运行。
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
# 墙钟上限（秒）。由 api/chat.py 的 asyncio.timeout 执行，不在中间件里 ——
# 超时后仍要能推 SSE 事件。
WALL_CLOCK_S = float(os.getenv("WALL_CLOCK_S", "90"))
