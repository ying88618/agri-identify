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
# 单次请求的累计 token 上限（含 input + output + reasoning）。
# 实测一次模型调用约 1500~2300（推理模型 reasoning 占大头），
# 按 8 次调用估算约 18000，故默认 20000。
# 【口径】必须用 core/run_state 采集的真值，不能用 response.usage_metadata ——
# 后者在本 provider 上会被膨胀上百倍（见 core/llm_usage.py）。
MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "20000"))
# 同一工具 + 同一参数连续出现多少次判定为死循环。
# 取 3 而不是 2：模型偶发地用同样参数重查一次是正常的"再确认"，
# 连续 3 次才足以判定它没有新信息可拿了。
MAX_REPEAT_ACTIONS = int(os.getenv("MAX_REPEAT_ACTIONS", "3"))


# ---------------- 跨轮次上下文压缩（P4）----------------
# 只做"跨轮"，不做"单次运行内"：一次诊断最多 8 次模型调用，消息量不大；
# 而工具返回的文档原文是回答与溯源的依据，压掉反而丢证据。
SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "1") not in ("0", "false", "False")
# 会话累计超过这么多条才开始压缩（一条 user + 一条 assistant 算 2 条）
SUMMARY_TRIGGER_TURNS = int(os.getenv("SUMMARY_TRIGGER_TURNS", "12"))
# 保留最近多少条原文。与 api/chat.py 的 HISTORY_TURNS 对齐，避免"压了又读不到"
SUMMARY_KEEP_RECENT = int(os.getenv("SUMMARY_KEEP_RECENT", "6"))
# 摘要硬上限（字符）。模型偶发超长时截断，避免摘要本身把上下文撑爆。
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "600"))
