import os
import json
from dotenv import load_dotenv

load_dotenv()

import redis

_redis = redis.Redis.from_url(
    os.getenv("REDIS_URL"),
    decode_responses=True,
)

HISTORY_TTL = int(os.getenv("CHAT_HISTORY_TTL", "1800"))


def _key(user_id: int, session_id: str) -> str:
    return f"user_memory:{user_id}:{session_id}"


# ---------------- 跨轮压缩相关的读写 ----------------
# 都放在这一层：summary 模块只负责"算什么"，不碰 Redis 细节。
# 这样 memory 的测试用假 Redis 就能覆盖，summary 的测试用假存储也能覆盖。


def _summary_key(user_id: int, session_id: str) -> str:
    return f"{_key(user_id, session_id)}:summary"


def _upto_key(user_id: int, session_id: str) -> str:
    return f"{_key(user_id, session_id)}:upto"


def history_length(user_id: int, session_id: str) -> int:
    """当前会话累计条数（含首条）。压缩判断用它，不读内容。"""
    return _redis.llen(_key(user_id, session_id))


def read_summary(user_id: int, session_id: str) -> str | None:
    return _redis.get(_summary_key(user_id, session_id)) or None


def read_upto(user_id: int, session_id: str) -> int:
    """摘要已覆盖到的位置（不含）。默认 1 —— 首条永远保留原文，从第 2 条开始压。"""
    try:
        return int(_redis.get(_upto_key(user_id, session_id)) or 1)
    except (TypeError, ValueError):
        return 1


def write_summary(user_id: int, session_id: str, text: str, upto: int) -> None:
    """摘要与游标一起写，各自续 TTL。

    不用 pipeline：压缩只在跨阈值时发生一次，多两个 RTT 可接受，
    换来的是测试里用一个几十行的假 Redis 就能验证。
    """
    _redis.set(_summary_key(user_id, session_id), text)
    _redis.expire(_summary_key(user_id, session_id), HISTORY_TTL)
    _redis.set(_upto_key(user_id, session_id), str(upto))
    _redis.expire(_upto_key(user_id, session_id), HISTORY_TTL)


def read_range(user_id: int, session_id: str, start: int, end: int) -> list[dict]:
    """读 [start, end) 区间并解析；坏数据跳过，绝不让一条脏 JSON 打挂请求。"""
    if end <= start:
        return []
    raw = _redis.lrange(_key(user_id, session_id), start, end - 1)
    out: list[dict] = []
    for item in raw:
        try:
            obj = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def load_history(
    user_id: int,
    session_id: str,
    n: int = 10,
    *,
    include_summary: bool = True,
) -> list[dict]:
    """取对话历史, 按时间正序返回 [{role, content}]

    【首条常驻】首条 user 消息里带着图片的 VL 描述, 而后续轮次不再传图。
    若只用"最近 n 条", 第 5 轮起该描述就会被挤出窗口, 模型从此"失明"
    (实测: n=6 时第 5 轮丢失, n=10 时第 7 轮丢失)。
    故策略改为: 首条 + 最近 (n-1) 条。
    """
    key = _key(user_id, session_id)
    total = _redis.llen(key)
    if total <= n:
        raw = _redis.lrange(key, 0, -1)
    else:
        raw = _redis.lrange(key, 0, 0) + _redis.lrange(key, -(n - 1), -1)

    out = []
    for item in raw:
        try:
            obj = json.loads(item)
        except json.JSONDecodeError:
            continue
        # 只接受对象：Redis 里若混进合法 JSON 但非 dict 的值(如 "abc"/123),
        # 后续 out[0].get("role") 会抛 AttributeError 直接打挂整个请求。
        if isinstance(obj, dict):
            # 老数据里没有 partial 键（P2 之前写入的），统一补 False，
            # 调用方就不必到处写 .get("partial", False)。
            obj.setdefault("partial", False)
            out.append(obj)
    # 保证序列不以 assistant 开头(流式中断会留下不配对的尾巴)
    while out and out[0].get("role") != "user":
        out.pop(0)

    if include_summary:
        summary = read_summary(user_id, session_id)
        if summary:
            # 插在首条之后：首条必须留在最前（里面是图片的 VL 描述，丢了模型就"失明"）。
            # 用 system 角色而非 user/assistant，避免破坏对话轮次的配对关系。
            # 注意：此时返回条数是 n+1（摘要占一条），这是有意的 ——
            # 摘要替代的是"被压掉的十几条"，不是窗口内的内容。
            out.insert(
                1 if out and out[0].get("role") == "user" else 0,
                {"role": "system", "content": f"【早前对话摘要】\n{summary}"},
            )
    return out


def append_turn(
    user_id: int,
    session_id: str,
    role: str,
    content: str,
    *,
    partial: bool = False,
) -> None:
    """追加一条对话，并刷新TTL。

    partial=True 表示这条 assistant 回答是被熔断打断的半截内容。
    只在为 True 时写这个键：正常条目保持与旧数据同构，读侧用 setdefault 兜底。
    """
    payload = {"role": role, "content": content}
    if partial:
        payload["partial"] = True
    _redis.rpush(
        _key(user_id, session_id),
        json.dumps(payload, ensure_ascii=False),
    )

    _redis.expire(_key(user_id, session_id), HISTORY_TTL)
