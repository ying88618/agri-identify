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


def load_history(user_id: int, session_id: str, n: int = 10) -> list[dict]:
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
