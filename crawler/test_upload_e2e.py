# -*- coding: utf-8 -*-
"""端到端验证：本地图片上传 -> 带 image_id 对话 -> 多轮命中 VL 描述缓存。

覆盖链路（全部经过完整 ASGI 应用：multipart 解析、鉴权依赖、SSE 生成器）：
  1. 登录拿令牌
  2. POST /files 上传一张 PlantVillage 真实叶片图 -> image_id
  3. 第 1 轮 /chat/stream：应调用 VL，并写出 {image_id}.desc.json
  4. 第 2 轮 同一个 image_id（换问题）：应命中缓存，不再调 VL
  5. 校验 Redis 历史写在「令牌里的 user_id」下

判断「第 2 轮没再调 VL」的依据是缓存文件的 mtime 不变 ——
若第 2 轮又调了一次 VL，写缓存会把 mtime 刷新。

注意：
  · 会真实调用 LLM + VL，两轮约 30~90 秒，产生少量费用。
  · 导入 main 会拉起 core.agent（模块级 create_agent），启动约 10~20 秒。
  · 需要 Milvus / Redis / MySQL 都在运行。

用法：
    python crawler/test_upload_e2e.py
    set E2E_USER=xxx && set E2E_PASS=yyy && python crawler/test_upload_e2e.py
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from fastapi.testclient import TestClient  # noqa: E402

from api.security import decode_token  # noqa: E402
from core.config import VL_MODEL_NAME  # noqa: E402
from core.images import description_path  # noqa: E402
from main import app  # noqa: E402

USERNAME = os.getenv("E2E_USER", "farmer01")
PASSWORD = os.getenv("E2E_PASS", "secret123")
SESSION = "e2e_s1"
DATASET = Path(os.getenv(
    "E2E_DATASET", r"e:\RAG_agri\PlantVillage-Dataset-master\color"
))
# 对应 kb_agri 里的「番茄早疫病」，是 agri_eval_map.py 里 confidence=high 的类
SAMPLE_CLASS = "Tomato___Early_blight"


def pick_sample() -> Path:
    d = DATASET / SAMPLE_CLASS
    if not d.is_dir():
        raise SystemExit(f"数据集目录不存在: {d}\n可用 E2E_DATASET 环境变量指定")
    files = sorted(
        p for p in d.iterdir()
        if p.suffix.upper() in (".JPG", ".JPEG", ".PNG")
    )
    if not files:
        raise SystemExit(f"目录为空: {d}")
    # 取排序后第一条：确定性，可复现（与 eval_multiturn.py 的采样思路一致）
    return files[0]


def parse_sse(text: str) -> list[dict]:
    """从 SSE 响应体里取出所有 data 帧。

    不要靠字符串匹配 '{"type":"done"}'：分隔符里有没有空格取决于
    sse-starlette 的 json.dumps 参数，换个版本就会失效。
    """
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            out.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            pass
    return out


def ask(client, token: str, question: str, image_id: str | None) -> str:
    body = {"session_id": SESSION, "question": question}
    if image_id:
        body["image_id"] = image_id
    t0 = time.time()
    r = client.post(
        "/chat/stream", json=body, headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200, f"/chat/stream 返回 {r.status_code}: {r.text[:300]}"
    frames = parse_sse(r.text)
    done = [f for f in frames if f.get("type") == "done"]
    assert done, f"未收到 done 事件: {r.text[:300]}"
    print(f"      ({len(frames)} 个 SSE 帧, {time.time() - t0:.1f}s)")
    return done[-1]["content"]


def main() -> None:
    sample = pick_sample()
    print(f"样本: {sample.name}  ({sample.stat().st_size / 1024:.1f} KB)")
    print(f"VL 模型: {VL_MODEL_NAME or '(未配置! VL_MODEL_NAME 为空)'}")

    with TestClient(app) as client:
        # ---- 1. 登录 ----
        r = client.post("/auth/login",
                        json={"username": USERNAME, "password": PASSWORD})
        if r.status_code != 200:
            r = client.post("/auth/register",
                            json={"username": USERNAME, "password": PASSWORD})
            assert r.status_code == 201, f"登录/注册均失败: {r.status_code} {r.text}"
        token = r.json()["access_token"]
        user_id = decode_token(token)
        print(f"[1] 登录成功 user_id={user_id}")

        # ---- 2. 上传本地图片 ----
        with sample.open("rb") as f:
            r = client.post(
                "/files",
                headers={"Authorization": f"Bearer {token}"},
                # 字段名必须是 file，对应 api/files.py 的 File(...)
                files={"file": (sample.name, f.read(), "image/jpeg")},
            )
        assert r.status_code == 201, f"上传失败 {r.status_code}: {r.text}"
        info = r.json()
        image_id = info["image_id"]
        print(f"[2] 上传成功 {info}")

        desc_path = description_path(user_id, image_id)
        assert not desc_path.exists(), f"此时不该有缓存: {desc_path}"
        print(f"    缓存路径: {desc_path}")

        # ---- 3. 第 1 轮：应触发 VL 并写缓存 ----
        print("[3] 第 1 轮对话（预期真正调用 VL）...")
        a1 = ask(client, token, "这片叶子有什么症状？", image_id)
        print(f"    回答: {a1[:120].replace(chr(10), ' ')}...")
        assert desc_path.exists(), (
            "第 1 轮后没有写出缓存 —— 多半是 VL 调用失败（看日志里的 "
            "'describe_image failed'），或 VL_MODEL_NAME 未配置"
        )
        cached = json.loads(desc_path.read_text(encoding="utf-8"))
        mtime1 = desc_path.stat().st_mtime_ns
        assert cached.get("model") == VL_MODEL_NAME, \
            f"缓存记录的模型({cached.get('model')})与环境变量({VL_MODEL_NAME})不一致"
        assert len(cached.get("text", "")) >= 20, \
            f"VL 描述过短、疑似无效: {cached.get('text')!r}"
        print(f"    缓存描述: {cached['text'][:120].replace(chr(10), ' ')}...")

        # ---- 4. 第 2 轮：同一 image_id，应命中缓存 ----
        print("[4] 第 2 轮对话（预期命中缓存、不再调 VL）...")
        a2 = ask(client, token, "那该怎么防治？", image_id)
        print(f"    回答: {a2[:120].replace(chr(10), ' ')}...")
        mtime2 = desc_path.stat().st_mtime_ns
        assert mtime1 == mtime2, (
            "缓存文件被重写了 —— 说明第 2 轮又调了一次 VL，缓存没生效。"
            "检查 api/chat.py 里 read_cached_description 的返回值判断"
        )
        print("    缓存 mtime 未变 -> 确认命中缓存，第 2 轮没有调 VL")

        # ---- 5. Redis 历史的归属 ----
        from core.memory import load_history  # 延后导入，缩短启动时间

        hist = load_history(user_id, SESSION, n=10)
        print(f"[5] Redis 历史 {len(hist)} 条（键 user_memory:{user_id}:{SESSION}）")
        assert hist, "历史为空：append_turn 没写到令牌对应的 user_id 下"
        assert "图片内容描述如下" in hist[0]["content"], (
            "首条消息里应含 VL 描述 —— core/memory.py 的【首条常驻】机制依赖它，"
            f"实际首条: {hist[0]['content'][:120]}"
        )
        print(f"    首条含图片描述 ✓  {hist[0]['content'][:80]}...")

    print("\n--- 全部通过 ---")
    print("清理本轮产物（可选）：")
    print(f"  del {desc_path.parent}\\{image_id}.jpg")
    print(f"  del {desc_path}")


if __name__ == "__main__":
    main()
