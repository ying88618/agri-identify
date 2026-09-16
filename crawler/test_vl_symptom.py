# -*- coding: utf-8 -*-
"""
test_vl_symptom.py — 验证「VL 症状描述 → kb_agri 检索确诊」分层架构

核心假设: VL 直接闭集分类弱(实测 Top-1 36.5%), 但"客观描述症状"是 VL 强项;
让 VL 输出结构化症状文本 → 用文本去 kb_agri 检索 → 看命中正确病害的比例。

用法:
    python crawler/test_vl_symptom.py                 # 默认每类 3 张
    python crawler/test_vl_symptom.py --per-class 5 --pattern Tomato
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

from core.embeddings import embeddings
from pymilvus import MilvusClient

IMG_ROOT = r"E:\RAG_agri\PlantVillage-Dataset-master\color"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
CONCURRENCY = 2
TOPK = 3          # 判定 Top-3 命中

# PlantVillage 类 → (作物中文名, kb_agri 中的准确病害名)
MAP = {
    "Tomato___Early_blight": ("番茄", "番茄早疫病"),
    "Tomato___Late_blight": ("番茄", "番茄晚疫病"),
    "Tomato___Leaf_Mold": ("番茄", "番茄叶霉病"),
    "Tomato___Septoria_leaf_spot": ("番茄", "番茄斑枯病"),
    "Tomato___Tomato_mosaic_virus": ("番茄", "番茄花叶病毒病"),
    "Corn_(maize)___Common_rust_": ("玉米", "玉米锈病"),
    "Corn_(maize)___Northern_Leaf_Blight": ("玉米", "玉米大斑病"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("玉米", "玉米灰斑病"),
    "Potato___Early_blight": ("马铃薯", "马铃薯早疫病"),
    "Potato___Late_blight": ("马铃薯", "马铃薯晚疫病"),
}

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      base_url=os.getenv("OPENAI_BASE_URL"))
_milvus = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
_milvus.load_collection("kb_agri")

DESCRIBE_PROMPT = """你是一名农业植保专家。请仔细观察这张叶片/植株照片，用中文客观描述可见症状，供后续病害诊断参考。

请按以下结构输出：
1) 作物判断与整体状态（是否健康、萎蔫、发黄等）
2) 病斑细节：颜色、形状、边缘特征、分布位置
3) 表面附着物：有无霉层/粉状物/孢子堆/虫体/网丝等
4) 若叶片健康无明显病症，直接说明"无明显病症"

要求：用农业术语客观描述，但【不要臆断或说出具体病害名称】，控制在 150 字内。"""


def to_b64_dataurl(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def search_kb(query_text: str, crop: str) -> list[str]:
    """向量检索 kb_agri(限定作物), 返回 Top-K disease 名"""
    vec = embeddings.embed_query(query_text)
    res = _milvus.search("kb_agri", data=[vec], anns_field="vector", limit=TOPK,
                         filter=f'crop == "{crop}"',
                         output_fields=["crop", "disease", "section"])
    return [h["entity"].get("disease", "") for h in res[0]] if res else []


async def describe(img: str) -> str:
    resp = await _client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": to_b64_dataurl(img)}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--pattern", default="Tomato,Corn_(maize),Potato")
    args = ap.parse_args()
    patterns = [p for p in args.pattern.split(",") if p]

    sem = asyncio.Semaphore(CONCURRENCY)
    stats = {}   # class -> [hit_top1, hit_top3, n]
    total = hit1 = hit3 = 0
    os.makedirs(OUT_DIR, exist_ok=True)

    async def run(cls_name, img):
        async with sem:
            desc = await describe(img)
            if "healthy" in cls_name:
                return cls_name, img, desc, None
            crop, truth = MAP.get(cls_name, (None, None))
            top = search_kb(desc, crop) if crop else []
            t1 = bool(top and top[0] == truth)
            t3 = truth in top
            return cls_name, img, desc, (crop, truth, top, t1, t3)

    tasks = []
    for cls_dir in sorted(glob.glob(os.path.join(IMG_ROOT, "*"))):
        name = os.path.basename(cls_dir)
        if not any(name.startswith(p) for p in patterns):
            continue
        if name not in MAP and "healthy" not in name:
            continue   # 只测映射到的病害类 + healthy
        imgs = [f for f in glob.glob(os.path.join(cls_dir, "*"))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        tasks += [run(name, p) for p in imgs[: args.per_class]]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    with open(os.path.join(OUT_DIR, "vl_symptom_result.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            if isinstance(r, Exception):
                print(f"  [异常] {r}")
                continue
            cls_name, img, desc, detail = r
            rec = {"class": cls_name, "img": os.path.basename(img), "desc": desc}
            if detail is None:   # healthy
                f.write(json.dumps(rec | {"note": "healthy"}, ensure_ascii=False) + "\n")
                print(f"\n[健康对照] {cls_name} | {os.path.basename(img)}")
                print(f"  描述: {desc[:80]}...")
                continue
            crop, truth, top, t1, t3 = detail
            total += 1
            hit1 += t1
            hit3 += t3
            s = stats.setdefault(cls_name, [0, 0, 0])
            s[0] += t1
            s[1] += t3
            s[2] += 1
            mark = "✅TOP1" if t1 else ("🔸TOP3" if t3 else "❌")
            print(f"\n[{mark}] {cls_name} | 真实={truth}")
            print(f"  描述: {desc[:100]}...")
            print(f"  检索Top{TOPK}: {top}")
            f.write(json.dumps(rec | {"truth": truth, "crop": crop,
                                      "top": top, "top1": t1, "top3": t3},
                               ensure_ascii=False) + "\n")

    if total:
        print(f"\n===== 分层诊断验证 (VL描述 → kb_agri检索, 模型 {MODEL}) =====")
        print(f"病害样本: {total}, Top-1 命中: {hit1 / total:.1%}, Top-{TOPK} 命中: {hit3 / total:.1%}")
        for cls, (h1, h3, n) in sorted(stats.items()):
            print(f"  {cls}: Top1 {h1}/{n} ({h1 / n:.0%}), Top{TOPK} {h3}/{n} ({h3 / n:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
