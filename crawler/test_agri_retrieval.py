# -*- coding: utf-8 -*-
"""验证 kb_agri 向量检索质量: 农业问题召回测试(含 crop 过滤对比)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import MilvusClient

from core.embeddings import embeddings

URI = os.getenv("MILVUS_URI", "http://localhost:19530")
c = MilvusClient(uri=URI)
c.load_collection("kb_agri")

QUESTIONS = [
    ("水稻得了稻瘟病怎么防治", "水稻"),
    ("番茄叶片出现褐色圆形病斑，边缘有黄色晕圈，是什么病", "番茄"),
    ("小麦赤霉病有什么防治方法", "小麦"),
    ("玉米螟怎么防治", "玉米"),
]


def search(q: str, crop: str | None, topk: int = 5):
    vec = embeddings.embed_query(q)
    flt = f'crop == "{crop}"' if crop else ""
    res = c.search("kb_agri", data=[vec], anns_field="vector", limit=topk,
                   filter=flt or None,
                   output_fields=["crop", "disease", "section", "text"])
    return res[0]


for q, crop in QUESTIONS:
    print(f"\n=== 问: {q}")
    # ① 不限作物
    print("  [不限定作物]")
    for i, hit in enumerate(search(q, None, 5)):
        e = hit["entity"]
        print(f"    {i+1}. [{e['crop']}] {e['disease']} | {e['section']} | {e['text'][:36]}")
    # ② 限定作物
    if crop:
        print(f"  [限定作物: {crop}]")
        for i, hit in enumerate(search(q, crop, 5)):
            e = hit["entity"]
            print(f"    {i+1}. [{e['crop']}] {e['disease']} | {e['section']} | {e['text'][:36]}")
