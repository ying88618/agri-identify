# -*- coding: utf-8 -*-
"""列出 kb_agri 指定作物的全部 disease 名(用于建 PlantVillage 类名映射)"""
from pymilvus import MilvusClient

c = MilvusClient(uri="http://localhost:19530")
c.load_collection("kb_agri")
for crop in ["番茄", "玉米", "马铃薯"]:
    r = c.query("kb_agri", filter=f'crop == "{crop}"',
                output_fields=["disease"], limit=5000)
    names = sorted({x["disease"] for x in r})
    print(f"\n[{crop}] {len(names)} 种:")
    print("  " + " | ".join(names))
