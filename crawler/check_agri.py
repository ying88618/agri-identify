# -*- coding: utf-8 -*-
"""验证 kb_agri 数据: collection 列表 + 稻瘟病抽样 + 作物覆盖"""
from pymilvus import MilvusClient

c = MilvusClient(uri="http://localhost:19530")
print("collections:", c.list_collections())

c.load_collection("kb_agri")
print("\n[稻瘟病 相关 chunk]")
r = c.query("kb_agri", filter='disease == "稻瘟病"',
            output_fields=["crop", "disease", "section", "text"], limit=10)
print("数量:", len(r))
for x in r[:5]:
    print(f"  {x['crop']} | {x['disease']} | {x['section']} | {x['text'][:40]}")

print("\n[水稻 crop 覆盖抽样]")
r2 = c.query("kb_agri", filter='crop == "水稻"',
             output_fields=["disease", "section"], limit=300)
diseases = sorted({x["disease"] for x in r2})
print(f"水稻相关 chunks: {len(r2)}, 涉及病害/虫害: {len(diseases)} 种")
print("前 10 种:", diseases[:10])

print("\n[元数据字段完整度抽查: 有无 crop 为空/缺失]")
r3 = c.query("kb_agri", filter='crop == "跨作物/其他"',
             output_fields=["disease", "section"], limit=5)
print(f"跨作物(crop为空)的 chunks: {len(r3)} 条(检疫/入侵类, 预期存在)")
