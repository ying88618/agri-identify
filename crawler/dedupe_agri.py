# -*- coding: utf-8 -*-
"""dedupe_agri.py — 清除 kb_agri 里的重复 chunk(原地按主键删除, 不重新 embedding)

重复判定键: (text, crop, disease, category, source_file, section) 全等。
六个字段只要有一个不同就当不同记录, 不做模糊匹配, 不会误删。

成因: bulk_ingest_agri.py 先在 --limit 20 下冒烟(111 chunks), 随后全量又写入
同一批前 20 条; auto_id=True 不具备去重语义, 于是库里留下 111 条镜像。

用法:
    python crawler/dedupe_agri.py            # 试运行: 只统计, 打印将删除的条数
    python crawler/dedupe_agri.py --apply    # 真正删除
"""
import os
import sys

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

COLLECTION = "kb_agri"
KEY_FIELDS = ("text", "crop", "disease", "category", "source_file", "section")
SCAN_BATCH = 1000
DELETE_BATCH = 100


def main():
    apply = "--apply" in sys.argv
    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
    client.load_collection(COLLECTION)
    print(f"collection={COLLECTION} "
          f"row_count={client.get_collection_stats(COLLECTION)['row_count']}")

    keep: dict = {}          # 键 -> 保留的 pk(先入库的那条)
    dup_pks: list = []
    scanned = 0
    it = client.query_iterator(
        collection_name=COLLECTION,
        batch_size=SCAN_BATCH,
        filter="",
        output_fields=["pk", *KEY_FIELDS],
    )
    try:
        while True:
            rows = it.next()
            if not rows:
                break
            for r in rows:
                scanned += 1
                key = tuple(r.get(f) for f in KEY_FIELDS)
                if key in keep:
                    dup_pks.append(r["pk"])
                else:
                    keep[key] = r["pk"]
    finally:
        it.close()

    print(f"扫描 {scanned} 行 -> 唯一 {len(keep)} 条, 重复 {len(dup_pks)} 条")
    if not dup_pks:
        print("没有重复, 无需处理。")
        return
    if not apply:
        print(f"[试运行] 将删除 {len(dup_pks)} 条, 删除后 row_count = {len(keep)}")
        print("确认无误后加 --apply 执行。")
        return

    for i in range(0, len(dup_pks), DELETE_BATCH):
        client.delete(collection_name=COLLECTION, ids=dup_pks[i:i + DELETE_BATCH])
    client.flush(COLLECTION)
    client.load_collection(COLLECTION)
    print(f"已删除 {len(dup_pks)} 条, 当前 row_count = "
          f"{client.get_collection_stats(COLLECTION)['row_count']}")


if __name__ == "__main__":
    main()
