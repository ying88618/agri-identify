# -*- coding: utf-8 -*-
"""dump_agri.py — 把 Milvus 集合(含向量)导出到本地 jsonl, 用于灾难恢复

【为什么把向量也导出来】
恢复时直接把备份里的向量写回去, 不需要重新调 embedding 接口 ——
既省钱, 也避免"重灌一次的向量和原来不完全一致"造成检索结果漂移。

【用法】
    python crawler/dump_agri.py                          # 默认导 kb_agri
    python crawler/dump_agri.py --collection kb_default
    python crawler/dump_agri.py --out D:\\bak_agri

【产出】(默认目录 backups/)
    agri-20260914-143012.jsonl        每行一条: 全部标量字段 + 向量(base64 打包的 float32)
    agri-20260914-143012.meta.json    集合名 / 字段列表 / 向量维度 / embedding 模型 / 条数

字段是**动态探测**的, 所以 kb_default 那种字段不同的集合也能直接导。
"""
import argparse
import array
import base64
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from pymilvus import MilvusClient

from core.config import EMBEDDING_MODEL, KB_COLLECTION, MILVUS_URI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "backups")
BATCH = 500
VECTOR_TYPE = 101          # Milvus DataType.FLOAT_VECTOR


def pack_vector(vec) -> str:
    """float32 打包再 base64: 1024 维从上千个 JSON 数字压到约 5.5KB 文本"""
    return base64.b64encode(array.array("f", vec).tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default=KB_COLLECTION)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    client = MilvusClient(uri=MILVUS_URI)
    all_cols = client.list_collections()
    if args.collection not in all_cols:
        print(f"x 集合 {args.collection} 不存在, 当前库中有: {all_cols}")
        return 1
    client.load_collection(args.collection)

    desc = client.describe_collection(args.collection)
    fields = [f for f in desc["fields"] if not f.get("is_primary")]
    names = [f["name"] for f in fields]
    vec_fields = [f["name"] for f in fields if f["type"] == VECTOR_TYPE]
    dims = {f["name"]: f["params"].get("dim") for f in fields if f["type"] == VECTOR_TYPE}
    print(f"集合 {args.collection}: 字段 {names}, 向量字段 {vec_fields} {dims}")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = os.path.join(args.out, f"{args.collection}-{stamp}")
    data_path, meta_path = base + ".jsonl", base + ".meta.json"

    n = 0
    it = client.query_iterator(
        collection_name=args.collection,
        batch_size=BATCH,
        filter="",
        output_fields=names,
    )
    try:
        with open(data_path, "w", encoding="utf-8") as f:
            while True:
                rows = it.next()
                if not rows:
                    break
                for r in rows:
                    for vf in vec_fields:
                        if r.get(vf) is not None:
                            r[vf] = pack_vector(r[vf])
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
                print(f"\r导出 {n} 条...", end="", flush=True)
    finally:
        it.close()

    index_info = []
    try:
        for ix_name in client.list_indexes(args.collection):
            ix = client.describe_index(args.collection, ix_name)
            index_info.append({
                "index_name": ix.get("index_name") or ix_name,
                "field_name": ix.get("field_name"),
                "index_type": ix.get("index_type"),
                "metric_type": ix.get("metric_type"),
            })
    except Exception as e:          # 索引信息只用于还原, 拿不到不影响数据本身
        print(f"! 读取索引信息失败(不影响备份数据): {e}")

    meta = {
        "collection": args.collection,
        "row_count": n,
        "fields": names,
        "vector_fields": vec_fields,
        "dims": dims,
        "auto_id": desc.get("auto_id", True),
        "enable_dynamic_field": desc.get("enable_dynamic_field", False),
        "raw_fields": desc["fields"],   # 完整字段定义: restore 靠它原样重建 schema
        "index_info": index_info,       # 索引类型 / 度量方式
        "embedding_model": EMBEDDING_MODEL,
        "milvus_uri": MILVUS_URI,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": os.path.basename(data_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(data_path) / 1024 / 1024
    print(f"\n完成: {n} 条 ({size_mb:.1f} MB) -> {data_path}")
    print(f"      元信息 -> {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
