# -*- coding: utf-8 -*-
"""restore_agri.py — 用 dump 出来的 jsonl 恢复集合(直接写回备份里的向量, 不调 embedding)

【用法】
    python crawler/restore_agri.py --file backups\\kb_agri-20260914-140000.jsonl
    python crawler/restore_agri.py --file ... --collection kb_agri
    python crawler/restore_agri.py --file ... --append      # 往非空集合追加

【安全性】
    * 目标集合不存在      -> 按备份 meta 里的 raw_fields 原样重建 schema + 索引;
    * 目标集合已存在且非空 -> 默认**拒绝**, 避免灌成双份(要追加必须显式 --append);
    * 备份的 embedding 模型与当前 .env 不一致时告警(向量是原模型算的, 混用会让检索失真)。

注意: 不要用 core.vectorstore.build_vs() 来建集合 —— langchain_milvus 是**懒创建**,
构造时不会真的建表, 后面 insert 会报 collection not found。
"""
import argparse
import array
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from pymilvus import DataType, MilvusClient

from core.config import EMBEDDING_MODEL, KB_COLLECTION, MILVUS_URI

BATCH = 200

# Milvus 字段类型编号 -> pymilvus DataType(未列出的类型会明确报错, 不静默降级)
_DTYPE = {
    1: DataType.BOOL,
    2: DataType.INT8,
    3: DataType.INT16,
    4: DataType.INT32,
    5: DataType.INT64,
    10: DataType.FLOAT,
    11: DataType.DOUBLE,
    21: DataType.VARCHAR,
    22: DataType.ARRAY,
    23: DataType.JSON,
    101: DataType.FLOAT_VECTOR,
}


def unpack_vector(s: str):
    a = array.array("f")
    a.frombytes(base64.b64decode(s))
    return a.tolist()


def create_collection(client: MilvusClient, name: str, meta: dict) -> str:
    """按备份里的字段定义重建集合, 返回向量字段名"""
    raw_fields = meta.get("raw_fields")
    if not raw_fields:
        raise SystemExit(
            "x 这个备份没有 raw_fields 元信息(旧版脚本产物), 无法原样重建集合。\n"
            "  请用当前版本的 crawler/dump_agri.py 重新导出一份。"
        )

    auto_id = meta.get("auto_id", True)
    schema = client.create_schema(
        auto_id=auto_id,
        enable_dynamic_field=meta.get("enable_dynamic_field", False),
    )
    vector_field = None
    for f in raw_fields:
        dt = _DTYPE.get(f["type"])
        if dt is None:
            raise SystemExit(f"x 未知字段类型 type={f['type']} (字段 {f['name']}), 请手工建集合")
        kw = {}
        if dt == DataType.VARCHAR:
            kw["max_length"] = (f.get("params") or {}).get("max_length", 65535)
        if dt == DataType.FLOAT_VECTOR:
            kw["dim"] = (f.get("params") or {}).get("dim")
            vector_field = f["name"]
        if f.get("is_primary"):
            kw["is_primary"] = True
            kw["auto_id"] = f.get("auto_id", auto_id)
        schema.add_field(f["name"], dt, **kw)

    idx = (meta.get("index_info") or [{}])[0]
    ip = client.prepare_index_params()
    ip.add_index(
        field_name=idx.get("field_name") or vector_field,
        index_type=idx.get("index_type") or "AUTOINDEX",
        metric_type=idx.get("metric_type") or "COSINE",
    )
    client.create_collection(collection_name=name, schema=schema, index_params=ip)
    print(f"已建集合 {name}: {len(raw_fields)} 个字段, "
          f"索引 {idx.get('index_type') or 'AUTOINDEX'}/{idx.get('metric_type') or 'COSINE'}")
    return vector_field


def count_of(client: MilvusClient, coll: str) -> int:
    client.load_collection(coll)
    res = client.query(coll, filter="", output_fields=["count(*)"])
    return res[0]["count(*)"] if res else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="dump 出来的 .jsonl 路径")
    ap.add_argument("--collection", help="目标集合名, 默认沿用备份里的")
    ap.add_argument("--append", action="store_true", help="允许往非空集合追加")
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"x 文件不存在: {args.file}")
        return 1

    meta_path = os.path.splitext(args.file)[0] + ".meta.json"
    if not os.path.exists(meta_path):
        print(f"x 缺少元信息文件: {meta_path}")
        return 1
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    target = args.collection or meta.get("collection") or KB_COLLECTION
    vec_fields = set(meta.get("vector_fields") or ["vector"])
    # 只投喂"非主键"字段: query 出来的行里一定带 pk, 而 auto_id 集合不允许显式传主键,
    # 整行 insert 会报 "more fieldData has pass in: expected=N actual=N+1"。
    ins_fields = [f["name"] for f in (meta.get("raw_fields") or [])
                  if not f.get("is_primary")] or list(meta.get("fields") or [])
    print(f"备份: 集合={meta.get('collection')} 条数={meta.get('row_count')} "
          f"模型={meta.get('embedding_model')} -> 恢复到 {target}")

    if meta.get("embedding_model") and meta["embedding_model"] != EMBEDDING_MODEL:
        print(f"! 告警: 备份用 {meta['embedding_model']}, 当前 .env 是 {EMBEDDING_MODEL}。"
              f" 向量是原模型产的, 混用会让检索结果失真。")

    client = MilvusClient(uri=MILVUS_URI)
    if target in client.list_collections():
        cur = count_of(client, target)
        if cur and not args.append:
            print(f"x 集合 {target} 已有 {cur} 条, 拒绝覆盖(会灌成双份)。")
            print("  请先清空/删除该集合, 或显式加 --append。")
            return 1
        print(f"集合已存在, 当前 {cur} 条, 追加写入")
    else:
        create_collection(client, target, meta)

    n = 0
    buf = []
    with open(args.file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            missing = [k for k in ins_fields if k not in raw]
            if missing:
                raise SystemExit(f"x 备份行缺少字段 {missing}, 与集合 schema 不匹配")
            row = {k: raw[k] for k in ins_fields}
            for vf in vec_fields:
                if isinstance(row.get(vf), str):
                    row[vf] = unpack_vector(row[vf])
            buf.append(row)
            if len(buf) >= BATCH:
                client.insert(collection_name=target, data=buf)
                n += len(buf)
                buf = []
                print(f"\r恢复 {n} 条...", end="", flush=True)
    if buf:
        client.insert(collection_name=target, data=buf)
        n += len(buf)

    client.flush(target)
    total = count_of(client, target)
    print(f"\n完成: 本次写入 {n} 条, 集合 {target} 现在共 {total} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
