# 混合检索通道
import os
import re

import jieba
from pymilvus import MilvusClient
from rank_bm25 import BM25Okapi

_MILVUS_URL = os.getenv("MILVUS_URI", "http://localhost:19530")

# 各库命名不一致: 旧库用 file_name, 农业库用 source_file; 其余业务字段按存在与否透传
_META_FIELDS = ("file_name", "source_file", "doc_id",
                "disease", "crop", "category", "section")


def tokenize(text: str) -> list[str]:
    """中英文分开治理，英文保留术语，中文走jieba"""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9_+#.@-]+", text)
    zh = re.sub(r"[^\u4e00-\u9fff]", "", text)
    tokens += [t for t in jieba.cut(zh) if t.strip()]
    return tokens


def _load_all_chunks(collection_name: str) -> list[dict]:
    """从Milvus拉全量chunk; 输出字段按 collection 实际 schema 动态探测, 避免字段不存在报错"""
    client = MilvusClient(uri=_MILVUS_URL)
    desc = client.describe_collection(collection_name)
    available = {f["name"] for f in desc.get("fields", [])}
    out_fields = ["text"] + [f for f in _META_FIELDS if f in available]

    client.load_collection(collection_name)
    # Milvus query 单次上限 16384; 分页拉取以覆盖大库(kb_agri≈1.5万条)
    rows: list[dict] = []
    page = 16384
    offset = 0
    while True:
        batch = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=out_fields,
            limit=page,
            offset=offset,
        )
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page

    chunks = []
    for r in rows:
        src = r.get("file_name") or r.get("source_file") or "未知"
        chunks.append(
            {
                **{k: r.get(k) for k in _META_FIELDS if k in available},
                "content": r.get("text", ""),
                "file_name": src,
                "source_file": src,
            }
        )
    return chunks


class Bm25Index:
    def __init__(self, collection_name: str):
        chunks = _load_all_chunks(collection_name)
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])

    def search(self, query: str, k: int = 20) -> list[dict]:
        if not self.chunks:
            return []
        toks = tokenize(query)
        if not toks:
            return []
        scores = self.bm25.get_scores(toks)
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        out = []
        for i in top:
            c = dict(self.chunks[i])
            c["bm25_score"] = float(scores[i])
            out.append(c)
        return out


_BM25_CACHE: dict[str, Bm25Index] = {}


def get_bm25(collection_name: str) -> Bm25Index:
    if collection_name not in _BM25_CACHE:
        _BM25_CACHE[collection_name] = Bm25Index(collection_name)
    return _BM25_CACHE[collection_name]
