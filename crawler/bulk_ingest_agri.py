# -*- coding: utf-8 -*-
"""
bulk_ingest_agri.py — 农业病虫害语料入库 → Milvus kb_agri

用法:
    python crawler/bulk_ingest_agri.py                 # 全量入库(2940 条)
    python crawler/bulk_ingest_agri.py --limit 20      # 只入前 20 条(验证用)

输入: crawler/data/agri_pests.jsonl (excel_to_jsonl.py 产物)
chunk 设计: 每条病虫害按「字段」切 chunk(用户问"防治方法"应命中对应字段);
            单字段超 600 字再用 RecursiveCharacterTextSplitter 二次切分。
metadata:  crop(作物) / disease(名称) / category / section(字段名) / source_file
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.vectorstore import build_vs

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "agri_pests.jsonl")
COLLECTION = "kb_agri"
CHUNK_LIMIT = 600      # 单字段超此值二次切分
OVERLAP = 60
BATCH = 100            # 调小: 控制单次 embedding 请求 token, 避免硅基流动 TPM 限流
CONCURRENCY = 2        # 降并发: 配合 TPM 限制
START = 0              # 断点续传: 中断后从 --start N 继续

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_LIMIT, chunk_overlap=OVERLAP)


def make_chunks(rec: dict) -> list[tuple[str, dict]]:
    """按字段切 chunk; 返回 [(text, metadata)]"""
    meta_base = {
        "crop": rec.get("crop") or "跨作物/其他",
        "disease": rec["name"],
        "category": rec["category"],
        "source_file": rec["source_file"],
    }
    out = []
    for section, text in rec["fields"].items():
        t = (text or "").strip()
        if len(t) < 20:
            continue
        pieces = [t]
        if len(t) > CHUNK_LIMIT:
            pieces = [p for p in _splitter.split_text(t) if p.strip()]
        for p in pieces:
            m = dict(meta_base)
            m["section"] = section
            out.append((p, m))
    return out


async def main():
    limit = None
    start = START          # 局部副本: 直接在函数里给 START 赋值会把它变成局部变量,
                           # 导致下方 chunks[START:] 抛 UnboundLocalError
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--start" in sys.argv:
        start = int(sys.argv[sys.argv.index("--start") + 1])

    with open(DATA, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if limit:
        recs = recs[:limit]
    print(f"待入库: {len(recs)} 条病虫害")

    chunks: list[tuple[str, dict]] = []
    for r in recs:
        chunks.extend(make_chunks(r))
    print(f"切出 chunk: {len(chunks)} 个")

    vs = build_vs(COLLECTION)
    sem = asyncio.Semaphore(CONCURRENCY)

    async def write_batch(batch):
        async with sem:
            await vs.aadd_texts(
                [t for t, _ in batch],
                metadatas=[m for _, m in batch],
            )

    chunks = chunks[start:]
    done = 0
    total = start + len(chunks)
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        # 429/网络抖动时退避重试
        for attempt in range(5):
            try:
                await write_batch(batch)
                break
            except Exception as e:
                if attempt == 4:
                    raise
                wait = 30 * (attempt + 1)
                print(f"  写入失败({type(e).__name__}), 等待 {wait}s 重试...")
                await asyncio.sleep(wait)
        done += len(batch)
        print(f"[入库] {start + done}/{total}")
        await asyncio.sleep(0.3)      # 批次间隔, 稳住 TPM
    print(f"\n完成: {total} chunks → {COLLECTION}")


if __name__ == "__main__":
    asyncio.run(main())
