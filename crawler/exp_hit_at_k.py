# -*- coding: utf-8 -*-
"""验证「hit@5 的含义是否变弱」——即 top-5 到底有多少来自"排序"，多少来自"运气"。

【背景】
前面的探针发现 rerank 分数是饱和的二分类置信度（同 query 下 top8 跨度 <0.01，
离题 query 全部恰好 0.0000）。若真如此，那么"取前 5 条"这个动作在饱和簇内部
基本是任意的，hit@5 测的就只是"正确病害有没有落进这个簇"。

【把模糊猜测拆成三个可证伪的量】
1) 随机基线对照
   若排序无信息，正确病害落进 top-k 的概率 = k/N（N 为池子大小）。
   实测 P(rank<=k | 在池中) 与它的比值就是 lift。lift≈1 表示排序无用。
2) 失手归因分解（对没进 top-5 的样本）
   A 不在池子里            -> 召回问题（embedding / BM25 / RRF）
   B 在池子里但分数<阈值   -> rerank 判它"非同话题"（rerank 出错）
   C 在同话题簇内但排名靠后 -> 排序问题（被并列的文档挤出去）
   三者的修复手段完全不同，混在一起看就只是一个 "hit@5=51%" 的数字。
3) 并列度
   正确病害的分数与多少条文档完全相同。并列时"排名"只是 API 返回顺序，
   不是判别结果。若命中里有大量并列边界，说明 hit@5 含运气成分。

用法：
    python crawler/exp_hit_at_k.py            # 每作物 4 条 = 40 样本
    set PER_CROP=2 && python crawler/exp_hit_at_k.py
"""
import asyncio
import json
import os
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from pymilvus import MilvusClient  # noqa: E402

from core.config import DEFAULT_SCORE_THRESHOLD, KB_COLLECTION, MILVUS_URI  # noqa: E402
from core.retriever import retrieve  # noqa: E402

DATA = ROOT / "crawler" / "data"
DESC_FILE = os.getenv("DESC_FILE", "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl")
PER_CROP = int(os.getenv("PER_CROP", "4"))
CONC = int(os.getenv("CONC", "4"))
MAXK = 100
HEALTHY = "__HEALTHY__"
KS = [1, 3, 5, 10, 20, 50, 100]
TIE_DP = 4

# NO_RERANK=1 时把 _rerank 打桩成"永远失败"，于是 retrieve 走降级路径
# （candidates[:k]，即 RRF 融合顺序）。用来分离两件事：
#   返回顺序里有多少来自 rerank，多少来自上游的 RRF。
# 注意降级路径下 score_threshold 是按向量余弦分过滤的（尺度与 rerank 分不同），
# 所以这里不传阈值，两边都拿纯顺序，比较才公平。
NO_RERANK = os.getenv("NO_RERANK", "") not in ("", "0")


def preflight() -> None:
    try:
        names = MilvusClient(uri=MILVUS_URI).list_collections()
    except Exception as e:
        raise SystemExit(f"Milvus 不可用（{MILVUS_URI}）。先 docker compose up -d\n{e}")
    if KB_COLLECTION not in names:
        raise SystemExit(f"集合 {KB_COLLECTION} 不存在: {names}")
    print(f"Milvus OK  collections={names}", flush=True)


def load_valid_pairs() -> set[tuple[str, str]]:
    pairs = set()
    with open(DATA / "agri_pests.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("crop") and r.get("name"):
                    pairs.add((r["crop"], r["name"]))
    return pairs


def load_samples(valid: set[tuple[str, str]]) -> list[dict]:
    by_crop: dict[str, list[dict]] = defaultdict(list)
    with open(DATA / DESC_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("truth") == HEALTHY or not (r.get("desc") or "").strip():
                continue
            if (r.get("crop"), r.get("truth")) not in valid:
                continue
            by_crop[r["crop"]].append(r)
    out = []
    for crop in sorted(by_crop):
        out += by_crop[crop][:PER_CROP]
    return out


async def probe_one(s: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        rows = await retrieve(
            s["desc"], KB_COLLECTION, k=MAXK, recall_k=MAXK, hybrid=True,
            crop=s["crop"], section="危害症状",
        )
    scores = [float(r.get("score", 0.0)) for r in rows]
    diseases = [r.get("disease") for r in rows]
    truth = s["truth"]
    pool = len(rows)

    rank = diseases.index(truth) + 1 if truth in diseases else None
    tscore = scores[rank - 1] if rank else None
    n_tied = n_above = None
    if tscore is not None:
        key = round(tscore, TIE_DP)
        n_tied = sum(1 for x in scores if round(x, TIE_DP) == key)
        n_above = sum(1 for x in scores if x > tscore)

    return {
        "crop": s["crop"], "truth": truth, "desc": s["desc"],
        "pool": pool, "rank": rank, "truth_score": tscore,
        "top1_disease": diseases[0] if diseases else None,
        "top1_score": scores[0] if scores else 0.0,
        "n_tied_at_truth_score": n_tied,
        "n_above_truth": n_above,
        "cluster_ge_threshold": sum(1 for x in scores if x >= DEFAULT_SCORE_THRESHOLD),
        "scores": [round(x, TIE_DP) for x in scores],
    }


async def main() -> None:
    if NO_RERANK:
        import core.retriever as _r

        async def _always_fail(*_a, **_k):
            return None          # -> retrieve 降级为 RRF 顺序

        _r._rerank = _always_fail
        print("!! 对照组：已禁用 rerank，走降级路径（RRF 顺序）", flush=True)

    preflight()
    valid = load_valid_pairs()
    samples = load_samples(valid)
    print(f"样本: {len(samples)}（每作物 {PER_CROP}）  阈值={DEFAULT_SCORE_THRESHOLD}\n",
          flush=True)

    sem = asyncio.Semaphore(CONC)
    t0 = time.time()
    rows = []
    for i, s in enumerate(samples, 1):
        rows.append(await probe_one(s, sem))
        print(f"  [{i}/{len(samples)}] {s['crop']} {s['truth'][:14]:16s} "
              f"pool={rows[-1]['pool']:3d} rank={rows[-1]['rank']}", flush=True)
    print(f"\n耗时 {time.time() - t0:.0f}s\n", flush=True)
    got = [r for r in rows if r["pool"] > 0]
    if not got:
        raise SystemExit("没有任何样本返回候选，后续统计无意义")

    # ---- 1) hit@k 曲线 + 随机基线对照 ----
    print("=" * 82)
    print("① hit@k 与随机基线对照")
    print(f"{'k':>5s} {'实测':>8s} {'随机基线':>10s} {'lift':>7s}   说明")
    print("-" * 82)
    for k in KS:
        obs = [r for r in got if r["rank"] and r["rank"] <= k]
        # 随机基线只对"真值确在池中"的样本有意义，分母也用同一批样本
        inpool = [r for r in got if r["rank"]]
        if not inpool:
            continue
        p_obs = len(obs) / len(inpool)
        p_rand = sum(min(k, r["pool"]) / r["pool"] for r in inpool) / len(inpool)
        lift = (p_obs / p_rand) if p_rand else 0
        print(f"{k:5d} {p_obs:7.1%} {p_rand:9.1%} {lift:7.2f}"
              f"   {'排序有效' if lift > 2 else '≈随机，排序无信息'}")
    print(f"\n真值在池中的样本: {len([r for r in got if r['rank']])}/{len(got)}"
          f"（{len([r for r in got if r['rank']])/len(got):.0%}）"
          f" —— 这是 hit@k 的**上限**")
    print("=" * 82)

    # ---- 2) 失手归因分解 ----
    miss = [r for r in got if not (r["rank"] and r["rank"] <= 5)]
    if miss:
        cat_a = [r for r in miss if r["rank"] is None]
        cat_b = [r for r in miss if r["rank"] and r["truth_score"] < DEFAULT_SCORE_THRESHOLD]
        cat_c = [r for r in miss if r["rank"] and r["truth_score"] >= DEFAULT_SCORE_THRESHOLD]
        print(f"\n② 未进 top-5 的 {len(miss)} 例，按原因分解:")
        print(f"   A 不在池子里（召回问题）            : {len(cat_a):3d}  ({len(cat_a)/len(miss):.0%})")
        print(f"   B 在池里但被 rerank 判为非同话题     : {len(cat_b):3d}  ({len(cat_b)/len(miss):.0%})")
        print(f"   C 在同话题簇内但排名靠后（排序问题） : {len(cat_c):3d}  ({len(cat_c)/len(miss):.0%})")
    else:
        print("\n② 没有未进 top-5 的样本")

    # ---- 3) 并列度 ----
    hits = [r for r in got if r["rank"] and r["rank"] <= 5]
    tied = [r for r in hits if (r["n_tied_at_truth_score"] or 0) > 1]
    print(f"\n③ 并列度（真值分数与多少条文档完全相同，4 位小数）")
    if hits:
        print(f"   命中的 {len(hits)} 例中，分数有并列的: {len(tied)}"
              f"  ({len(tied)/len(hits):.0%})")
        print(f"   命中样本里真值平均被 {st.mean([r['n_tied_at_truth_score'] for r in hits]):.1f}"
              f" 条文档并列，平均被 {st.mean([r['n_above_truth'] for r in hits]):.1f} 条压在上方")
    if [r for r in got if r["rank"]]:
        ranks = [r["rank"] for r in got if r["rank"]]
        print(f"   真值排名: 中位数={st.median(ranks):.0f}  最小={min(ranks)}"
              f"  最大={max(ranks)}")
        print(f"   排名前 5 的分布: " + " ".join(
            f"{k}->{len([x for x in ranks if x == k])}" for k in range(1, 6)))

    # ---- 4) 同话题簇（阈值语义）----
    print(f"\n④ 若把阈值 {DEFAULT_SCORE_THRESHOLD} 应用到正常路径（候选>=阈值的条数）")
    empty = [r for r in got if r["cluster_ge_threshold"] == 0]
    print(f"   同话题簇为空的样本: {len(empty)}/{len(got)}"
          f"  ({len(empty)/len(got):.0%}) <- 这些会变成显式的\"未发现病害\"")
    if got:
        print(f"   簇大小: 中位数={st.median([r['cluster_ge_threshold'] for r in got]):.0f}"
              f"  均值={st.mean([r['cluster_ge_threshold'] for r in got]):.1f}")
    tr_in_cluster = [r for r in got if r["rank"] and r["truth_score"] >= DEFAULT_SCORE_THRESHOLD]
    print(f"   真值也在簇内的样本: {len(tr_in_cluster)}/{len(got)}"
          f"  ({len(tr_in_cluster)/len(got):.0%}) <- 这是应用阈值后 hit 的**上限**")
    print("=" * 82)

    # 两个 arm 分开存，避免互相覆盖
    out = DATA / ("exp_hit_at_k_norerank.jsonl" if NO_RERANK
                  else "exp_hit_at_k.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
