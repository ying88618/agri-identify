# -*- coding: utf-8 -*-
"""一次性探针：rerank 分数的分布形态（是否双峰 / 中段是否被掏空）。

【为什么查这个】
实验 exp_query_rewrite.py 里发现"未命中的平均分比命中的还高"，
怀疑分数不是连续的相关性分，而是接近二分类的置信度
（Qwen3-Reranker 由 yes/no token 的概率导出 -> P(yes) 天然压向两端）。

若成立，含义是：
  · 分数不能用于细粒度排序（大量并列 0.99x）
  · 分数不能作为置信度
  · 0.5 阈值实际是"是不是同一话题"的分界，不是"对不对"的分界

做法：同一条 query 取 recall_k=100 的候选池，让 rerank 全量打分，
打印分桶直方图；再对一条已知真值的样本，列出 top15（病害, 分数）看区分度。

用法：python crawler/_probe_score_dist.py
"""
import asyncio
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from core.config import KB_COLLECTION  # noqa: E402
from core.retriever import retrieve  # noqa: E402

K = 100


def show(tag: str, scores: list[float]) -> None:
    if not scores:
        print(f"{tag}: (无结果)")
        return
    bins = [0] * 10
    for s in scores:
        bins[min(int(s * 10), 9)] += 1
    mid = sum(1 for s in scores if 0.3 <= s < 0.7)
    print(f"\n{tag}")
    print(f"  n={len(scores)}  min={min(scores):.3f}  median={st.median(scores):.3f} "
          f" max={max(scores):.3f}")
    for i, c in enumerate(bins):
        bar = "#" * c
        print(f"  {i/10:.1f}-{(i+1)/10:.1f} {c:4d} {bar}")
    print(f"  中段[0.3,0.7) 占比: {mid}/{len(scores)} = {mid/len(scores):.0%}"
          f"   <- 越接近 0 越说明分布是双峰/被压平的")


async def run(query: str, crop: str | None, section: str | None, tag: str) -> list[dict]:
    rows = await retrieve(query, KB_COLLECTION, k=K, recall_k=K, hybrid=True,
                          crop=crop, section=section)
    show(tag, [float(r.get("score", 0.0)) for r in rows])
    return rows


async def main() -> None:
    # ---- 1) 三种 query 的形状对比 ----
    ra = await run("叶片褐色斑点", "番茄", "危害症状", "A) 泛症状 query（番茄/危害症状）")
    print("  top8 分数:", [round(float(r.get("score", 0)), 4) for r in ra[:8]])
    rb = await run("早疫病 同心轮纹 褐色病斑 黄晕", "番茄", "危害症状",
                   "B) 术语化 query（番茄/危害症状）")
    print("  top8 分数:", [round(float(r.get("score", 0)), 4) for r in rb[:8]])
    # ↑ 若这些值挤在千分位以内，说明分数已饱和：既无法排序，Top1/Top2 分差也无意义
    await run("如何做红烧肉", None, None, "C) 完全离题 query（无过滤）")

    # ---- 2) 一条已知真值的样本：看正确病害能不能被区分出来 ----
    path = ROOT / "crawler" / "data" / "exp_query_rewrite.jsonl"
    samples = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    s = samples[0]
    print("\n" + "=" * 76)
    print(f"D) 已知真值样本  作物={s['crop']}  真值={s['truth']}")
    print(f"   desc: {s['desc'][:90]}")
    rows = await run(s["desc"], s["crop"], "危害症状", "   （同一 query 的分数分布）")
    print("\n   top15 明细（★ = 真值）:")
    for i, r in enumerate(rows[:15], 1):
        mark = "★" if r.get("disease") == s["truth"] else " "
        print(f"   {i:2d}. {mark} {str(r.get('disease'))[:22]:24s} "
              f"{float(r.get('score', 0)):.4f}  [{r.get('section')}]")


if __name__ == "__main__":
    asyncio.run(main())
