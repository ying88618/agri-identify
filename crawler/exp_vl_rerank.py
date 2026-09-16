# -*- coding: utf-8 -*-
"""
expb2_vl_rerank.py — 实验B2: 多模态 listwise rerank (让原图直接参与排序)

背景: 平台 /rerank 接口只收字符串, Qwen3-VL-Reranker-8B 拿不到图(实测 image_tokens=0),
      所以改用「多模态 LLM 做 listwise 重排」这条可行路径。

做法: 对每个样本
  1) 用基线配置(混合+作物过滤+危害症状, bge rerank)召回去重后的候选病害 Top-N
     —— 候选池与基线完全一致, 从而把"重排"的效果单独隔离出来
  2) 把 【原图】 + 【VL症状描述】 + 【候选病害及其危害症状】 一起给 VL 模型, 让它输出排序
  3) 按新顺序算 HitRate@K

对照: 同一样本集的基线顺序(来自 expA_8B.jsonl)

用法: python .codebuddy/expb2_vl_rerank.py [--per-class 5] [--concurrency 4] [--model ...]
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from openai import AsyncOpenAI
from core import retriever as R

IMG_ROOT = r"E:\RAG_agri\PlantVillage-Dataset-master\color"
DATA = os.path.join(ROOT, "crawler", "data")
CLIENT = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_BASE_URL"))

# 「排序弱」的 5 个类 + 1 个对照类(玉米灰斑病, 基线本就 100%)
TARGET = ["Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust",
          "Tomato___Late_blight", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
          "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot"]

PROMPT = """你是农业植保专家。下面给你一张真实的叶片照片，以及该叶片的一段症状描述。
请从候选病害中，按「与该照片症状的吻合程度」从高到低排序。

叶片症状描述：{desc}

候选病害及其危害症状：
{cands}

要求：
1. 必须结合照片上你实际看到的特征（病斑颜色/形状/大小/边缘晕圈/是否有霉层或孢子堆等）来判断；
2. 严格只输出候选编号，从最吻合到最不吻合，用英文逗号分隔，例如：3,1,7,2,...；
3. 必须包含全部 {n} 个编号，不要输出任何解释文字。"""


def dataurl(path, max_side=512):
    raw = open(path, "rb").read()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    try:
        from PIL import Image
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        raw, mime = buf.getvalue(), "image/jpeg"
    except Exception:
        pass
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


async def rerank_one(sem, s, model, n_cand):
    """返回 (新顺序的病害名列表, 候选池, 原始顺序, err)"""
    async with sem:
        try:
            res = await R.retrieve(s["desc"], "kb_agri", k=n_cand, hybrid=True,
                                   crop=s["crop"], section="危害症状",
                                   recall_k=n_cand)
        except Exception as e:
            return None, [], [], f"retrieve {type(e).__name__}: {e}"

        pool, seen, texts = [], set(), {}
        for r in res:
            d = r.get("disease")
            if d and d not in seen:
                seen.add(d)
                pool.append(d)
                texts[d] = (r.get("content") or "").replace("\n", " ")[:160]
        if not pool:
            return None, [], [], "空候选池"

        cands = "\n".join(f"{i+1}. {d}：{texts[d]}" for i, d in enumerate(pool))
        p = PROMPT.format(desc=s["desc"][:200], cands=cands, n=len(pool))
        img = os.path.join(IMG_ROOT, s["cls"], s["img"])
        try:
            resp = await CLIENT.chat.completions.create(
                model=model, temperature=0.0, max_tokens=200,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": p},
                    {"type": "image_url", "image_url": {"url": dataurl(img)}},
                ]}])
            txt = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            return None, pool, pool, f"vl {type(e).__name__}: {e}"

        nums = [int(x) for x in re.findall(r"\d+", txt)]
        order, used = [], set()
        for x in nums:
            if 1 <= x <= len(pool) and x not in used:
                used.add(x)
                order.append(pool[x - 1])
        # 模型漏掉的候选按原顺序补在后面(保证是完整排序, 便于算 @20)
        for d in pool:
            if d not in used:
                order.append(d)
        return order, pool, pool, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--n-cand", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(DATA, "expB2_vlrerank.jsonl"))
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            open(os.path.join(DATA, "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl"),
                 encoding="utf-8") if l.strip()]
    sel, cnt = [], {}
    for r in rows:
        c = r["cls"]
        if c in TARGET and cnt.get(c, 0) < args.per_class:
            cnt[c] = cnt.get(c, 0) + 1
            sel.append(r)
    print(f"样本 {len(sel)} 条 | 模型 {args.model} | 并发 {args.concurrency} | "
          f"候选 {args.n_cand}\n" + "-" * 70)

    sem = asyncio.Semaphore(args.concurrency)
    t = time.time()
    out = await asyncio.gather(*[rerank_one(sem, s, args.model, args.n_cand)
                                 for s in sel])
    dt = time.time() - t

    recs, nerr = [], 0
    agg = {}
    for s, (order, pool, orig, err) in zip(sel, out):
        if err:
            nerr += 1
        rec = {"cls": s["cls"], "truth": s["truth"], "img": s["img"],
               "crop": s["crop"], "desc": s["desc"][:200],
               "pool": pool, "orig": orig, "order": order or [], "err": err}
        recs.append(rec)
        if err or not order:
            continue
        a = agg.setdefault(s["cls"].split("___")[-1][:26], [0, 0, 0, 0, 0, 0])
        a[0] += 1
        a[1] += s["truth"] in order[:1]
        a[2] += s["truth"] in order[:5]
        a[3] += s["truth"] in order[:20]
        a[4] += s["truth"] in pool[:5]
        a[5] += s["truth"] in pool[:20]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n耗时 {dt:.1f}s ({len(sel)/dt:.2f}/s), 失败 {nerr}")
    print("\n" + "=" * 92)
    print(f"{'类':<28}{'n':>3}{'新@1':>8}{'新@5':>8}{'新@20':>8} | "
          f"{'基线@5':>8}{'基线@20':>9}")
    print("-" * 92)
    for c, (n, n1, n5, n20, o5, o20) in sorted(agg.items(), key=lambda x: x[1][2]):
        print(f"{c:<28}{n:>3}{n1/n:>8.1%}{n5/n:>8.1%}{n20/n:>8.1%} | "
              f"{o5/n:>8.1%}{o20/n:>9.1%}")
    tot = [sum(a[i] for a in agg.values()) for i in range(6)]
    N = tot[0] or 1
    print("-" * 92)
    print(f"{'合计':<28}{tot[0]:>3}{tot[1]/N:>8.1%}{tot[2]/N:>8.1%}{tot[3]/N:>8.1%} | "
          f"{tot[4]/N:>8.1%}{tot[5]/N:>9.1%}")
    print(f"\n明细 → {args.out}")


asyncio.run(main())
