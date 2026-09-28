# -*- coding: utf-8 -*-
"""一次性实验：量化「LLM 查询改写」对检索的收益。

【要回答的问题】
POST /diagnose 打算加一步"LLM 把症状改写成检索术语串"，这一步值不值一次 LLM 调用？

【为什么不能只测一个 arm】
数据集里的 desc 是 VL 模型产出的**结构化描述**，本身已经相当术语化
（"叶面可见细小黑点，分布于叶脉间"）。而真实农户说的是"有点黄、背面好像有霉层"。
只测前者会得到"改写收益很小"的结论 —— 那是**错的**，因为输入本来就不需要改写。
所以这里同时构造口语化输入，把收益按输入风格分开看。

【五个 arm（配对比较，同一批样本）】
  V1 desc                  原始 VL 描述（术语化较强）
  V2 rewrite(desc)         术语化改写
  V3 colloquial(desc)      口语化，模拟真实农户
  V4 rewrite(colloquial)   口语 -> 术语（真实场景）
  V5 guess(desc)           允许模型给病害猜测（= 当前 /chat/stream 的做法）

V5 单独列出来，是为了量化**确认偏差**：如果模型猜错了病害，检索会不会照样给出
高分（因为它在查自己猜的那个病）？若是，说明让 LLM 把病害名写进检索词很危险。

【判定】返回的 top-k 里是否含 truth 病害。确定性判定，不用 LLM judge。
【样本筛选】只保留知识库里确实存在 (crop, truth) 条目的样本 —— 否则永远命中不了，
两个 arm 一起被拉平、只增噪声（对应 agri_eval_map.py 里 NO_MATCH 那条教训）。

用法:
    python crawler/exp_query_rewrite.py                 # 每作物 2 条
    set PER_CROP=4 && python crawler/exp_query_rewrite.py
"""
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from openai import AsyncOpenAI  # noqa: E402

from core.config import KB_COLLECTION  # noqa: E402
from core.retriever import retrieve  # noqa: E402

DATA = ROOT / "crawler" / "data"
# 与 crawler/eval_multiturn.py 用同一个描述文件，便于和已有基线对照
DESC_FILE = os.getenv("DESC_FILE", "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl")
PER_CROP = int(os.getenv("PER_CROP", "2"))
CONC = int(os.getenv("CONC", "4"))
TOP_K = 5
HEALTHY = "__HEALTHY__"

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("MODEL_NAME")

REWRITE_PROMPT = """你是农业植保检索专家。把农户对作物症状的描述改写成**检索资料库用的症状术语串**。

规则：
1. 只输出一行，用空格分隔术语；不要标点、不要换行、不要任何解释
2. 保留有鉴别力的特征：部位/颜色/形状/边缘/分布/有无霉层或孢子堆/发病先后
3. 可以补充规范术语（如"病斑受叶脉限制""同心轮纹""水渍状""边缘黄晕"）
4. 绝对不要写出任何病害名称，也不要下结论 —— 检索由症状驱动，结论由资料决定
5. 20~40 字

作物：{crop}
症状描述：{text}"""

COLLOQUIAL_PROMPT = """把下面这段农技人员式的症状描述，改写成一位普通农户在微信里向专家求教时的说法。

规则：
1. 用口语、短句，可以含糊，可以有"好像""感觉""不太确定"这类表达
2. 不要说得面面俱到：农户通常只注意到最显眼的 2~3 个特征，其余略过
3. 不要使用任何专业术语（如"水渍状""同心圆纹""受叶脉限制"）
4. 只输出农户说的话，不要任何解释或引号
5. 40 字以内

症状描述：{text}"""

GUESS_PROMPT = """你是农业植保检索专家。根据农户描述，给出一个用于检索资料库的查询串。

规则：
1. 只输出一行，用空格分隔；不要标点、不要换行、不要解释
2. 先给出你推测的病害名称，再跟 2~4 个最关键的鉴别症状术语
3. 15~30 字

作物：{crop}
症状描述：{text}"""


async def ask(prompt: str, sem: asyncio.Semaphore) -> str:
    """单次非流式调用，返回去空白后的一行；失败返回空串。"""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL, temperature=0.2, max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip().replace("\n", " ")
        except Exception as e:
            print(f"    [LLM 失败] {type(e).__name__}: {e}", flush=True)
            return ""


def load_valid_pairs() -> set[tuple[str, str]]:
    """知识库真实存在的 (crop, disease) 集合，来源是灌库前的源文件。"""
    pairs = set()
    with open(DATA / "agri_pests.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            crop, name = r.get("crop"), r.get("name")
            if crop and name:
                pairs.add((crop, name))
    return pairs


def load_samples(valid: set[tuple[str, str]]) -> tuple[list[dict], int, int]:
    by_crop: dict[str, list[dict]] = defaultdict(list)
    healthy = out_of_kb = 0
    with open(DATA / DESC_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("truth") == HEALTHY:
                healthy += 1
                continue
            if not (r.get("desc") or "").strip():
                continue
            if (r.get("crop"), r.get("truth")) not in valid:
                out_of_kb += 1
                continue
            by_crop[r["crop"]].append(r)

    samples = []
    for crop in sorted(by_crop):                      # 确定性采样
        samples += by_crop[crop][:PER_CROP]
    return samples, healthy, out_of_kb


def preflight() -> None:
    """先确认 Milvus 可用。

    加这一步是因为上一次跑的时候 Milvus 没启动：100 次检索全在等连接超时，
    白等 388 秒，最后报表里所有 arm 都是"(无数据)"。
    让它在开头就失败 —— 依赖不可用应该立刻可见，不该伪装成"实验做完了但没结果"。
    """
    from pymilvus import MilvusClient

    from core.config import MILVUS_URI

    try:
        names = MilvusClient(uri=MILVUS_URI).list_collections()
    except Exception as e:
        raise SystemExit(
            f"Milvus 不可用（{MILVUS_URI}）。先启动依赖：docker compose up -d\n"
            f"原始错误: {type(e).__name__}: {e}"
        ) from e
    if KB_COLLECTION not in names:
        raise SystemExit(f"集合 {KB_COLLECTION} 不存在，现有集合: {names}")
    print(f"Milvus OK: {MILVUS_URI}  collections={names}", flush=True)


async def build_queries(s: dict, sem: asyncio.Semaphore) -> dict[str, str]:
    """为一个样本生成五个 arm 的查询串。"""
    desc, crop = s["desc"], s["crop"]
    rewritten, colloquial, guess = await asyncio.gather(
        ask(REWRITE_PROMPT.format(crop=crop, text=desc), sem),
        ask(COLLOQUIAL_PROMPT.format(text=desc), sem),
        ask(GUESS_PROMPT.format(crop=crop, text=desc), sem),
    )
    rewritten_colloquial = (
        await ask(REWRITE_PROMPT.format(crop=crop, text=colloquial), sem)
        if colloquial else ""
    )
    return {
        "V1 desc": desc,
        "V2 rewrite(desc)": rewritten,
        "V3 colloquial": colloquial,
        "V4 rewrite(colloquial)": rewritten_colloquial,
        "V5 guess(desc)": guess,
    }


async def run_query(q: str, crop: str, s: dict, sem: asyncio.Semaphore) -> dict | None:
    if not q:
        return None
    async with sem:
        try:
            results = await retrieve(
                q, KB_COLLECTION, k=TOP_K, recall_k=100, hybrid=True,
                crop=crop, section="危害症状",
            )
        except Exception as e:
            print(f"    [检索失败] {type(e).__name__}: {e}", flush=True)
            return None
    diseases = [r.get("disease") for r in results]
    rank = diseases.index(s["truth"]) + 1 if s["truth"] in diseases else None
    return {
        "query": q,
        "top1_score": float(results[0].get("score", 0.0)) if results else 0.0,
        "diseases": diseases,
        "truth_rank": rank,
        "returned": len(results),
    }


async def main() -> None:
    preflight()
    valid = load_valid_pairs()
    samples, healthy, out_of_kb = load_samples(valid)
    print(f"知识库 (crop,disease) 条目: {len(valid)}")
    print(f"样本: {len(samples)}（每作物 {PER_CROP}）"
          f"  跳过 健康 {healthy} / 库中无对应条目 {out_of_kb}", flush=True)

    sem = asyncio.Semaphore(CONC)
    arms = ["V1 desc", "V2 rewrite(desc)", "V3 colloquial",
            "V4 rewrite(colloquial)", "V5 guess(desc)"]
    rows: list[dict] = []

    t0 = time.time()
    for i, s in enumerate(samples, 1):
        queries = await build_queries(s, sem)
        jobs = [run_query(queries[a], s["crop"], s, sem) for a in arms]
        results = await asyncio.gather(*jobs)
        rec = {"crop": s["crop"], "truth": s["truth"], "desc": s["desc"]}
        for arm, r in zip(arms, results):
            rec[arm] = r
        rows.append(rec)

        marks = "".join(
            "·" if rec[a] is None else ("O" if rec[a]["truth_rank"] else "x") for a in arms
        )
        print(f"[{i}/{len(samples)}] {s['crop']} {s['truth'][:12]:14s} {marks}",
              flush=True)
    print(f"\n取数耗时 {time.time() - t0:.0f}s\n", flush=True)

    # ---- 报表 ----
    print("=" * 78)
    print(f"{'arm':24s} {'hit@1':>7s} {'hit@3':>7s} {'hit@5':>7s} "
          f"{'meanTop1':>9s} {'meanTruth':>10s}")
    print("-" * 78)
    for arm in arms:
        got = [r[arm] for r in rows if r[arm] is not None]
        if not got:
            print(f"{arm:24s}  (无数据)")
            continue
        n = len(got)
        h1 = sum(1 for g in got if g["truth_rank"] == 1) / n
        h3 = sum(1 for g in got if g["truth_rank"] and g["truth_rank"] <= 3) / n
        h5 = sum(1 for g in got if g["truth_rank"]) / n
        mt = sum(g["top1_score"] for g in got) / n
        print(f"{arm:24s} {h1:6.1%} {h3:6.1%} {h5:6.1%} {mt:9.3f} {'-':>10s}")
    print("=" * 78)

    # ---- 配对比较：相对 V1 的胜负（按 truth 排名提升）----
    def better(a, b):
        """a 是否优于 b（None 视为未命中，排最差）。"""
        ra = a["truth_rank"] if a and a["truth_rank"] else 999
        rb = b["truth_rank"] if b and b["truth_rank"] else 999
        return ra < rb

    print("\n相对 V1 desc 的配对结果（按 truth 排名）:")
    for arm in arms[1:]:
        w = t = l = 0
        for r in rows:
            base, cur = r["V1 desc"], r[arm]
            if base is None or cur is None:
                continue
            if better(cur, base):
                w += 1
            elif better(base, cur):
                l += 1
            else:
                t += 1
        print(f"  {arm:24s} 改好 {w:2d} / 持平 {t:2d} / 变差 {l:2d}")

    # ---- V5 的确认偏差检查 ----
    print("\nV5 确认偏差检查（V5 把推测的病害名写进了检索词）:")
    n_ok = n_bad = 0
    bad_high = 0
    for r in rows:
        g = r["V5 guess(desc)"]
        if not g:
            continue
        guessed = g["query"].split()[0] if g["query"].split() else ""
        hit = guessed in r["truth"] or r["truth"] in guessed
        if hit:
            n_ok += 1
        else:
            n_bad += 1
            if g["top1_score"] >= 0.5:      # 阈值：core.config.DEFAULT_SCORE_THRESHOLD
                bad_high += 1
    if n_ok + n_bad:
        print(f"  猜测命中 truth 的: {n_ok} / {n_ok + n_bad}")
        if n_bad:
            print(f"  猜错但 top1 分数仍 >=0.5 的: {bad_high} / {n_bad}"
                  f"  ← 这几个就是确认偏差，检索在替错误的猜测背书")

    # ---- 抽样打印，让数字可核对 ----
    print("\n样例（前 3 条）:")
    for r in rows[:3]:
        print(f"\n  [{r['crop']}] 真值 {r['truth']}")
        # 注意 None 保护：某个 arm 的检索失败时 r[arm] 是 None，
        # 直接下标会让整个报表在最后一步崩掉（上一次就是这样）
        v1 = r["V1 desc"]
        print(f"    {'V1 原始':10s}: {(v1 or {}).get('query', '(无)')[:70]}")
        for arm in arms[1:]:
            q = (r[arm] or {}).get("query", "(无)")
            print(f"    {arm:10s}: {q[:70]}")

    out = DATA / "exp_query_rewrite.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
