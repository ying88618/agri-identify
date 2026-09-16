# -*- coding: utf-8 -*-
"""
eval_agri_recall.py — 阶段2: 农业知识库检索召回评测

评测链路: VL症状描述(query) → kb_agri 检索 → 判断是否命中正确病害

关键口径:
  · 命中去重: 同一病害的多个 chunk 只算一次, 按"首次出现的排名"计
    (否则一个正确病害会被自己的重复 chunk 挤掉名额, 指标失真)
  · 健康样本: 也跑检索, 若返回了"具体病害"即记为误报

对比矩阵:
  · 检索方式: 纯向量 / 混合检索(向量+BM25+RRF)
  · 作物过滤: 不过滤 / 限定 crop

指标:
  · HitRate@K   正确病害出现在去重后 Top-K 的比例
  · MRR         首个正确病害排名的倒数均值
  · 健康误报率   健康样本被检索出具体病害的比例

用法:
    python crawler/eval_agri_recall.py
    python crawler/eval_agri_recall.py --k 1,3,5 --model Qwen/Qwen3-VL-8B-Instruct
    python crawler/eval_agri_recall.py --by-class   # 额外输出按类命中率
"""
import argparse
import asyncio
import json
import os
import sys
from math import comb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from agri_eval_map import MAP, HEALTHY, NO_MATCH

from core.config import DEFAULT_SCORE_THRESHOLD, KB_COLLECTION
from core.retriever import retrieve

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(DATA_DIR, "agri_recall_result.jsonl")
COLLECTION = KB_COLLECTION
RECALL_K = 20          # 召回候选数(与线上一致)
CONCURRENCY = 5
# 健康误报判定阈值: top1 分数 >= 此值才算"报了病"。
# 直接引用 core.config.DEFAULT_SCORE_THRESHOLD, 与线上永远保持一致
# (原先硬编码数字, 换 rerank 模型后极易与线上脱节, 报出的误报率就没有业务含义)。
# 该阈值随 rerank 模型而变(不同模型分数尺度不同), 换模型后必须重新标定 —— 实测:
#   bge-reranker-v2-m3 在 0.3 与 0.5 下都是 6.5%;
#   Qwen3-Reranker-8B  在 0.3 下高达 27.0%, 到 0.5 才回到 6.5%。
# 沿用旧阈值会得出"换模型导致误报爆炸"的错误结论。
FA_THRESHOLD = DEFAULT_SCORE_THRESHOLD
MAX_K = 5              # 取足够多, 供去重后按 K 切分


def cache_path(model: str, pv: int) -> str:
    slug = model.replace("/", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"vl_desc_{slug}_v{pv}.jsonl")


def load_samples(model: str, pv: int) -> list[dict]:
    """读取描述缓存, 并剔除未纳入评测的类

    描述缓存里可能残留 MAP 中已标为 NO_MATCH 的类(例如 kb_agri 无对应条目的
    Grape___Esca)。若不过滤, 这些"注定召不回"的样本会稀释整体指标。
    """
    path = cache_path(model, pv)
    if not os.path.exists(path):
        raise SystemExit(f"未找到描述缓存: {path}\n"
                         f"请先运行: python crawler/gen_vl_desc.py --model {model} --pv {pv}")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    kept = [r for r in rows if MAP.get(r["cls"]) is not NO_MATCH]
    dropped = len(rows) - len(kept)
    if dropped:
        bad = sorted({r["cls"] for r in rows if MAP.get(r["cls"]) is NO_MATCH})
        print(f"  [已剔除] {dropped} 条样本属于未纳入评测的类: {', '.join(bad)}")
    return kept


async def eval_one(sem, sample, hybrid, use_crop, section=None):
    """检索并返回去重后的病害序列(保持首次出现顺序) + top1 分数"""
    async with sem:
        crop = sample["crop"] if use_crop else None
        try:
            res = await retrieve(sample["desc"], COLLECTION, k=RECALL_K,
                                 hybrid=hybrid, crop=crop, section=section,
                                 recall_k=RECALL_K)
        except Exception as e:
            return {"sample": sample, "diseases": [], "top_score": 0.0,
                    "err": f"{type(e).__name__}: {e}"}

        # 按病害去重(保持顺序)
        seen, diseases = set(), []
        for r in res:
            d = r.get("disease")
            if d and d not in seen:
                seen.add(d)
                diseases.append(d)
        top_score = res[0]["score"] if res else 0.0
        return {"sample": sample, "diseases": diseases, "top_score": top_score}


async def run_pv(model: str, pv: int, ks: list[int], quiet: bool = False,
                 concurrency: int = CONCURRENCY, sections: list = None,
                 only_best: bool = False, crop_only: bool = False):
    """跑某个 prompt 版本的全部检索配置, 返回 (summary, all_rows)

    sections: 要对比的 section 列表(含 None 表示不过滤)
    only_best: 只跑 作物过滤+混合(最省时间)
    crop_only: 只跑 限定作物的配置(向量+crop / 混合+crop), 跳过不过滤作物的两个
    """
    samples = load_samples(model, pv)
    n_dis = sum(1 for s in samples if s["truth"] != HEALTHY)
    n_hea = len(samples) - n_dis
    if not quiet:
        print(f"\n模型: {model} | prompt v{pv}")
        print(f"样本: {len(samples)} 条 (病害 {n_dis}, 健康 {n_hea})")

    sections = sections if sections is not None else [None]
    if only_best:
        base = [(True, True)]
    elif crop_only:
        base = [(h, True) for h in (False, True)]
    else:
        base = [(h, c) for h in (False, True) for c in (False, True)]
    configs = [(h, c, s) for (h, c) in base for s in sections]

    all_rows, summary = [], []

    for hybrid, use_crop, section in configs:
        sem = asyncio.Semaphore(concurrency)
        results = await asyncio.gather(
            *[eval_one(sem, s, hybrid, use_crop, section) for s in samples])

        disease = [r for r in results if r["sample"]["truth"] != HEALTHY]
        healthy = [r for r in results if r["sample"]["truth"] == HEALTHY]
        n = len(disease)

        for k in ks:
            hits, mrr_sum = 0, 0.0
            for r in disease:
                top = r["diseases"][:k]
                if r["sample"]["truth"] in top:
                    hits += 1
                    mrr_sum += 1.0 / (top.index(r["sample"]["truth"]) + 1)
            # 健康误报: top1 分数超过阈值才算是"报了病"(否则检索总返回结果, 无意义)
            fa = sum(1 for r in healthy if r["top_score"] >= FA_THRESHOLD)
            summary.append({
                "pv": pv, "section": section,
                "k": k, "hybrid": hybrid, "use_crop": use_crop,
                "hitrate": hits / n if n else 0.0,
                "mrr": mrr_sum / n if n else 0.0,
                "n_healthy": len(healthy), "false_alarm": fa,
                "fa_rate": fa / len(healthy) if healthy else 0.0,
                "healthy_top1_mean": (sum(r["top_score"] for r in healthy) / len(healthy)) if healthy else 0.0,
                "disease_top1_mean": (sum(r["top_score"] for r in disease) / n) if n else 0.0,
            })

        for r in results:
            all_rows.append({
                "pv": pv, "section": section,
                "hybrid": hybrid, "use_crop": use_crop,
                "cls": r["sample"]["cls"], "truth": r["sample"]["truth"],
                "img": r["sample"]["img"],          # 保留图片名, 便于跨配置配对检验
                "crop": r["sample"]["crop"],
                "desc": r["sample"]["desc"][:200],
                "diseases": r["diseases"],
                "top_score": r["top_score"],
                "err": r.get("err"),
            })
    return summary, all_rows


def _mcnemar(a: dict, b: dict):
    """配对 McNemar 精确检验 -> (b01, b10, p)"""
    keys = sorted(set(a) & set(b))
    b01 = sum(1 for x in keys if (not a[x]) and b[x])
    b10 = sum(1 for x in keys if a[x] and (not b[x]))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = min(sum(comb(n, i) for i in range(min(b01, b10) + 1)) / (2 ** n) * 2, 1.0)
    return b01, b10, p


def print_significance(all_rows, sections, ks=(1, 3, 5, 20)):
    """基于明细做配对 McNemar 检验(用 img 作配对键)"""
    print("\n" + "=" * 100)
    print("配对 McNemar 显著性检验 (配对键 = 类+图片)")
    print("-" * 100)

    def hit_map(hybrid, section, k):
        out = {}
        for r in all_rows:
            if (r["hybrid"] == hybrid and r.get("section") == section
                    and r["use_crop"] and r["truth"] != HEALTHY):
                out[(r["cls"], r["img"])] = r["truth"] in r["diseases"][:k]
        return out

    for k in ks:
        # ① 向量 -> 混合 (section=None)
        a = hit_map(False, None, k)
        b = hit_map(True, None, k)
        if a and b:
            b01, b10, p = _mcnemar(a, b)
            print(f"  K={k:>2} 向量→混合(不限section): 救回{b01} 丢掉{b10} "
                  f"p={p:.4f} {'✅显著' if p < 0.05 else '❌不显著'}")
        # ② 不限 -> 限定危害症状 (混合)
        c = hit_map(True, "危害症状", k)
        if b and c:
            b01, b10, p = _mcnemar(b, c)
            print(f"  K={k:>2} 混合: 不限→限定危害症状: 救回{b01} 丢掉{b10} "
                  f"p={p:.4f} {'✅显著' if p < 0.05 else '❌不显著'}")


def print_by_class(all_rows, ks=(1, 3, 5, 20), section=None, hybrid=True):
    """按 PlantVillage 类拆开命中率 —— 区分「召不回」与「排不上」两层问题

    动机: 只看总 HitRate 会把两种完全不同的故障混在一起, 导致判断错层:
      · 正确病害根本进不了候选池      -> 召回/知识库匹配问题
      · 进了候选池但排不进 Top-K      -> 排序问题
    按类拆开后二者立刻分开。实例: 玉米灰斑病 @5=@20=100%(检索无忧, 0/4 的确诊
    失败应归因于 agent 推理); 而番茄细菌性斑疹病 @20=0% 才是真的召不回。

    只取一组最贴近线上的配置(默认 混合检索 + 作物过滤), 因为按类拆开后每组
    样本量很小(约 20), 再多配置叠加反而看不清。
    """
    RECALL_OK = 0.70   # @20 低于此值 -> 「召回弱」: 正确病害常进不了候选池
    RANK_OK = 0.40     # @5 低于此值且召回正常 -> 「排序弱」: 召回了但挤不进前5

    if section is None and any(r.get("section") == "危害症状" for r in all_rows):
        section = "危害症状"   # 有此口径时默认用它(实测最贴近线上)
    rows = [r for r in all_rows
            if r["hybrid"] == hybrid and r["use_crop"]
            and r.get("section") == section and r["truth"] != HEALTHY]
    if not rows:
        print("\n(按类命中率: 无匹配配置, 跳过)")
        return

    ks = sorted(ks)
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["cls"], []).append(r)

    stat = []
    for cls, lst in by.items():
        n = len(lst)
        rec = {"cls": cls, "n": n}
        for k in ks:
            rec[k] = sum(1 for r in lst if r["truth"] in r["diseases"][:k]) / n
        mrr = 0.0
        for r in lst:
            top = r["diseases"][:RECALL_K]
            if r["truth"] in top:
                mrr += 1.0 / (top.index(r["truth"]) + 1)
        rec["mrr"] = mrr / n
        stat.append(rec)

    k5 = 5 if 5 in ks else ks[-1]
    k20 = 20 if 20 in ks else ks[-1]

    def verdict(r):
        if r[k20] < RECALL_OK:
            return "❌召回弱"
        if r[k5] < RANK_OK:
            return "⚠️排序弱"
        return "✅"

    rank = {"❌召回弱": 0, "⚠️排序弱": 1, "✅": 2}
    # 先按诊断分组, 组内问题最严重的排最前 —— 便于一眼扫出需要处理的类
    stat.sort(key=lambda r: (rank[verdict(r)], r[k20], r[k5]))

    print("\n" + "=" * 100)
    print(f"按类命中率 (配置: {'混合' if hybrid else '向量'} + 作物过滤 + "
          f"section={section or '不限'}) —— 用于区分「召不回」与「排不上」")
    print("-" * 100)
    head = f"{'PlantVillage 类':<50}{'n':>3}"
    for k in ks:
        head += f"{'@' + str(k):>8}"
    head += f"{'MRR':>8}  {'诊断'}"
    print(head)
    print("-" * 100)
    for r in stat:
        line = f"{r['cls']:<50}{r['n']:>3}"
        for k in ks:
            line += f"{r[k]:>8.1%}"
        line += f"{r['mrr']:>8.3f}  {verdict(r)}"
        print(line)
    print("-" * 100)
    n_bad = sum(1 for r in stat if verdict(r) != "✅")
    print(f"诊断口径: ❌召回弱=@{k20} < {RECALL_OK:.0%}(正确病害进不了候选池) | "
          f"⚠️排序弱=@{k20}≥{RECALL_OK:.0%} 且 @{k5}<{RANK_OK:.0%}(召回了但排不进前{k5})"
          f"\n共 {len(stat)} 类, 其中 {n_bad} 类存在问题")


def print_summary(summary, title=""):
    if title:
        print(f"\n{title}")
    print("=" * 108)
    print(f"{'K':>3} {'检索方式':<6} {'作物过滤':<10} {'section':<12} {'HitRate':>9} {'MRR':>8} "
          f"{'健康误报率':>10} {'健康top1均值':>12} {'病害top1均值':>12}")
    print("-" * 108)
    for s in summary:
        mode = "混合" if s["hybrid"] else "向量"
        filt = "crop过滤" if s["use_crop"] else "不过滤"
        sec = s.get("section") or "不过滤"
        print(f"{s['k']:>3} {mode:<6} {filt:<10} {sec:<12} {s['hitrate']:>9.1%} "
              f"{s['mrr']:>8.3f} {s['fa_rate']:>10.1%} "
              f"{s['healthy_top1_mean']:>12.3f} {s['disease_top1_mean']:>12.3f}")
    print(f"\n注: 健康误报率 = 健康样本 top1 分数 >= {FA_THRESHOLD} 的比例(分数分布见后两列)")


def print_section_compare(summary, sections, title=""):
    """全矩阵对比: 向量/混合 × section"""
    if title:
        print(f"\n{title}")
    rows = [x for x in summary if x["use_crop"]]
    ks = sorted({x["k"] for x in rows})
    print("=" * 100)
    print("检索方式 × section 对比 (均限定作物)")
    print("-" * 100)
    combos = [(h, s) for h in (False, True) for s in sections]
    head = f"{'K':>3}"
    for h, s in combos:
        head += f" {(('混合' if h else '向量')+'/'+(s or '不限')):>18}"
    print(head)
    print("-" * 100)
    for k in ks:
        line = f"{k:>3}"
        for h, s in combos:
            r = next((x for x in rows if x["k"] == k and x["hybrid"] == h
                      and x.get("section") == s), None)
            line += f" {r['hitrate']:>18.1%}" if r else f" {'-':>18}"
        print(line)
    print("-" * 100)
    print("健康误报率:")
    for h, s in combos:
        r = next((x for x in rows if x["k"] == ks[0] and x["hybrid"] == h
                  and x.get("section") == s), None)
        if r:
            print(f"  {'混合' if h else '向量'}/{(s or '不限'):<10} {r['fa_rate']:.1%}")


def print_compare(sums: dict[int, list], pvs: list[int]):
    """多版本对比(最佳配置: crop过滤 + 混合)"""
    print("\n" + "=" * 90)
    print(f"prompt 版本对比: {' vs '.join('v'+str(p) for p in pvs)} (crop过滤 + 混合检索)")
    print("-" * 90)
    head = f"{'K':>3}"
    for p in pvs:
        head += f" {'v'+str(p)+' HitRate':>13} {'MRR':>8}"
    print(head)
    print("-" * 90)
    base = sums[pvs[0]]
    ks = sorted({x["k"] for x in base})
    for k in ks:
        line = f"{k:>3}"
        for p in pvs:
            row = next((x for x in sums[p] if x["k"] == k and x["use_crop"] and x["hybrid"]), None)
            line += f" {row['hitrate']:>13.1%} {row['mrr']:>8.3f}" if row else f" {'-':>13} {'-':>8}"
        print(line)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="1,3,5,20")
    ap.add_argument("--model", default=os.getenv("VL_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--pv", type=int, default=3, help="prompt 版本")
    ap.add_argument("--compare", default="", help="对比多个 prompt 版本, 如 1,3")
    ap.add_argument("--sections", default="", help="对比多个 section, 逗号分隔; 空=只跑不过滤")
    ap.add_argument("--only-best", action="store_true", help="只跑 作物过滤+混合 配置(省时间)")
    ap.add_argument("--crop-only", action="store_true", help="只跑限定作物的配置(向量+crop/混合+crop)")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    ap.add_argument("--by-class", action="store_true",
                    help="额外输出按类命中率(区分「召不回」与「排不上」两层问题)")
    ap.add_argument("--out", default=OUT, help="明细输出路径")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.k.split(",") if x)

    # section 列表: 空字符串 -> [None]; "危害症状" -> [None, "危害症状"]
    if args.sections:
        sections = [None] + [x.strip() for x in args.sections.split(",") if x.strip()]
    else:
        sections = [None]

    if args.compare:
        pvs = [int(x) for x in args.compare.split(",") if x]
        sums, all_rows = {}, []
        for p in pvs:
            s, r = await run_pv(args.model, p, ks, quiet=True,
                                concurrency=args.concurrency, sections=sections,
                                only_best=args.only_best, crop_only=args.crop_only)
            sums[p] = s
            all_rows.extend(r)
        print(f"模型: {args.model}")
        print_compare(sums, pvs)
    else:
        summary, all_rows = await run_pv(
            args.model, args.pv, ks, concurrency=args.concurrency,
            sections=sections, only_best=args.only_best, crop_only=args.crop_only)
        if len(sections) > 1:
            print_section_compare(summary, sections,
                                  f"模型: {args.model} | prompt v{args.pv}")
        else:
            print_summary(summary, f"模型: {args.model} | prompt v{args.pv}")
        print_significance(all_rows, sections, ks=tuple(ks))
        if args.by_class:
            print_by_class(all_rows, ks=tuple(ks))

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n明细已导出 → {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
