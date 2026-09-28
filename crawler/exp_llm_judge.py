# -*- coding: utf-8 -*-
"""验证 ③ 的「对照判断层」实际上限——即把 top-10 候选交给 LLM，它能判对多少。

【要回答的问题】
core/diagnosis.py 打算把 top-10 候选连同「危害症状」原文交给 LLM，让它在候选内
判断哪几个吻合、并给出对照理由。这一层能到多准？值不值这次 LLM 调用？

【为什么必须把"检索"和"判断"分开测】
上一轮实验已知：top-10 里含真值的比例是 **82.5%**，那是**检索层的上限**。
本轮测的是**条件准确率** P(LLM 选中真值 | 真值在候选里)。
两者相乘才是 ③ 的实际上限。混在一起测，就分不清错在检索还是错在判断。

【四个必测项】
1. 条件准确率 —— 判断层的能力（本项目前只有一个数字：未知）
2. 越界率     —— LLM 是否真的守住"只能从候选里选"（期望 0）
                这是防幻觉的**结构性**保证：它最坏只能选错候选，不能凭空造病害
3. 弃权行为   —— 真值不在候选里时，它是诚实返回空，还是硬选一个
4. JSON 解析失败率 —— 用 LLM 输出换结构化的真实代价

用法：
    python crawler/exp_llm_judge.py               # 每作物 4 条 = 40 样本
    set PER_CROP=2 && python crawler/exp_llm_judge.py
    set JSON_MODE=0 && python crawler/exp_llm_judge.py   # 关掉 response_format 看看解析率
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
from pymilvus import MilvusClient  # noqa: E402

from core.config import KB_COLLECTION, MILVUS_URI  # noqa: E402
from core.retriever import retrieve  # noqa: E402

DATA = ROOT / "crawler" / "data"
DESC_FILE = os.getenv("DESC_FILE", "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl")
PER_CROP = int(os.getenv("PER_CROP", "4"))
CONC = int(os.getenv("CONC", "4"))
TOP_N = 10                      # 上一轮实验定下的候选窗口
HEALTHY = "__HEALTHY__"
JSON_MODE = os.getenv("JSON_MODE", "1") not in ("", "0")

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("MODEL_NAME")

# 用 __PLACEHOLDER__ 而不是 {} 做占位：prompt 里含 JSON 花括号，
# 用 f-string 或 .format() 都会被花括号吃掉，改成 replace 最省心。
PROMPT = """你是农业植保诊断专家。下面是农户描述的症状，以及从资料库检索到的候选病害及其「危害症状」原文。

请判断：哪些候选的症状描述与农户描述**真正吻合**。

要求：
1. 只能从给出的候选里选择，**严禁新增任何候选病害**（这是硬约束）
2. 每个选中的候选，必须列出农户描述与资料原文**对上的具体症状特征**（2~4 个）
3. 若没有任何候选真正吻合，返回空列表 —— 宁可说不知道，也不要牵强选一个
4. 按吻合程度从高到低排序，最多返回 3 个

只输出 JSON，不要任何解释、不要 markdown 代码块。格式：
{"matches": [{"disease": "候选里出现的病害名", "features": ["特征1", "特征2"], "level": "high"}]}
其中 level 取 high / medium / low。

作物：__CROP__

农户描述：
__DESC__

候选（共 __N__ 个）：
__CANDS__"""


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


def parse_json(text: str) -> dict | None:
    """容忍 markdown 代码块与前后杂语的 JSON 提取。"""
    if not text:
        return None
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        obj = json.loads(text[i:j + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def ask(prompt: str, sem: asyncio.Semaphore) -> tuple[str, bool]:
    """返回 (原始文本, 是否用上了 json 模式)。

    先试 response_format={"type":"json_object"}；该供应商/模型不支持时自动退回
    普通模式 —— 并把这个事实记录下来（它本身就是要测的"解析失败率"的一部分）。
    """
    modes = [True, False] if JSON_MODE else [False]
    last = None
    async with sem:
        for use_json in modes:
            try:
                kw = {"response_format": {"type": "json_object"}} if use_json else {}
                resp = await client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=900,
                    messages=[{"role": "user", "content": prompt}], **kw,
                )
                return (resp.choices[0].message.content or ""), use_json
            except Exception as e:
                last = e
    print(f"    [LLM 失败] {type(last).__name__}: {last}", flush=True)
    return "", False


def fill(tpl: str, crop: str, desc: str, cands_block: str, n: int) -> str:
    return (tpl.replace("__CROP__", crop)
               .replace("__DESC__", desc)
               .replace("__N__", str(n))
               .replace("__CANDS__", cands_block))


def feature_in_source(feature: str, source: str) -> bool:
    """粗核对：引用的特征能否在原文里找到。

    局限：LLM 常改写措辞（原文"同心轮纹"、引用"病斑呈同心轮纹状"），
    子串匹配会判为不中。所以这个指标只能当**下界**看，
    价值在于抓"完全编造"——引用的特征在原文里连续 4 个字都找不到。
    """
    f = "".join((feature or "").split())
    s = "".join((source or "").split())
    if not f or not s:
        return False
    if f in s:
        return True
    if len(f) >= 4:
        return any(f[i:i + 4] in s for i in range(len(f) - 3))
    return False


async def run_one(s: dict, sem: asyncio.Semaphore) -> dict:
    # ---- 检索（只取危害症状章节）----
    async with sem:
        rows = await retrieve(s["desc"], KB_COLLECTION, k=TOP_N, recall_k=100,
                              hybrid=True, crop=s["crop"], section="危害症状")
    if not rows:
        return {"crop": s["crop"], "truth": s["truth"], "desc": s["desc"],
                "n_cands": 0, "truth_in_cands": False, "matches": [],
                "parse_ok": None, "used_json_mode": None, "raw": "",
                "cited": [], "cited_ok": [], "pool_empty": True}

    # 同一个病害可能有多个 chunk，按病害聚合（症状原文拼接）
    by_disease: dict[str, list[str]] = {}
    for r in rows:
        d = r.get("disease") or "?"
        by_disease.setdefault(d, []).append((r.get("content") or "").strip())
    cands = list(by_disease)                       # 保持检索顺序
    src_text = {d: " ".join(v) for d, v in by_disease.items()}
    block = "\n".join(
        f"[{i}] 病害名：{d}\n危害症状：{src_text[d][:400]}"
        for i, d in enumerate(cands, 1)
    )
    prompt = fill(PROMPT, s["crop"], s["desc"], block, len(cands))

    raw, used_json = await ask(prompt, sem)
    obj = parse_json(raw)
    matches = (obj or {}).get("matches") if isinstance(obj, dict) else None
    if not isinstance(matches, list):
        matches = []

    # ---- 越界检查：选中的病害是否都在候选集合内 ----
    norm = {d.strip() for d in cands}
    out_of_scope = [m.get("disease") for m in matches
                    if isinstance(m, dict) and str(m.get("disease", "")).strip() not in norm]

    # ---- 真的被选中了吗 ----
    picked = [str(m.get("disease", "")).strip() for m in matches if isinstance(m, dict)]
    first = picked[0] if picked else None
    high = [str(m.get("disease", "")).strip() for m in matches
            if isinstance(m, dict) and m.get("level") == "high"]

    # ---- 特征引用核对 ----
    cited, cited_ok = [], []
    for m in matches:
        if not isinstance(m, dict):
            continue
        d = str(m.get("disease", "")).strip()
        for feat in (m.get("features") or []):
            cited.append(str(feat))
            cited_ok.append(feature_in_source(str(feat), src_text.get(d, "")))

    return {
        "crop": s["crop"], "truth": s["truth"], "desc": s["desc"],
        "n_cands": len(cands), "cands": cands,
        "truth_in_cands": s["truth"] in norm,
        "pool_empty": False,
        "parse_ok": obj is not None, "used_json_mode": used_json,
        "picked": picked, "first_pick": first, "high_picks": high,
        "hit_in_pick": s["truth"] in picked,
        "hit_first": first == s["truth"],
        "hit_high": s["truth"] in high,
        "abstained": len(picked) == 0,
        "out_of_scope": out_of_scope,
        "cited": cited, "cited_ok": cited_ok,
        "raw": raw[:600],
    }


async def main() -> None:
    preflight()
    valid = load_valid_pairs()
    samples = load_samples(valid)
    print(f"样本: {len(samples)}（每作物 {PER_CROP}）  候选窗口 top-{TOP_N}  "
          f"json_mode={JSON_MODE}\n", flush=True)

    sem = asyncio.Semaphore(CONC)
    t0 = time.time()
    rows = []
    for i, s in enumerate(samples, 1):
        rec = await run_one(s, sem)
        rows.append(rec)
        mark = "!" if rec["pool_empty"] else (
            "O" if rec["hit_in_pick"] else ("·" if not rec["truth_in_cands"] else "x"))
        print(f"  [{i}/{len(samples)}] {s['crop']} {s['truth'][:12]:14s} "
              f"cands={rec['n_cands']:3d} truthIn={'Y' if rec['truth_in_cands'] else 'n'} "
              f"pick={len(rec['picked'])} {mark}", flush=True)
    print(f"\n耗时 {time.time() - t0:.0f}s\n", flush=True)

    got = [r for r in rows if not r["pool_empty"]]
    tic = [r for r in got if r["truth_in_cands"]]      # 检索层已把真值交到手上
    tno = [r for r in got if not r["truth_in_cands"]]

    print("=" * 80)
    print("① 判断层条件准确率（分母 = 真值确实在候选里的样本）")
    print(f"   样本数: {len(tic)}/{len(got)}  = {len(tic)/len(got):.0%}"
          f"  <- 检索层上限，也决定 ③ 的硬上限")
    if tic:
        print(f"   被选中（任意位次）: {sum(r['hit_in_pick'] for r in tic)}"
              f"  = {sum(r['hit_in_pick'] for r in tic)/len(tic):.0%}")
        print(f"   被排在第一位      : {sum(r['hit_first'] for r in tic)}"
              f"  = {sum(r['hit_first'] for r in tic)/len(tic):.0%}")
        print(f"   被标为 level=high : {sum(r['hit_high'] for r in tic)}"
              f"  = {sum(r['hit_high'] for r in tic)/len(tic):.0%}")
        print(f"\n   ③ 的整体上限 = 检索层({len(tic)/len(got):.0%}) × 判断层"
              f"({sum(r['hit_in_pick'] for r in tic)/len(tic):.0%})"
              f" = {len(tic)/len(got) * sum(r['hit_in_pick'] for r in tic)/len(tic):.0%}"
              f"（全量 {len(got)} 样本口径）")

    print("\n② 越界检查：LLM 是否新增了候选以外的病害")
    bad = [r for r in got if r["out_of_scope"]]
    print(f"   越界样本: {len(bad)}/{len(got)}"
          f"  {'← 约束守住了' if not bad else '← 约束被突破，需在解析层强制过滤'}")
    for r in bad[:3]:
        print(f"     越界项: {r['out_of_scope']}  (真值 {r['truth']})")

    print("\n③ 弃权行为")
    if tno:
        ab = [r for r in tno if r["abstained"]]
        print(f"   真值不在候选里的 {len(tno)} 例中，弃权 {len(ab)}"
              f"  = {len(ab)/len(tno):.0%}")
        print(f"   （其余 {len(tno)-len(ab)} 例选了别的病害 —— 这不算错，"
              f"是检索层没给到答案；但要看它有没有硬选）")
    if tic:
        wrong_ab = [r for r in tic if r["abstained"]]
        print(f"   真值在候选里却弃权: {len(wrong_ab)}/{len(tic)}"
              f"  = {len(wrong_ab)/len(tic):.0%}  <- 误弃权，是判断层的漏检")

    print("\n④ JSON 解析与特征引用")
    fail = [r for r in got if not r["parse_ok"]]
    used = sum(1 for r in got if r["used_json_mode"])
    print(f"   解析失败: {len(fail)}/{len(got)}  = {len(fail)/len(got):.0%}"
          f"   （json 模式生效 {used}/{len(got)}）")
    all_cited = [x for r in got for x in r["cited"]]
    all_ok = [x for r in got for x in r["cited_ok"]]
    if all_cited:
        print(f"   特征引用共 {len(all_cited)} 条，可在原文中核对到 {sum(all_ok)}"
              f"  = {sum(all_ok)/len(all_cited):.0%}  ← 只能当下界看（改写措辞会被判不中）")
        miss = [x for r in got for x, ok in zip(r["cited"], r["cited_ok"]) if not ok]
        if miss:
            print(f"   核对不到的样例（人工看是改写还是编造）: {miss[:6]}")
    print("=" * 80)

    out = DATA / "exp_llm_judge.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细已写入 {out}")


if __name__ == "__main__":
    asyncio.run(main())
