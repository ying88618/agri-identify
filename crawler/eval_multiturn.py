# -*- coding: utf-8 -*-
"""
eval_multiturn.py — 多轮对话端到端评测(40 样本 = 10 作物 × 前 4 条描述缓存)

指标:
  ① 3 轮内确诊成功率
  ② 5 轮内确诊成功率
  ③ 5 轮内给出防治方案成功率
  ④ 单轮基线(第 1 轮确诊), 用于对比多轮增益

协议(与 v3 一致):
  用户初始给图片描述+提问; 之后 agent 追问 -> 用户按 KB 症状回答;
  若 agent 不提问, 用户主动补充一条还没说过的症状。
  采样是【确定性的】(每作物取描述缓存前 4 条), 故结果可复现、可跨配置逐条配对比较。

两个易踩的坑:
  · 本脚本绕过 chat.py 直接设检索上下文, 因此阈值必须引用
    core.config.DEFAULT_SCORE_THRESHOLD; 写死数字会导致换 rerank 模型后结果与线上不一致。
  · 采样只按 truth != HEALTHY 过滤, 【不排除】MAP 中 NO_MATCH 的类(如 Grape___Esca)。
    这些类在知识库里没有正确可匹配的条目, 会系统性拖低分数, 解读时须单独剔除
    (原因见 crawler/agri_eval_map.py 中 Esca 的注释)。

历史基线(crawler/data/):
  · multiturn_40_baseline_bge.jsonl   bge-reranker-v2-m3 + 阈值 0.25(旧配置)
  · multiturn_40_qwen3rerank.jsonl    Qwen3-Reranker-8B + 阈值 0.5(新配置)
  · multiturn_40.jsonl                最近一次运行(脚本默认输出, 会被覆盖)

用法:
    python crawler/eval_multiturn.py
"""
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

from langchain_core.messages import AIMessageChunk
from pymilvus import MilvusClient
from openai import AsyncOpenAI

from core.config import DEFAULT_SCORE_THRESHOLD, KB_COLLECTION
from core.agent import get_agent, set_request_context
from core.memory import load_history, append_turn
from core.llm import SYSTEM_PROMPT

DATA = os.path.join(ROOT, "crawler", "data")
COLLECTION = KB_COLLECTION
MAX_TURNS = 5
PER_CROP = 4
CONC = 3
USER_ID = 9999

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                     base_url=os.getenv("OPENAI_BASE_URL"))
MODEL = os.getenv("MODEL_NAME")
milvus = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
milvus.load_collection(COLLECTION)

cache = [json.loads(l) for l in open(
    os.path.join(DATA, "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl"),
    encoding="utf-8") if l.strip()]
byc = defaultdict(list)
for r in cache:
    if r["truth"] != "__HEALTHY__":
        byc[r["crop"]].append(r)
samples = []
for c, lst in byc.items():
    samples += lst[:PER_CROP]
MAX_TURNS = MAX_TURNS
print(f"样本 {len(samples)} (每作物 {PER_CROP})  并发 {CONC}", flush=True)


def kb_symptom(crop, disease):
    try:
        rows = milvus.query(COLLECTION,
                            filter=f'crop == "{crop}" and disease == "{disease}"'
                                   f' and section == "危害症状"',
                            output_fields=["text"], limit=1)
        return rows[0]["text"] if rows else ""
    except Exception:
        return ""


SIM = """你是一位种植{crop}的农民。你地里的作物出了问题，拍了一张照片发给助手。
你不知道这是什么病，正在向助手求助。

【你只知道下面这些，除此之外一无所知】
{symptom}

回答规则（必须严格遵守）：
1. 只能复述上面**明确写过**的内容。上面没写到的，一律回答"这个我没太注意"。
2. 【绝对禁止】编造上面没有的任何观察——包括气味、手感、菌脓、虫体、天气等。
3. 【绝对禁止】说出任何病害名称，你并不知道那是什么病。
4. 助手问了什么，就针对那个问题回答，1~2 句。
5. 如果助手**没有向你提问**，你就主动补充一条自己观察到的、之前还没说过的症状
   （从上面内容里挑一条），推动诊断继续。
6. 如果上面所有信息都已经说过了，就回复："其他的我确实没太注意，您再看看照片吧。"
"""

J_DIAG = """下面是一段「助手与农民」的多轮对话。真实病害是：{truth}

请判断：助手在第几轮**首次**给出了正确的诊断？
- 病名允许同义/别称（如"番茄细菌性斑点病"与"番茄细菌性斑疹病"视为同一病害）
- 必须收敛到 {truth} 这个具体病害；只罗列一堆候选、没有倾向的，不算
- 从未正确诊断则返回 0

只输出一个数字，不要解释。"""

J_SOLU = """下面是一段「助手与农民」的多轮对话。真实病害是：{truth}

请只看**最后一轮**助手的回复，判断它是否给出了针对该病害的**可执行防治方案**。
合格的方案应包含具体措施，例如：农业防治（清园/轮作/控湿等）、化学防治（具体药剂名或药剂类型、稀释倍数、喷药时机/频次）等。
只要泛泛而谈"注意通风、加强管理"、或没有给出任何可执行的措施，算不合格；
如果最后一轮给的方案是针对**错误病害**的，也算不合格。

只输出 1（合格）或 0（不合格），不要解释。"""

JUDGE_TXT = lambda turns: "\n".join(
    f"第{t['turn']}轮 助手：{(t.get('agent') or '')[:600]}" for t in turns)


async def sim_reply(crop, symptom, agent_said):
    r = await client.chat.completions.create(
        model=MODEL, temperature=0.3, max_tokens=120,
        messages=[
            {"role": "system", "content": SIM.format(crop=crop, symptom=symptom)},
            {"role": "user", "content": f"助手刚才说：\n{agent_said[-400:]}\n\n"
                                        f"请以农民身份回复助手（1~2 句）。"},
        ])
    return (r.choices[0].message.content or "").strip()


async def _ask(prompt, max_tokens=8):
    r = await client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()


async def agent_reply(messages):
    # 必须与线上一致: 本脚本绕过 chat.py 直接设上下文, 阈值写死会导致
    # 更换 rerank 模型后的多轮结果与线上不一致(见 graph.DEFAULT_SCORE_THRESHOLD 的标定说明)
    set_request_context(COLLECTION, score_threshold=DEFAULT_SCORE_THRESHOLD)
    agent = get_agent()
    cfg = {"configurable": {"score_threshold": DEFAULT_SCORE_THRESHOLD}}
    full = []
    async for chunk, _meta in agent.astream({"messages": messages},
                                            config=cfg, stream_mode="messages"):
        if not isinstance(chunk, AIMessageChunk):
            continue
        t = getattr(chunk, "content", "")
        if t and isinstance(t, str):
            full.append(t)
    return "".join(full).strip()


async def run_one(i, s):
    kb = kb_symptom(s["crop"], s["truth"])
    sid = f"mt_{int(time.time())}_{i}"
    user_text = (f"用户上传了一张图片，图片内容描述如下：\n{s['desc']}\n\n"
                 f"用户问题：我家的{s['crop']}叶子出问题了，这是什么病？")
    append_turn(USER_ID, sid, "user", user_text)

    turns, n_asked = [], 0
    for t in range(1, MAX_TURNS + 1):
        history = load_history(USER_ID, sid, n=10)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        reply = await agent_reply(msgs)
        append_turn(USER_ID, sid, "assistant", reply)
        asked = ("？" in reply) or ("?" in reply)
        n_asked += asked
        turns.append({"turn": t, "agent": reply, "asked": asked})
        if t == MAX_TURNS:
            break
        sr = await sim_reply(s["crop"], kb, reply)
        append_turn(USER_ID, sid, "user", sr)
        turns[-1]["sim"] = sr

    txt = JUDGE_TXT(turns)
    d = await _ask(J_DIAG.format(truth=s["truth"]) + "\n\n" + txt)
    digits = "".join(ch for ch in d if ch.isdigit())
    first_hit = int(digits[:1]) if digits else 0
    last = (turns[-1].get("agent") or "")[:1500]
    sv = await _ask(J_SOLU.format(truth=s["truth"]) + "\n\n" + f"最后一轮 助手：{last}")
    return {"crop": s["crop"], "truth": s["truth"], "kb_len": len(kb),
            "turns": turns, "first_hit": first_hit, "n_asked": n_asked,
            "solution_ok": sv.startswith("1"),
            "leak": sum(1 for t in turns if "来源:" in t["agent"])}


async def main():
    sem = asyncio.Semaphore(CONC)
    done = [0]

    async def w(i, s):
        async with sem:
            try:
                r = await run_one(i, s)
            except Exception as e:
                r = {"crop": s["crop"], "truth": s["truth"], "err": repr(e)}
            done[0] += 1
            print(f"  [{done[0]}/{len(samples)}] {r.get('crop')}/{r.get('truth')} "
                  f"first_hit={r.get('first_hit')} sol={r.get('solution_ok')} "
                  f"{r.get('err','')}", flush=True)
            return r

    res = await asyncio.gather(*[w(i, s) for i, s in enumerate(samples)])
    ok = [r for r in res if not r.get("err")]
    n = len(ok)
    print(f"\n{'='*66}\n成功 {n}/{len(res)}\n{'-'*66}")
    print(f"① 3 轮内确诊      : {sum(1 for r in ok if 0 < r['first_hit'] <= 3)}/{n}"
          f"  = {sum(1 for r in ok if 0 < r['first_hit'] <= 3)/n:.1%}")
    print(f"② 5 轮内确诊      : {sum(1 for r in ok if r['first_hit'])}/{n}"
          f"  = {sum(1 for r in ok if r['first_hit'])/n:.1%}")
    print(f"③ 5 轮内给出防治方案: {sum(1 for r in ok if r['solution_ok'])}/{n}"
          f"  = {sum(1 for r in ok if r['solution_ok'])/n:.1%}")
    print(f"④ 单轮基线(第1轮确诊): {sum(1 for r in ok if r['first_hit']==1)}/{n}"
          f"  = {sum(1 for r in ok if r['first_hit']==1)/n:.1%}")
    print(f"   追问率 {sum(1 for r in ok if r['n_asked']>0)/n:.1%}"
          f" | 工具原文泄漏轮次 {sum(r['leak'] for r in ok)}")
    print(f"{'-'*66}\n按作物:")
    bc = defaultdict(list)
    for r in ok:
        bc[r["crop"]].append(r)
    for c, lst in sorted(bc.items()):
        print(f"  {c:<6} n={len(lst):<3} 3轮内 "
              f"{sum(1 for r in lst if 0<r['first_hit']<=3)}/{len(lst)}"
              f"  5轮内 {sum(1 for r in lst if r['first_hit'])}/{len(lst)}"
              f"  方案 {sum(1 for r in lst if r['solution_ok'])}/{len(lst)}")
    with open(os.path.join(DATA, "multiturn_40.jsonl"), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细 → {os.path.join(DATA, 'multiturn_40.jsonl')}")


asyncio.run(main())
