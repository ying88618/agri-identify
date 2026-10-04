# -*- coding: utf-8 -*-
"""
eval_multiturn.py — 多轮对话端到端评测(40 样本 = 10 作物 × 前 4 条描述缓存)

指标:
  ① 3 轮内确诊成功率
  ② 5 轮内确诊成功率
  ③ 给出可执行防治方案成功率
  ④ 单轮基线(第 1 轮确诊), 用于对比多轮增益

【③ 的口径在 2026-10-04 变过】旧口径只判**最后一轮**的回复，而本脚本强制跑满
MAX_TURNS 轮 —— agent 在第 3~4 轮给了方案、第 5 轮改成回答农民追问时，那份方案
就被判丢了。实测旧基线 40 条里 25 次方案失败，其中 12 次属于这种盲区
（见 crawler/_check_solution_sections.py --plans），即 37.5% 只是下界而非真实值。
现在 solution_ok = "任意一轮给过"，并保留 solution_last_only 记录旧口径 ——
与 history 里的 multiturn_40_*.jsonl 对比 ③ 时只能用后者。

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
import hashlib
import json
import logging
import os
import subprocess
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

from core.config import (
    BM25_FILTER_POOL,
    DEFAULT_SCORE_THRESHOLD,
    DESCRIBE_PROMPT_VERSION,
    EMBEDDING_MODEL,
    KB_COLLECTION,
    RECALL_K,
    RERANK_MODEL,
    TOP_K,
)
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

# 日志文件。评测要跑 30 分钟，最容易被忘掉的就是 `> log 2>&1`；而事后想确认
# "这次是不是撞了 rerank 限流降级"，唯一证据在 core/retriever._rerank 的 warning 里
# （它自己注明降级会让指标失真 15%~63%）。所以让日志默认就有，不靠命令行记得加。
logger = logging.getLogger("eval_multiturn")

LOG_PATH = os.path.join(DATA, f"eval_run_{time.strftime('%Y%m%d_%H%M%S')}.log")


class _Tee:
    """把 print 同时写到终端和日志文件。

    logging 那一路不能靠替换 sys.stderr 来捕获：core/agent.py 在 import 时就调了
    logging.basicConfig，它的 StreamHandler 绑死的是**那一刻的** sys.stderr，
    之后再替换 sys.stderr 也拦不到。所以另外挂一个 StreamHandler（见 _install_log）。
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for st in self._streams:
            try:
                st.write(data)
            except Exception:
                pass  # 日志文件写失败不该中断评测

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _install_log():
    """把 print 与 logging 两路输出并进同一份日志，返回 (文件对象, handler)。"""
    f = open(LOG_PATH, "w", encoding="utf-8")
    # 与 print 共用同一个文件句柄：两个句柄各持偏移会把日志写乱
    handler = logging.StreamHandler(f)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, f)
    sys.stderr = _Tee(orig_err, f)
    return f, handler

# 样本源：(组名, 文件名, 每作物取几条)。None = 不限。
#
# 【基线必须逐字节不动】PER_CROP=4 且取"文件里前 4 条"，是为了让
# multiturn_40_*.jsonl 之间能逐条配对比较。改动采样就等于让历史基线全部失效。
#
# 【反馈集不并进基线】这些是靠 crawler/feedback_to_samples.py 从用户判错里还原的
# badcase，准确率天然低于随机抽的样本。混进基线会让"badcase 变多"看起来
# 像是"基线退化"。所以分组成两组，分开报数，基线组的结果仍可与历史对齐。
SOURCES = [
    ("baseline", "vl_desc_Qwen_Qwen3-VL-8B-Instruct_v3.jsonl", PER_CROP),
    ("feedback", "vl_desc_feedback.jsonl", None),
]

SOURCE_META = []   # 样本源指纹，写进 META；用来证明两次运行的采样集是否一致
samples = []
for group, fname, per_crop in SOURCES:
    path = os.path.join(DATA, fname)
    if not os.path.exists(path):
        print(f"  [跳过] 样本源不存在: {fname}", flush=True)
        continue
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    missing = sum(1 for r in rows if not (r.get("crop") and r.get("truth")))
    if missing:
        # 少 crop 或 truth 的样本没法评测（kb_symptom 要用 crop 查库），
        # 静默丢掉会让"样本数不对"变成一个查不出原因的谜题。
        print(f"  [跳过] {fname}: {missing} 行缺 crop/truth", flush=True)
    byc = defaultdict(list)
    for r in rows:
        if r.get("crop") and r.get("truth") and r["truth"] != "__HEALTHY__":
            byc[r["crop"]].append(r)
    for c, lst in byc.items():
        picked = lst if per_crop is None else lst[:per_crop]
        samples += [dict(s, group=group) for s in picked]
    SOURCE_META.append({
        "group": group, "file": fname, "per_crop": per_crop,
        "rows": len(rows), "missing_crop_or_truth": missing,
        # sha1 是关键：判断"两次运行采样是否一致"比对比文件名强得多 ——
        # 文件被追加/改写时文件名不会变，但哈希会变。
        "sha1": hashlib.sha1(open(path, "rb").read()).hexdigest()[:12],
    })

# 结果文件名。只跑一组时换成另一个名字，避免一次子集运行把全量结果盖掉 ——
# 否则 multiturn_40.jsonl 会变成"只有 1 条"的文件，而名字里还写着 40。
OUT_NAME = "multiturn_40.jsonl"

# --group baseline|feedback：只跑某一组。用途是"只重跑 badcase 看有没有改善"，
# 不必为了几条反馈把 40 条基线也重新烧一遍 API。
if "--group" in sys.argv:
    want = sys.argv[sys.argv.index("--group") + 1]
    samples = [s for s in samples if s["group"] == want]
    OUT_NAME = f"multiturn_40_{want}.jsonl"

# --samples N：只跑前 N 条。改完这个脚本先用它做一次分钟级冒烟，
# 而不是等 20 分钟才发现某个字段名写错了。
if "--samples" in sys.argv:
    samples = samples[: int(sys.argv[sys.argv.index("--samples") + 1])]

print(f"样本 {len(samples)}  并发 {CONC}", flush=True)
for g in dict.fromkeys(s["group"] for s in samples):
    print(f"  {g}: {sum(1 for s in samples if s['group'] == g)}", flush=True)

# --dry-run：只列样本源并退出，不烧 API 额度。开跑前用它确认反馈集有没有并进来。
if "--dry-run" in sys.argv:
    sys.exit(0)

META_PATH = os.path.join(DATA, OUT_NAME.replace(".jsonl", ".meta.json"))


def _git_info():
    """代码版本。跨时间对比时，"配置没变但指标变了"最常见的解释就是代码变了 ——
    没有这个字段就只能靠回忆。不在 git 仓库或没装 git 时返回空值，不影响评测。"""

    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    return {"commit": run("rev-parse", "--short", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def _config_fingerprint():
    """本次运行的全部配置变量。

    含 SYSTEM_PROMPT 的哈希：prompt 是最大的隐性变量 —— 改它会让指标整体漂移，
    而它不在 config.py 里，只对着配置项一个个比是发现不了的。
    """
    return {
        "MODEL_NAME": MODEL,
        "RERANK_MODEL": RERANK_MODEL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "recall_k": RECALL_K,
        "top_k": TOP_K,
        "bm25_filter_pool": BM25_FILTER_POOL,
        "describe_prompt_version": DESCRIBE_PROMPT_VERSION,
        "system_prompt_sha1": hashlib.sha1(
            SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
    }


# 跨时间对比时，指标差异必须能归因到"配置变了 / 代码变了 / 样本变了 / 只是抖动"。
# 光有结果文件做不到 —— 手里那个 multiturn_40_qwen3rerank.jsonl 就只有指标。
# 这份 sidecar 就是为了回答这个归因问题。
META = {
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "log": os.path.basename(LOG_PATH),
    "out": OUT_NAME,
    "n_samples": len(samples),
    "groups": {g: sum(1 for s in samples if s["group"] == g)
               for g in dict.fromkeys(s["group"] for s in samples)},
    "sources": SOURCE_META,
    "config": _config_fingerprint(),
    "code": _git_info(),
    "runtime": {"max_turns": MAX_TURNS, "conc": CONC, "per_crop": PER_CROP},
    # 指标口径写进元数据。2026-10-04 把 solution_ok 从"最后一轮"改成"任意一轮"，
    # 不写下来的话，将来对比时会拿两个不同口径的数字互相印证。
    "metrics": {
        "first_hit": "首次给出正确诊断的轮次, 0 = 从未; 判定看全部轮次",
        "solution_ok": "任意一轮给出针对真值病害的可执行防治方案（2026-10-04 起）",
        "solution_last_only": "仅看最后一轮（旧口径），用于与该日期之前的基线对比",
    },
}


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

# 两个问题一次问完，输出两位数字；不拆成两次调用是为了不多花一轮 judge 成本。
#
# 【为什么加问 ①】旧版只看**最后一轮**，但本脚本强制跑满 MAX_TURNS 轮 ——
# agent 在第 3~4 轮给了方案、第 5 轮改成回答农民追问时，那份方案就被判丢了。
# 实测旧基线 40 条里 25 次方案失败，其中 12 次属于这种盲区。
#
# 【为什么保留 ②】旧口径要留着：history 里 multiturn_40_*.jsonl 的 solution_ok
# 全是"只看最后一轮"，不留一份同口径的数字就无法与它们对齐。
J_SOLU = """下面是一段「助手与农民」的多轮对话。真实病害是：{truth}

请回答两个问题。合格的防治方案应包含具体措施，例如农业防治（清园/轮作/控湿等）、
化学防治（具体药剂名或药剂类型、稀释倍数、喷药时机/频次）等；
只要泛泛而谈"注意通风、加强管理"、或没有给出任何可执行措施，即算不合格；
针对**错误病害**给出的方案也算不合格。

① 助手在**任意一轮**中，是否给出过针对 {truth} 的可执行防治方案？
② **最后一轮**的回复，本身是否就是一个针对 {truth} 的可执行防治方案？

只输出两个数字，用空格分隔，顺序为 ①②。例如 "1 0" 表示任意一轮给过、但最后一轮没给。
不要解释。"""

# 每轮截断 1500 字，与旧版对"最后一轮"的截断长度一致 ——
# 这样 solution_last_only 与旧数字同口径，可以直接对齐历史基线。
# 刻意不复用 JUDGE_TXT（那边是 600）：改它会让 J_DIAG 的输入一起变、进而动到确诊率，
# 那是另一个指标，不该被这次改动连带影响。
J_SOLU_TXT = lambda turns: "\n".join(
    f"第{t['turn']}轮 助手：{(t.get('agent') or '')[:1500]}" for t in turns)

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
    sv = await _ask(J_SOLU.format(truth=s["truth"]) + "\n\n" + J_SOLU_TXT(turns))
    sdigits = "".join(ch for ch in sv if ch.isdigit())
    if len(sdigits) < 2:
        # 少一位就意味着 ② 会被判成 0，静默拉低旧口径指标；
        # 宁可留痕，也不要让一次返回格式异常变成看不出来的数字变化。
        logger.warning("J_SOLU 未返回两位数字: %r (truth=%s)", sv, s["truth"])
    return {"crop": s["crop"], "truth": s["truth"],
            "group": s.get("group", "baseline"),   # 基线 / 反馈集，用于分组报数
            "kb_len": len(kb),
            "turns": turns, "first_hit": first_hit, "n_asked": n_asked,
            # 口径变更（2026-10-04）：solution_ok 现在是"任意一轮给过"，
            # 不再是"最后一轮给过"。与历史文件对比请用 solution_last_only。
            "solution_ok": sdigits[:1] == "1",
            "solution_last_only": sdigits[1:2] == "1",
            "leak": sum(1 for t in turns if "来源:" in t["agent"])}


async def _run():
    sem = asyncio.Semaphore(CONC)
    done = [0]

    async def w(i, s):
        async with sem:
            try:
                r = await run_one(i, s)
            except Exception as e:
                r = {"crop": s["crop"], "truth": s["truth"],
                     "group": s.get("group", "baseline"), "err": repr(e)}
            done[0] += 1
            print(f"  [{done[0]}/{len(samples)}] {r.get('crop')}/{r.get('truth')} "
                  f"first_hit={r.get('first_hit')} sol={r.get('solution_ok')} "
                  f"{r.get('err','')}", flush=True)
            return r

    res = await asyncio.gather(*[w(i, s) for i, s in enumerate(samples)])
    ok = [r for r in res if not r.get("err")]
    n = len(ok)
    print(f"\n{'='*66}\n成功 {n}/{len(res)}\n{'-'*66}")

    def report(rs, title):
        # 基线组的结果可与 multiturn_40_*.jsonl 直接比对；反馈组是用户判错的 badcase，
        # 准确率天然更低 —— 分开看才知道变化来自哪一边。
        if not rs:
            return
        m = len(rs)
        print(f"[{title}] n={m}")
        print(f"  ① 3 轮内确诊      : {sum(1 for r in rs if 0 < r['first_hit'] <= 3)}/{m}"
              f"  = {sum(1 for r in rs if 0 < r['first_hit'] <= 3)/m:.1%}")
        print(f"  ② 5 轮内确诊      : {sum(1 for r in rs if r['first_hit'])}/{m}"
              f"  = {sum(1 for r in rs if r['first_hit'])/m:.1%}")
        n_any = sum(1 for r in rs if r["solution_ok"])
        n_last = sum(1 for r in rs if r.get("solution_last_only"))
        print(f"  ③ 给出防治方案(任意一轮): {n_any}/{m} = {n_any/m:.1%}")
        print(f"      └ 仅看最后一轮(旧口径): {n_last}/{m} = {n_last/m:.1%}"
              f"   两者之差 = 判定盲区 {n_any - n_last} 条")
        print(f"  ④ 单轮基线(第1轮确诊): {sum(1 for r in rs if r['first_hit']==1)}/{m}"
              f"  = {sum(1 for r in rs if r['first_hit']==1)/m:.1%}")
        print(f"     追问率 {sum(1 for r in rs if r['n_asked']>0)/m:.1%}"
              f" | 工具原文泄漏轮次 {sum(r['leak'] for r in rs)}")

    report(ok, "全部")
    for g in dict.fromkeys(r.get("group", "baseline") for r in ok):
        report([r for r in ok if r.get("group", "baseline") == g], g)
    print(f"{'-'*66}\n按作物:")
    bc = defaultdict(list)
    for r in ok:
        bc[r["crop"]].append(r)
    for c, lst in sorted(bc.items()):
        print(f"  {c:<6} n={len(lst):<3} 3轮内 "
              f"{sum(1 for r in lst if 0<r['first_hit']<=3)}/{len(lst)}"
              f"  5轮内 {sum(1 for r in lst if r['first_hit'])}/{len(lst)}"
              f"  方案 {sum(1 for r in lst if r['solution_ok'])}/{len(lst)}")
    with open(os.path.join(DATA, OUT_NAME), "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n明细 → {os.path.join(DATA, OUT_NAME)}")
    return res


def _summarize(res):
    """指标摘要，与终端输出同源。写进元数据后，只看 meta 就知道这次跑了什么。"""
    ok = [r for r in res if not r.get("err")]

    def m(rs):
        n = len(rs)
        if not n:
            return {}
        return {
            "n": n,
            "hit3": sum(1 for r in rs if 0 < r["first_hit"] <= 3),
            "hit5": sum(1 for r in rs if r["first_hit"]),
            "solution_ok": sum(1 for r in rs if r["solution_ok"]),
            "solution_last_only": sum(1 for r in rs if r.get("solution_last_only")),
            "hit1": sum(1 for r in rs if r["first_hit"] == 1),
            "asked": sum(1 for r in rs if r["n_asked"] > 0),
            "leak": sum(r["leak"] for r in rs),
        }

    return {
        "total": len(res),
        "ok": len(ok),
        "errors": [r for r in res if r.get("err")],
        "all": m(ok),
        "by_group": {
            g: m([r for r in ok if r.get("group", "baseline") == g])
            for g in dict.fromkeys(r.get("group", "baseline") for r in ok)
        },
    }


def _finalize(log_f, handler, t0):
    """元数据与日志必须在 finally 里落盘。

    跑 30 分钟后才崩的话，这份日志与元数据正是唯一能定位原因的线索，
    不能因为走了异常路径就丢掉。
    """
    META["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    META["duration_sec"] = round(time.time() - t0, 1)
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(META, f, ensure_ascii=False, indent=2)
        print(f"元数据 → {META_PATH}", flush=True)
    finally:
        logging.getLogger().removeHandler(handler)
        log_f.flush()
        log_f.close()


async def main():
    log_f, handler = _install_log()
    print(f"日志 → {LOG_PATH}", flush=True)
    t0 = time.time()
    try:
        META["results"] = _summarize(await _run())
    except BaseException as e:
        # 用 BaseException 而不是 Exception：Ctrl-C 中断也要留下元数据，
        # 否则"跑了一半被打断"和"从没跑过"在文件系统上分不出来。
        META["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        _finalize(log_f, handler, t0)


if __name__ == "__main__":
    # 这层守卫是为了让本模块能被 import（例如单独调 J_SOLU 验证判定输出格式、
    # 或写单测覆盖这里的解析逻辑）；否则一 import 就会跑满整个评测。
    asyncio.run(main())
