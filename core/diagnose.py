"""结构化诊断：把对话里已经得出的结论搬出来，附上不可编造的知识库依据。

【与 /chat/stream 的分工】
对话负责判断（追问 → 排除 → 收敛），本模块只负责**搬运与核对**：
把结论从对话里读出来当查阅键，再去知识库取原文。它不是第二次诊断。

【字段来源 —— 这是本模块的全部设计】
    crop / diseases      ← 抽取（只取对话里说过的，且事后做确定性校验）
    evidence / plan      ← core.retriever 的原文，**LLM 碰不到**
    in_kb                ← 确定性比对（病名是否出现在检索池里）
    status / reason      ← 确定性规则

被刻意省略的字段（matched_features / support / conflicts / 置信度数字）：
它们是对话正文的再包装——用户刚读过；而字段名会让前端误当成"已核实的事实"。
与 core/sources.py 里"relevance 不叫 confidence"是同一条原则。

【两路独立，才有互校价值】
抽取走"对话"，检索走"症状描述"，两条路的输入不同，所以 in_kb 能证伪。
若检索也用病名当 query，就成了自证。

【阈值必须在本模块自己判】
core/retriever.retrieve() 的 score_threshold 只在 rerank 降级分支生效
（core/retriever.py 的 else 分支），rerank 正常返回时完全不看它。
依赖它会得到一个静默失效的阈值。

⚠ 已知局限：rerank 连续失败降级后，分数是双塔相似度混着 0.0，相互不可比，
此时下面 low_relevance 的判定不可靠（可观察 core/retriever._rerank 的 warning 日志）。

【失败语义】
抽取失败抛 DiagnoseError，**不退化成 abstain**：
"服务故障"与"确实没查出结论"在前端必须能区分开（见 api/diagnose.py）。
"""
import asyncio
import json
import logging
import re
import time

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from .config import (
    DEFAULT_SCORE_THRESHOLD,
    DIAGNOSE_EVIDENCE_K,
    DIAGNOSE_HISTORY_TURNS,
    DIAGNOSE_MAX_CANDIDATES,
    DIAGNOSE_MAX_EVIDENCE,
    DIAGNOSE_MAX_PLAN,
    DIAGNOSE_PLAN_K,
    DIAGNOSE_PROMPT_VERSION,
    KB_COLLECTION,
    MODEL_NAME,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RECALL_K,
    SECTION_CONTROL,
    SECTION_SYMPTOM,
)
from .memory import load_history
from .retriever import retrieve
from .sources import kb_sources

logger = logging.getLogger("diagnose")

# 独立非流式客户端。理由与 core/vision.py 相同：不复用 core/llm.py 那个
# 开了 streaming、temperature=0.3 的主对话实例 —— 抽取要的是完整 JSON 与确定性输出。
_client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)

# 单次抽取的硬超时。输出只有几十个 token，30s 已是宽裕值。
EXTRACT_TIMEOUT = 30

# 检索 query 的拼装上限。属本模块实现细节，故不放进 config。
# 上限存在的理由：query 过长会稀释向量检索的语义焦点。
QUERY_MAX_CHARS = 2000
QUERY_HEAD_CHARS = 600

NOTES = (
    "relevance 为 rerank 相关性分，仅同一次查询内可比，不是概率。"
    "evidence / plan 为知识库原文摘录，未经模型改写。"
)


class DiagnoseError(RuntimeError):
    """诊断不可用（抽取调用失败 / 结果无法解析）。api 层转成 503。"""


# 用 replace("{conversation}", ...) 而不是 str.format：
# 下面 prompt 里有 JSON 示例的大括号，.format 会把它们当成占位符直接抛 KeyError。
EXTRACT_PROMPT = """下面是一段农业病虫害诊断对话。请抽取其中**已经得出的结论**，只做抽取，不做推理。

输出一个 json 对象，字段如下：
{
  "crop": "作物名，或 null",
  "diseases": ["候选病害名", "..."]
}

抽取规则：
1. crop —— 对话中明确提到的作物（如"番茄"）。只填作物本身，不要带"已知作物："这类
   前缀，不要带标点。对话没说就填 null。
2. diseases —— 助手在回答中给出的候选病害名，**按可能性从高到低**排列，最多 3 个。
   · 必须**逐字复制对话里的原文**：不要补全作物前缀、不要规范化、不要翻译。
   · 只取助手**给出了结论**的病名；仅被提及用于"排除"的不算。
   · 同一种病只出现一次。
   · 对话还停在追问阶段、尚未给出任何结论时，返回空数组 []。

对话：
{conversation}
"""


class _Extraction(BaseModel):
    crop: str | None = None
    diseases: list[str] = Field(default_factory=list)


def _strip_fence(text: str) -> str:
    """剥掉 markdown 代码块围栏。json_object 模式下理论上不会有，但代价极小。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _clean_crop(raw: str | None) -> str | None:
    """清洗作物名。

    对话里的作物标记形如「已知作物：番茄。」（api/chat.py 的 CROP_NOTE_PREFIX 注入），
    抽取结果可能把前缀或句末标点一并带回来。

    刻意不 import api.chat 的 CROP_NOTE_PREFIX：core 不应反向依赖 api 层
    （见 api/__init__.py 的分工说明）。这里重复一个字符串字面量是可接受的代价。
    """
    if not raw:
        return None
    s = re.sub(r"^已知作物\s*[:：]\s*", "", raw.strip())
    s = s.strip("。．.,，;；:： \t")
    return s or None


def _format_conversation(history: list[dict]) -> str:
    lines = []
    for m in history:
        text = (m.get("content") or "").strip()
        if text:
            lines.append(f"{'用户' if m.get('role') == 'user' else '助手'}：{text}")
    return "\n".join(lines)


def _build_query(history: list[dict]) -> str:
    """检索 query = 对话全文（首条优先保留）。

    首条必须保住：图片的 VL 描述只在首条里（core/memory.py 的【首条常驻】），
    丢了它等于让检索看不到图片信息 —— 而图片是本项目最主要的入口。
    """
    text = "\n".join(
        (m.get("content") or "").strip() for m in history if (m.get("content") or "").strip()
    )
    if len(text) <= QUERY_MAX_CHARS:
        return text
    # 超长时保留头(图片描述)+尾(最近的症状描述)，掐掉中间
    head = text[:QUERY_HEAD_CHARS]
    tail = text[-(QUERY_MAX_CHARS - QUERY_HEAD_CHARS):]
    return f"{head}\n{tail}"


def _appears_in(conversation: str, name: str, crop: str | None) -> bool:
    """病名是否真的出现在对话里。

    抽取已要求模型逐字复制，但实际可能带/不带作物前缀，两种都认。
    这是启发式，目的是拦住"番茄的对话里冒出柑橘黄龙病"这类纯编造，
    不是做严格的字符串相等。
    """
    if not name:
        return False
    if name in conversation:
        return True
    if crop and name.startswith(crop):
        rest = name[len(crop):].strip()
        if rest and rest in conversation:
            return True
    if crop and f"{crop}{name}" in conversation:
        return True
    return False


def _valid_names(conversation: str, raw_names: list[str], crop: str | None) -> list[str]:
    """去重 → 校验出处 → 截断到 DIAGNOSE_MAX_CANDIDATES，保持抽取给出的顺序。"""
    out: list[str] = []
    for raw in raw_names or []:
        name = (raw or "").strip()
        if not name or name in out:
            continue
        if not _appears_in(conversation, name, crop):
            # 唯一能拦住"结论没有出处"的闸门，故留 warning 便于事后统计编造率
            logger.warning("丢弃对话中找不到的病名: %r", name)
            continue
        out.append(name)
        if len(out) >= DIAGNOSE_MAX_CANDIDATES:
            break
    return out


async def _extract(conversation: str) -> _Extraction:
    """调 LLM 抽取结论。失败抛 DiagnoseError，绝不返回空结果冒充"没查出结论"。"""
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            _client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0,  # 抽取要确定性，不要发挥
                max_tokens=300,
                # 走 json_object 模式。若该 provider 不支持这个参数，会在首次调用
                # 就报 400（验证阶段即可发现），届时删掉这一行即可 ——
                # prompt 本身已要求只输出 json，_strip_fence 也能兜住围栏。
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": EXTRACT_PROMPT.replace("{conversation}", conversation),
                    }
                ],
            ),
            timeout=EXTRACT_TIMEOUT,
        )
    except Exception as e:
        # 必须带 type(e).__name__：TimeoutError 的 str() 是空串，
        # 否则日志只剩一句 "抽取失败: "，看不出是超时（与 core/vision.py 同一个坑）。
        logger.warning(
            "抽取失败 after %.1fs: %s: %s", time.monotonic() - t0, type(e).__name__, e
        )
        raise DiagnoseError("诊断结论抽取失败，请稍后重试") from e

    raw = (resp.choices[0].message.content or "").strip()
    try:
        return _Extraction.model_validate(json.loads(_strip_fence(raw)))
    except Exception as e:
        logger.warning(
            "抽取结果无法解析: %s: %s / raw=%r", type(e).__name__, e, raw[:200]
        )
        raise DiagnoseError("诊断结论抽取结果无法解析") from e


def _result(
    session_id: str,
    status: str,
    crop: str | None,
    candidates: list[dict],
    abstain_reason: str | None,
) -> dict:
    return {
        "session_id": session_id,
        "status": status,
        "crop": crop,
        "prompt_version": DIAGNOSE_PROMPT_VERSION,
        "candidates": candidates,
        "abstain_reason": abstain_reason,
        "notes": NOTES,
    }


def _assemble(
    session_id: str,
    crop: str | None,
    names: list[str],
    symptom_pool: list[dict],
    plan_pool: list[dict],
) -> dict:
    """纯函数：把抽取结果与检索池拼成响应。

    不碰网络、不碰 Redis、不碰 LLM，故可直接单测（tests/test_diagnose.py）——
    它是这个模块里唯一值得、也唯一能够被穷举覆盖的部分。
    """
    # in_kb 的判据就是这张集合：病名有没有出现在**独立检索**的结果里。
    # 只认症状与防治两个章节 —— 某病害若在这两章都没有内容，我们本来也服务不了它。
    pool_diseases = {
        r.get("disease") for r in (*symptom_pool, *plan_pool) if r.get("disease")
    }

    candidates: list[dict] = []
    best_score = 0.0
    for rank, name in enumerate(names, start=1):
        # 检索池内部已按 relevance 降序，故每组取前 N 条即该病害最相关的 N 条。
        ev = [r for r in symptom_pool if r.get("disease") == name][:DIAGNOSE_MAX_EVIDENCE]
        pl = [r for r in plan_pool if r.get("disease") == name][:DIAGNOSE_MAX_PLAN]
        if ev:
            best_score = max(best_score, float(ev[0].get("score") or 0.0))
        candidates.append(
            {
                "rank": rank,
                "disease": name,
                "in_kb": name in pool_diseases,
                # evidence/plan 用 core.sources.kb_sources 构造，
                # 与 /chat/stream 的 sources 事件同构 —— 前端一套渲染组件通吃。
                "evidence": kb_sources(ev),
                "plan": kb_sources(pl),
            }
        )

    # 弃权时仍然返回 candidates：'疑似 X 但知识库里查不到依据'比空数组有用得多。
    if not any(c["evidence"] for c in candidates):
        status, abstain_reason = "abstain", "out_of_kb"
    elif best_score < DEFAULT_SCORE_THRESHOLD:
        status, abstain_reason = "abstain", "low_relevance"
    else:
        status, abstain_reason = "ok", None

    return _result(session_id, status, crop, candidates, abstain_reason)


async def diagnose(
    user_id: int, session_id: str, crop_override: str | None = None
) -> dict:
    """读该 session 的对话，产出结构化诊断结果。

    user_id 来自令牌（api/deps.py），session_id 来自请求体 —— 二者决定读哪段历史。
    对话内容本身不接受客户端传入：客户端能塞对话，就能塞一句
    "我确定这是番茄晚疫病"，然后拿到一个结论。
    """
    history = load_history(user_id, session_id, n=DIAGNOSE_HISTORY_TURNS)
    if not history:
        # 没聊过，或已过 CHAT_HISTORY_TTL(30min)。这两种情况前端文案相同。
        return _result(session_id, "abstain", None, [], "session_empty")

    conversation = _format_conversation(history)
    extraction = await _extract(conversation)  # 失败抛 DiagnoseError，不降级

    # 显式传入优先：用户自己知道种的是什么，比让模型从对话里认更可靠。
    # 认错作物会让检索被过滤到错误的作物上，表现为"知识库无相关资料"——
    # 是个会误导人的失败模式，而 crop_override 是最便宜的规避手段。
    crop = _clean_crop(crop_override) or _clean_crop(extraction.crop)

    names = _valid_names(conversation, extraction.diseases, crop)
    if not names:
        return _result(session_id, "abstain", crop, [], "no_conclusion")

    # query 用对话本身，不用病名 —— 这是与抽取相互独立的另一路，in_kb 才成立（见模块头）。
    query = _build_query(history)

    # 两次检索互不依赖，并发发起。
    # 【为什么这里敢并发】/diagnose 一次会话只调一次（不是评测脚本那种 40 连发），
    # 峰值只有 2 个 rerank 请求；而 core/retriever.py 的 _rerank 本身带指数退避。
    # 若日后观察到 429 降级日志变多，改回顺序 await 即可。
    symptom_pool, plan_pool = await asyncio.gather(
        retrieve(
            query,
            KB_COLLECTION,
            k=DIAGNOSE_EVIDENCE_K,
            hybrid=True,  # 与 /chat/stream 的检索配置一致，否则两边结果不可比
            recall_k=RECALL_K,
            crop=crop,
            section=SECTION_SYMPTOM,
        ),
        retrieve(
            query,
            KB_COLLECTION,
            k=DIAGNOSE_PLAN_K,
            hybrid=True,
            recall_k=RECALL_K,
            crop=crop,
            section=SECTION_CONTROL,
        ),
    )

    result = _assemble(session_id, crop, names, symptom_pool, plan_pool)
    logger.info(
        "diagnose session=%s crop=%r names=%d pools=%d/%d status=%s reason=%s",
        session_id,
        crop,
        len(names),
        len(symptom_pool),
        len(plan_pool),
        result["status"],
        result["abstain_reason"],
    )
    return result
