"""结构化诊断测试。

分两层：
  · 纯函数层（_clean_crop / _valid_names / _build_query / _assemble）
    —— 不需要 LLM、不需要 Milvus、不需要 Redis，毫秒级。
    _assemble 是本模块唯一值得穷举覆盖的部分，弃权规则全在这里。
  · HTTP 层 —— 只 patch core.diagnose 内部的两个调用点，不打真实外部依赖。

不依赖 MySQL：TestClient 不用 with 形式，lifespan 的 init_db() 就不会执行，
而 /diagnose 本身不碰数据库。
"""
import pytest
from fastapi.testclient import TestClient

from core.config import SECTION_CONTROL, SECTION_SYMPTOM
from core.diagnose import (
    DIAGNOSE_MAX_EVIDENCE,
    _Extraction,
    _assemble,
    _build_query,
    _clean_crop,
    _strip_fence,
    _valid_names,
    DiagnoseError,
)


# ---------------------------------------------------------------- 纯函数层

def test_clean_crop_strips_injected_prefix():
    """作物标记是 api/chat.py 注入的「已知作物：番茄。」，抽取可能连前缀带标点一起回来。"""
    assert _clean_crop("已知作物：番茄。") == "番茄"
    assert _clean_crop("  番茄 , ") == "番茄"
    assert _clean_crop(None) is None
    assert _clean_crop("   ") is None


def test_valid_names_drops_fabricated_disease():
    """病名必须能在对话里找到 —— 这是拦纯编造的唯一闸门。"""
    conversation = "助手：最可能是番茄晚疫病。"
    assert _valid_names(conversation, ["番茄晚疫病", "柑橘黄龙病"], "番茄") == ["番茄晚疫病"]


def test_valid_names_accepts_both_crop_prefixed_and_bare():
    """抽取可能带或不带作物前缀，两种都应认（一处带、一处不带）。"""
    assert _valid_names("助手：是晚疫病", ["番茄晚疫病"], "番茄") == ["番茄晚疫病"]
    assert _valid_names("助手：是番茄晚疫病", ["晚疫病"], "番茄") == ["晚疫病"]


def test_valid_names_dedups_and_caps():
    conversation = "助手：番茄晚疫病、番茄早疫病、番茄斑枯病"
    got = _valid_names(conversation, ["番茄晚疫病"] * 3 + ["番茄早疫病", "番茄斑枯病"], "番茄")
    assert got == ["番茄晚疫病", "番茄早疫病", "番茄斑枯病"]  # 去重 + 截到 DIAGNOSE_MAX_CANDIDATES=3


def test_build_query_keeps_head_on_truncation():
    """超长时头尾都留：首条带图片 VL 描述（core/memory.py 的【首条常驻】），丢了就等于检索看不到图。"""
    history = [{"role": "user", "content": "图片描述：叶背有白色霉层" + "甲" * 3000}]
    q = _build_query(history)
    assert q.startswith("图片描述：叶背有白色霉层")
    assert len(q) <= 2000 + 1  # 头 + 换行 + 尾


def test_strip_fence():
    assert _strip_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fence('{"a": 1}') == '{"a": 1}'


def _kb(disease: str, score: float, section: str) -> dict:
    return {
        "disease": disease,
        "crop": "番茄",
        "section": section,
        "source_file": "f.pdf",
        "content": f"{disease}的{section}原文",
        "score": score,
    }


def test_assemble_ok_path():
    r = _assemble(
        "s1", "番茄", ["番茄晚疫病"],
        [_kb("番茄晚疫病", 0.91, SECTION_SYMPTOM)], [_kb("番茄晚疫病", 0.88, SECTION_CONTROL)],
    )
    assert r["status"] == "ok" and r["abstain_reason"] is None
    c = r["candidates"][0]
    assert c["disease"] == "番茄晚疫病" and c["in_kb"] is True and c["rank"] == 1
    assert c["evidence"][0]["section"] == SECTION_SYMPTOM
    assert c["plan"][0]["section"] == SECTION_CONTROL
    # evidence/plan 走 core.sources.kb_sources，字段名必须与 SSE 的 sources 事件同构
    assert set(c["evidence"][0]) == {"kind", "crop", "disease", "section", "source_file", "relevance", "excerpt"}


def test_assemble_evidence_never_leaks_across_candidates():
    """两个候选共用同一个检索池时，各自的 evidence 不能串味。"""
    r = _assemble(
        "s1", "番茄", ["番茄晚疫病", "番茄早疫病"],
        [_kb("番茄晚疫病", 0.9, SECTION_SYMPTOM), _kb("番茄早疫病", 0.8, SECTION_SYMPTOM)],
        [],
    )
    a, b = r["candidates"]
    assert [e["disease"] for e in a["evidence"]] == ["番茄晚疫病"]
    assert [e["disease"] for e in b["evidence"]] == ["番茄早疫病"]


def test_assemble_out_of_kb_when_no_evidence():
    r = _assemble("s1", "番茄", ["番茄晚疫病"], [], [])
    assert r["status"] == "abstain" and r["abstain_reason"] == "out_of_kb"
    # 弃权时仍返回 candidates：'疑似 X 但查不到依据' 比空数组有用
    assert r["candidates"][0]["disease"] == "番茄晚疫病"
    assert r["candidates"][0]["in_kb"] is False


def test_assemble_in_kb_true_but_no_symptom_evidence():
    """命中了防治方法、但危害症状章节没有内容 —— evidence 为空即视为无依据。"""
    r = _assemble("s1", "番茄", ["番茄晚疫病"], [], [_kb("番茄晚疫病", 0.9, SECTION_CONTROL)])
    assert r["status"] == "abstain" and r["abstain_reason"] == "out_of_kb"
    assert r["candidates"][0]["in_kb"] is True   # 在池里
    assert r["candidates"][0]["plan"]            # 有方案
    assert r["candidates"][0]["evidence"] == []  # 没症状原文


def test_assemble_low_relevance_boundary():
    """阈值是 DEFAULT_SCORE_THRESHOLD=0.5，判据为 < —— 恰好等于阈值应当放行。"""
    just_below = _assemble("s1", "番茄", ["番茄晚疫病"], [_kb("番茄晚疫病", 0.49, SECTION_SYMPTOM)], [])
    assert just_below["status"] == "abstain" and just_below["abstain_reason"] == "low_relevance"

    at_threshold = _assemble("s1", "番茄", ["番茄晚疫病"], [_kb("番茄晚疫病", 0.5, SECTION_SYMPTOM)], [])
    assert at_threshold["status"] == "ok"


def test_assemble_truncates_evidence():
    pool = [_kb("番茄晚疫病", 0.9 - i * 0.01, SECTION_SYMPTOM) for i in range(10)]
    r = _assemble("s1", "番茄", ["番茄晚疫病"], pool, [])
    assert len(r["candidates"][0]["evidence"]) == DIAGNOSE_MAX_EVIDENCE


def test_assemble_always_reports_prompt_version():
    """prompt_version 必须始终回传 —— 反馈数据要靠它区分是哪版 prompt 产出的。"""
    r = _assemble("s1", "番茄", ["番茄晚疫病"], [], [])
    assert isinstance(r["prompt_version"], int)


# ---------------------------------------------------------------- HTTP 层

@pytest.fixture
def client():
    from main import app

    return TestClient(app)


def _auth(user_id: int = 1) -> dict:
    from api.security import create_token

    return {"Authorization": f"Bearer {create_token(user_id)}"}


def test_diagnose_requires_token(client):
    r = client.post("/diagnose", json={"session_id": "s1"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_diagnose_empty_session_is_abstain_not_500(client, monkeypatch):
    """没聊过 / 已过 TTL —— 必须是 abstain，不能是 500，也不能是 ok。"""
    monkeypatch.setattr("core.diagnose.load_history", lambda *a, **k: [])
    r = client.post("/diagnose", json={"session_id": "never-used"}, headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "abstain" and body["abstain_reason"] == "session_empty"
    assert body["candidates"] == []


def test_diagnose_extraction_failure_is_503_not_abstain(client, monkeypatch):
    """★ 关键区分：抽取失败不能伪装成'没查出结论'，否则前端会把服务故障渲染成诊断结果。"""
    async def _boom(_conversation):
        raise DiagnoseError("抽取失败")

    monkeypatch.setattr("core.diagnose.load_history", lambda *a, **k: [{"role": "user", "content": "hi"}])
    monkeypatch.setattr("core.diagnose._extract", _boom)
    r = client.post("/diagnose", json={"session_id": "s1"}, headers=_auth())
    assert r.status_code == 503


def test_diagnose_happy_path(client, monkeypatch):
    """端到端（把两个外部调用点换成假的）：对话已有结论 → 绑定 KB 原文 → ok。"""
    conversation = [{"role": "user", "content": "已知作物：番茄。叶背有白色霉层"},
                    {"role": "assistant", "content": "最可能是番茄晚疫病。"}]

    async def _fake_extract(_conversation):
        return _Extraction(crop="番茄", diseases=["番茄晚疫病"])

    async def _fake_retrieve(question, collection_name, **kw):
        section = kw.get("section")
        if section == SECTION_SYMPTOM:
            return [_kb("番茄晚疫病", 0.91, SECTION_SYMPTOM)]
        assert section == SECTION_CONTROL
        return [_kb("番茄晚疫病", 0.88, SECTION_CONTROL)]

    monkeypatch.setattr("core.diagnose.load_history", lambda *a, **k: conversation)
    monkeypatch.setattr("core.diagnose._extract", _fake_extract)
    monkeypatch.setattr("core.diagnose.retrieve", _fake_retrieve)

    r = client.post("/diagnose", json={"session_id": "s1"}, headers=_auth())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["crop"] == "番茄"
    assert body["candidates"][0]["disease"] == "番茄晚疫病"
    assert body["candidates"][0]["evidence"] and body["candidates"][0]["plan"]


def test_diagnose_crop_override_beats_extraction(client, monkeypatch):
    """显式传入的作物优先 —— 用户比模型更知道自己种的是什么，这也是规避'认错作物'的手段。"""
    conversation = [{"role": "user", "content": "番茄晚疫病"}]
    seen: dict = {}

    async def _fake_extract(_conversation):
        return _Extraction(crop="黄瓜", diseases=["番茄晚疫病"])

    async def _fake_retrieve(question, collection_name, **kw):
        seen["crop"] = kw.get("crop")
        return []

    monkeypatch.setattr("core.diagnose.load_history", lambda *a, **k: conversation)
    monkeypatch.setattr("core.diagnose._extract", _fake_extract)
    monkeypatch.setattr("core.diagnose.retrieve", _fake_retrieve)

    body = client.post(
        "/diagnose", json={"session_id": "s1", "crop": "番茄"}, headers=_auth()
    ).json()
    assert body["crop"] == "番茄"
    assert seen["crop"] == "番茄"
