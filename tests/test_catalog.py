"""core/catalog.py 与 core/retriever.py 过滤净化的单测。

刻意不依赖 Milvus / Redis / 网络：catalog 的核心逻辑是纯函数
build_crop_list()，净化逻辑是纯函数 _safe_filter_value()，
最后一个用例用 monkeypatch 替换向量库来验证真实调用路径。
"""
import asyncio

import pytest

from core.catalog import _NOT_A_CROP, build_crop_list
from core.retriever import _safe_filter_value, retrieve


def _row(crop, disease, section="危害症状", category="病害"):
    """构造一条 Milvus 元数据行（chunk 是按 section 切分的，所以元数据会重复）。"""
    return {"crop": crop, "disease": disease, "category": category, "section": section}


# ---------------------------------------------------------------------------
# 作物列表派生
# ---------------------------------------------------------------------------

def test_counts_diseases_and_dedupes_sections():
    """同一病害的多个 section 是多个 chunk，但只应算一个病害。"""
    rows = [
        _row("番茄", "番茄早疫病"),
        _row("番茄", "番茄早疫病", section="防治方法"),
        _row("番茄", "番茄早疫病", section="病原"),
        _row("番茄", "番茄晚疫病"),
    ]
    assert build_crop_list(rows) == [{"crop": "番茄", "disease_count": 2}]


def test_pseudo_crop_is_filtered():
    """bulk_ingest_agri.py 的兜底值不是真实作物，不该出现在列表里。"""
    rows = [_row(_NOT_A_CROP, "跨作物条目"), _row("番茄", "番茄早疫病")]
    assert build_crop_list(rows) == [{"crop": "番茄", "disease_count": 1}]


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_blank_crop_is_filtered(blank):
    rows = [_row(blank, "x"), _row("番茄", "番茄早疫病")]
    assert [c["crop"] for c in build_crop_list(rows)] == ["番茄"]


def test_sort_is_count_desc():
    rows = (
        [_row("玉米", f"玉米病{i}") for i in range(3)]
        + [_row("番茄", f"番茄病{i}") for i in range(5)]
        + [_row("苹果", "苹果病1")]
    )
    got = [(c["crop"], c["disease_count"]) for c in build_crop_list(rows)]
    assert got == [("番茄", 5), ("玉米", 3), ("苹果", 1)]


def test_sort_is_deterministic_on_ties():
    """病害数相同时，顺序必须只由作物名决定、与输入顺序无关。

    不在这里断言具体的 Unicode 次序（中文码位顺序对人不直观、也不必是"对"的），
    只断言确定性 —— 这才是可复现、可测试所依赖的性质。
    """
    rows = [_row("乙作物", "a"), _row("甲作物", "b")]
    first = build_crop_list(rows)
    assert first == build_crop_list(list(reversed(rows)))
    assert {c["crop"] for c in first} == {"甲作物", "乙作物"}


def test_empty_rows():
    assert build_crop_list([]) == []


def test_crop_name_is_not_stripped():
    """crop 必须原样返回。

    这个值会被前端原样回传，最终进到 f'crop == "{crop}"'。若这里 strip 过，
    库里带首尾空格的记录就再也匹配不上，且是静默失败（答案照常返回，
    只是检索少了作物约束）。
    """
    assert build_crop_list([_row("番茄 ", "番茄早疫病")])[0]["crop"] == "番茄 "


# ---------------------------------------------------------------------------
# 过滤值净化（Milvus 表达式注入防护）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ok", ["番茄", "危害症状", "跨作物/其他"])
def test_safe_filter_value_accepts_normal(ok):
    assert _safe_filter_value(ok) == ok


@pytest.mark.parametrize("bad", [
    '番茄" or crop != "番茄',   # 最典型的注入载荷
    '番茄"',
    "番茄\\",
    "番茄\n",
    "番茄\r",
])
def test_safe_filter_value_rejects_injection(bad):
    assert _safe_filter_value(bad) is None


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_safe_filter_value_rejects_blank(empty):
    assert _safe_filter_value(empty) is None


@pytest.mark.parametrize("bad_type", [123, ["番茄"], {"a": 1}])
def test_safe_filter_value_rejects_non_str(bad_type):
    assert _safe_filter_value(bad_type) is None


def test_retrieve_drops_malicious_crop_before_building_expr(monkeypatch):
    """净化必须在真实调用路径上生效，而不只是 _safe_filter_value 自己好用。

    直接检查传给向量库的 expr —— 这是"畸形值到底有没有落到表达式里"的
    唯一可靠证据。monkeypatch 掉 build_vs，避免连真 Milvus；
    hybrid=False 也就不触发 BM25 通道。
    """
    captured = {}

    class _FakeVS:
        async def asimilarity_search_with_score(self, query, k=None, expr=None):
            captured["expr"] = expr
            return []          # 空结果让 retrieve 提前返回，不碰 rerank

    monkeypatch.setattr("core.retriever.build_vs", lambda name: _FakeVS())

    out = asyncio.run(
        retrieve(
            "叶片有褐色斑点",
            "kb_agri",
            crop='番茄" or crop != "番茄',
            section="危害症状",
            hybrid=False,
        )
    )

    assert out == []
    assert captured["expr"] == 'section == "危害症状"', \
        "畸形 crop 应被整体丢弃，只留下合法的 section 条件"
