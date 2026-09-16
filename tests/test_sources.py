"""core/sources.py 的单测：纯函数，不依赖 Milvus / Tavily / 网络。"""
from core.sources import EXCERPT_CHARS, excerpt, kb_sources, merge_sources, web_sources


def _kb(disease="番茄早疫病", section="危害症状", crop="番茄",
        score=0.83, content="叶片出现褐色病斑"):
    return {
        "crop": crop,
        "disease": disease,
        "section": section,
        "source_file": "病害/番茄/番茄病害.xlsx",
        "score": score,
        "content": content,
    }


def _web(url="https://a.example", title="T", content="摘要"):
    return {"title": title, "url": url, "content": content}


# ---------------------------------------------------------------------------
# kb_sources
# ---------------------------------------------------------------------------

def test_kb_sources_maps_fields():
    s = kb_sources([_kb()])[0]
    assert s["kind"] == "kb"
    assert s["crop"] == "番茄"
    assert s["disease"] == "番茄早疫病"
    assert s["section"] == "危害症状"
    assert s["source_file"] == "病害/番茄/番茄病害.xlsx"
    assert s["relevance"] == 0.83
    assert s["excerpt"] == "叶片出现褐色病斑"


def test_field_name_is_relevance_not_confidence():
    """刻意锁住命名。

    这个分数是 rerank 的排序分，不是概率，不同 query 之间不可比。
    叫 confidence 会被前端当成"置信度"展示 —— 农业用药场景下，
    一个被误读成"83% 确定"的数字比不给数字更危险。
    """
    assert "confidence" not in kb_sources([_kb()])[0]


def test_kb_sources_tolerates_missing_fields():
    s = kb_sources([{"content": "x"}])[0]
    assert s["disease"] == "" and s["crop"] == "" and s["relevance"] == 0.0


def test_kb_sources_falls_back_to_file_name():
    """旧库(kb_default)用 file_name 字段。"""
    s = kb_sources([{"file_name": "旧库.docx", "content": "x"}])[0]
    assert s["source_file"] == "旧库.docx"


def test_kb_sources_handles_none_score():
    assert kb_sources([{"score": None, "content": "x"}])[0]["relevance"] == 0.0


def test_kb_sources_handles_empty_input():
    assert kb_sources(None) == []
    assert kb_sources([]) == []


# ---------------------------------------------------------------------------
# web_sources
# ---------------------------------------------------------------------------

def test_web_sources_maps_fields():
    assert web_sources([_web()])[0] == {
        "kind": "web", "title": "T", "url": "https://a.example", "excerpt": "摘要",
    }


def test_web_sources_skips_entries_without_url():
    """没有 url 的来源无法溯源，直接丢弃。"""
    out = web_sources([{"title": "无链接", "content": "x"}, _web()])
    assert len(out) == 1 and out[0]["url"] == "https://a.example"


def test_web_sources_skips_blank_url():
    assert web_sources([{"url": "   ", "content": "x"}]) == []


def test_web_sources_handles_empty_input():
    assert web_sources(None) == []


# ---------------------------------------------------------------------------
# excerpt
# ---------------------------------------------------------------------------

def test_excerpt_collapses_whitespace():
    assert excerpt("a\n\nb   c\td") == "a b c d"


def test_excerpt_truncates_with_ellipsis():
    out = excerpt("字" * (EXCERPT_CHARS + 50))
    assert len(out) == EXCERPT_CHARS + 1
    assert out.endswith("…")


def test_excerpt_leaves_short_text_alone():
    assert excerpt("短文本") == "短文本"
    assert excerpt("") == ""
    assert excerpt(None) == ""


# ---------------------------------------------------------------------------
# merge_sources
# ---------------------------------------------------------------------------

def test_merge_dedupes_same_chunk():
    """同一篇文档的同一章节被两次检索召回时只保留一条。"""
    dup = _kb()
    assert len(merge_sources([dup], [dict(dup)])) == 1


def test_merge_keeps_different_sections():
    """同一病害的不同章节是不同依据，不能合并掉。"""
    grouped = merge_sources([_kb(section="危害症状")], [_kb(section="防治方法")])
    assert len(grouped) == 2


def test_merge_keeps_different_diseases():
    grouped = merge_sources([_kb(disease="A")], [_kb(disease="B")])
    assert len(grouped) == 2


def test_merge_dedupes_same_url():
    u = web_sources([_web()])[0]
    assert len(merge_sources([u], [dict(u)])) == 1


def test_merge_kb_and_web_do_not_collide():
    assert len(merge_sources([_kb()], web_sources([_web()]))) == 2


def test_merge_preserves_call_order():
    a, b = _kb(disease="A"), _kb(disease="B")
    assert [s["disease"] for s in merge_sources([a], [b])] == ["A", "B"]


def test_merge_handles_none_and_empty_groups():
    """merge_sources 只做合并，不做字段转换，条目应原样透出。"""
    item = kb_sources([_kb()])[0]
    assert merge_sources(None, [], [item]) == [item]
    assert merge_sources() == []
