"""检索层单元测试：只测纯函数，不依赖 Milvus / Redis / 外部 API。

这些函数是混合检索的核心：分词质量决定 BM25 通道能否召回农业术语，
RRF 融合决定两路候选如何合并去重。
"""
from core.bm25_index import tokenize
from core.retriever import _rrf_fusion


class TestTokenize:
    def test_keeps_english_terms_lowercased(self):
        toks = tokenize("Tomato Early Blight")
        assert "tomato" in toks
        assert "early" in toks
        assert "blight" in toks

    def test_chinese_goes_through_jieba(self):
        toks = tokenize("番茄叶片有褐色病斑")
        assert any("番茄" in t for t in toks), "中文应由 jieba 切分而非逐字"

    def test_mixed_text_and_punctuation(self):
        toks = tokenize("番茄病斑 2~3mm，边缘黄色晕圈！")
        assert all("," not in t and "，" not in t and "！" not in t for t in toks), \
            "标点不应进入 token"
        assert any("3mm" in t for t in toks), "数字+单位应作为术语保留"

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize("   ") == []


class TestRRFFusion:
    def test_item_ranked_high_in_both_channel_wins(self):
        # 向量通道: X, Y     BM25 通道: Y, Z
        vec = [{"content": "X"}, {"content": "Y"}]
        bm25 = [{"content": "Y"}, {"content": "Z"}]
        out = _rrf_fusion([vec, bm25])
        assert [o["content"] for o in out] == ["Y", "X", "Z"], \
            "两路都靠前的条目应排第一"

    def test_dedupes_same_content_across_channels(self):
        vec = [{"content": "同一条片段"}]
        bm25 = [{"content": "同一条片段"}]
        out = _rrf_fusion([vec, bm25])
        assert len(out) == 1, "两路命中的同一片段应在融合后去重"

    def test_single_channel_preserves_order(self):
        only = [{"content": "A"}, {"content": "B"}, {"content": "C"}]
        out = _rrf_fusion([only])
        assert [o["content"] for o in out] == ["A", "B", "C"]
