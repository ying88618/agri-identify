"""自纠正闭环测试：拒答检测。

该检测决定了「是否触发 query 改写 → 重检索」的纠错分支，
漏报会让该纠错的内容直接输出，误报会把正常答案反复重生成。
"""
import pytest

try:
    from core.self_correct import detect_refusal
    _IMPORT_ERR = None
except Exception as e:  # 缺少完整 .env（例如 CI 无密钥）时跳过而非报错
    _IMPORT_ERR = e

pytestmark = pytest.mark.skipif(
    _IMPORT_ERR is not None, reason=f"依赖完整 .env 配置: {_IMPORT_ERR}"
)


def test_detects_known_refusal_phrases():
    for text in ["知识库中暂无相关资料", "未找到相关资料", "资料不足，无法回答"]:
        assert detect_refusal(text) is True, f"应识别为拒答: {text}"


def test_empty_or_blank_answer_counts_as_refusal():
    assert detect_refusal("") is True
    assert detect_refusal("   ") is True


def test_normal_answer_is_not_refusal():
    answer = "番茄早疫病的防治方法是加强通风，并喷施代森锰锌或百菌清。"
    assert detect_refusal(answer) is False
