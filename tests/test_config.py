"""配置契约测试：防止旧项目残留值与硬编码回潮。

这两个坑都真实发生过：
  · 集合名 `kb_default` 是第一版项目的遗留，混进来会让检索静默查错库；
  · rerank 阈值曾在多个脚本里各写一遍，换模型后线上与评测脱节。
"""
from pathlib import Path

from core import config

ROOT = Path(__file__).resolve().parent.parent


def test_collection_is_fixed_to_agri():
    assert config.KB_COLLECTION == "kb_agri"
    assert config.MCP_DEFAULT_COLLECTION == config.KB_COLLECTION, \
        "MCP server 必须与主服务共用同一个集合，否则两边检索结果对不上"


def test_retrieval_params_are_coherent():
    assert config.RECALL_K >= config.TOP_K, "宽召回池必须不小于精排返回条数"
    assert 0 < config.DEFAULT_SCORE_THRESHOLD < 1
    assert config.BM25_FILTER_POOL >= 1


def test_threshold_is_defined_only_in_config():
    """阈值只允许在 core/config.py 定义一次，其它模块必须 import。"""
    agent_src = (ROOT / "core" / "agent.py").read_text(encoding="utf-8")
    assert "DEFAULT_SCORE_THRESHOLD =" not in agent_src, \
        "禁止在 core/agent.py 里重复定义阈值，必须 from .config import"
    assert "from .config import" in agent_src


def test_false_alarm_threshold_follows_config():
    """评测脚本的误报阈值必须引用配置，不能硬编码数字。"""
    src = (ROOT / "crawler" / "eval_agri_recall.py").read_text(encoding="utf-8")
    assert "FA_THRESHOLD = DEFAULT_SCORE_THRESHOLD" in src, \
        "误报阈值必须引用 core.config.DEFAULT_SCORE_THRESHOLD，否则换模型后与线上脱节"


def test_no_legacy_absolute_paths_in_scripts():
    """禁止再把上个项目的绝对路径写进脚本（曾导致评测偷偷加载旧工程）。"""
    for name in ("eval_multiturn.py", "exp_vl_rerank.py"):
        src = (ROOT / "crawler" / name).read_text(encoding="utf-8")
        assert "IdeaProjects" not in src, f"{name} 仍残留旧项目绝对路径"
