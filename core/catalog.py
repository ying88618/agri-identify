"""知识库目录：作物等元数据的枚举与进程级缓存。

供两个场景使用：
  · API 层给前端输入框做自动补全（见 api/kb.py）
  · 将来做"作物 -> 病害"二级联动，或对 crop/section 做白名单校验

【数据来源，以及为什么不复用 bm25_index 的缓存】
本模块只取轻量元数据（crop/disease/category/section），不取正文。
core/bm25_index.py 里已经有全量 chunk 的内存缓存，看着可以白捡 ——
但它拉的是 ["text"] + 元数据，正文有几十 MB（kb_agri 约 1.5 万条 chunk），
而且取它会触发 jieba 分词 + BM25Okapi 构建。

调用时机恰恰是最坏的那种：前端进入对话页第一件事就是渲染作物输入框，
这时 BM25 索引还没建。若走那条路，用户为了看一个列表要先付整条索引的代价，
而这份代价本来是在他真正开始对话时才该付的。

【缓存】
进程级懒加载，刻意不做启动预热：预热会让"服务启动"依赖 Milvus 可用，
而其余端点（鉴权、对话鉴权部分）并不需要它。Milvus 抖动时只影响本模块。
重灌知识库后调 reset()，或访问端点时带 refresh=true。

【缓存的粒度比接口的粒度粗】
缓存的是**原始元数据行**，接口只负责投影。这样将来加"病害列表""章节列表"
都是从同一份行里派生，不会再查一次 Milvus。
"""
import logging

from pymilvus import MilvusClient

from core.config import KB_COLLECTION, MILVUS_URI

logger = logging.getLogger("catalog")

# 只取元数据：正文（text）本模块完全用不到，不取它能把单次查询的
# 传输量从几十 MB 降到几百 KB。
_META_FIELDS = ("crop", "disease", "category", "section")

# Milvus query 单次上限 16384，而 kb_agri 已有约 1.5 万条 chunk，贴着上限。
# 必须分页：库再长一点就会被**静默截断**（不报错，只是少几个作物）。
_PAGE = 16384

# bulk_ingest_agri.py 里 crop 的兜底值（rec.get("crop") or "跨作物/其他"）。
# 它不是真实作物，不能出现在列表或白名单里。
_NOT_A_CROP = "跨作物/其他"

_rows: list[dict] | None = None


def _load_metadata() -> list[dict]:
    """从 Milvus 分页拉取全部元数据行。

    这里刻意不像 bm25_index.py 那样先 describe_collection 探测字段是否存在：
    本模块只对农业库有意义，字段缺失应当**显式报错**（→ 端点返回 503），
    而不是静默返回空列表、让前端看到一个空白的输入框列表。
    """
    client = MilvusClient(uri=MILVUS_URI)
    client.load_collection(KB_COLLECTION)

    rows: list[dict] = []
    offset = 0
    while True:
        batch = client.query(
            collection_name=KB_COLLECTION,
            filter="",
            output_fields=list(_META_FIELDS),
            limit=_PAGE,
            offset=offset,
        )
        rows.extend(batch)
        if len(batch) < _PAGE:
            break
        offset += _PAGE
    return rows


def _rows_cached(refresh: bool = False) -> list[dict]:
    global _rows
    if _rows is None or refresh:
        _rows = _load_metadata()
        logger.info("catalog loaded: %d rows", len(_rows))
    return _rows


def reset() -> None:
    """丢弃缓存，下次访问重新拉取。重灌知识库后调用（或端点带 refresh=true）。"""
    global _rows
    _rows = None


def build_crop_list(rows: list[dict]) -> list[dict]:
    """由元数据行派生作物列表：[{"crop": ..., "disease_count": N}, ...]

    纯函数，不碰 Milvus，因此可以完整单测（见 tests/test_catalog.py）。

    【排序】disease_count 降序 -> 作物名 Unicode 升序。
    中文按 Unicode 码位排序对用户是"随机"顺序，但用病害数降序对用户有实际意义
    （资料最全的排前面，没输入时先看到的就是它们）；同数时用名字做二级排序，
    保证结果与输入顺序无关、可复现、可测试。
    没有引入 pypinyin：为排序多加一个依赖不划算。

    【crop 值必须原样返回，不做 strip】
    这个值会被前端原样回传给 /chat/stream，最终进到
    core/retriever.py 的 f'crop == "{crop}"'。库里若存在带首尾空格的脏值
    （"番茄 " 与 "番茄" 是两行不同记录），strip 之后回传会让过滤**静默**
    匹配不到任何东西 —— 答案照常返回，只是检索少了作物约束。
    真要先修，应该修在灌库侧，而不是在这里做归一化。

    副作用提示：若库里确实存在这种脏值，列表里会同时出现"番茄"和"番茄 "两项。
    这正好是个可见的信号 —— 看到两条相似项，说明该去查源头数据了。
    """
    diseases: dict[str, set[str]] = {}
    for r in rows:
        crop = r.get("crop") or ""
        # 只按 strip 判断"是否为空/伪值"，但保留原始值
        if not crop.strip() or crop.strip() == _NOT_A_CROP:
            continue
        diseases.setdefault(crop, set()).add(r.get("disease") or "")

    return [
        {"crop": crop, "disease_count": len(names)}
        for crop, names in sorted(
            diseases.items(), key=lambda kv: (-len(kv[1]), kv[0])
        )
    ]


def crop_list(refresh: bool = False) -> list[dict]:
    """作物列表（含病害数）。首次访问或 refresh=True 时从 Milvus 拉取。"""
    return build_crop_list(_rows_cached(refresh))
