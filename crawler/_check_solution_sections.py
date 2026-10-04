"""排查"确诊率 75% 而方案率只有 37.5%"这个缺口到底出在哪一层。

三种可能的层，各有对应的检查手段：

  数据层 —— KB 里根本没有该病害的防治内容
      默认模式：直接查 Milvus 数每个病害各章节的切片数（确定性，不依赖 LLM）。
      `--text`：把「防治方法」原文打出来，判断里面有没有可执行内容
      （药剂名 / 稀释倍数 / 喷药频次）—— 只有 1 条切片也不能说明它有内容。

  生成层 —— 内容在库里、模型也拿到了，但没写成合格方案
      若前两项都正常，就只能落在这里。

  判定层 —— 方案其实给了，但评测脚本看不见
      `--plans`：J_SOLU 只截取**最后一轮**的回复做判断。若 agent 在第 3~4 轮
      就给了方案、第 5 轮改成回答农民的追问，那份方案就被漏掉了。
      这个模式逐轮扫方案特征，把"测量盲区"和"确实没给"分开。

用法：
    python crawler/_check_solution_sections.py            # 章节切片数 + 与方案成败交叉
    python crawler/_check_solution_sections.py --text     # 防治方法原文
    python crawler/_check_solution_sections.py --plans    # 方案出现在第几轮
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))

from pymilvus import MilvusClient  # noqa: E402

from core.config import (  # noqa: E402
    KB_COLLECTION,
    SECTION_CONTROL,
    SECTION_SYMPTOM,
)

DATA = os.path.join(ROOT, "crawler", "data")
RESULT = os.path.join(DATA, "multiturn_40_baseline.jsonl")

client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
client.load_collection(KB_COLLECTION)


def chunks(crop: str, disease: str, section: str) -> int:
    """该病害该章节有多少条切片。

    crop/disease 直接取自结果文件 —— 它们本来就是喂给评测的同一个值，
    与入库时的写法一致，所以这里不需要做任何归一化。
    """
    expr = f'crop == "{crop}" and disease == "{disease}" and section == "{section}"'
    return client.query(KB_COLLECTION, filter=expr, output_fields=["count(*)"])[0][
        "count(*)"
    ]


# 判定「这一轮的回复里有没有可执行方案」的特征词。
# 取"倍液/可湿性粉剂/乳油/悬浮剂"这类制剂术语，而不是"防治/药剂"这种泛词 ——
# 泛词在追问轮里也会出现，会把盲区判成"给了方案"。
_PLAN_HINT = re.compile(
    r"倍液|可湿性粉剂|乳油|悬浮剂|水分散|微乳剂|每隔|喷\s*\d\s*次|喷药|稀释"
)


def main() -> None:
    rows = [json.loads(l) for l in open(RESULT, encoding="utf-8") if l.strip()]

    if "--plans" in sys.argv:
        blind = real_miss = 0
        for r in rows:
            turns = r.get("turns") or []
            hits = [t["turn"] for t in turns
                    if _PLAN_HINT.search(t.get("agent") or "")]
            last_has = bool(turns) and bool(
                _PLAN_HINT.search(turns[-1].get("agent") or "")
            )
            ok = bool(r.get("solution_ok"))
            is_blind = (not ok) and bool(hits) and not last_has
            blind += is_blind
            real_miss += (not ok) and not is_blind
            print(f"{'OK' if ok else 'NG':<4}{r['crop']:<6}{str(hits):<12}"
                  f"{'← 方案在更早轮次，判定看不到' if is_blind else ''}")
        failed = sum(1 for r in rows if not r.get("solution_ok"))
        print("-" * 62)
        print(f"方案失败 {failed} = 判定盲区 {blind} + 确实没给 {real_miss}")
        if blind:
            print("  ↑ 盲区那部分说明 37.5% 低估了真实方案率，"
                  "修 J_SOLU 的取轮逻辑比改 prompt 更划算")
        return

    if "--text" in sys.argv:
        # 直接读「防治方法」原文：方案失败到底是"库里没有可执行内容"还是
        # "有内容但模型没写成方案"，只有读了原文才能判断 ——
        # 前者是数据问题（检索救不回来，/diagnose 的 plan[] 也救不回来），
        # 后者是生成问题（换个交付方式就能好）。
        for crop, disease in dict.fromkeys((r["crop"], r["truth"]) for r in rows):
            got = client.query(
                KB_COLLECTION,
                filter=f'crop == "{crop}" and disease == "{disease}" '
                       f'and section == "{SECTION_CONTROL}"',
                output_fields=["text"], limit=1,
            )
            text = (got[0]["text"] if got else "").replace("\n", " ")
            print(f"\n【{crop} / {disease}】len={len(text)}\n{text[:420]}")
        return

    print(f"{'方案':<4}{'作物':<6}{'病害':<24}{'症状':>5}{'防治':>5}")
    print("-" * 62)

    no_ctrl, ctrl_but_failed, no_sym = [], [], []
    for r in rows:
        ok = bool(r.get("solution_ok"))
        n_sym = chunks(r["crop"], r["truth"], SECTION_SYMPTOM)
        n_ctl = chunks(r["crop"], r["truth"], SECTION_CONTROL)
        if n_ctl == 0:
            no_ctrl.append(r)
        elif not ok:
            ctrl_but_failed.append(r)
        if n_sym == 0:
            no_sym.append(r)
        print(f"{'OK' if ok else 'NG':<4}{r['crop']:<6}{r['truth']:<24}"
              f"{n_sym:>5}{n_ctl:>5}")

    n = len(rows)
    failed = sum(1 for r in rows if not r.get("solution_ok"))
    print("-" * 62)
    print(f"样本 {n}   方案成功 {n - failed}   方案失败 {failed}")
    print(f"防治章节为空            : {len(no_ctrl)}/{n}"
          f"（其中方案失败 {sum(1 for r in no_ctrl if not r.get('solution_ok'))}）"
          f"  <- 数据缺失，检索救不回来")
    print(f"防治章节非空但方案失败  : {len(ctrl_but_failed)}/{n}"
          f"  <- 内容在库里却没写成方案，属生成/判定问题")
    print(f"危害症状章节为空        : {len(no_sym)}/{n}")


if __name__ == "__main__":
    main()
