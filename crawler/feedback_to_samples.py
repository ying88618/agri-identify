import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))
from api.chat import CROP_NOTE_PREFIX
from api.models import Feedback, SessionLocal

OUT = os.path.join(ROOT, "crawler", "data", "vl_desc_feedback.jsonl")

# 首条消息里的图片描述块。格式由 api/chat.py 的 _build_user_text 决定。
_IMG_BLOCK = re.compile(r"图片内容描述如下：\s*\n(?P<desc>.*?)\n\s*\n用户问题：", re.S)
# 作物标记。直接复用 api.chat 的前缀常量
_CROP_NOTE = re.compile(re.escape(CROP_NOTE_PREFIX) + r"(?P<crop>[^。\n\r]{1,32})。")

def parse_first_turn(text: str) -> tuple[str | None, str | None]:
    """从首条 user 消息解析出 (desc, crop)。解析不到就返回 None。"""
    desc = crop = None
    m = _IMG_BLOCK.search(text or "")
    if m:
        desc = m.group("desc").strip() or None
    m = _CROP_NOTE.search(text or "")
    if m:
        crop = m.group("crop").strip() or None
    return desc, crop

def to_sample(row: Feedback) -> tuple[dict | None, str | None]:
    """Feedback 行 -> 样本行。返回 (样本, 跳过原因)，两者恰有一个非 None。"""
    if not row.snapshot:
        return None, "no_snapshot"
    if not row.correct_disease:
        return None, "no_truth"
    first = next(
        (t.get("content") or "" for t in row.snapshot if t.get("role") == "user"), ""
    )
    desc, crop = parse_first_turn(first)
    if not desc:
        return None, "no_desc"
    return {
        "cls": "",     # 真实用户照片无 PlantVillage 标签，故留空
        "crop": crop or "",
        "truth": row.correct_disease,
        "img": "",
        "desc": desc,
        "source": f"feedback:{row.id}",   # 额外字段，eval_multiturn 会忽略
    }, None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"追加写入 {OUT}")
    args = ap.parse_args()

    db = SessionLocal()
    try:
        rows = db.query(Feedback).filter(Feedback.verdict == "wrong").all()
    finally:
        db.close()

    samples, skipped, seen = [], Counter(), set()
    for row in rows:
        s, why = to_sample(row)
        if s is None:
            skipped[why] += 1
            continue
        # 同一张照片被多个用户报错 -> desc 相同 -> 只留一条，避免样本集被重复项拉偏
        key = (s["crop"], s["truth"], s["desc"])
        if key in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(key)
        samples.append(s)

    print(f"反馈记录(verdict=wrong): {len(rows)}")
    print(f"可入样本集: {len(samples)}")
    for why, n in sorted(skipped.items()):
        print(f"  跳过 {why}: {n}")
    if not samples:
        return

    if args.write:
        # 与已落盘的内容再按 (crop, truth, desc) 去重：--write 是**追加**语义，
        # 跑第二次会把同一批 badcase 再写一遍 —— 样本集被重复项拉偏且不报错。
        existing = set()
        if os.path.exists(OUT):
            with open(OUT, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        existing.add((e.get("crop"), e.get("truth"), e.get("desc")))
        new = [s for s in samples if (s["crop"], s["truth"], s["desc"]) not in existing]
        if len(new) < len(samples):
            print(f"  已在文件中，跳过 {len(samples) - len(new)} 条")
        with open(OUT, "a", encoding="utf-8") as f:
            for s in new:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"\n新增 {len(new)} 条 → {OUT}")
        print("重跑评测（基线组与反馈组会分开报数）：")
        print("  python crawler/eval_multiturn.py")
    else:
        print("\n前 3 条预览（加 --write 才落盘）:")
        for s in samples[:3]:
            print(" ", json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()