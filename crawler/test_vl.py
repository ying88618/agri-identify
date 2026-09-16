# -*- coding: utf-8 -*-
"""
test_vl.py — 视觉模型病虫害识别评测(PlantVillage)

用法:
    python crawler/test_vl.py                       # 默认: 番茄+玉米+马铃薯, 每类 15 张
    python crawler/test_vl.py --per-class 5         # 每类只测 5 张(快速验证)
    python crawler/test_vl.py --pattern Tomato      # 只看番茄
    python crawler/test_vl.py --model Qwen/Qwen3-VL-32B-Instruct   # 换更强的模型

评测: Top-1 / Top-5 准确率(closed-set 分类, 类别=PlantVillage 英文类名)
输出: crawler/data/vl_result.jsonl
"""
import argparse
import asyncio
import base64
import glob
import json
import os

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

IMG_ROOT = r"E:\RAG_agri\PlantVillage-Dataset-master\color"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# 默认覆盖的作物模式(文件夹前缀)
DEFAULT_PATTERNS = ("Tomato", "Corn_(maize)", "Potato")
CONCURRENCY = 2        # 图片 token 大, 并发要低, 防 TPM 限流

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      base_url=os.getenv("OPENAI_BASE_URL"))


def pick_classes(patterns) -> list[tuple[str, list[str]]]:
    """返回 [(类名, [图片路径...]), ...], 按 pattern 过滤"""
    result = []
    for cls_dir in sorted(glob.glob(os.path.join(IMG_ROOT, "*"))):
        if not os.path.isdir(cls_dir):
            continue
        name = os.path.basename(cls_dir)
        if not any(name.startswith(p) for p in patterns):
            continue
        imgs = [f for f in glob.glob(os.path.join(cls_dir, "*"))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            result.append((name, imgs))
    return result


def to_b64_dataurl(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def norm(s: str) -> str:
    """归一化用于匹配(去空白/引号/换行)"""
    return "".join(ch for ch in (s or "").lower() if ch.isalnum() or ch == "_")


def build_prompt(classes: list[str]) -> str:
    lst = "\n".join(f"- {c}" for c in classes)
    return (
        "你是一名农业植保专家。这是一张植物叶片/植株照片。\n"
        "请从下列病害类别中鉴定出最可能的类别。\n"
        "规则: 先输出你最有把握的 1 个类别名, 若不确定则继续输出次可能的类别, 最多 5 个, 用逗号分隔。\n"
        "只输出类别名(与列表完全一致), 不要解释, 不要补充说明。\n\n"
        f"类别列表:\n{lst}"
    )


async def classify(model: str, img: str, classes: list[str]):
    resp = await _client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": build_prompt(classes)},
            {"type": "image_url", "image_url": {"url": to_b64_dataurl(img)}},
        ]}],
        max_tokens=120,
    )
    return resp.choices[0].message.content or ""


def topk_hit(pred_raw: str, truth: str) -> tuple[bool, bool]:
    """解析预测, 返回 (top1命中, top5命中)"""
    truth_n = norm(truth)
    parts = [norm(p) for p in pred_raw.replace("，", ",").split(",") if p.strip()]
    if not parts:
        return False, False
    return (parts[0] == truth_n), (truth_n in parts[:5])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--pattern", default=",".join(DEFAULT_PATTERNS))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()
    patterns = [p for p in args.pattern.split(",") if p]

    classes = pick_classes(patterns)
    total_imgs = sum(len(v) for _, v in classes)
    print(f"类别: {len(classes)} 类, 图片总数可用: {total_imgs}, 每类抽 {args.per_class} 张")

    sem = asyncio.Semaphore(CONCURRENCY)
    per_class_ok = {}

    async def run(cls_name, img):
        async with sem:
            try:
                pred = await classify(args.model, img, [c for c, _ in classes])
                t1, t5 = topk_hit(pred, cls_name)
                return cls_name, os.path.basename(img), pred, t1, t5
            except Exception as e:
                return cls_name, os.path.basename(img), f"(ERROR {e})", False, False

    tasks = []
    for cls_name, imgs in classes:
        sample = imgs[: args.per_class]
        tasks += [run(cls_name, p) for p in sample]

    total = len(tasks)
    ok1 = ok5 = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "vl_result.jsonl"), "w", encoding="utf-8") as f:
        for i in range(0, total, 10):
            batch = tasks[i : i + 10]
            for cls_name, fname, pred, t1, t5 in await asyncio.gather(*batch):
                ok1 += t1
                ok5 += t5
                per_class_ok.setdefault(cls_name, [0, 0])  # (top1数, 总数)
                per_class_ok[cls_name][1] += 1
                per_class_ok[cls_name][0] += t1
                f.write(json.dumps({"class": cls_name, "img": fname,
                                    "pred": pred, "top1": t1, "top5": t5},
                                   ensure_ascii=False) + "\n")
            print(f"[评测] {min(i + 10, total)}/{total}")
            await asyncio.sleep(0.2)

    print(f"\n===== VL 识别评测 (模型 {args.model}) =====")
    print(f"样本: {total} 张, Top-1 准确率: {ok1 / total:.1%}, Top-5: {ok5 / total:.1%}")
    print("按类别 Top-1:")
    for cls, (ok, n) in sorted(per_class_ok.items()):
        print(f"  {cls}: {ok}/{n} = {ok / n:.0%}" if n else f"  {cls}: 0")


if __name__ == "__main__":
    asyncio.run(main())
