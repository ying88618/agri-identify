# -*- coding: utf-8 -*-
"""
gen_vl_desc.py — 阶段1: 采样 PlantVillage 图片, 调 VL 生成症状描述并缓存

为什么要独立成阶段:
  VL 调用慢且贵(每张图数秒), 而检索评测要反复跑多种配置(向量/混合 × 是否crop过滤)。
  先把描述生成一次并落盘缓存, 之后调检索参数就不用再花 VL 的钱。

用法:
    python crawler/gen_vl_desc.py --per-class 5
    python crawler/gen_vl_desc.py --per-class 5 --pattern Tomato,Corn   # 只跑部分类
    python crawler/gen_vl_desc.py --per-class 5 --model Qwen/Qwen3-VL-32B-Instruct

输出: crawler/data/vl_desc_cache.jsonl
      {"cls", "crop", "truth", "img", "desc"}
      · truth="__HEALTHY__" 表示健康对照
      · 支持断点续传: 已有缓存 (cls+img) 会跳过
"""
import argparse
import asyncio
import base64
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI

from agri_eval_map import MAP, HEALTHY, NO_MATCH

IMG_ROOT = r"E:\RAG_agri\PlantVillage-Dataset-master\color"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CONCURRENCY = 2          # 图片 token 大, 低并发防 TPM 限流
SEED = 42                # 固定随机种子, 保证采样可复现


def out_path(model: str, pv: int) -> str:
    """按 模型 + prompt版本 分文件缓存: 便于横向对比"""
    slug = model.replace("/", "_").replace(":", "_")
    return os.path.join(DATA_DIR, f"vl_desc_{slug}_v{pv}.jsonl")

_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                      base_url=os.getenv("OPENAI_BASE_URL"))

# ============ prompt v1 (原始版, 保留用于对比) ============
PROMPT_V1 = """你是一名农业植保专家。请仔细观察这张叶片/植株照片，用中文客观描述可见症状，供后续病害诊断参考。

请按以下结构输出：
1) 作物判断与整体状态（是否健康、萎蔫、发黄等）
2) 病斑细节：颜色、形状、边缘特征、分布位置
3) 表面附着物：有无霉层/粉状物/孢子堆/虫体/网丝等
4) 若叶片健康无明显病症，直接说明"无明显病症"

要求：用农业术语客观描述，但【不要臆断或说出具体病害名称】，控制在 150 字内。"""

# ============ prompt v2 (修复版) ============
# v1 的三个问题:
#   a) 第4条被当成必填项 → 带病叶片也照抄"无明显病症", 污染 query
#   b) 未强制输出判别特征(尺寸/是否受叶脉限制/边缘晕圈) → 描述笼统不可分
#   c) 让模型判断作物 → 实测会认错(番茄认成马铃薯), 污染 query
PROMPT_V2 = """你是一名农业植保专家。请看这张叶片照片，只用中文客观描述【肉眼可见的症状特征】，供后续检索病害资料。

先判断：如果整片叶子完全健康、没有任何异常，就只回复这四个字：无明显病症

否则，严格按下面每一行输出（看不清就写"未见"，不要省略任何一行）：
- 病斑大小：___mm（或写"针尖大小/小于1mm/大于1cm"）
- 病斑形状：圆形/近圆形/不规则形/多角形/梭形/长条形/水渍状
- 病斑颜色：中心___色，边缘___色
- 边缘特征：有无黄色晕圈/有无隆起/界线是否清晰
- 是否受叶脉限制：受叶脉限制呈多角形 / 不受叶脉限制
- 分布位置：叶尖/叶缘/叶脉间/叶面中上部/老叶/新叶
- 表面附着物：有无霉层(颜色)/粉状物/锈色孢子堆/黑色小点/虫体/网丝
- 整体状态：是否萎蔫/卷曲/黄化/枯死

严格要求：
1. 只描述你确实看到的，不要推测是什么病；
2. 【禁止】说出任何病害名称；
3. 【禁止】提及作物种类；
4. 总字数 120 字以内。"""

# ============ prompt v3 (在 v2 基础上修正) ============
# v2 实测教训:
#   a) 强制逐条填字段 → 模型量不准就填默认值, 所有病害描述同质化(模板塌缩)
#   b) 不适用字段被填"未见" → query 变噪声
#   c) 健康判断置于开头 → 病害被误判为健康(4/44)
# v3 对策: 保留自然语言 + 只要求"描述能看到的" + 健康判断置尾且加严 + 禁用"未见"灌水
PROMPT_V3 = """你是一名农业植保专家。请看这张叶片照片，用中文客观描述你在图上【实际看到】的症状特征，供后续检索病害资料。

请用一段通顺的话描述（不要逐条列字段、不要分行罗列），尽量包含你确实观察到的以下信息：
· 病斑的大小与形状
· 病斑中心色、边缘色，边缘是否有黄色晕圈或隆起
· 是否受叶脉限制（例如呈多角形）
· 分布位置（叶尖/叶缘/叶脉间/老叶/新叶）
· 表面是否有霉层、粉状物、锈色孢子堆、黑色小点、虫体或网丝
· 叶片整体状态（萎蔫/卷曲/黄化/枯死）

严格要求：
1. 只写你确实看到的，不确定的就不写；【禁止】写"未见""无异常"等占位词；
2. 【禁止】说出任何病害名称；
3. 【禁止】提及作物种类；
4. 总字数 100 字以内；
5. 只有当叶片确实完全健康、毫无症状时，才回复这六个字：无明显病症。"""

# ============ prompt v4 (在 v3 基础上修正) ============
# v3 实测教训: 逐条列判别特征时用了"带示例"的写法 ——
#     "· 是否受叶脉限制（例如呈多角形）"
#   模型把这个示例当成了标准答案照抄。实测 620 条描述里 59.4% 都写了"多角形"
#   (番茄类 76%, 番茄斑枯/玉米灰斑/马铃薯早疫/苹果锈病/草莓蛇眼 均 100%)。
#   一条特征若出现在六成描述里, 就不再具备区分度, 反而把检索推向知识库里
#   同样用"多角形/受叶脉限制"措辞的【错误】病害 —— 这是番茄类召不回的根因之一。
# v4 对策: 去掉括号里的示例; 并明确"看不出形状就说形状不明显, 不要硬猜"。
PROMPT_V4 = """你是一名农业植保专家。请看这张叶片照片，用中文客观描述你在图上【实际看到】的症状特征，供后续检索病害资料。

请用一段通顺的话描述（不要逐条列字段、不要分行罗列），只写你确实观察到的以下信息：
· 病斑的大小与形状（若病斑细小/模糊、判断不出形状，就写"形状不明显"，不要硬套）
· 病斑中心色、边缘色，边缘是否有黄色晕圈或隆起
· 是否受叶脉限制
· 分布位置（叶尖/叶缘/叶脉间/老叶/新叶）
· 表面是否有霉层、粉状物、锈色孢子堆、黑色小点、虫体或网丝
· 叶片整体状态（萎蔫/卷曲/黄化/枯死）

严格要求：
1. 只写你确实看到的，不确定的就不写；【禁止】写"未见""无异常"等占位词；
2. 【禁止】说出任何病害名称；
3. 【禁止】提及作物种类；
4. 总字数 100 字以内；
5. 只有当叶片确实完全健康、毫无症状时，才回复这六个字：无明显病症。"""

# ============ prompt v5 (在 v4 基础上修正) ============
# v4 实测教训: 去掉"多角形"示例后, 该词从 76.2% 降到 0.0%, 但"受叶脉限制"仍高达 73.8%
#   (与 v3 完全一致)。说明根因不只是"示例被照抄", 而是【清单式提示本身】:
#   模型会把列出的每一条都"填上", 以显得描述完整 —— 即便图上并不明显。
#   一条出现在 74% 描述里的"特征"没有区分度, 只会把检索推向同样用该措辞的错误病害。
# v5 对策: 删掉"是否受叶脉限制"这一条; 并明确告知"各项只是提示方向, 不要求逐条覆盖,
#          图上不明显就不要写"。
PROMPT_V5 = """你是一名农业植保专家。请看这张叶片照片，用中文客观描述你在图上【实际看到】的症状特征，供后续检索病害资料。

请用一段通顺的话描述（不要逐条列字段、不要分行罗列）。下面几点只是提示可以从哪些角度观察，**不是必须逐条覆盖**；图上不明显的角度就跳过，不要为了写全而硬凑：
· 病斑的大小与形状（若病斑细小/模糊、判断不出形状，就写"形状不明显"，不要硬套）
· 病斑中心色、边缘色，边缘是否有黄色晕圈或隆起
· 分布位置（叶尖/叶缘/叶脉间/老叶/新叶）
· 表面是否有霉层、粉状物、锈色孢子堆、黑色小点、虫体或网丝
· 叶片整体状态（萎蔫/卷曲/黄化/枯死）

严格要求：
1. 只写你确实看到的，不确定的就不写；【禁止】写"未见""无异常"等占位词；
2. 【禁止】说出任何病害名称；
3. 【禁止】提及作物种类；
4. 总字数 100 字以内；
5. 只有当叶片确实完全健康、毫无症状时，才回复这六个字：无明显病症。"""

PROMPTS = {1: PROMPT_V1, 2: PROMPT_V2, 3: PROMPT_V3, 4: PROMPT_V4, 5: PROMPT_V5}


def to_b64_dataurl(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def load_cache(path: str) -> set[tuple[str, str]]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["cls"], r["img"]))
    return done


async def describe(model: str, img_path: str, prompt: str) -> str:
    resp = await _client.chat.completions.create(
        model=model, temperature=0.0, max_tokens=300,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": to_b64_dataurl(img_path)}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--pattern", default="", help="按类名前缀过滤, 逗号分隔")
    ap.add_argument("--model", default=os.getenv("VL_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct"))
    ap.add_argument("--pv", type=int, default=3, help="prompt 版本: 1=原始, 2=结构化, 3=自然语言修正版")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    if args.pv not in PROMPTS:
        raise SystemExit(f"未知 prompt 版本: {args.pv}, 可选 {list(PROMPTS)}")
    prompt = PROMPTS[args.pv]

    OUT = out_path(args.model, args.pv)
    rng = random.Random(SEED)
    done = load_cache(OUT)
    if done:
        print(f"已有缓存 {len(done)} 条, 将跳过重复 (cls+img)")

    # 组装待跑任务
    tasks = []
    for cls, v in MAP.items():
        if v is NO_MATCH:
            continue
        if args.pattern and not any(cls.startswith(p) for p in args.pattern.split(",") if p):
            continue
        cls_dir = os.path.join(IMG_ROOT, cls)
        if not os.path.isdir(cls_dir):
            print(f"  [跳过] 目录不存在: {cls}")
            continue
        imgs = sorted(f for f in glob.glob(os.path.join(cls_dir, "*"))
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if len(imgs) > args.per_class:
            imgs = rng.sample(imgs, args.per_class)
        crop, truth, _conf = v      # healthy 也带作物, 便于评测误报
        for p in imgs:
            name = os.path.basename(p)
            if (cls, name) in done:
                continue
            tasks.append((cls, crop, truth, p))

    print(f"待生成: {len(tasks)} 张 (模型 {args.model}, prompt v{args.pv}, 并发 {args.concurrency})")
    sem = asyncio.Semaphore(args.concurrency)
    ok = fail = 0

    os.makedirs(DATA_DIR, exist_ok=True)
    fout = open(OUT, "a", encoding="utf-8")

    async def run(cls, crop, truth, p):
        nonlocal ok, fail
        async with sem:
            desc = None
            for attempt in range(3):        # 大规模跑必须有重试, 否则限流/抖动会丢样本
                try:
                    desc = await describe(args.model, p, prompt)
                    break
                except Exception as e:
                    if attempt == 2:
                        fail += 1
                        print(f"  [失败] {cls}/{os.path.basename(p)}: {type(e).__name__}: {e}")
                        return
                    await asyncio.sleep(2 * (attempt + 1))
            rec = {"cls": cls, "crop": crop, "truth": truth,
                   "img": os.path.basename(p), "desc": desc}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            ok += 1
            if ok % 20 == 0:
                print(f"  已生成 {ok}/{len(tasks)}")

    await asyncio.gather(*[run(*t) for t in tasks])
    fout.close()
    print(f"\n完成: 成功 {ok}, 失败 {fail} → {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
