# -*- coding: utf-8 -*-
"""
agri_eval_map.py — 农业检索评测的类别映射表(评测地基)

作用: 把 PlantVillage 的英文类名映射到 kb_agri 的 {作物, 病害} 中文名,
      用于判定"检索结果是否命中正确病害"。

统一格式: 类名 -> (作物, 病害名, 置信度) | NO_MATCH
  · 病害名 = HEALTHY 表示健康对照(仍带作物, 用于测"健康样本会不会被误报病害")
  · NO_MATCH 表示 kb_agri 无对应条目, 不纳入评测

confidence:
  high   = 名称明确对应, 可直接用于正式指标
  medium = 语义近似, 建议人工抽检后再引用

用法:
    python crawler/agri_eval_map.py        # 校验映射是否都存在于 kb_agri, 输出覆盖度报告
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HEALTHY = "__HEALTHY__"
NO_MATCH = None

# PlantVillage 类名 -> (作物, 病害/HEALTHY, 置信度)
MAP: dict[str, tuple | None] = {
    # ===== 番茄 (kb_agri 84 种) =====
    "Tomato___Bacterial_spot": ("番茄", "番茄细菌性斑疹病", "high"),
    "Tomato___Early_blight": ("番茄", "番茄早疫病", "high"),
    "Tomato___Late_blight": ("番茄", "番茄晚疫病", "high"),
    "Tomato___Leaf_Mold": ("番茄", "番茄叶霉病", "high"),
    "Tomato___Septoria_leaf_spot": ("番茄", "番茄斑枯病", "high"),
    "Tomato___Tomato_mosaic_virus": ("番茄", "番茄花叶病毒病", "high"),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("番茄", "朱砂叶螨", "medium"),
    "Tomato___Target_Spot": NO_MATCH,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": NO_MATCH,
    "Tomato___healthy": ("番茄", HEALTHY, "high"),

    # ===== 玉米 (kb_agri 59 种) =====
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("玉米", "玉米灰斑病", "high"),
    "Corn_(maize)___Common_rust_": ("玉米", "玉米锈病", "high"),
    "Corn_(maize)___Northern_Leaf_Blight": ("玉米", "玉米大斑病", "high"),
    "Corn_(maize)___healthy": ("玉米", HEALTHY, "high"),

    # ===== 马铃薯 =====
    "Potato___Early_blight": ("马铃薯", "马铃薯早疫病", "high"),
    "Potato___Late_blight": ("马铃薯", "马铃薯晚疫病", "high"),
    "Potato___healthy": ("马铃薯", HEALTHY, "high"),

    # ===== 苹果 =====
    "Apple___Apple_scab": ("苹果", "苹果黑星病", "high"),
    "Apple___Black_rot": ("苹果", "苹果黑腐病", "high"),
    "Apple___Cedar_apple_rust": ("苹果", "苹果锈病", "high"),
    "Apple___healthy": ("苹果", HEALTHY, "high"),

    # ===== 葡萄 =====
    # Esca(葡萄干枯病/黑麻疹) 在 kb_agri 中【无对应条目】, 故不纳入评测。
    # 历史: 曾被以 medium 近似映射到"葡萄轮斑病"。人工核对全部 20 张图后确认症状不符 ——
    #   图为 红褐色坏死沿【叶脉间】分布、受叶脉限制呈不规则块状(典型"虎纹状"叶症);
    #   而 葡萄轮斑病 为【圆形】病斑 + 【同心环纹】 + 反面【浅褐色霉层】。三项全不符。
    # 副作用: 该映射使葡萄在历次评测中系统性偏低(检索@20 仅 40%, 多轮测试 0/4 确诊),
    #         属评测集问题而非系统问题, 故整体剔除而非改用其他近似条目。
    "Grape___Esca_(Black_Measles)": NO_MATCH,
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("葡萄", "葡萄叶斑病", "high"),
    "Grape___Black_rot": NO_MATCH,
    "Grape___healthy": ("葡萄", HEALTHY, "high"),

    # ===== 桃 / 辣椒 / 草莓 =====
    "Peach___Bacterial_spot": ("桃", "桃细菌性穿孔病", "high"),
    "Peach___healthy": ("桃", HEALTHY, "high"),
    "Pepper,_bell___Bacterial_spot": ("辣椒", "辣椒疮痂病", "medium"),
    "Pepper,_bell___healthy": ("辣椒", HEALTHY, "high"),
    "Strawberry___Leaf_scorch": ("草莓", "草莓蛇眼病", "high"),
    "Strawberry___healthy": ("草莓", HEALTHY, "high"),

    # ===== 西葫芦 / 柑橘 =====
    "Squash___Powdery_mildew": ("西葫芦", "西葫芦白粉病", "high"),
    "Orange___Haunglongbing_(Citrus_greening)": ("柑橘", "柑橘黄龙病", "high"),

    # ===== 樱桃: 库里有该作物但无白粉病; healthy 仍可测误报 =====
    "Cherry___Powdery_mildew": NO_MATCH,
    "Cherry_(including_sour)___healthy": ("樱桃", HEALTHY, "high"),

    # ===== kb_agri 无此作物 =====
    "Blueberry___healthy": NO_MATCH,
    "Raspberry___healthy": NO_MATCH,
    "Soybean___healthy": ("大豆", HEALTHY, "high"),
}


def validate():
    """校验映射中的病害/作物是否真实存在于 kb_agri, 输出覆盖度报告"""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URI", "http://localhost:19530"))
    client.load_collection("kb_agri")
    rows = client.query("kb_agri", filter="",
                        output_fields=["crop", "disease"], limit=16384)
    exist = {(r.get("crop"), r.get("disease")) for r in rows}
    crops = {r.get("crop") for r in rows}

    ok, bad = [], []
    healthy_n = nomatch_n = 0
    for cls, v in MAP.items():
        if v is NO_MATCH:
            nomatch_n += 1
            continue
        crop, disease, conf = v
        if disease == HEALTHY:
            healthy_n += 1
            if crop not in crops:
                bad.append((cls, crop, "(healthy)", conf))
            else:
                ok.append((cls, crop, "(healthy)", conf))
        elif (crop, disease) in exist:
            ok.append((cls, crop, disease, conf))
        else:
            bad.append((cls, crop, disease, conf))

    print("=" * 72)
    print(f"映射总类数: {len(MAP)}   纳入评测: {len(ok) + len(bad)}   "
          f"无对应: {nomatch_n}")
    print(f"  其中 病害样本: {len(ok) + len(bad) - healthy_n}   "
          f"健康样本: {healthy_n}")
    print("=" * 72)

    print(f"\n[校验通过] {len(ok)} 类:")
    for cls, crop, disease, conf in sorted(ok, key=lambda x: (x[1], x[2])):
        flag = "OK " if conf == "high" else "中等"
        print(f"  [{flag}] {cls:52s} -> {crop}/{disease}")

    if bad:
        print(f"\n[校验失败] {len(bad)} 类 (kb_agri 中不存在, 需修正):")
        for cls, crop, disease, conf in bad:
            hint = "" if crop in crops else " (作物不存在)"
            print(f"   ! {cls:52s} -> {crop}/{disease}{hint}")

    print(f"\n可用于评测的类: {len(ok)}")


if __name__ == "__main__":
    validate()
