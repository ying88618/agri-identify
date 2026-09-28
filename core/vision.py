# vision.py — 图像理解模块(方案B): 图片 → 结构化症状描述文本
# 说明: 独立 VL 客户端, 不污染主对话的 llm.py 流式实例;
#       输出为"客观症状描述"而非直接结论, 结论交给下游 RAG Agent 结合知识库给出。
import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .config import DESCRIBE_PROMPT_VERSION

load_dotenv()   # 本模块 import 时即读取 OPENAI_API_KEY/VL_MODEL_NAME，须先加载 .env

logger = logging.getLogger("vision")

# 独立非流式 VL 客户端(图片理解专用)
_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
VL_MODEL = os.getenv("VL_MODEL_NAME", "").strip()
DESCRIBE_TIMEOUT = 30

DESCRIBE_PROMPT = """你是一名农业植保专家。请看这张照片，用中文客观描述你在图上【实际看到】的症状特征，供后续检索病害资料。

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
5. 只有当叶片确实完全健康、毫无症状时，才回复这六个字：无明显病症；
6. 若图中没有植物叶片、或图片过于模糊无法辨认，就如实说明你看到了什么，不要编造症状。"""


async def describe_image(image_url: str, question: str = "", max_len: int = 300) -> str:
    """调用 VL 模型把图片转成文本描述; 失败或模型未配置时返回空串(不阻断主流程)。

    返回: 描述文本; 若 image_url 为空 / VL_MODEL 未配置 / 调用失败, 返回 ""。
    """
    if not image_url or not VL_MODEL:
        logger.info("describe_image skipped (image_url=%s vl_model=%s)", bool(image_url), VL_MODEL)
        return ""
    try:
        text_part = DESCRIBE_PROMPT
        if question:
            text_part += f"\n\n用户的问题(供参考): {question}"
        resp = await asyncio.wait_for(
            _client.chat.completions.create(
                model=VL_MODEL,
                temperature=0.1,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": text_part},
                        ],
                    }
                ],
            ),
            timeout=DESCRIBE_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if len(text) > max_len:
            text = text[:max_len]
        logger.info("describe_image done len=%d prompt_v=%d",
                    len(text), DESCRIBE_PROMPT_VERSION)
        return text
    except Exception as e:
        logger.warning("describe_image failed: %s", e)
        return ""
