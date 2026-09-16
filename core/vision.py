# vision.py — 图像理解模块(方案B): 图片 → 结构化症状描述文本
# 说明: 独立 VL 客户端, 不污染主对话的 llm.py 流式实例;
#       输出为"客观症状描述"而非直接结论, 结论交给下游 RAG Agent 结合知识库给出。
import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()   # 本模块 import 时即读取 OPENAI_API_KEY/VL_MODEL_NAME，须先加载 .env

logger = logging.getLogger("vision")

# 独立非流式 VL 客户端(图片理解专用)
_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
VL_MODEL = os.getenv("VL_MODEL_NAME", "").strip()
DESCRIBE_TIMEOUT = 30


DESCRIBE_PROMPT = """你是一名农业植保专家。请仔细观察用户提供的图片，输出一份**客观、结构化**的症状描述，供后续检索知识库判断病虫害使用。

要求:
1. 只描述图片中客观可见的信息(部位/颜色/形状/大小/分布/数量), 不要直接给出病虫害名称结论;
2. 若图片不清晰或无法判断, 如实说明, 不要编造;
3. 若图中没有植物/病虫害相关对象, 说明你看到的内容;
4. 用简洁的中文分条输出, 控制在 150 字以内。
示例输出:
- 部位: 叶片正面
- 症状: 圆形黄褐色病斑, 边缘有黄色晕圈, 直径约 3~5mm
- 分布: 老叶较多, 新叶未见
- 其他: 无明显虫体"""


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
        logger.info("describe_image done len=%d", len(text))
        return text
    except Exception as e:
        logger.warning("describe_image failed: %s", e)
        return ""
