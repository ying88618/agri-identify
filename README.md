# Agri-Agent · 农业病虫害识别与防治问答 Agent

面向农业病虫害的**多轮对话诊断 Agent**。农民上传叶片照片、用口语描述症状，Agent 结合私域知识库
（2940 条病虫害资料）给出病害判断与防治方案；鉴别点不足时**主动追问**，知识库无命中时联网兜底。

## 实测指标

评测集：PlantVillage 21 类 / 620 样本（病害 420 + 健康 200），完整留档见
[`crawler/data/PROJECT2_RESULTS.md`](crawler/data/PROJECT2_RESULTS.md)。

| 指标 | 结果 |
|---|---|
| 检索命中 @1 / @3 / @5 / @20 | 15.7% / 35.7% / **51.2%** / 67.6% |
| 健康样本误报率（top1 ≥ 0.5） | 6.5% |
| 多轮对话 5 轮内确诊（40 例） | **75.0%** |
| 首轮即确诊 | 40.0% |
| 防治方案合格率 | 65.0% |

> 线上配置：**v3 图片描述 + Qwen3-Reranker-8B + 阈值 0.5**。相比旧配置（bge-reranker + 0.25）
> @5 提升 **+12.2pt**（39.0% → 51.2%，McNemar p<0.0001），@20 持平（67.6%，救回 0 / 丢掉 0）
> —— 改善 100% 来自重排序，而非扩大召回。

## 功能

- **图片识别**：上传叶片照片转成客观文字描述（颜色 / 形状 / 部位 / 边缘 / 有无霉层），
  描述带缓存，同一 `image_id` 只调用一次 VL 模型
- **多轮问诊**：基于缺失的鉴别点主动追问（如"叶背有没有霉层"），而非只给一次性结论
- **混合检索**：向量召回 Top-100 + BM25 → RRF 融合 → Reranker 精排 → Top-6；
  支持 `crop` / `section` 过滤，rerank 失败自动降级
- **溯源**：SSE `sources` 事件回传命中文档来源
- **联网兜底**：知识库无相关内容时走 Tavily 搜索
- **MCP 服务**：知识库检索同时以标准 MCP 工具对外暴露，可挂载到任意 MCP 客户端
- **鉴权与隔离**：JWT + bcrypt；上传图片按 `user_id` 分目录存放

## 架构

```
上传图片 ──┐
          ├──→ FastAPI  POST /chat/stream (SSE)
文字提问 ──┘         │
                     ├─ VL 描述        core/vision.py（按 image_id 缓存）
                     ├─ 多轮记忆       core/memory.py（Redis，近 6 轮 + 首条常驻）
                     ↓
              LangGraph Agent           core/agent.py
                     ├─ knowledge_base_search ──→ 混合检索 core/retriever.py
                     │                              ├─ Milvus 向量召回 Top-100
                     │                              ├─ BM25 召回 → RRF 融合
                     │                              └─ Reranker 精排 → Top-6
                     └─ web_search ──→ Tavily 兜底
                     ↓
              流式回答 + sources 溯源事件
```

## 技术栈

| 层 | 选型 |
|---|---|
| Web | FastAPI + SSE（sse-starlette） |
| Agent | LangChain 1.x `create_agent` + LangGraph |
| 向量库 | Milvus 2.4.24（standalone + etcd + MinIO） |
| 关键词检索 | rank-bm25 + jieba 中文分词 |
| 重排 | Qwen3-Reranker-8B（OpenAI 兼容 `/rerank`） |
| Embedding | BAAI/bge-m3 |
| LLM / VL | OpenAI 兼容接口（模型名走 `.env`） |
| 对话记忆 | Redis 8（AOF 持久化） |
| 用户数据 | MySQL 8.0 + SQLAlchemy |
| 工具协议 | MCP（fastmcp） |
| 测试 | pytest |

## 目录结构

```
agri-agent/
├── main.py                    # FastAPI 入口，挂载 api/* 各 router
├── mcp_server.py              # MCP server：knowledge_search 工具
├── docker-compose.yml         # Milvus + etcd + MinIO + Redis
├── init_db.sql                # MySQL 库 / 账号 / 授权初始化
├── requirements.txt
├── api/
│   ├── auth.py                # 注册 / 登录
│   ├── chat.py                # SSE 流式问答（核心路由）
│   ├── files.py               # 图片上传
│   ├── kb.py                  # 作物目录
│   ├── deps.py                # Bearer 令牌 → user_id
│   ├── models.py              # SQLAlchemy 模型 + 引擎
│   └── security.py            # bcrypt / JWT
├── core/
│   ├── agent.py               # Agent 与工具定义（检索 / 联网）
│   ├── retriever.py           # 两阶段混合检索 + 过滤值净化
│   ├── vectorstore.py         # Milvus 客户端
│   ├── bm25_index.py          # BM25 索引（进程内缓存）
│   ├── vision.py              # VL 图片描述
│   ├── images.py              # 上传落盘 / 缩放 / 描述缓存
│   ├── memory.py              # Redis 多轮记忆
│   ├── llm.py                 # LLM 实例 + SYSTEM_PROMPT
│   ├── catalog.py             # 作物目录缓存
│   ├── sources.py             # 来源合并去重
│   ├── web_search.py          # Tavily 联网搜索
│   ├── self_correct.py        # 检索结果自校验
│   └── config.py              # 环境变量 + 全局常量（阈值唯一出口）
├── crawler/                   # 语料采集、入库与评测脚本
│   ├── crawl_agri.py          # 爬取「中国农业农村信息网」病虫害栏目
│   ├── excel_to_jsonl.py      # Excel 数据集 → agri_pests.jsonl
│   ├── bulk_ingest_agri.py    # 入库 Milvus
│   ├── eval_agri_recall.py    # 620 样本检索评测
│   ├── eval_multiturn.py      # 40 例多轮对话端到端评测
│   ├── exp_vl_rerank.py       # 多模态 rerank 实验
│   ├── gen_vl_desc.py         # VL 描述生成（含 v3/v4/v5 prompt 迭代注释）
│   ├── dump_agri.py           # 集合备份 / restore_agri.py 恢复
│   └── data/                  # 语料与评测留档（见下方「数据」）
└── tests/                     # pytest（检索 / 鉴权 / 记忆 / 召回契约）
```

## 快速开始

### 0. 前置

Python 3.11+、Docker Desktop、MySQL 8.0。

### 1. 启动基础设施

```bash
docker compose up -d          # Milvus 19530 / 9091，Redis 6380
```

> `docker-compose.yml` 里的 `name: knowledge` 是**故意固定**的，不要改名：Docker 卷名是
> `<项目名>_<卷名>`，改名会挂载到一套全新的空卷，现象是"collection 全没了"。

### 2. 初始化 MySQL

```bash
mysql -u root -p < init_db.sql
```

脚本会创建 `agri_agent` 库、`agri` 账号并授权（刻意不给 DROP 权限）。
**执行前请先把脚本里的 `<YOUR_DB_PASSWORD>` 换成你的密码**，并同步更新 `.env` 的 `DATABASE_URL`。

### 3. 配置环境变量

```bash
copy .env.example .env        # Windows
cp   .env.example .env        # Linux / macOS
```

必须填写：`OPENAI_API_KEY`、`MODEL_NAME`、`VL_MODEL_NAME`、`RERANK_MODEL`、`TAVILY_API_KEY`、
`JWT_SECRET`、`DATABASE_URL`。键名与含义见 `.env.example`（该文件不含任何密钥，`.env` 已被忽略）。

> **换 reranker 必须重新标定阈值**：`core/config.py` 的 `DEFAULT_SCORE_THRESHOLD` 依赖具体模型的
> 分数尺度。沿用旧阈值不会报错，只会让误报率隐性漂移（实测沿用 0.25 时健康误报从 6.5% 涨到 27%）。

### 4. 灌知识库

仓库已带 `crawler/data/agri_pests.jsonl`（2940 条），可直接入库：

```bash
python crawler/bulk_ingest_agri.py               # 全量
python crawler/bulk_ingest_agri.py --limit 20    # 试跑
python crawler/bulk_ingest_agri.py --start 500   # 断点续传
```

### 5. 启动服务

```bash
uvicorn main:app --reload --port 8000
```

### 6. 接入 MCP（可选）

```bash
python mcp_server.py
```

客户端配置示例：

```json
{
  "mcpServers": {
    "private-kb-server": {
      "command": "python",
      "args": ["E:/agri-agent/mcp_server.py"]
    }
  }
}
```

### 7. 跑测试

```bash
pytest
```

## API

| 方法 | 路径 | 说明 | 鉴权 |
|---|---|---|---|
| POST | `/auth/register` | 注册，返回 JWT（201） | 否 |
| POST | `/auth/login` | 登录，返回 JWT | 否 |
| POST | `/files` | 上传图片，返回 `image_id`（≤8MB） | Bearer |
| POST | `/chat/stream` | SSE 流式问答 | Bearer |
| GET | `/kb/crops` | 作物列表及病害数（`refresh=true` 清缓存） | Bearer |

`/chat/stream` 请求体：

```json
{ "session_id": "s1", "question": "叶片有黑点，边缘干枯", "crop": "番茄", "image_id": "abc123" }
```

事件流顺序：`token`（增量文本，多次） → `sources`（溯源，可选） → `done`。

> `/kb/crops` 返回的 `crop` 值与 Milvus 中**逐字节一致**，前端必须原样回传，
> 任何 trim / 大小写处理都会让过滤条件静默匹配不到（表现为"检索变差但不报错"）。

## 数据

### 语料

`crawler/data/agri_pests.jsonl` — **2940 条**，每条一种病虫害：

```json
{
  "crop": "丝瓜",
  "category": "病害",
  "name": "丝瓜霜霉病",
  "source_file": "病害/丝瓜/丝瓜病害.xlsx",
  "fields": {
    "简介": "...",
    "危害症状": "...",
    "病原": "...",
    "侵染循环": "...",
    "发生因素": "...",
    "防治方法": "..."
  }
}
```

入库时**按 `fields` 的键切 chunk**（问"怎么防治"直接命中 `防治方法` 字段），
单字段超 600 字再用 `RecursiveCharacterTextSplitter` 二次切分；
metadata 携带 `crop` / `disease` / `category` / `section` / `source_file`，供检索期过滤。

### 评测留档

| 文件 | 内容 |
|---|---|
| `crawler/data/PROJECT2_RESULTS.md` | 12 节完整评测报告（口径、显著性、按类诊断、prompt 迭代、阈值标定） |
| `baseline_v3_bge.jsonl` | 620 样本 × bge-reranker（旧线上基线） |
| `baseline_v3_qwen3rerank.jsonl` | 620 样本 × Qwen3-Reranker（当前线上基线） |
| `multiturn_40*.jsonl` | 多轮对话 40 例（新 / 旧配置完整记录） |
| `vl_desc_*.jsonl` | VL 描述缓存（v1 / v3 / v4 / v5 各 prompt 版本） |

> 上述评测明细（约 30MB）默认不入库，`.gitignore` 只放行了 README 引用到的最小文件集，
> 其余可用 `crawler/eval_agri_recall.py`、`crawler/eval_multiturn.py` 重新生成。

## 关键评测结论

### 1. 检索基线（620 样本）

| 配置 | @1 | @3 | @5 | @20 | 健康误报 |
|---|---|---|---|---|---|
| v3 描述 + bge-reranker-v2-m3（旧线上） | 11.2% | 31.7% | 39.0% | 67.6% | 6.5% |
| **v3 描述 + Qwen3-Reranker-8B（当前线上）** | 15.7% | 35.7% | **51.2%** | 67.6% | 6.5% |
| 32B 描述 + Qwen3-Reranker | **21.0%** | 35.5% | 47.1% | **71.7%** | 26.5% |
| v4 描述 + Qwen3-Reranker | 17.1% | 34.8% | 49.3% | 66.0% | 11.5% |
| v5 描述 + Qwen3-Reranker | 16.0% | 36.0% | 51.7% | 69.8% | 17.0% |

### 2. 换 reranker 的显著性（配对 McNemar）

| K | 救回 | 丢掉 | p 值 | 判定 |
|---|---|---|---|---|
| 1 | 51 | 32 | 0.0475 | 显著 |
| 5 | **62** | **11** | **0.0000** | 强显著 |
| 20 | 0 | 0 | 1.0000 | 完全不变 |

@20 救回 0 / 丢掉 0 说明两代 reranker 的**候选池完全一致**，收益 100% 来自重排序。

### 3. 按类诊断（21 类）

| 诊断 | 类数 | 平均 @5 变化 |
|---|---|---|
| 召回弱（@20 < 70%） | 9 | 18.9% → 21.7%（+2.8pt） |
| 排序弱（@20 ≥ 70% 但 @5 < 40%） | 2 | 2.5% → 5.0% |
| 健康（其余） | 10 | 64.5% → 87.0%（**+22.5pt**） |

**核心结论：换 reranker 的收益几乎全部落在"召回了但排不上"的类上；"召不回"的 9 个类几乎没有改善。
排序手段救不了召回问题** —— 瓶颈在知识库症状描述与图片的可匹配性。

### 4. 番茄专项（7 类 × 20 = 140 样本，全项目最弱作物）

| 描述版本 | @1 | @3 | @5 | @20 |
|---|---|---|---|---|
| v3 + bge | 1.4% | 7.1% | 12.9% | 32.9% |
| v3 + Qwen3 | 5.0% | 15.0% | 25.0% | 32.9% |
| v4 + Qwen3 | 2.9% | 12.9% | 28.6% | 36.4% |
| **v5 + Qwen3** | **7.1%** | **20.0%** | **37.9%** | **47.1%** |

番茄 7 个类里 6 个属"召回弱"，@20 本身就低 —— 指向知识库描述问题，而非排序问题。

### 5. VL 描述 prompt 迭代（示例被模型照抄）

| 版本 | 平均字数 | 含「多角形」 | 含「受叶脉限制」 |
|---|---|---|---|
| v3 | 72.7 | 84.3% | 81.4% |
| v4 | 64.5 | 0.0% | 81.4% |
| v5 | 56.1 | 0.0% | 0.0% |

**根因不是模型能力，而是提示词写法**：v3 的判别特征清单写成"是否受叶脉限制（**例如呈多角形**）"，
模型把括号里的示例当标准答案照抄，620 条中 59.4% 出现该短语（5 个类 100%），失去区分度，
反而把检索推向知识库中用了同样措辞的错误病害。v4 删示例、v5 删整条清单项后归零，番茄 @5
25.0% → 37.9%。

### 6. 阈值标定

| 阈值 | bge 误报 | Qwen3 误报 |
|---|---|---|
| 0.25（旧线上） | 7.0% | **27.0%** |
| 0.50（当前线上） | 6.5% | **6.5%** |

两个 reranker 的分数尺度不同（健康样本 top1 中位数：bge 0.084 / Qwen3 0.025），
沿用旧阈值会造成"误报从 7% 跳到 27%"的阈值假象。

## 已知问题

| 优先级 | 问题 |
|---|---|
| 1 | `core/vision.py` 的 `DESCRIBE_PROMPT` 仍是含「示例输出」的旧版（与 v3 同类反模式），未切到 v5 |
| 2 | 番茄 6/7 类召回弱（@20 本身偏低），需改知识库症状描述 |
| 3 | 柑橘黄龙病 @20 = 40%、玉米锈病 / 大斑病 @20 = 60%，同属召回问题 |
| 4 | `Peach___Bacterial_spot` @5 回退 −25pt，是换 reranker 唯一明显回退的类 |

完整列表见 [`PROJECT2_RESULTS.md` 第 11 节](crawler/data/PROJECT2_RESULTS.md)。

## 数据来源与免责

- 病虫害语料：公开数据集《农业病虫害信息检索数据集》（Excel），经 `excel_to_jsonl.py` 结构化
- 补充语料：`crawl_agri.py` 抓取「中国农业农村信息网」作物病虫害栏目公开文章
- 评测图片：PlantVillage 公开数据集
- 本项目为技术验证与学习用途。**防治方案均为语料原文的检索结果，实际用药请以当地植保部门指导为准。**
