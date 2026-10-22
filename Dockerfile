# Agri-Agent 运行时镜像。
# 注意：不能装 requirements.lock.txt —— 那是 Windows 上 freeze 出来的，含 pywin32，
# Linux 下 pip 会直接报找不到匹配分发。锁定版本的需求由 requirements.txt 里的 == 满足。
FROM python:3.12-slim

# PYTHONUNBUFFERED：不加的话 stdout 被缓冲，docker logs 看不到实时输出
# PYTHONDONTWRITEBYTECODE：容器里不需要 .pyc，少一层无用写入
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

# tzdata：slim 镜像里没有 /usr/share/zoneinfo，只设 ENV TZ 是无效的
# （docker logs 的时间戳仍是 UTC，和本地差 8 小时，对着日志排查会误判）
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独成层：只改业务代码时不会触发重装（构建从几秒到几十秒的差别）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 其余代码（具体排除项见 .dockerignore；.env 不会被带进来）
COPY . .

# 上传目录。容器内以 root 运行，绑定挂载 ./uploads 时权限天然匹配；
# 若要改成非 root 用户，必须同时处理宿主目录属主，否则上传会 500。
RUN mkdir -p /app/uploads

EXPOSE 8000

# --workers 1 是刻意的：core/bm25_index.py 的 _BM25_CACHE 是进程内缓存，
# 首次检索会把整库（约 1.5 万 chunk）从 Milvus 拉进内存建 BM25 索引。
# 开 N 个 worker 就是 N 份索引、N 倍内存，且各自冷启动各拖一次全库。
# 这个项目要横向扩展，得先把 BM25 挪出进程（如独立服务），不是改个参数。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
