# 部署手册 · Agri-Agent

> 定位：这份文档不是"怎么敲命令"，而是"每个决策为什么这么做、坏了怎么判断"。
> 所有标注 **【实测】** 的内容都在本机（WSL2 + Docker Desktop）真实跑过，输出原文保留；
> 标注 **【预防】** 的是设计上已经处理、但没有实际踩到的点；**【未复现】** 的是理论分析。
>
> 复现所需：`Dockerfile`、`.dockerignore`、`deploy/nginx.conf`、`docker-compose.yml`（含 `profiles: deploy`）。

---

## 1. 目标架构

```
              ┌──────────────── 公网 / 宿主机 localhost ────────────────┐
              │                                                        │
       80/tcp │                                                        │ 6380 / 19530 / 9091
              ▼                                                        ▼   （仅 127.0.0.1）
        ┌───────────┐        service name         ┌──────────────┐
        │   nginx   │ ─────── "app:8000" ───────▶ │     app      │
        │ (反代/SSE)│                              │  (uvicorn)   │
        └───────────┘                              └──────┬───────┘
                                                          │
                            ┌──────────────┬──────────────┼──────────────┐
                            ▼              ▼              ▼              ▼
                      ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
                      │  mysql   │   │  redis   │   │  milvus  │   │ 外部 API │
                      │ (用户/反馈)│   │(对话记忆) │   │(向量检索) │   │ LLM/VL/  │
                      └──────────┘   └──────────┘   └────┬─────┘   │ Embed/   │
                                                          │         │ Rerank   │
                                              ┌───────────┴────┐    └──────────┘
                                              │ etcd  +  minio │
                                              └────────────────┘
```

| 服务 | 暴露方式 | 数据落位 | 说明 |
|---|---|---|---|
| nginx | **`80:80`（唯一公网入口）** | 无状态 | 挂载 `deploy/nginx.conf` |
| app | `expose: 8000`，不映射宿主 | `./uploads` 绑定挂载 | 单 worker |
| mysql | **不映射** | `mysql_data` 卷 | 仅 compose 网络内可达 |
| redis | `127.0.0.1:6380:6379` | `redis_data` 卷 | 本机开发要连，故绑回环 |
| milvus | `127.0.0.1:19530` / `9091` | `milvus_data` 卷 | 同上，且 Milvus 无认证 |
| etcd / minio | 不映射 | 各自的卷 | Milvus 的元数据与对象存储 |

**核心原则：只有 nginx 一个公网入口。** 数据库、缓存、向量库全部不可从公网访问，
其中 redis 和 milvus 即使需要给本机开发用，也只绑 `127.0.0.1`——
它们**没有认证机制**，映射到 `0.0.0.0` 等于把可读写的库送人。

---

## 2. 三个关键设计决策

### 2.1 app 放进 compose，但用 profile 隔离

```yaml
services:
  app:
    profiles: ["deploy"]
```

- **本机开发**：`docker compose up -d` → 只起 redis / etcd / minio / milvus 四个基础设施，
  app 仍用 `uvicorn main:app --reload` 跑，MySQL 仍用宿主机的 3306。
- **部署**：`docker compose --profile deploy up -d --build` → 追加 mysql / app / nginx。

不隔离会有一个具体问题：本机 3306 上已经有 MySQL 在跑，
mysql 容器一启动就端口冲突（虽然最终没映射端口，但早期版本会）。

### 2.2 单 worker 是刻意的

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

`core/bm25_index.py` 的 `_BM25_CACHE` 是**进程内缓存**：首次检索会把整库（约 1.5 万 chunk）
从 Milvus 拉进内存，用 `rank_bm25` 建索引。开 N 个 worker 就是：

- N 份索引、N 倍内存
- N 个进程各自冷启动各拖一次全库

**要横向扩展，必须先把这个索引挪出进程**（独立服务或预建索引文件），
这不是调参能解决的。面试里这条比"我会 `--workers 4`"有分量得多。

### 2.3 数据必须落卷，只有一种例外

| 数据 | 方式 | 丢失的后果 |
|---|---|---|
| Milvus 向量库 | `milvus_data` + `etcd_data` + `minio_data` | 2940 条文档要重灌，embedding 重算（小时级 + 花钱） |
| 用户 / 反馈 | `mysql_data` | 用户全注销，badcase 全丢 |
| 上传图片 | `./uploads` 绑定挂载 | Redis 里的 `image_id` 还在，图没了 → "图片不存在或不属于当前用户" |

图片用**绑定挂载而非命名卷**，是为了能直接 `ls`、直接 `tar` 备份。

---

## 3. 镜像构建的四个决策

```dockerfile
FROM python:3.12-slim
```

**① 不用 `requirements.lock.txt`。** 它是 Windows 上 `pip freeze` 的产物，第 83 行是
`pywin32==312`，Linux 上没有对应 wheel，`pip install` 直接失败。版本锁定由
`requirements.txt` 里的 `==` 承担。 **【预防】**（静态发现，未实际踩）

**② 依赖单独成层。** `COPY requirements.txt` → `pip install` → `COPY . .`。
只改业务代码时不会触发重装。实测：首次构建 226 秒，其中 `pip install` 159 秒；
改一行 Python 再构建，只需几秒。

**③ 装 `tzdata`。** slim 镜像里没有 `/usr/share/zoneinfo`，只写 `ENV TZ` 是无效的，
`docker logs` 时间戳仍是 UTC，和本地差 8 小时，对着日志排查会误判时区。

**④ `.dockerignore` 先于 Dockerfile 写。** `COPY . .` 会把 `.env` 打进镜像层，
而镜像层是只读的——删文件、`docker rmi` 之后，仍能用 `docker history` 扒出来。
另外排除 `crawler/data/*.jsonl`（约 30MB 评测产物），但保留
`!crawler/data/agri_pests.jsonl`（灌库要用）。

---

## 4. 反代的三处必配

```nginx
client_max_body_size 10m;      # ①
proxy_buffering off;           # ②
proxy_read_timeout 300s;       # ③
```

**① 上传体积。** nginx 默认 1MB，而 `/files` 的上限是 8MB（`core/config.py` 的
`MAX_UPLOAD_BYTES`）。不改这里，手机原图直接 413，**且报错在 nginx 层，FastAPI 日志一片空白**，
很容易误判成"前端没传上来"。

**② SSE 缓冲。** `/chat/stream` 是 SSE。默认 `buffering on` 会把整段回答攒完再一次性吐出，
前端表现是"转圈很久 → 答案整段蹦出来"，流式白做。**这项不关不会报错，只会悄悄退化**，
所以验证方式只能是给每一行打时间戳（见第 6 节）。

**③ 超时。** 一次诊断要串 VL 描述 + 混合检索 + 多轮 LLM，默认 60s 容易 504。

另外 `proxy_http_version 1.1` + `proxy_set_header Connection ""`：
默认 HTTP/1.0 + close，SSE 的长连接会被掐。

---

## 5. 部署流程

```bash
# 1. 代码进入 WSL 原生文件系统（不要在 /mnt/e 下构建）
mkdir -p ~/agri-agent
cd /mnt/e/agri-agent
tar -cf - --exclude=.git --exclude=uploads --exclude=__pycache__ . | tar -xf - -C ~/agri-agent

# 2. 配置（.env 不进 git、不进镜像）
cd ~/agri-agent
test -f .env || cp .env.example .env
# 必填：OPENAI_API_KEY / MODEL_NAME / VL_MODEL_NAME / RERANK_MODEL / TAVILY_API_KEY
#       JWT_SECRET / MYSQL_ROOT_PASSWORD / MYSQL_PASSWORD

# 3. 只校验渲染结果，不启动任何东西（能看到 localhost 变成服务名）
docker compose --profile deploy config | head -40

# 4. 构建
docker compose --profile deploy build

# 5. 启动
docker compose --profile deploy up -d

# 6. 看状态
docker compose ps
```

**【实测】** 构建结果：

```
=> [5/7] RUN pip install --no-cache-dir -r requirements.txt    159.5s
=> => naming to docker.io/library/knowledge-app:latest          0.0s
✔ Image knowledge-app Built                                    226.2s
```

`docker compose ps` 期望 7 个容器 Up，其中 `agri-app` / `agri-mysql` / `agri-redis` 带 `(healthy)`；
`milvus-standalone` 没有 healthcheck，`Up` 即可。

---

## 6. 验证清单（命令 + 真实输出）

### 6.1 反代 + app 进程

```bash
curl -i http://localhost/healthz
```

**【实测】**

```
HTTP/1.1 200 OK
Server: nginx/1.31.7
Content-Type: application/json
Content-Length: 11

{"ok":true}
```

### 6.2 MySQL 写入 + JWT

```bash
TOKEN=$(curl -s -X POST http://localhost/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"deploy_test","password":"Test123456"}' \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
echo "token 前 20 位: ${TOKEN:0:20}"
```

**【实测】** `token 前 20 位: eyJhbGciOiJIUzI1NiIs`

### 6.3 Milvus 数据完好

```bash
curl -s http://localhost/kb/crops -H "Authorization: Bearer $TOKEN" | head -c 400
```

**【实测】**

```json
[{"crop":"苹果","disease_count":85},{"crop":"番茄","disease_count":84},
 {"crop":"梨","disease_count":80},{"crop":"棉花","disease_count":76}, ...]
```

这一条同时证明：`knowledge_milvus_data` 卷完好、collection `kb_agri` 可查、
**2940 条文档不需要重灌**。

### 6.4 SSE 是否真流式（唯一可靠的验证方法）

```bash
curl -N -s -X POST http://localhost/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"s1","question":"番茄叶子有黑点，边缘干枯，怎么回事？","crop":"番茄"}' \
  | while IFS= read -r line; do echo "$(date +%T) $line"; done
```

**【实测】** 时间戳逐行递增，跨秒：

```
23:05:49 data: {"type":"token","content":"纹"}
23:05:49 data: {"type":"token","content":"的大"}
23:05:50 data: {"type":"token","content":"病"}
23:05:50 data: {"type":"token","content":"斑"}
```

如果所有行时间戳相同、或憋到最后一起出现，说明 `proxy_buffering off` 没生效。
**只看"有输出"是验不出来的**——缓冲时也有输出，只是晚。

事件顺序：`token`（多次）→ `sources`（可选）→ `done`。

---

## 7. 故障复盘

> 这一节是面试时最值钱的部分。共同点是：**都在容器显示 healthy 的情况下发生**。
>
> 来源说明：7.2 / 7.3 在同构的最小复现环境（nginx + Python 两容器，端口 8080）里实测，
> 链路结构与本项目完全一致（同为 nginx 反代到上游容器）；
> 7.4 / 7.5 是本次部署过程本身踩到的。区分开来写，是为了不把"练习环境的现象"
> 说成"生产项目的数据"。

### 7.1 502 的三种措辞 = 三层故障

| error.log 措辞 | 含义 | 查哪里 |
|---|---|---|
| `connect() failed (111: Connection refused)` | 连不上，对端没进程监听 | 容器没起 / 端口不对 |
| `host not found in upstream "api"` | 服务名解析失败 | DNS / 网络 / 启动顺序 |
| `upstream prematurely closed connection while reading response header` | 连上了、请求发了、**对端中途掉线** | 上游进程处理该请求时崩了 |

前两种（7.2 / 7.3）在同构的最小复现环境里分别实测到，第三种（DNS）属理论分析，**【未复现】**。

> 记住一个方法：**不要一看到 502 就怀疑反代配置**。先看 `error.log` 的措辞，
> 它已经告诉你故障在哪一层了。

### 7.2 案例 A：`NameError` 导致的 502 【实测】

**现象**

```
curl http://localhost:8080/api/hello  →  502
Server: nginx/1.31.7
```

但同时：两个容器都 `(healthy)`，`docker compose exec nginx wget http://api:8000/healthz` 返回 `ok`。

**证据**

```
[error] 23#23: *112 upstream prematurely closed connection while reading response header
        from upstream, request: "GET /api/hello", upstream: "http://172.20.0.2:8000/api/hello"
```

服务名已解析成 IP、连接已建立、请求已发出 —— 故障在最后一环。

**定位**

```bash
docker compose logs api | grep -v healthz | tail -30
```

```
File "/app/app.py", line 25, in do_GET
    self._send(200, json.dumps(payload), "application/json")
NameError: name 'json' is not defined
```

**根因**：`import json` 丢失。`/healthz` 分支不碰 `json`，所以健康检查一直通过。

**修法**：补 `import json`，`docker compose restart api`（bind mount 是文件实时同步，
但**已加载的代码不会热更新**，必须重启进程）。

**教训**

1. **健康检查只能证明"进程活着、能应答指定路径"**，证明不了业务路径可用。
   真实项目里这是高频事故：上线时全员绿灯，用户一点就炸。
2. 异常被 `socketserver` 的 `handle_error` 捕获，**主进程不死**，`docker ps` 永远是 Up。
   这类问题只能靠日志发现。

### 7.3 案例 B：改错端口导致的 502 【实测】

把 `/api/` 的 `proxy_pass` 从 `8000` 改成 `8001`：

```bash
docker compose exec nginx nginx -t          # → syntax is ok / test is successful
docker compose exec nginx nginx -s reload
curl -i http://localhost:8080/api/hello     # → 502
```

```
[error] 1969#1969: *642 connect() failed (111: Connection refused) while connecting to upstream,
        upstream: "http://172.20.0.2:8001/api/hello"
```

同时 `/healthz` 仍然 200（那条 `proxy_pass` 没动），所以 `docker compose ps` 依旧显示 healthy。

**教训**

1. **`nginx -t` 通过 ≠ 功能正常。** 它只查语法，不查端口后面有没有人。
2. `reload` 是热加载：老 worker 处理完手头请求再退出，连接不断。
   证据是 error.log 里出现了 `worker process 25 exited with code 0`，且后续日志的 PID 变了。
   线上改配置一律 `nginx -t && nginx -s reload`，不要 `restart`。
3. 配置有错时 reload 会失败，但**老 worker 仍用旧配置继续服务**——这是 nginx 的保护机制。

### 7.4 案例 C：容器名冲突，暴露了"项目名即命名空间" 【实测】

```
Error response from daemon: Conflict. The container name "/agri-redis" is already in use
by container "412cfa94..." ...
```

`docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Label "com.docker.compose.project"}}'`：

```
agri-redis        Exited (0) 2 weeks ago   agri-agent     ← 旧项目名
milvus-standalone Exited (137) 2 weeks ago knowledge      ← 当前项目名
```

**根因**：compose 文件里的 `name: knowledge` 是**后加的**，早期用目录名当项目名（`agri-agent`），
而 `container_name` 是硬编码的，于是同名容器归属不同项目，compose 拒绝接管。

**修法**：`docker rm -f agri-redis`（**删容器不删卷**），compose 会用新项目名重建。
数据无损，因为 Milvus 那三个容器的 label 本来就是 `knowledge`，卷名 `knowledge_milvus_data` 一致。

**教训**

- Docker 的卷名是 `<项目名>_<卷名>`。**项目名一改，就会挂到一套全新的空卷上**，
  现象是"collection 全没了"，而数据其实还在旧卷里。
- 判定规则要记住：

| 操作 | 卷还在吗 |
|---|---|
| `docker rm -f 容器` | 在 |
| `docker compose down` | 在 |
| `docker compose down -v` | **没了** |
| 换项目名重新 up | 旧卷在，但新建一套空卷 → 数据"看不见了" |

### 7.5 案例 D：代码没进构建上下文 【实测】

```
failed to solve: failed to read dockerfile: open Dockerfile: no such file or directory
```

**根因**：`cp -r /mnt/e/agri-agent ~/agri-agent`，当目标目录已存在时，
`cp` 会往里**再套一层**（`~/agri-agent/agri-agent/`），导致外层缺文件。
后续用 `cp -a` 覆盖时又撞上第二个问题：`.git/objects` 里的 loose object 权限是 `0444`，
`cp` 改不动，报一屏 `Permission denied`。

**修法**：`tar` 管道，一次性排除 `.git`：

```bash
mkdir -p ~/agri-agent
cd /mnt/e/agri-agent
tar -cf - --exclude=.git --exclude=uploads --exclude=__pycache__ . | tar -xf - -C ~/agri-agent
```

`tar` 解包是**新建**文件而非覆盖，不会撞只读属性。

**教训**：`cp -r 源 目标` 不是"同步"，是"复制"。要同步用 `rsync -a`，或者 `cp` 时写 `源/.`。
另外从 `/mnt/e`（drvfs）复制到 `~/`（ext4）时，`.env` 可能带 CRLF，
`MYSQL_PASSWORD` 末尾混进 `\r` 会让 MySQL 认证失败并报 `Access denied`——
**排查方向会被完全带偏**，所以先 `grep -c $'\r' .env` 确认。

---

## 8. 数据与备份

```bash
# MySQL（表由 main.py 的 init_db() 自动建，幂等）
docker compose exec mysql sh -c 'mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" agri_agent' > backup.sql

# 上传图片（绑定挂载，直接打包）
tar czf uploads.tgz uploads/

# Milvus（走项目自带脚本，含向量，几十 MB）
python crawler/dump_agri.py
```

**恢复顺序**：先起 mysql/milvus，再 `restore_agri.py` 恢复向量库，最后起 app。

**迁移到新机器时注意**：`docker-compose.yml` 的 `name: knowledge` 必须保持一致，
或者把老卷 `external: true` 指过去，否则会新建空卷。

---

## 9. 成本方案（学生版）

| 方案 | 成本 | 能得到 | 得不到 |
|---|---|---|---|
| 纯本地 + 本文档 | ¥0 | 完整链路、踩坑记录、可复现输出 | 公网、SSH 运维 |
| GitHub Codespaces | ¥0（免费额度） | 4C8G 真实 Linux、能跑 compose、端口可公开转发 | 长期在线、SSH/安全组 |
| Azure for Students / GCP 试用 | ¥0（$100–300 额度） | 真 ECS、SSH、安全组、公网 IP | 额度用完即止 |
| 按量付费，只开 6 小时 | **≈¥15** | 全部真实经验（含公网访问） | 只能撑一次"直播" |
| 轻量包月 | ¥150–400/月 | 长期在线 | 对学生是负担 |

**推荐组合**：免费额度打底 + 面试前开一次 6 小时按量付费 + 这份文档。

按量付费下单要点：4C8G / Ubuntu 24.04 / 系统盘 50G / **公网按流量计费**（不用固定带宽，
SSE 文本流的出网流量极小）/ 按量计费 / **纯净镜像，不要预装宝塔等面板**（会占 80 端口）。
开完立刻打快照，用完立刻释放。

> 按量付费实例通常**不能用于 ICP 备案**（要求包年包月且剩余 ≥3 个月）。
> 所以"先按量验证、再决定是否包月备案"这个顺序是对的，反过来不行。

---

## 10. 面试怎么讲（90 秒版本）

> "这个项目是一个农业病虫害的多轮诊断 Agent，FastAPI + LangGraph，
> 依赖 Milvus 向量库、Redis 对话记忆、MySQL 用户数据，LLM 和 embedding 走外部 API。
>
> 我把它容器化了：7 个服务，只有 nginx 一个公网入口，数据库和向量库全部不映射端口——
> 因为 Redis 和 Milvus 都没有认证，暴露到公网等于送人。app 用单 worker，
> 因为 BM25 索引是进程内缓存，多 worker 会让内存按倍数涨，横向扩展得先把索引挪出去。
>
> 上线前我在本地把整条链路验了一遍，踩到几个坑：一是 502 有三种不同写法，
> `Connection refused` 是连不上、`prematurely closed` 是上游进程处理请求时崩了，
> 后者我最后定位到一个漏掉的 `import json`——而当时容器一直是 healthy，
> 因为健康检查只打了 `/healthz`，没覆盖业务路径。
>
> 二是 SSE 流式，nginx 默认缓冲会把整段回答攒完再吐，功能不报错但体验退化成非流式。
> 这项只能靠给每行打时间戳验证，光看'有输出'是看不出来的。
>
> 三是数据卷和 compose 项目名绑定，改项目名会挂到空卷上，现象是 collection 全没了。"

### 可能的追问

| 问题 | 要点 |
|---|---|
| 为什么不用 K8s？ | 单机 7 个容器，K8s 的调度/自愈收益抵不过运维成本；compose 的 `depends_on: service_healthy` 已经解决启动顺序 |
| 怎么保证密钥不进镜像？ | `.dockerignore` 挡 `.env` + `env_file` 注入 + `docker history` 可验证 |
| 健康检查为什么不做深？ | 深检查会让依赖抖动触发应用重启，把可恢复的抖动放大成连环重启；就绪判断应该交给负载均衡摘节点 |
| 500 和 502 的区别？ | 500 是应用返回的错误（能连上），502 是反代连不上/收不到上游有效响应 |
| 怎么备份 Milvus？ | `dump_agri.py` 导出（含向量，免重算 embedding），恢复用 `restore_agri.py` |
| 单点故障怎么办？ | 承认是单机架构；真要高可用，Milvus 换集群模式、app 无状态化+多副本、BM25 外置 |

---

## 11. 已知未做（诚实清单）

- 没有 HTTPS / 域名（需要证书 + 备案，学生阶段用不到）
- 没有限流与防刷（`/auth/register` 可被批量调用）
- 日志没有轮转（`./logs` 靠 docker 默认的 json-file，长了要配 `max-size`）
- app 以 root 运行（改非 root 要同时处理 `./uploads` 属主，否则上传 500）
- 没有 CI（当前是手动 `build` + `up`）
- 没有监控/告警（只有容器 healthcheck）

> 面试时主动说出这些，比被问出来强。**"我知道它哪里不行、以及为什么现在不做"
> 是工程判断力，不是缺陷。**
