# -*- coding: utf-8 -*-
"""
crawl_agri.py — 爬取「中国农业农村信息网·作物病虫害」栏目 (agri.cn)

用法:
    python crawl_agri.py                # 全量: 翻 20 页列表, 抓全部文章正文
    python crawl_agri.py --pages 3      # 只翻前 3 页列表
    python crawl_agri.py --limit 5      # 每页最多抓 5 篇正文(试跑用)

输出:
    crawler/data/agri_zwbch.jsonl       # 每行一条 {url,title,publish_date,source,text}
    (自动断点续跑: 已成功抓取的 url 会跳过)

依赖:
    pip install httpx beautifulsoup4
"""
import asyncio
import json
import os
import re
import sys

import httpx
from bs4 import BeautifulSoup

BASE = "https://www.agri.cn/sc/zxjc/zwbch/"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(OUT_DIR, "agri_zwbch.jsonl")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
              "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
CONCURRENCY = 5          # 详情页并发数
LIST_DELAY = 0.6         # 翻页间隔(秒), 避免给网站压力
MAX_PAGES = 200          # 列表翻页上限(20页全量后自然 404 停止)


def parse_args():
    """--pages N 只翻前 N 页; --limit N 每页最多抓 N 篇正文(试跑用)"""
    pages, limit = MAX_PAGES, None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--pages" and i + 1 < len(args):
            pages = int(args[i + 1])
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
    return pages, limit


async def fetch(client: httpx.AsyncClient, url: str):
    """GET 一个页面; 失败指数退避重试; 404 返回 None(表示翻页到头)"""
    for attempt in range(3):
        try:
            r = await client.get(url)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 403:
                print(f"  [HTTP 403] {url}")
                print(f"  body 前300字: {r.text[:300]!r}")
            else:
                print(f"  [HTTP {r.status_code}] {url}")
        except Exception as e:
            print(f"  [重试 {attempt + 1}/3] {url} -> {type(e).__name__}: {e}")
        await asyncio.sleep(2 ** attempt)
    return None


def date_from_url(url: str) -> str:
    """文章链接 t20260826_123.htm → 2026-08-26"""
    m = re.search(r"t(\d{8})_", url)
    return f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}" if m else ""


def extract_body(soup: BeautifulSoup) -> str:
    """启发式提取正文: 常见政府网站正文容器里取文本最长的那个"""
    best = ""
    for sel in ("#zoom", ".TRS_Editor", ".article-content", "#article",
                ".content", ".view", ".Custom_UnionStyle"):
        node = soup.select_one(sel)
        if node:
            text = node.get_text("\n", strip=True)
            if len(text) > len(best):
                best = text
    if len(best) > 50:
        return re.sub(r"\n{3,}", "\n\n", best)
    # 兜底: 拼接正文里的 <p> (跳过过短的导航/链接文字)
    ps = [p.get_text(" ", strip=True) for p in soup.select("p")
          if len(p.get_text(strip=True)) > 20]
    return "\n".join(ps)


async def list_articles(client: httpx.AsyncClient, max_pages: int, limit):
    """翻列表页, 收集 (url, title, publish_date); 去重"""
    items, seen = [], set()
    for page in range(1, max_pages + 1):
        url = BASE + ("index.htm" if page == 1 else f"index_{page - 1}.htm")
        r = await fetch(client, url)
        if r is None:
            print(f"[列表] 翻到第 {page} 页已无内容, 停止")
            break
        soup = BeautifulSoup(r.text, "html.parser")
        got = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if not re.search(r"t\d{8}_\d+\.htm", href):   # 文章链接特征
                continue
            title = a.get_text(strip=True)
            if not title or href in seen:
                continue
            full = href if href.startswith("http") else BASE + href.lstrip("./")
            seen.add(href)
            items.append({"url": full, "title": title, "publish_date": date_from_url(full)})
            got += 1
            if limit and got >= limit:      # 试跑模式: 每页只取前 N 篇
                break
        print(f"[列表 第{page}页] 新增 {got} 篇 (累计 {len(items)})")
        await asyncio.sleep(LIST_DELAY)
    return items


def load_done() -> set:
    """读已抓成功的 url(断点续跑)"""
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        done.add(json.loads(line)["url"])
                    except json.JSONDecodeError:
                        pass
    return done


def append_record(rec: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def crawl_detail(client: httpx.AsyncClient, sem, item: dict):
    async with sem:
        r = await fetch(client, item["url"])
        if r is None:
            print(f"  [失败] {item['url']}")
            return False
        soup = BeautifulSoup(r.text, "html.parser")
        text = extract_body(soup)
        if len(text) < 50:       # 正文过短 = 抓错/页面异常
            print(f"  [正文过短] {item['title']} ({len(text)} 字) 跳过")
            return False
        append_record({
            "url": item["url"], "title": item["title"],
            "publish_date": item["publish_date"],
            "source": "agri_zwbch", "crop": None, "disease": None,
            "text": text,
        })
        return True


async def main():
    pages, limit = parse_args()
    done = load_done()
    print(f"输出: {OUT}\n已抓 {len(done)} 篇(断点续跑自动跳过)")

    async with httpx.AsyncClient(headers=HEADERS, timeout=20, follow_redirects=True) as client:
        # 先访问根域一次: 部分政府站 WAF 首次访问会给会话种 Cookie
        await fetch(client, "https://www.agri.cn/")
        items = await list_articles(client, pages, limit)
        todo = [it for it in items if it["url"] not in done]
        print(f"待抓详情: {len(todo)} 篇")

        sem = asyncio.Semaphore(CONCURRENCY)
        ok = 0
        for i in range(0, len(todo), 20):
            batch = todo[i:i + 20]
            results = await asyncio.gather(*[crawl_detail(client, sem, it) for it in batch])
            ok += sum(1 for r in results if r)
            print(f"[详情] 进度 {min(i + 20, len(todo))}/{len(todo)}, 本次成功 {sum(results)}")
    print(f"\n完成: 本次新增 {ok} 篇, 累计见 {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
