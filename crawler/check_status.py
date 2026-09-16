# -*- coding: utf-8 -*-
"""check_status.py — 一键自查: 数据还在不在

【为什么必须用它, 而不是随手 query 一下】
Milvus 重启后有一个"看起来数据没了"的窗口: load_state 已经是 Loaded、
get_collection_stats 的 row_count 也已经恢复, 但 query node 的数据还没加载完,
这时 count(*) 会返回 0。
在这个窗口里下结论、甚至去删容器/删卷, 会真的把数据弄丢。
所以本脚本默认**轮询等待数据回来**, 只有等超时了才算异常。

【用法】
    python crawler/check_status.py                # 自查(默认最多等 180 秒) + 与基线对比
    python crawler/check_status.py --wait 300     # 多等一会儿
    python crawler/check_status.py --no-wait      # 不等, 立刻看(用来确认服务是不是刚起来)
    python crawler/check_status.py --save         # 把当前条数/集合ID 存成基线(入库后跑一次)

【退出码】0 = 正常; 1 = 有集合读不到数据, 或相比基线掉了
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from pymilvus import MilvusClient

from core.config import MILVUS_URI

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(PROJECT_ROOT, "backups", "baseline.json")


def container_status():
    try:
        out = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}} | {{.Status}} | {{.CreatedAt}}"],
            capture_output=True, text=True, timeout=30,
        )
        return [l for l in out.stdout.strip().splitlines() if l]
    except Exception as e:
        return [f"(读不到容器状态: {e})"]


def snapshot(client, coll: str) -> dict:
    """读一个集合的三件套: 可见条数 / 物理行数 / 集合ID"""
    client.load_collection(coll)
    cnt = client.query(coll, filter="", output_fields=["count(*)"])[0]["count(*)"]
    stats = client.get_collection_stats(coll)["row_count"]
    try:
        cid = client.describe_collection(coll).get("collection_id")
    except Exception:
        cid = None
    return {"count": cnt, "stats": stats, "id": cid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", type=int, default=180, help="最长等待秒数, 默认 180")
    ap.add_argument("--no-wait", action="store_true", help="不等待, 立刻出结果")
    ap.add_argument("--save", action="store_true", help="把当前状态存成基线, 供以后对比")
    args = ap.parse_args()

    print("=== 容器(看 Up 了多久: 刚起来别急着下结论) ===")
    for line in container_status():
        print("  " + line)

    client = MilvusClient(uri=MILVUS_URI)
    # v2.4.24 起 list_collections() 会返回重复项(实时层+快照层各一份), 去重保序
    cols = list(dict.fromkeys(client.list_collections()))
    print(f"\n=== 集合: {cols} ===")

    start = time.time()
    deadline = start + (0 if args.no_wait else args.wait)
    snap: dict = {}
    while True:
        snap, empty = {}, []
        for col in cols:
            try:
                s = snapshot(client, col)
            except Exception as e:
                s = {"count": -1, "stats": -1, "id": None, "err": repr(e)}
            snap[col] = s
            if s["count"] == 0:
                empty.append(col)
        if not empty or time.time() >= deadline:
            break
        print(f"  等待数据加载... 已等 {int(time.time() - start)}s (还没回来: {empty})")
        time.sleep(5)

    ok = True
    for col, s in snap.items():
        if s["count"] > 0:
            print(f"  OK  {col:14s} count(*)={s['count']:<8} stats={s['stats']:<8} id={s['id']}")
        else:
            ok = False
            detail = s.get("err") or "查询返回 0 条"
            print(f"  !!  {col:14s} count(*)={s['count']:<8} stats={s['stats']:<8} -> {detail}")

    if args.save:
        os.makedirs(os.path.dirname(BASELINE), exist_ok=True)
        data = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "milvus_uri": MILVUS_URI,
            "collections": {c: {"count": s["count"], "id": s["id"]} for c, s in snap.items()},
        }
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n基线已保存 -> {BASELINE}")
        return 0 if ok else 1

    if os.path.exists(BASELINE):
        with open(BASELINE, encoding="utf-8") as f:
            base = json.load(f)
        print(f"\n=== 与基线对比 (基线存于 {base.get('saved_at')}) ===")
        for col, want in base.get("collections", {}).items():
            got = snap.get(col)
            if got is None:
                print(f"  !! {col}: 集合不见了 (基线 {want['count']} 条, id={want.get('id')})")
                ok = False
                continue
            if want.get("id") is not None and got["id"] != want["id"]:
                print(f"  !! {col}: 集合被重建过(基线 id={want['id']} -> 现在 id={got['id']}), "
                      f"旧数据已不在这个集合里")
                ok = False
            if got["count"] == want["count"]:
                print(f"  OK {col}: {got['count']} 条, 与基线一致")
            elif got["count"] < want["count"]:
                print(f"  !! {col}: {got['count']} 条 < 基线 {want['count']} 条, 少了 {want['count'] - got['count']} 条")
                ok = False
            else:
                print(f"  .. {col}: {got['count']} 条 > 基线 {want['count']} 条 (+{got['count'] - want['count']}, 应该是有新入库)")
        for col in snap:
            if col not in base.get("collections", {}):
                print(f"  .. {col}: 基线里没有的新集合, {snap[col]['count']} 条")
    else:
        print("\n(还没有基线: 入库正常时跑一次 --save, 以后就能自动对比)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
