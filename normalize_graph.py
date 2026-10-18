"""
图谱归一化：实体别名归并 + 脏节点清理
=====================================
在**已经建好的** LightRAG 图谱上做一次确定性整理，两件事：

1. 删掉不是知识的节点（作者单位、年份、街道地址、期刊名、DOI、``et al.`` 残片）；
2. 把同物异名的节点并成一个（``CO2`` ← ``Carbon dioxide (CO2)``，
   ``CCS`` ← 7 种写法，``Caprock`` ← ``Cap Rock`` …）。

为什么单独一个脚本，而不是塞进建图流程
--------------------------------------
建图（``ingest_ccus_batch.py``）要做 23 次 LLM 实体抽取，慢且贵；归一化只做
存储操作，快且免费。两者节奏完全不同 —— 归一化规则改一版，不应该触发
一次 20 分钟的重新抽取。分开之后可以反复调规则、反复应用。

为什么走 LightRAG 的 API，而不是直接改 graphml
----------------------------------------------
``amerge_entities`` / ``adelete_by_entity`` 会同时维护图存储、实体向量库、
关系向量库、chunk 追踪表。手工改 graphml 会让向量库里留下指向"已不存在的
节点"的向量 —— 检索时命中一个图里没有的实体，不报错，只是召回悄悄变差。
这类不一致最难排查，所以宁可慢一点也要走官方入口。

``--dry-run`` 会**预演**结果：用并查集在内存里把"合并 + 删除"应用一遍，
直接算出归一化后的连通分量数。这样在花时间真正执行之前，就能判断规则
是不是有用 —— 如果预演显示分量数没降，那就不该执行。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from utils.entity_aliases import build_merge_plan, plan_summary
from utils.graph_analysis import (
    connected_components,
    format_summary,
    load_graph,
    summarize,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHML = os.path.join(ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml")
REPORT = os.path.join(ROOT, "docs", "graph-normalization.json")


def project(plan: dict, nodes, edges) -> dict:
    """
    在内存里预演归一化结果，不碰任何存储。

    把每个存活节点映射到它的合并目标，删掉噪音节点，然后重算连通分量。
    这是 ``--dry-run`` 的核心：**先算清楚再动手**。
    """
    noise = {item["name"] for item in plan["noise"]}
    remap: dict[str, str] = {}
    for merge in plan["merges"]:
        for source in merge["sources"]:
            remap[source] = merge["target"]

    def resolve(name: str) -> str:
        return remap.get(name, name)

    survivors = [n for n in nodes if n not in noise]
    kept = {resolve(n) for n in survivors}

    projected_edges: set[tuple[str, str]] = set()
    for src, tgt in edges:
        if src in noise or tgt in noise:
            continue          # 边的任一端是脏节点，整条边跟着消失
        a, b = resolve(src), resolve(tgt)
        if a == b:
            continue          # 合并后两端重合 → 自环，丢掉
        projected_edges.add((a, b) if a <= b else (b, a))

    comps = connected_components(sorted(kept), sorted(projected_edges))
    largest = len(comps[0]) if comps else 0
    return {
        "nodes": len(kept),
        "edges": len(projected_edges),
        "components": len(comps),
        "largest": largest,
        "largest_ratio": round(largest / (len(kept) or 1), 4),
        "top": [len(c) for c in comps[:10]],
    }


def print_plan(plan: dict, limit: int = 12) -> None:
    print(f"\n合并组 {len(plan['merges'])} 个，脏节点 {len(plan['noise'])} 个")
    print("\n--- 合并示例（按覆盖节点数降序）---")
    for merge in sorted(plan["merges"], key=lambda m: -len(m["sources"]))[:limit]:
        others = [s for s in merge["sources"] if s != merge["target"]]
        print(f"  {merge['target']!r}  ←  {others}")

    print("\n--- 删除示例（按原因分组）---")
    by_reason: dict[str, list[str]] = {}
    for item in plan["noise"]:
        by_reason.setdefault(item["reason"], []).append(item["name"])
    for reason, names in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        sample = ", ".join(names[:3])
        print(f"  [{reason}] ×{len(names)}: {sample}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LightRAG 图谱归一化（别名归并 + 脏节点清理）")
    ap.add_argument("--apply", action="store_true",
                    help="真正执行（默认只预演）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个操作（小样验证执行链路）")
    ap.add_argument("--skip-noise", action="store_true", help="只合并，不删脏节点")
    ap.add_argument("--skip-merge", action="store_true", help="只删脏节点，不合并")
    ap.add_argument("--report", default=REPORT, help="预演/执行报告的输出路径")
    args = ap.parse_args(argv)

    if not os.path.exists(GRAPHML):
        print(f"❌ 找不到图谱文件：{GRAPHML}")
        return 1

    before = summarize(GRAPHML)
    print(f"归一化前：{format_summary(before)}")

    nodes, edges, attrs = load_graph(GRAPHML)
    types = {n: attrs.get(n, {}).get("entity_type", "") for n in nodes}
    plan = build_merge_plan(nodes, types)
    print(f"计划：{plan_summary(plan)}")
    print_plan(plan)

    after = project(plan, nodes, edges)
    print("\n--- 预演结果 ---")
    print(f"归一化后：{after['nodes']} 节点 / {after['edges']} 边 | "
          f"连通分量 {after['components']} | "
          f"最大分量 {after['largest']} 节点（{after['largest_ratio']:.1%}）")
    print(f"变化：节点 {before['nodes']} → {after['nodes']}，"
          f"边 {before['edges']} → {after['edges']}，"
          f"分量 {before['components']} → {after['components']}")

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(
            {"before": before, "projected": after, "plan": plan},
            fh, ensure_ascii=False, indent=2,
        )
    print(f"报告已写入 {args.report}")

    if not args.apply:
        print("\n（预演模式，未改动任何存储。加 --apply 执行。）")
        return 0

    # ─── 真正执行 ────────────────────────────────────────
    from lightrag_retriever import delete_entity, merge_entities

    noise = [] if args.skip_noise else plan["noise"]
    merges = [] if args.skip_merge else plan["merges"]
    if args.limit:
        noise = noise[: args.limit]
        merges = merges[: args.limit]

    print(f"\n开始执行：删除 {len(noise)} 个脏节点，合并 {len(merges)} 组…")
    deleted = failed_delete = 0
    for i, item in enumerate(noise, 1):
        if delete_entity(item["name"]):
            deleted += 1
        else:
            failed_delete += 1
        if i % 20 == 0 or i == len(noise):
            print(f"  删除进度 {i}/{len(noise)}（成功 {deleted}）")

    merged = failed_merge = 0
    for i, merge in enumerate(merges, 1):
        if merge_entities(merge["sources"], merge["target"]):
            merged += 1
        else:
            failed_merge += 1
        if i % 10 == 0 or i == len(merges):
            print(f"  合并进度 {i}/{len(merges)}（成功 {merged}）")

    # 执行完必须重新从磁盘读，不能沿用预演的数字 ——
    # 失败的删除/合并会让实际结果与预演不同，报告必须是实测值。
    final = summarize(GRAPHML)
    print(f"\n归一化后（实测）：{format_summary(final)}")
    print(f"变化：节点 {before['nodes']} → {final['nodes']}，"
          f"边 {before['edges']} → {final['edges']}，"
          f"分量 {before['components']} → {final['components']}")
    if failed_delete or failed_merge:
        print(f"⚠️  删除失败 {failed_delete}，合并失败 {failed_merge}")

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "before": before,
                "projected": after,
                "actual": final,
                "deleted": deleted,
                "failed_delete": failed_delete,
                "merged": merged,
                "failed_merge": failed_merge,
                "plan": plan,
            },
            fh, ensure_ascii=False, indent=2,
        )
    print(f"报告已写入 {args.report}")
    return 0 if not (failed_delete or failed_merge) else 2


if __name__ == "__main__":
    sys.exit(main())
