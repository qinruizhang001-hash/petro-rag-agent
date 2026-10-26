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

``--report-md`` 把同一次运行的结果写成一份 Markdown 统计（默认
``docs/graph-noise-report.md``）。它存在的理由是"零命中的规则"：
``build_merge_plan`` 只返回命中项，于是"某条规则从没命中过"与"这条规则
不存在"在结果里长得一样。报告会把**全部**规则列出来，零命中的也列，
并额外实测几条"考虑过但没写成规则"的模式（会议名等）——
把"我们查过、当时是 0 条"变成可复现的结论，而不是一句记忆。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from utils.entity_aliases import (
    build_merge_plan,
    noise_rule_names,
    plan_summary,
)
from utils.graph_analysis import (
    connected_components,
    format_summary,
    load_graph,
    summarize,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHML = os.path.join(ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml")
REPORT = os.path.join(ROOT, "docs", "graph-normalization.json")
REPORT_MD = os.path.join(ROOT, "docs", "graph-noise-report.md")


# ─── 报告用的说明文字 ──────────────────────────────────────
#
# 这些是"这个原因是什么意思"，不是统计结果。数字一律从 payload 取，
# 绝不写在这里 —— 语料一变，手写的数字就会静默过期，而且没有任何断言会红。
REASON_NOTES: dict[str, str] = {
    "empty": "空字符串节点",
    "doi": "DOI / URL / 网址，是引用标识不是概念",
    "email": "邮箱",
    "date-label": "`Received date` 一类投稿流程标签",
    "volume-issue": "卷号 / 期号",
    "section-label": "`Abstract` / `References` / `Acknowledg…` 一类章节标题",
    "citation": "`et al.` 残片，是引文碎片不是实体",
    "year": "单独的四位年份",
    "decade": "`1990s` 一类年代",
    "bare-quantity": "只有数值加单位的碎片（`2500 m`、`15%`）",
    "street-address": "街道地址片段",
    "postal-code": "邮编",
    "affiliation": "作者单位（大学 / 研究所 / 系 / 实验室）",
    "publisher": "出版社名",
    "journal": "期刊名",
    "funding-body": "资助 / 行政机构（部委、研究联盟），连的是经费关系不是地质知识",
    "meta-filler": "`this study` / `corresponding author` 一类元话语",
    "geo-political": "国家 / 地区 / 城市名，会变成跨论文的假枢纽",
    "too-short": "≤2 字符的拉丁字母名字（CJK 与化学式不受此限）",
    "title-like": "超过 10 个词，基本是论文标题而不是概念",
}

# 类型整类丢弃的说明（`type:xxx` 这种原因）
_TYPE_NOTES: dict[str, str] = {
    "person": "人名。语料里没有需要保留的人物实体",
    "publication": "出版物条目，已被上面的期刊 / 出版社规则覆盖",
    "content": "`content` 是模型无类型可归时的垃圾桶",
}

# 考虑过、但**故意没写成规则**的模式。
#
# 判据是实测命中数：写死一条没人命中的规则，只会扩大误杀面而不减少噪声。
# 这里把命中数当场算出来，而不是写"实测 0 条" —— 后者一旦语料变了就变成
# 假陈述，而且没有任何断言会红。
CONSIDERED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("会议 / 论文集名",
     r"\b(?:conference|symposium|workshop|proceedings|GHGT|"
     r"international\s+meeting)\b"),
    ("会议举办地式命名",
     r"^(?:\d{1,2}(?:st|nd|rd|th)\s+)?(?:international\s+)?"
     r"(?:conference|workshop|symposium)\b"),
    ("机构名（无 affiliation 关键词）",
     r"\b(?:agency|authority|ministry|bureau|consortium|alliance)\b"),
)


def _reason_note(reason: str) -> str:
    """把原因映射成一句话说明。未知原因也要有输出，不能留空。"""
    if reason in REASON_NOTES:
        return REASON_NOTES[reason]
    if reason.startswith("type:"):
        return _TYPE_NOTES.get(reason[5:], "按实体类型整类丢弃")
    return "（未登记说明，请补 REASON_NOTES）"



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
    isolated = sum(1 for c in comps if len(c) == 1)
    return {
        "nodes": len(kept),
        "edges": len(projected_edges),
        "components": len(comps),
        "largest": largest,
        "largest_ratio": round(largest / (len(kept) or 1), 4),
        # 孤立节点数：消融对照里"合并到底救回了什么"要看这个，
        # 只看分量数会漏掉"合并把 3 个孤立点并成 1 个孤立点"这种情形。
        "isolated": isolated,
        "isolated_ratio": round(isolated / (len(kept) or 1), 4),
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


def measure_considered(nodes) -> list[dict]:
    """
    实测"考虑过但没写成规则"的模式在**当前语料**里的命中数。

    为什么要有这一步：加一条噪声规则是有代价的（扩大误杀面），所以"我们查过
    会议名、当时 0 条"这个结论必须能复现。把它写成常量（"实测 0 条"）就成了
    会过期的断言 —— 语料换了，这句话还是 0 条，而且没有任何东西会红。
    """
    out: list[dict] = []
    for label, pattern in CONSIDERED_PATTERNS:
        rx = re.compile(pattern, re.I)
        hits = sorted({str(n) for n in nodes if rx.search(str(n))})
        out.append({
            "label": label,
            "pattern": pattern,
            "hits": len(hits),
            "samples": hits[:5],
        })
    return out


def render_noise_report(payload: dict) -> str:
    """
    把一次运行的统计渲染成 Markdown。

    数字全部来自 ``payload``（``summarize`` / ``project`` / ``build_merge_plan``
    的返回值），这里只负责排版与解释。手写数字会静默过期，见
    ``docs/evidence-table.md`` 的同类教训。
    """
    before = payload["before"]
    projected = payload["projected"]
    actual = payload.get("actual")
    plan = payload["plan"]
    noise = plan["noise"]
    merges = plan["merges"]

    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    for item in noise:
        reason = item["reason"]
        counts[reason] = counts.get(reason, 0) + 1
        samples.setdefault(reason, []).append(item["name"])
    total_noise = len(noise) or 1

    merged_nodes = sum(len(m["sources"]) for m in merges)
    net_merge = merged_nodes - len(merges)

    lines: list[str] = [
        "# 图谱噪声报告",
        "",
        "> 本文件由 `python normalize_graph.py` 生成，**不要手改**。",
        "> 表里的数字全部取自当次运行的实测/预演结果；语料或规则一变，"
        "重新跑一次即可，手改的数字会静默过期。",
        "",
        "## 1. 归一化前后",
        "",
        "| 口径 | 节点 | 边 | 连通分量 | 最大分量 | 孤立节点 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| 归一化前（实测） | {before['nodes']} | {before['edges']} | "
        f"{before['components']} | {before.get('largest', '—')} | "
        f"{before.get('isolated', '—')} |",
        f"| 归一化后（预演） | {projected['nodes']} | {projected['edges']} | "
        f"{projected['components']} | {projected['largest']} | "
        f"{projected['isolated']} |",
    ]
    if actual:
        lines.append(
            f"| 归一化后（实测） | {actual['nodes']} | {actual['edges']} | "
            f"{actual['components']} | {actual.get('largest', '—')} | "
            f"{actual.get('isolated', '—')} |"
        )
    lines += [
        "",
        f"合并组 {len(merges)} 个，覆盖 {merged_nodes} 个节点（净减 {net_merge}）；"
        f"待删脏节点 {len(noise)} 个。",
    ]
    if "deleted" in payload:
        lines += [
            "",
            f"本次执行：删除成功 {payload['deleted']} / 失败 "
            f"{payload['failed_delete']}；合并成功 {payload['merged']} / 失败 "
            f"{payload['failed_merge']}。",
            "",
            "失败数非零时，预演与实测的差额就是失败造成的 —— "
            "不要把预演数字当成已达成。",
        ]
    lines += [
        "",
        "## 2. 命中的噪声规则",
        "",
        "| 原因 | 命中 | 占脏节点 | 这类节点是什么 | 样例 |",
        "| --- | ---: | ---: | --- | --- |",
    ]

    for reason in noise_rule_names():
        hit = counts.get(reason, 0)
        if not hit:
            continue
        sample = "、".join(f"`{s}`" for s in samples[reason][:3])
        lines.append(
            f"| `{reason}` | {hit} | {hit / total_noise:.1%} | "
            f"{_reason_note(reason)} | {sample} |"
        )

    zero = [r for r in noise_rule_names() if not counts.get(r)]
    lines += [
        "",
        "## 3. 零命中的规则",
        "",
        "下面这些规则**在本语料一次都没命中**。这不等于「已验证有效」，"
        "只等于「还没被验证」：它们是为已知的脏数据类型准备的防线，"
        "当前语料恰好没触发。",
        "",
        "| 原因 | 命中 | 说明 |",
        "| --- | ---: | --- |",
    ]
    for reason in zero:
        lines.append(f"| `{reason}` | 0 | {_reason_note(reason)} |")

    considered = payload.get("considered") or []
    lines += [
        "",
        "## 4. 考虑过、但没有写成规则的模式",
        "",
        "判据是**实测命中数**：写死一条零命中的规则只会扩大误杀面，"
        "不会减少噪声。下表每次运行都会重算，所以它不是记忆，是结论。",
        "",
        "| 模式 | 命中 | 处理 | 样例 |",
        "| --- | ---: | --- | --- |",
    ]
    for item in considered:
        if item["hits"]:
            action = "**有命中，需要人工看样例后再决定是否加规则**"
        else:
            action = "零命中 → 不加规则"
        sample = "、".join(f"`{s}`" for s in item["samples"]) or "—"
        lines.append(f"| {item['label']} | {item['hits']} | {action} | {sample} |")

    lines += [
        "",
        "已知的一处覆盖关系：机构名（大学 / 研究所 / 系 / 实验室）由 "
        "`affiliation` 规则按**名字**处理，而不是按 `organization` 类型整类删除 —— "
        "因为 `IPCC`、`International Energy Agency` 这类政策主体也是 "
        "`organization`，整类删会连监管实体一起删掉。",
        "",
        "## 5. 怎么复现",
        "",
        "```bash",
        "# 只预演，不改存储（本文件就是这样生成的）",
        "python normalize_graph.py",
        "",
        "# 真正执行",
        "python normalize_graph.py --apply",
        "```",
        "",
        "预演用并查集在内存里把「合并 + 删除」应用一遍，因此"
        "「归一化后（预演）」这一行是**算出来的**，不是执行后的实测值；"
        "执行后本文件会多一行「实测」。",
        "",
    ]
    return "\n".join(lines)


def write_noise_report(path: str, payload: dict) -> None:
    """写 Markdown 统计。目录不存在就建，避免第一次跑报错。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_noise_report(payload))
    print(f"统计报告已写入 {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LightRAG 图谱归一化（别名归并 + 脏节点清理）")
    ap.add_argument("--apply", action="store_true",
                    help="真正执行（默认只预演）")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个操作（小样验证执行链路）")
    ap.add_argument("--skip-noise", action="store_true", help="只合并，不删脏节点")
    ap.add_argument("--skip-merge", action="store_true", help="只删脏节点，不合并")
    ap.add_argument("--report", default=REPORT, help="预演/执行报告的输出路径")
    ap.add_argument("--report-md", default=REPORT_MD,
                    help="噪声统计（Markdown）的输出路径；传空串可关闭")
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

    # 统计报告的 payload 与 JSON 报告共用，但额外带"考虑过但没写成规则"的
    # 实测命中数 —— 那部分只有这里算得出来（要看全部节点名）。
    md_payload = {
        "before": before,
        "projected": after,
        "plan": plan,
        "considered": measure_considered(nodes),
    }
    if args.report_md:
        write_noise_report(args.report_md, md_payload)

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
    if args.report_md:
        # 执行完重写一次：让「实测」那一行有值，且失败数与预演并列可见。
        md_payload.update({
            "actual": final,
            "deleted": deleted,
            "failed_delete": failed_delete,
            "merged": merged,
            "failed_merge": failed_merge,
        })
        write_noise_report(args.report_md, md_payload)
    return 0 if not (failed_delete or failed_merge) else 2


if __name__ == "__main__":
    sys.exit(main())
