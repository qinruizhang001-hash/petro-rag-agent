"""
把策展的领域本体合并进 LightRAG 图谱（纯标准库，确定性）
========================================================
为什么需要这一步
----------------
``scripts/corpus_audit.py`` 实测：4,826 个节点里盖层只有 18 个（0.37%），
而 parameter 有 1,310 个。这不是"论文读得不够"——22 篇论文都在讲 Zechstein，
再读 22 篇也不会让第二个盖层地层出现。抽取只能给出"这几篇提到了什么"，
给不出"这个领域有哪些东西"。

所以补一份人工策展的领域本体（``data/ccus_domain_ontology.json``）：
六维度、每条带出处、可审计。本脚本负责把它确定性地并进图谱，
并**明确区分**"抽取出来的"和"策展补进去的"——把两者的和当成抽取能力的
提升是自欺。

为什么直接改 XML 而不是重建 graphml
-----------------------------------
节点与边带 ``entity_id`` / ``source_id`` / ``file_path`` / ``created_at``
等 LightRAG 自己写入的字段，重建时漏一个字段不会报错，只会在下次检索时
悄悄少一条证据。用 ElementTree 在原有树上追加元素，原有内容逐字节保留。

幂等性
------
按名字去重（大小写不敏感）：``Zechstein caprock`` 与 ``Zechstein Caprock``
在图谱里已经是两个节点，本体再加第三个只会更糟。重复执行本脚本，
第二次的 ``added_nodes`` 必须为 0。

用法::

    python scripts/augment_graph.py                # 只预演，打印将要发生的改动
    python scripts/augment_graph.py --write        # 就地写入（先备份）
    python scripts/augment_graph.py --check        # 只校验本体文件
    python scripts/augment_graph.py --out x.graphml  # 写到别处，不动原文件
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.graph_analysis import _NS, load_graph  # noqa: E402

DEFAULT_GRAPHML = os.path.join(
    _ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml"
)
DEFAULT_ONTOLOGY = os.path.join(_ROOT, "data", "ccus_domain_ontology.json")

# 与 utils/kg_prompt.py 的 ---Allowed Entity Types--- 保持一致。
# 本体不能引入抽取阶段不允许的类型，否则两边对同一张图的类型统计不可比。
ALLOWED_TYPES = {
    "geologicalbody", "wellborecomponent", "material", "process",
    "technology", "parameter", "risk", "regulation",
}

# 本体节点的 file_path 标记。报告里要能一眼看出哪些节点不是从 PDF 抽出来的，
# 以及下次重新索引时哪些节点会消失（LightRAG 重建图会丢掉非抽取节点）。
ONTOLOGY_SOURCE = "domain_ontology"


def load_ontology(path: str = DEFAULT_ONTOLOGY) -> dict:
    """读取并校验本体文件。校验失败直接抛错 —— 宁可炸，不要静默降质。"""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    for key in ("version", "entities", "relations"):
        if key not in data:
            raise ValueError(f"本体缺少字段 {key}")

    seen: dict[str, str] = {}
    for e in data["entities"]:
        for key in ("name", "entity_type", "dimension", "description", "reference"):
            if not e.get(key):
                raise ValueError(f"实体 {e.get('name')!r} 缺少字段 {key}")
        etype = e["entity_type"].strip().lower()
        if etype not in ALLOWED_TYPES:
            raise ValueError(f"实体 {e['name']!r} 的类型 {etype!r} 不在允许清单内")
        low = e["name"].strip().lower()
        if low in seen:
            raise ValueError(f"本体内有重复实体：{e['name']!r} 与 {seen[low]!r}")
        seen[low] = e["name"]

    for r in data["relations"]:
        for key in ("from", "to", "keywords", "description", "reference"):
            if not r.get(key):
                raise ValueError(f"关系 {r.get('from')!r}->{r.get('to')!r} 缺少字段 {key}")

    return data


def _index(nodes: list[str]) -> dict[str, str]:
    """小写名 → 图中原名。用于大小写不敏感去重。

    **首次出现优先**（``setdefault``），不用字典推导。图里已经存在
    ``Zechstein Caprock`` 与 ``Zechstein caprock`` 这类大小写变体，
    取"最后一次"会让同一份本体在两次运行里解析到不同节点，
    结果就是每次重跑都多写几条重复边 —— 幂等性当场失效。
    """
    idx: dict[str, str] = {}
    for n in nodes:
        idx.setdefault(n.strip().lower(), n)
    return idx


def merge(nodes: list[str], edges: list[tuple[str, str]], attrs: dict[str, dict],
          ontology: dict) -> dict:
    """在内存里合并本体，返回合并后的结构与统计。

    不写文件，供 ``corpus_audit`` 直接调用 —— 审计必须反映"图谱 + 本体"
    的真实覆盖，否则报告的数字和系统实际能用的知识不是一回事。
    """
    idx = _index(nodes)
    out_nodes = list(nodes)
    out_attrs = dict(attrs)
    overrides: dict[str, list[str]] = {}
    added: list[str] = []
    deduped: list[str] = []

    for e in ontology["entities"]:
        name = e["name"].strip()
        low = name.lower()
        existing = idx.get(low)
        if existing is not None:
            deduped.append(f"{name} → {existing}" if existing != name else name)
            # 既有节点也拿到本体的维度声明：图里叫 "Caprock"，本体里叫
            # "Shale Caprock"，两者同属盖层维度，声明应当生效
            overrides.setdefault(existing, [])
            if e["dimension"] not in overrides[existing]:
                overrides[existing].append(e["dimension"])
            continue
        out_nodes.append(name)
        out_attrs[name] = {
            "entity_type": e["entity_type"],
            "description": e["description"],
            "file_path": ONTOLOGY_SOURCE,
            "source_id": f"ontology:{ontology['version']}",
            "reference": e["reference"],
        }
        overrides[name] = [e["dimension"]]
        idx[low] = name
        added.append(name)

    existing_edges = {frozenset((s, t)) for s, t in edges}
    out_edges = list(edges)
    added_edges: list[tuple[str, str]] = []
    duplicate_edges = 0
    unresolved: list[tuple[str, str]] = []

    for r in ontology["relations"]:
        src, dst = r["from"].strip(), r["to"].strip()
        s_real = idx.get(src.lower())
        d_real = idx.get(dst.lower())
        if s_real is None or d_real is None:
            unresolved.append((src, dst))
            continue
        key = frozenset((s_real, d_real))
        if key in existing_edges:
            duplicate_edges += 1
            continue
        existing_edges.add(key)
        out_edges.append((s_real, d_real))
        added_edges.append((s_real, d_real))

    per_dim = Counter()
    for e in ontology["entities"]:
        per_dim[e["dimension"]] += 1

    return {
        "nodes": out_nodes,
        "edges": out_edges,
        "attrs": out_attrs,
        "overrides": overrides,
        "stats": {
            "ontology_version": ontology["version"],
            "added_nodes": len(added),
            "added_node_names": added,
            "deduped_nodes": len(deduped),
            "deduped_examples": deduped[:8],
            "added_edges": len(added_edges),
            "duplicate_edges": duplicate_edges,
            "unresolved_edges": len(unresolved),
            "unresolved_examples": [f"{s} -> {t}" for s, t in unresolved[:8]],
            "added_by_dimension": dict(per_dim),
            "nodes_before": len(nodes),
            "nodes_after": len(out_nodes),
            "edges_before": len(edges),
            "edges_after": len(out_edges),
        },
    }


# ─── 写回 graphml ────────────────────────────────────────────


def _key_ids(root: ET.Element, scope: str) -> dict[str, str]:
    """attr.name → key id，按 for="node"/"edge" 分开取。

    硬编码 d0/d1 会在 LightRAG 升级换 key 编号后静默写错字段。
    """
    out: dict[str, str] = {}
    for key in root.findall(f"{_NS}key"):
        if key.get("for") != scope:
            continue
        kid, name = key.get("id"), key.get("attr.name")
        if kid and name:
            out[name] = kid
    return out


def write_augmented(graphml: str, ontology: dict, out_path: str | None = None,
                    backup: bool = True) -> dict:
    """把本体写进 graphml。``out_path`` 为空时就地写入（默认先备份）。"""
    nodes, edges, attrs = load_graph(graphml)
    merged = merge(nodes, edges, attrs, ontology)
    stats = dict(merged["stats"])

    if stats["added_nodes"] == 0 and stats["added_edges"] == 0:
        stats["written"] = False
        stats["reason"] = "无新增（幂等：本体已在图中）"
        return stats

    tree = ET.parse(graphml)
    root = tree.getroot()
    graph = root.find(f"{_NS}graph")
    if graph is None:
        raise ValueError(f"{graphml} 里没有 <graph> 元素")

    nk = _key_ids(root, "node")
    ek = _key_ids(root, "edge")
    for required in ("entity_id", "entity_type", "description", "source_id", "file_path"):
        if required not in nk:
            raise ValueError(f"graphml 缺少节点属性键 {required}")

    existing = {n.get("id") for n in graph.findall(f"{_NS}node")}
    # 必须与 merge() 用同一套解析规则（按节点顺序、首次出现优先）。
    # 从 set 里建映射会得到任意顺序，大小写变体就会解析到不同节点。
    by_low = _index(list(merged["nodes"]))

    for name in stats["added_node_names"]:
        el = ET.SubElement(graph, f"{_NS}node", {"id": name})
        meta = merged["attrs"][name]
        # entity_id 在原图里等于节点 id，LightRAG 用它做反查；留空会让新节点
        # 在按 id 索引时查不到
        values = {"entity_id": name, **meta}
        for attr_name in ("entity_id", "entity_type", "description",
                          "source_id", "file_path", "reference"):
            if attr_name not in nk:
                continue
            child = ET.SubElement(el, f"{_NS}data", {"key": nk[attr_name]})
            child.text = str(values.get(attr_name, ""))
        existing.add(name)
        by_low[name.lower()] = name

    edge_pairs = {frozenset((e.get("source"), e.get("target")))
                  for e in graph.findall(f"{_NS}edge")}
    for r in ontology["relations"]:
        s = by_low.get(r["from"].strip().lower())
        t = by_low.get(r["to"].strip().lower())
        if s is None or t is None:
            continue
        key = frozenset((s, t))
        if key in edge_pairs:
            continue
        edge_pairs.add(key)
        el = ET.SubElement(graph, f"{_NS}edge", {"source": s, "target": t})
        for attr_name, value in (
            ("weight", "1.0"),
            ("description", r["description"]),
            ("keywords", r["keywords"]),
            ("source_id", f"ontology:{ontology['version']}"),
            ("file_path", ONTOLOGY_SOURCE),
        ):
            if attr_name not in ek:
                continue
            child = ET.SubElement(el, f"{_NS}data", {"key": ek[attr_name]})
            child.text = value

    target = out_path or graphml
    if out_path is None and backup:
        bak = graphml + ".pre-ontology.bak"
        shutil.copy2(graphml, bak)
        stats["backup"] = bak
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    tree.write(target, encoding="utf-8", xml_declaration=True)
    stats["written"] = True
    stats["target"] = target
    return stats


def render(stats: dict) -> str:
    lines = [
        f"本体版本 {stats['ontology_version']}",
        f"节点 {stats['nodes_before']} → {stats['nodes_after']}"
        f"（新增 {stats['added_nodes']}，去重 {stats['deduped_nodes']}）",
        f"边   {stats['edges_before']} → {stats['edges_after']}"
        f"（新增 {stats['added_edges']}，重复 {stats['duplicate_edges']}，"
        f"端点未解析 {stats['unresolved_edges']}）",
        "按维度新增: " + " · ".join(
            f"{k}={v}" for k, v in sorted(stats["added_by_dimension"].items())
        ),
    ]
    if stats["deduped_examples"]:
        lines.append("去重示例: " + "；".join(stats["deduped_examples"]))
    if stats["unresolved_examples"]:
        lines.append("未解析示例: " + "；".join(stats["unresolved_examples"]))
    if not stats.get("written", True):
        lines.append(f"未写入：{stats.get('reason', '')}")
    elif stats.get("target"):
        lines.append(f"已写入 {stats['target']}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="把策展领域本体合并进 LightRAG 图谱")
    ap.add_argument("--graphml", default=DEFAULT_GRAPHML)
    ap.add_argument("--ontology", default=DEFAULT_ONTOLOGY)
    ap.add_argument("--out", default=None, help="输出路径；默认就地写入")
    ap.add_argument("--write", action="store_true", help="真的写入（否则只预演）")
    ap.add_argument("--check", action="store_true", help="只校验本体文件")
    ap.add_argument("--json", action="store_true", help="输出 JSON 统计")
    args = ap.parse_args()

    ontology = load_ontology(args.ontology)
    if args.check:
        print(f"本体校验通过：{len(ontology['entities'])} 实体 / "
              f"{len(ontology['relations'])} 关系 / 版本 {ontology['version']}")
        return 0

    if not os.path.exists(args.graphml):
        print(f"graphml 不存在：{args.graphml}")
        return 2

    if args.write:
        stats = write_augmented(args.graphml, ontology, out_path=args.out)
    else:
        nodes, edges, attrs = load_graph(args.graphml)
        stats = merge(nodes, edges, attrs, ontology)["stats"]
        stats["written"] = False
        stats["reason"] = "预演模式（加 --write 才写入）"

    print(json.dumps(stats, ensure_ascii=False, indent=2) if args.json else render(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
