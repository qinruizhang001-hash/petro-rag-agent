"""
图谱连通性分析（纯标准库）
==========================
回答一个具体问题：**这张图是一张网，还是一堆互不相连的碎片？**

指标只有两个，但都很难糊弄：
- ``components``：连通分量个数。23 篇论文如果讲的是同一件事，这个数应该远小于
  论文数；如果它接近节点数，说明每个实体都是孤岛。
- ``largest``：最大分量的节点数。它占总节点的比例，才是"知识网络"的真实形状。

为什么不直接用 networkx
------------------------
CI 的 ``unit-core`` 作业是零安装的（只跑标准库），而这条分析逻辑值得被单测
覆盖 —— 分量算法写错（比如把有向边当无向用、或者漏掉自环）不会报错，
只会给出一个看起来很合理但错误的数字。用 ElementTree + 并查集实现，
既能进 CI，也让算法本身可读。

graphml 的 ``edgedefault="undirected"`` 在 LightRAG 里恒成立（关系按无向处理），
所以这里直接用并查集，不做有向性判断。
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter

# graphml 的命名空间。LightRAG 输出的文件带默认 xmlns，
# 不带命名空间前缀地查找会一个元素都找不到。
_NS = "{http://graphml.graphdrawing.org/xmlns}"


def load_graph(path: str) -> tuple[list[str], list[tuple[str, str]], dict[str, dict]]:
    """
    读取 graphml，返回 ``(节点名列表, 边列表, 属性字典)``。

    只取节点 id / 边 source-target / 以及 ``entity_type`` 等节点属性；
    其余字段（描述、权重、时间戳）与连通性无关，读了只是浪费内存。
    """
    tree = ET.parse(path)
    root = tree.getroot()

    # key id → attr.name，先把 graphml 的元数据表建起来，
    # 否则 <data key="d1"> 不知道对应哪个属性
    key_names: dict[str, str] = {}
    for key in root.findall(f"{_NS}key"):
        kid = key.get("id")
        name = key.get("attr.name")
        if kid and name:
            key_names[kid] = name

    graph = root.find(f"{_NS}graph")
    if graph is None:
        return [], [], {}

    nodes: list[str] = []
    attrs: dict[str, dict] = {}
    for node in graph.findall(f"{_NS}node"):
        nid = node.get("id")
        if nid is None:
            continue
        nodes.append(nid)
        data = {}
        for item in node.findall(f"{_NS}data"):
            field = key_names.get(item.get("key") or "", item.get("key") or "")
            data[field] = (item.text or "").strip()
        attrs[nid] = data

    edges: list[tuple[str, str]] = []
    for edge in graph.findall(f"{_NS}edge"):
        src, tgt = edge.get("source"), edge.get("target")
        if src and tgt:
            edges.append((src, tgt))

    return nodes, edges, attrs


class _UnionFind:
    """并查集。路径压缩 + 按秩合并，O(α(n))。"""

    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.rank = {i: 0 for i in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # 路径压缩
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_components(nodes, edges) -> list[list[str]]:
    """返回按大小降序排列的连通分量。孤立节点各自成一个分量。"""
    uf = _UnionFind(nodes)
    known = set(nodes)
    for src, tgt in edges:
        # 边端点可能在节点表之外（手工改过图时会出现），补进去而不是丢弃 ——
        # 丢弃会让分量数偏大，这正是我们要测量的指标，不能悄悄失真。
        for endpoint in (src, tgt):
            if endpoint not in known:
                uf.parent[endpoint] = endpoint
                uf.rank[endpoint] = 0
                known.add(endpoint)
        uf.union(src, tgt)

    buckets: dict[str, list[str]] = {}
    for item in known:
        buckets.setdefault(uf.find(item), []).append(item)
    # 分量内部排序：`known` 是 set，迭代顺序随 PYTHONHASHSEED 变化。不排序的话
    # 同一个图两次运行会给出 `['B','A']` 和 `['A','B']`，测试随机失败、
    # 报告也不可复现 —— 一个只用来"看数字"的工具，输出不稳定比算错更烦人。
    comps = [sorted(members) for members in buckets.values()]
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def summarize(path: str) -> dict:
    """
    一次性给出图谱的连通性全貌。

    返回字段：
    - ``nodes`` / ``edges``
    - ``components``：连通分量数
    - ``largest``：最大分量节点数
    - ``largest_ratio``：最大分量占全部节点的比例
    - ``isolated``：孤立节点数（度为 0）
    - ``top``：前 10 大分量的节点数
    - ``types``：按 ``entity_type`` 的分布
    """
    nodes, edges, attrs = load_graph(path)
    comps = connected_components(nodes, edges)
    types = Counter(
        (attrs.get(n, {}).get("entity_type") or "unknown").lower() for n in nodes
    )
    largest = len(comps[0]) if comps else 0
    total = len(nodes) or 1
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "components": len(comps),
        "largest": largest,
        "largest_ratio": round(largest / total, 4),
        "isolated": sum(1 for c in comps if len(c) == 1),
        "top": [len(c) for c in comps[:10]],
        "types": dict(types.most_common()),
    }


def format_summary(stats: dict) -> str:
    """把 summarize() 的结果排成一行可读文本。"""
    return (
        f"{stats['nodes']} 节点 / {stats['edges']} 边 | "
        f"连通分量 {stats['components']} | "
        f"最大分量 {stats['largest']} 节点（{stats['largest_ratio']:.1%}）| "
        f"孤立节点 {stats['isolated']}"
    )
