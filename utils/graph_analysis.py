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


# ══════════════════════════════════════════════════════════
# 领域维度覆盖（回答"广度够不够"）
# ══════════════════════════════════════════════════════════
# 总节点数说明不了广度。4,826 个节点里可以有 1,310 个是参数、只有 17 个是盖层，
# 那样"数千级实体"这句话在安全最关键的那个维度上是空的。所以按**领域维度**
# 再切一刀：真实 CCUS 项目要回答的问题，落在下面六类里，任何一类太薄，
# 多跳推理在那条线上就断。
#
# 口径声明：这是**词法**审计，不是语义分类。用正则匹配实体名（必要时看描述），
# 因此会漏掉换了个说法的同义实体，也会把 "Well Grid" 这类沾边的算进井筒。
# 它的用途是发现**量级失衡**（17 vs 1,310），不是给每个实体贴标签 ——
# 需要精确标签时用 ``entity_type``，那才是抽取阶段定的。
DOMAIN_DIMENSIONS: dict[str, dict] = {
    "reservoir": {
        "label": "储层类型",
        "label_en": "reservoir types",
        "pattern": (
            r"aquifer|reservoir|formation|sandstone|carbonate|basalt|coal|"
            r"depleted (oil|gas)|saline|porous media|储层|砂岩|碳酸盐|咸水层"
        ),
        # "Reservoir Simulation" 是方法不是储层类型；这一维问的是"能装 CO₂ 的
        # 地质体有哪几类"，把模拟器算进去会让储层维度虚胖。
        "exclude": r"simulation|simulator|modell?ing",
    },
    "fault": {
        "label": "断层体系",
        "label_en": "fault systems",
        "pattern": r"fault|fracture zone|fissure|断层|断裂",
        # "default" / "fault-tolerant" 不是断层
        "exclude": r"\bdefault\b|fault[- ]toleran",
    },
    "caprock": {
        "label": "盖层",
        "label_en": "caprock and seal",
        "pattern": (
            r"cap ?rock|\bseals?\b|top ?seal|shale|mudstone|clay|overburden|"
            r"盖层|泥岩|页岩|封闭层"
        ),
        # 聚合物/黏土纳米复合材料是固井材料，不是盖层岩性
        "exclude": r"nanocomposite|polymer",
    },
    "wellbore": {
        "label": "井筒",
        "label_en": "wellbore",
        "pattern": (
            # 必须带词边界：不加 \b 时 "Reinforcement" 和 "Displacement"
            # 里的 "cement" 会命中，把两个强化学习算法和一组位移场算成井筒
            r"\bwells?\b|borehole|\bcasing|\bcements?\b|\btubing\b|\bpackers?\b|"
            r"\bperforations?\b|井筒|套管|固井|射孔"
        ),
        # 水泥厂/水泥产量属于 CCS 产业链，不是井筒完整性
        "exclude": r"cement (plant|plants|production|consumption|industry)|and lime",
    },
    "monitoring": {
        "label": "监测",
        "label_en": "monitoring",
        "pattern": (
            r"monitor|seismic|insar|\bdas\b|\bdts\b|sensor|survey|sampling|"
            r"time-lapse|4d seismic|监测|地震|示踪"
        ),
    },
    "regulation": {
        "label": "法规",
        "label_en": "regulation",
        "pattern": (
            r"directive|regulation|\bact\b|standard|policy|permit|licen[cs]e|"
            r"45q|class vi|tax credit|法规|标准|政策|许可"
        ),
        # 强化学习术语里的 "policy" / 统计里的 "standard deviation" 不是法规
        "exclude": (
            r"standard deviation|policy (gradient|network|optimization|learning)|"
            r"optimal policy|deterministic policy"
        ),
        # 类型为 regulation 的实体无条件计入，避免 "UIC Class VI" 这类
        # 不含上述词的实体被漏掉。代价是继承上游抽取的类型错误
        # （例如 "Drucker-Prager Yield Criterion" 被抽成 regulation），
        # 这一部分错误来自 entity_type 而不是这里的正则。
        "types": {"regulation"},
    },
}

_DOMAIN_RES = {k: __import__("re").compile(v["pattern"], __import__("re").IGNORECASE)
               for k, v in DOMAIN_DIMENSIONS.items()}
_DOMAIN_EXCL = {k: (__import__("re").compile(v["exclude"], __import__("re").IGNORECASE)
                    if v.get("exclude") else None)
                for k, v in DOMAIN_DIMENSIONS.items()}

# 引用形态：``Borello et al.`` / ``Cheung et al. (2012)`` / ``Bolourinejad and Herber``。
# 抽取阶段会把作者名标成 regulation / institution，类型直通分支照收就会让
# 法规维度里混进一排人名。只在类型直通时过滤，正则命中不受影响。
_CITATION_LIKE = __import__("re").compile(
    r"\bet al\b|^[A-Z][\w'-]+ (and|&) [A-Z][\w'-]+$", __import__("re").IGNORECASE
)


def classify_domains(name: str, entity_type: str = "",
                     description: str = "") -> list[str]:
    """返回实体所属的领域维度（可能多个：`Monitoring Well` 同时属于两维）。

    只看名字与类型，不看描述 —— 描述里"本文用地震数据反演"会让几乎每个
    实体都沾上监测，那就不再是覆盖度而是噪声。描述只在**名字完全没命中**
    时作为补充，且仅当名字里含通用词（`data` / `method`）时才用。
    """
    hits = []
    etype = (entity_type or "").strip().lower()
    for key, spec in DOMAIN_DIMENSIONS.items():
        excl = _DOMAIN_EXCL[key]
        if excl is not None and excl.search(name or ""):
            continue
        if etype and etype in spec.get("types", ()):
            # 作者名被抽成 regulation 时不该算进法规维度
            if not _CITATION_LIKE.search(name or ""):
                hits.append(key)
            continue
        if _DOMAIN_RES[key].search(name or ""):
            hits.append(key)
    return hits


def domain_coverage(nodes: list[str], edges: list[tuple[str, str]],
                    attrs: dict[str, dict] | None = None,
                    overrides: dict[str, list[str]] | None = None) -> dict:
    """按领域维度统计实体数、占比、连通率。

    参数 ``overrides`` 是 ``{实体名: [维度, ...]}`` 的显式声明，优先于词法判断。
    人工策展的领域本体（``data/ccus_domain_ontology.json``）自带 dimension 字段，
    用它做覆盖统计时不能靠名字猜 —— 否则 "Zechstein Halite" 能不能算盖层，
    取决于我有没有把 halite 写进正则，统计就成了自证。

    返回::

        {
          "total_nodes": int,
          "dimensions": {
             "caprock": {"label", "label_en", "entities", "share",
                         "edges_touching", "isolated", "connected_ratio",
                         "examples": [...], "declared": int},
             ...
          },
          "untyped_ratio": float,   # entity_type 为空或 UNKNOWN 的占比
          "empty": [dimension keys with 0 entities],
        }

    ``connected_ratio`` 是该维度里**至少有一条边**的实体占比 —— 孤立实体
    再多也进不了多跳，这个比例比实体数更能说明"这一维能不能用"。
    ``declared`` 是其中由 ``overrides`` 显式声明的条数，用来把"抽取出来的"
    和"策展补进去的"分开报，避免把两者的和当成抽取能力的提升。
    """
    attrs = attrs or {}
    overrides = overrides or {}
    deg: Counter = Counter()
    for s, t in edges:
        deg[s] += 1
        deg[t] += 1

    total = len(nodes) or 1
    per_dim: dict[str, dict] = {}
    for key, spec in DOMAIN_DIMENSIONS.items():
        members = [n for n in nodes
                   if key in (overrides.get(n)
                              or classify_domains(
                                  n, attrs.get(n, {}).get("entity_type", "")))]
        declared = [n for n in members if key in overrides.get(n, ())]
        connected = [n for n in members if deg.get(n, 0) > 0]
        # 例子按度数取前 5：报告里要能一眼看出这一维到底装的是什么
        examples = sorted(members, key=lambda n: -deg.get(n, 0))[:5]
        per_dim[key] = {
            "label": spec["label"],
            "label_en": spec["label_en"],
            "entities": len(members),
            "share": round(len(members) / total, 4),
            "edges_touching": sum(deg.get(n, 0) for n in members),
            "isolated": len(members) - len(connected),
            "connected_ratio": round(len(connected) / len(members), 4) if members else 0.0,
            "examples": examples,
            "declared": len(declared),
        }

    untyped = sum(
        1 for n in nodes
        if (attrs.get(n, {}).get("entity_type") or "").strip().lower()
        in ("", "unknown", "other")
    )
    return {
        "total_nodes": len(nodes),
        "dimensions": per_dim,
        "untyped_ratio": round(untyped / total, 4),
        "empty": [k for k, v in per_dim.items() if v["entities"] == 0],
    }


def format_domain_table(cov: dict) -> str:
    """把 domain_coverage() 排成对齐的表格文本（报告/CLI 用）。

    ``策展`` 列是其中由本体显式声明的条数 —— 看报告的人要能立刻分清
    "这一维是靠抽论文抽出来的，还是靠人工本体补的"。
    """
    head = (f"{'维度':<8}{'实体':>7}{'策展':>7}{'占比':>9}"
            f"{'边':>7}{'孤立':>7}{'连通率':>9}  示例")
    rows = [head, "─" * 82]
    for key, d in cov["dimensions"].items():
        rows.append(
            f"{d['label']:<8}{d['entities']:>7}{d.get('declared', 0):>7}"
            f"{d['share']:>8.2%}{d['edges_touching']:>7}{d['isolated']:>7}"
            f"{d['connected_ratio']:>8.1%}  " + "、".join(d["examples"][:3])
        )
    return "\n".join(rows)
