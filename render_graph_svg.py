"""
把 LightRAG 的 graphml 渲染成 SVG 知识图谱（纯 Python，无需 matplotlib）。
输出：docs/knowledge-graph.svg

设计取舍：
- 只渲染**最大连通分量**。它是 CCUS 地质封存的核心因果链，其余碎片画上去
  只会让主图缩成一团，信息密度反而下降。标题里如实标注总量，不夸大。
- 每条边标注 LightRAG 抽取的关系关键词 —— 这是 GraphRAG 区别于向量检索的直接证据。
- **只给度数 ≥3 的节点画标签**。全画标签会互相压字，读者反而一个都看不清。
  低度节点保留悬浮提示，鼠标移上去能看到名字、类型和描述。

为什么要按度数抽样（v1.7 起默认只画前 700 个节点）
--------------------------------------------------
全文重建后最大连通分量有 2,992 个节点 / 4,311 条边。原样画出来的 SVG 是
**2.52 MB**，而 CI 有一条守卫：仓库里任何被跟踪的文件不得超过 1 MB
（`.github/workflows/ci.yml` 的 "Guard against corpus binaries"）。那条守卫
的本意是拦住误提交的语料/模型，但一个 2.5 MB 的派生图确实会撞上它 ——
而且每次重建都会给 git 历史永久加 2.5 MB。

三个可选做法，选了第三个：

1. 放宽 CI 上限 —— 等于为了这张图削弱一条真在防事故的守卫；
2. 只截断悬浮描述 —— 实测 `<title>` 只占全文件 42.5%，截完仍约 2.0 MB，不够；
3. **按度数取前 N 个节点**（默认 700），并在图标题里写明"这是最大分量的
   度数前 700 个节点，总量 2,992"。

第三个还顺带修了一个可读性问题：1560×1000 的画布上放 2,992 个节点，每个
节点平均不到 20×20 像素，图例里的类型颜色根本分辨不出来。抽到 700 个之后
节点是"看得见"的，而抽样依据（度数）恰好就是这张图要表达的"谁是枢纽"。

代价要写清楚：**抽样后的布局与全图不同**，图的连通性也只反映核心子图。
这不是"完整图谱的缩略图"，是"核心子图的示意图"，标题里必须说。
需要全图时用 ``--max-nodes 0``（约 9 分钟，且会超过 CI 上限，不要提交）。

注意：分量数量本身就是个诚实指标。
- 语料 5 篇时：46 节点 / 32 边，基本是一张手绘的种子图。
- 扩到 23 篇、用通用 prompt：765 节点散成 **247 个分量**，最大只有 83 个
  —— 多喂论文并没有自动产生连通性，因为 LightRAG 按**实体名完全相等**合并，
  而 `CO2` 和 `Carbon dioxide (CO2)` 是两个节点。
- 加了类型约束 + 别名归并后：389 节点 / 84 个分量 / 最大 244（62.7%）。
- 2026-10-25 全文重建后：4,826 节点 / 1,169 个分量 / 最大 2,992（62%）。
"""
import argparse
import html
import math
import os
from collections import Counter

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHML = os.path.join(ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml")
OUT = os.path.join(ROOT, "docs", "knowledge-graph.svg")

W, H = 1560, 1000
PAD_X, PAD_TOP, PAD_BOT = 90, 110, 70

# 默认只画度数最高的 700 个节点（见模块文档）。0 表示不抽样。
DEFAULT_MAX_NODES = 700

# CI 对单个被跟踪文件的硬上限（.github/workflows/ci.yml）。
REPO_FILE_LIMIT = 1024 * 1024

# 悬浮提示里描述文本的截断长度。
# 完整描述在 graphml 里（4.9 MB），SVG 里只需要"够认出来"：
# 700 节点 + 1,600 边一共 2,300 个 <title>，每个多留 100 字就是 230 KB。
# 实测：700 节点 / 1,597 边时，400/300 字 → 901 KB；120/140 字 → 800 KB，
# 给 1 MB 上限留出 22% 余量（图谱再长也会先撞 --max-nodes 的抽样上限）。
NODE_DESC_LIMIT = 120
EDGE_DESC_LIMIT = 140

# 低于这个度数的节点不画文字标签，只留悬浮提示。
LABEL_MIN_DEGREE = 4

# 标签总数上限（按度数取最高的若干个）。
# 这条上限是 v1.7 补的：LABEL_MIN_DEGREE=4 在 244 节点的图上只产生 43 个标签，
# 但在 700 节点的子图上会产生 **264** 个 —— 1560×1000 的画布上中心区必然糊成
# 一片，一个都读不出来。封顶到 70 之后，被标出来的正好是真正的枢纽
# （CO2 / Reservoir / Permeability / Fault / Caprock …），其余节点靠悬浮提示。
LABEL_TOP_K = 70

PALETTE = ["#4C9AFF", "#FF7A59", "#36B37E", "#B37FEB", "#FFC53D",
           "#40C4D0", "#FF5C8A", "#F2A65A", "#7FD1AE", "#9AA5B1"]
BG, INK, MUTED, EDGE = "#0E1624", "#E6EAF2", "#8A93A5", "#61728C"


def esc(s):
    return html.escape(str(s), quote=True)


def shorten(name, limit=16):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def halo_text(x, y, text, *, size, fill, anchor="middle", weight="500"):
    """
    带描边光晕的文字，用**两层 <text>** 实现而不是 `paint-order`。

    第一版用了 `paint-order="stroke"`，浏览器里正常，但 PyMuPDF 不认这个属性，
    会按默认顺序"先填充、后描边"，于是 3.5px 的深色描边把字整个盖住 ——
    导出的 PNG 上一个标签都看不见。双层写法在所有渲染器里都对：底层只负责
    描边（填充也用背景色），上层只负责填充。
    """
    common = (
        f'x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" font-family="ui-sans-serif,Segoe UI,Helvetica,Arial"'
    )
    return (
        f'<text {common} fill="{BG}" stroke="{BG}" stroke-width="3.5" '
        f'stroke-linejoin="round">{esc(text)}</text>'
        f'<text {common} fill="{fill}" fill-opacity="0.94">{esc(text)}</text>'
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render the LightRAG graphml into docs/knowledge-graph.svg",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=DEFAULT_MAX_NODES,
        help=(f"只渲染度数最高的 N 个节点（默认 {DEFAULT_MAX_NODES}，0 = 全部分量）。"
              "抽样是为了让文件留在 CI 的 1 MB 上限以内，并让节点在画布上看得见。"),
    )
    args = parser.parse_args(argv)
    max_nodes = int(args.max_nodes)

    full = nx.read_graphml(GRAPHML)
    comps = sorted(nx.connected_components(full), key=len, reverse=True)
    main_comp = full.subgraph(comps[0]).copy()
    g = main_comp
    if 0 < max_nodes < g.number_of_nodes():
        # 按度数取前 N 个：度数就是这张图要表达的"谁是枢纽"，用它抽样不会
        # 把核心结构抽掉，也不会引入"随手挑几个"的任意性。
        keep = sorted(g.degree, key=lambda kv: (-kv[1], str(kv[0])))[:max_nodes]
        g = g.subgraph([n for n, _ in keep]).copy()
    print(f"full {full.number_of_nodes()}/{full.number_of_edges()} → "
          f"main component {main_comp.number_of_nodes()}/{main_comp.number_of_edges()}"
          f" → rendered {g.number_of_nodes()}/{g.number_of_edges()}")

    types = [(d.get("entity_type") or "unknown").strip() for _, d in g.nodes(data=True)]
    order = [t for t, _ in Counter(types).most_common()]
    color_of = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(order)}

    try:
        pos = nx.kamada_kawai_layout(g)
    except Exception:
        pos = nx.spring_layout(g, k=2.0 / math.sqrt(g.number_of_nodes()),
                               iterations=3000, seed=7)

    xs = np.array([p[0] for p in pos.values()])
    ys = np.array([p[1] for p in pos.values()])
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
    scale = min((W - 2 * PAD_X) / max(xmax - xmin, 1e-6),
                (H - PAD_TOP - PAD_BOT) / max(ymax - ymin, 1e-6))

    def to_px(p):
        x = PAD_X + (p[0] - xmin) * scale + (W - 2 * PAD_X - (xmax - xmin) * scale) / 2
        y = PAD_TOP + (p[1] - ymin) * scale + (H - PAD_TOP - PAD_BOT - (ymax - ymin) * scale) / 2
        return x, H - y + PAD_BOT * 0.35

    deg = dict(g.degree())
    radius = {n: 7.0 + min(deg.get(n, 1), 6) * 1.7 for n in g.nodes}

    edges_svg, kw_svg = [], []
    for u, v, data in g.edges(data=True):
        x1, y1 = to_px(pos[u])
        x2, y2 = to_px(pos[v])
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        ax, ay = x1 + ux * radius[u], y1 + uy * radius[u]
        bx, by = x2 - ux * (radius[v] + 6), y2 - uy * (radius[v] + 6)
        mx, my = (ax + bx) / 2 - uy * 16, (ay + by) / 2 + ux * 16
        kws = [k.strip() for k in (data.get("keywords") or "").replace("<SEP>", "|").split("|") if k.strip()]
        # 只标注"短且信息量大"的关键词（中文多为 4-8 字，英文短语动辄 20+ 字符）。
        # 过长的英文短语一律截断，完整内容保留在 <title> 悬浮提示里。
        kw = kws[0] if kws and len(kws[0]) <= 12 else (shorten(kws[0], 10) if kws else "")
        desc = esc((data.get("description") or "")[:EDGE_DESC_LIMIT])
        edges_svg.append(
            f'<path d="M{ax:.1f},{ay:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" fill="none" '
            f'stroke="{EDGE}" stroke-opacity="0.6" stroke-width="1.6" marker-end="url(#arrow)">'
            f'<title>{esc(", ".join(kws))} — {desc}</title></path>'
        )
        if kw and len(kw) <= 8:
            kw_svg.append(halo_text(mx, my - 4, kw, size=9.5, fill=MUTED, weight="400"))

    node_svg, label_svg = [], []
    # 先算出"够格标名字"的节点，再按度数封顶 —— 顺序不能反：
    # 若在循环里边算边标，封顶就只能靠"先到先得"，与节点遍历顺序有关，
    # 而 GraphML 的遍历顺序不是稳定契约。
    label_candidates = sorted(
        (n for n in g.nodes if deg.get(n, 0) >= LABEL_MIN_DEGREE),
        key=lambda n: (-deg[n], str(n)),
    )[:LABEL_TOP_K]
    label_set = set(label_candidates)

    for n, data in g.nodes(data=True):
        x, y = to_px(pos[n])
        etype = (data.get("entity_type") or "unknown").strip()
        r = radius[n]
        desc = esc((data.get("description") or "")[:NODE_DESC_LIMIT])
        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color_of.get(etype, PALETTE[0])}" '
            f'fill-opacity="0.93" stroke="{BG}" stroke-width="2">'
            f'<title>{esc(n)} [{esc(etype)}] — {desc}</title></circle>'
        )
        # 没进前 K 的节点不画标签，名字与描述仍在 <title> 里，悬浮可见。
        if n not in label_set:
            continue
        # 标签放在远离图心的方向，减少互相压字
        cx, cy = to_px((0, 0))
        ox, oy = x - cx, y - cy
        norm = math.hypot(ox, oy) or 1.0
        lx, ly = x + ox / norm * (r + 14), y + oy / norm * (r + 14)
        anchor = "middle"
        if ox / norm > 0.45:
            anchor = "start"
        elif ox / norm < -0.45:
            anchor = "end"
        label_svg.append(
            halo_text(lx, ly + 4, shorten(n), size=11.5, fill=INK, anchor=anchor)
        )

    legend = []
    for i, t in enumerate(order):
        oy = 96 + i * 21
        legend.append(
            f'<circle cx="28" cy="{oy}" r="5" fill="{color_of[t]}"/>'
            f'<text x="42" y="{oy + 4}" font-size="11" '
            f'font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{MUTED}">'
            f'{esc(t)} ({Counter(types)[t]})</text>'
        )

    if g.number_of_nodes() < main_comp.number_of_nodes():
        # 抽样必须在图上一眼可见，否则读者会以为这就是完整图谱。
        shown = (f"largest component {main_comp.number_of_nodes()} nodes — "
                 f"top {g.number_of_nodes()} by degree shown")
    else:
        shown = (f"largest component shown ({g.number_of_nodes()} nodes, "
                 f"{g.number_of_nodes() / full.number_of_nodes():.0%} of the graph)")
    subtitle = (f"{full.number_of_nodes()} entities · {full.number_of_edges()} relations · "
                f"{len(comps)} components — {shown}")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="CCUS knowledge graph built by LightRAG">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
    <path d="M0,1 L9,5 L0,9 z" fill="{EDGE}"/>
  </marker>
</defs>
<rect width="{W}" height="{H}" fill="{BG}"/>
<text x="{W - 28}" y="42" text-anchor="end" font-size="18" font-weight="700"
      font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{INK}">CCUS Knowledge Graph</text>
<text x="{W - 28}" y="64" text-anchor="end" font-size="11.5"
      font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{MUTED}">{subtitle}</text>
{"".join(legend)}
{"".join(edges_svg)}
{"".join(kw_svg)}
{"".join(node_svg)}
{"".join(label_svg)}
</svg>
'''
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)
    size = os.path.getsize(OUT)
    print(f"saved {OUT} ({size / 1024:.0f} KB)")
    # 主动报告，而不是等 CI 红了再回来查：这张图是派生文件，
    # 重建时最容易不知不觉超限。
    if size > REPO_FILE_LIMIT:
        print(f"⚠️  {size / 1048576:.2f} MB 超过仓库单文件上限 1 MB"
              f"（CI: 'Guard against corpus binaries'）。"
              f"请减小 --max-nodes（当前 {max_nodes}）或不要提交这个文件。")


if __name__ == "__main__":
    main()
