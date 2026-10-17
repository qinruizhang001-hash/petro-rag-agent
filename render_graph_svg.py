"""
把 LightRAG 的 graphml 渲染成 SVG 知识图谱（纯 Python，无需 matplotlib）。
输出：docs/knowledge-graph.svg

设计取舍：
- 只渲染最大连通分量（23 实体 / 23 关系）。它是 CCUS 地质封存的核心因果链，
  其余碎片（4/3/1 个节点的小分量）画上去只会让主图缩成一团，信息密度反而下降。
  标题里如实标注 46 实体 / 32 关系的总量，不夸大。
- 每条边标注 LightRAG 抽取的关系关键词 —— 这是 GraphRAG 区别于向量检索的直接证据。
"""
import html
import math
import os
from collections import Counter

import networkx as nx
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPHML = os.path.join(ROOT, "lightrag_storage", "graph_chunk_entity_relation.graphml")
OUT = os.path.join(ROOT, "docs", "knowledge-graph.svg")

W, H = 1180, 780
PAD_X, PAD_TOP, PAD_BOT = 90, 110, 70

PALETTE = ["#4C9AFF", "#FF7A59", "#36B37E", "#B37FEB", "#FFC53D",
           "#40C4D0", "#FF5C8A", "#F2A65A", "#7FD1AE", "#9AA5B1"]
BG, INK, MUTED, EDGE = "#0E1624", "#E6EAF2", "#8A93A5", "#61728C"


def esc(s):
    return html.escape(str(s), quote=True)


def shorten(name, limit=16):
    return name if len(name) <= limit else name[: limit - 1] + "…"


def main():
    full = nx.read_graphml(GRAPHML)
    comps = sorted(nx.connected_components(full), key=len, reverse=True)
    g = full.subgraph(comps[0]).copy()
    print(f"full {full.number_of_nodes()}/{full.number_of_edges()} → "
          f"main component {g.number_of_nodes()}/{g.number_of_edges()}")

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
        desc = esc((data.get("description") or "")[:300])
        edges_svg.append(
            f'<path d="M{ax:.1f},{ay:.1f} Q{mx:.1f},{my:.1f} {bx:.1f},{by:.1f}" fill="none" '
            f'stroke="{EDGE}" stroke-opacity="0.6" stroke-width="1.6" marker-end="url(#arrow)">'
            f'<title>{esc(", ".join(kws))} — {desc}</title></path>'
        )
        if kw and len(kw) <= 8:
            kw_svg.append(
                f'<text x="{mx:.1f}" y="{my - 4:.1f}" text-anchor="middle" font-size="9.5" '
                f'font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{MUTED}" '
                f'paint-order="stroke" stroke="{BG}" stroke-width="3" stroke-linejoin="round">'
                f'{esc(kw)}</text>'
            )

    node_svg, label_svg = [], []
    for n, data in g.nodes(data=True):
        x, y = to_px(pos[n])
        etype = (data.get("entity_type") or "unknown").strip()
        r = radius[n]
        desc = esc((data.get("description") or "")[:400])
        node_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color_of.get(etype, PALETTE[0])}" '
            f'fill-opacity="0.93" stroke="{BG}" stroke-width="2">'
            f'<title>{esc(n)} [{esc(etype)}] — {desc}</title></circle>'
        )
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
            f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" font-size="11.5" '
            f'font-weight="500" font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{INK}" '
            f'fill-opacity="0.94" paint-order="stroke" stroke="{BG}" stroke-width="3.5" '
            f'stroke-linejoin="round">{esc(shorten(n))}</text>'
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
      font-family="ui-sans-serif,Segoe UI,Helvetica,Arial" fill="{MUTED}">{full.number_of_nodes()} entities · {full.number_of_edges()} relations extracted by LightRAG — largest connected component shown</text>
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
    print("saved", OUT)


if __name__ == "__main__":
    main()
